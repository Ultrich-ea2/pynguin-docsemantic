# SPDX-License-Identifier: MIT
"""Initialisiert die GA-Start­population mithilfe von Doctest-Beispielen."""

from __future__ import annotations

import ast
import importlib
import inspect
import logging
import textwrap
import traceback
from types import ModuleType
from typing import Any

from collections import defaultdict

from pynguin.ga.testcasechromosome import TestCaseChromosome
from pynguin.population.initialiser import Initialiser
from pynguin.testcase.defaulttestcase import DefaultTestCase
from pynguin.testcase.factory.primitivefactory import PrimitiveFactory
from pynguin.assertion.assertion import ObjectAssertion
from pynguin.analyses.constants import EmptyConstantProvider
from pynguin.testcase.testfactory import TestFactory

from . import docstring_constraints as dsc

_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.DEBUG)


# --------------------------------------------------------------------------- #
def _get_original_docstring(func) -> str | None:
    """Versucht, die *ursprüngliche* Docstring zu ermitteln.

    Wenn die Funktion bereits gepatcht oder generiert wurde, kann die Runtime-
    Docstring mit JSON-Headern versehen sein.  Wir greifen dann auf den Source
    Code zurück.
    """
    runtime_ds = inspect.getdoc(func)
    if runtime_ds and not runtime_ds.strip().startswith('{"code_object_id"'):
        return runtime_ds
    try:
        src = textwrap.dedent(inspect.getsource(func))
        node = ast.parse(src).body[0]
        return ast.get_docstring(node, clean=True)
    except (OSError, TypeError, SyntaxError, IndexError):
        return None


# --------------------------------------------------------------------------- #
class DocstringSeedInitialiser(Initialiser):
    """Extrahiert Beispiele aus Docstrings und wandelt sie in Testfälle um."""

    def __init__(self, population: list[TestCaseChromosome], test_cluster) -> None:
        super().__init__(population)
        self.population = population
        self._cluster = test_cluster

    # ......................................................... Initialiser API
    def initialise(self) -> None:  # noqa: D401
        """Gehe alle relevanten SUT-Module durch und baue Seeds."""
        if hasattr(self._cluster, "modules"):
            modules: set[ModuleType] = set(self._cluster.modules)  # type: ignore[attr-defined]
        else:
            modules = {
                importlib.import_module(acc.callable.__module__)  # type: ignore[attr-defined]
                for acc in getattr(self._cluster, "accessible_objects_under_test", [])
                if hasattr(acc, "callable")
            }

        for mod in modules:
            for name, func in inspect.getmembers(mod, inspect.isfunction):
                if func.__module__ == mod.__name__ and not name.startswith("_"):
                    _LOG.debug("Untersuche Funktion %s in Modul %s", name, mod.__name__)
                    self._try_seed_from_docstring(func)

    # ......................................................... Kernlogik
    def _try_seed_from_docstring(self, func) -> None:  # noqa: C901
        doc = _get_original_docstring(func) or ""
        examples, preconds = dsc.parse_docstring(doc)
        if not examples:
            return

        # ▶︎ Namen, die in Beispielen verwendet werden, müssen aufgelöst werden
        env: dict[str, Any] = {m.__name__: m for m in getattr(self._cluster, "modules", [])}
        for m in list(env.values()):
            env.update(
                {
                    n: f
                    for n, f in inspect.getmembers(m, inspect.isfunction)
                    if f.__module__ == m.__name__ and not n.startswith("_")
                }
            )

        for call_src, expected_src in examples:
            try:
                call_ast = ast.parse(call_src, mode="eval")
                if not isinstance(call_ast.body, ast.Call):
                    _LOG.warning("Überspringe Beispiel, kein Funktionsaufruf: %s", call_src)
                    continue
                arg_vals = [self._eval_ast(a, env) for a in call_ast.body.args]

                # Prä­kondi­tionen aus Docstring?
                if dsc.violates_preconditions(preconds, func, arg_vals, env):
                    _LOG.info("Beispiel %s ignoriert – Präcondition verletzt.", call_src)
                    continue

                # -------- Testfall-Gerüst aufbauen --------
                tf = TestFactory(test_cluster=self._cluster, constant_provider=EmptyConstantProvider())
                tc = TestCaseChromosome(DefaultTestCase(self._cluster), test_factory=tf)
                tc.prevent_minimization = True

                # Argumente als Primitive anlegen
                arg_refs = [
                    tc.test_case.add_variable_creating_statement(
                        PrimitiveFactory.make_primitive(v, tc.test_case)
                    )
                    for v in arg_vals
                ]

                # Passende AccessibleFunction im Cluster finden
                accessible_obj = None
                for acc in getattr(self._cluster, "accessible_objects_under_test", []):
                    if getattr(acc, "original_callable", None) is func or getattr(acc, "callable", None) is func:
                        accessible_obj = acc
                        break
                if accessible_obj is None:
                    _LOG.warning("Funktion %s nicht in TestCluster gefunden – Seed übersprungen.", func.__name__)
                    continue

                # Funktionsaufruf in den Test einfügen
                ret_var = tf.add_function(tc.test_case, accessible_obj)

                # Parameter-Binding korrigieren
                from pynguin.testcase.statement import FunctionStatement

                stmt_pos = ret_var.get_statement_position()
                func_stmt = tc.test_case.get_statement(stmt_pos)
                assert isinstance(func_stmt, FunctionStatement)

                param_names = list(inspect.signature(func).parameters)
                for name, ref in zip(param_names, arg_refs):
                    func_stmt._args[name] = ref  # noqa: SLF001  (internes Attribut)

                # Erwarteten Rückgabewert überprüfen
                if expected_src.strip():
                    exp_val = self._safe_eval(expected_src, env)
                    # ▶︎ Direkt Literal verwenden – kein VariableReference mehr!
                    func_stmt.add_assertion(ObjectAssertion(ret_var, exp_val))

                # ------- Markierungen & Fitness-Cache --------
                tc.is_docstring_seed = True
                tc._do_not_mutate = True
                tc.changed = True

                _cache = tc.computation_cache._fitness_cache
                tc.computation_cache._fitness_cache = defaultdict(lambda: float("inf"), _cache)

                self.population.append(tc)
                _LOG.debug("Docstring-Seed erzeugt: %s", tc.test_case)

            except Exception:
                _LOG.error(
                    "Fehler beim Bauen des Seeds für %s:\n%s",
                    call_src,
                    textwrap.indent(traceback.format_exc(), "    "),
                )

        _LOG.info(
            "→ %d Docstring-Seeds in Startpopulation",
            sum(1 for ind in self.population if getattr(ind, "is_docstring_seed", False)),
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _eval_ast(node: ast.AST, env: dict[str, Any]) -> Any:
        return eval(compile(ast.Expression(node), "<docstring>", "eval"), env)  # noqa: S307

    @staticmethod
    def _safe_eval(src: str, env: dict[str, Any]) -> Any:
        try:
            return eval(src, env)  # noqa: S307
        except Exception:
            # Fällt eval() fehl (z.B. bei Zeichenketten ohne Quotes),
            # behandeln wir den Wert als String-Literal
            return src.strip()
