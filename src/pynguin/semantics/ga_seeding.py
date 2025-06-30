# SPDX-License-Identifier: MIT
"""Initialisiert die GA-Startpopulation mithilfe von Doctest-Beispielen."""

from __future__ import annotations

import ast
import importlib
import inspect
import logging
import textwrap
import traceback
from types import ModuleType
from typing import Any

from pynguin.ga.testcasechromosome import TestCaseChromosome
from pynguin.population.initialiser import Initialiser
from pynguin.testcase.defaulttestcase import DefaultTestCase
from pynguin.testcase.factory.primitivefactory import PrimitiveFactory
from pynguin.assertion.assertion import ObjectAssertion


from . import docstring_constraints as dsc

_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.DEBUG)



def _get_original_docstring(func) -> str | None:
    runtime_ds = inspect.getdoc(func)
    if runtime_ds and not runtime_ds.strip().startswith('{"code_object_id"'):
        return runtime_ds
    try:
        src = textwrap.dedent(inspect.getsource(func))
        node = ast.parse(src).body[0]
        return ast.get_docstring(node, clean=True)
    except (OSError, TypeError, SyntaxError, IndexError):
        return None



class DocstringSeedInitialiser(Initialiser):
    """Verwandelt Docstring-Beispiele in TestCaseChromosomen und fügt sie hinzu."""

    def __init__(self, population: list[TestCaseChromosome], test_cluster) -> None:
        super().__init__(population)
        self.population = population
        self._cluster = test_cluster

  
    def initialise(self) -> None:
        """Durchsuche alle SUT-Module nach Beispielen."""
        modules: set[ModuleType]
        if hasattr(self._cluster, "modules"):
            modules = set(self._cluster.modules)  # type: ignore[attr-defined]
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

    def _try_seed_from_docstring(self, func) -> None:  # noqa: C901
        doc = _get_original_docstring(func) or ""
        examples, preconds = dsc.parse_docstring(doc)
        if not examples:
            return

        env: dict[str, Any] = {m.__name__: m for m in getattr(self._cluster, "modules", [])}
        for m in list(env.values()):
            env.update(
                {
                    n: f
                    for n, f in inspect.getmembers(m, inspect.isfunction)
                    if f.__module__ == m.__name__ and not n.startswith("_")
                }
            )

       
        try:
            prim_factory = PrimitiveFactory(test_cluster=self._cluster) 
        except TypeError:
            prim_factory = PrimitiveFactory()  

        # Docstring-Beispiele in Testfälle umwandeln
        for call_src, expected_src in examples:
            try:
                call_ast = ast.parse(call_src, mode="eval")
                if not isinstance(call_ast.body, ast.Call):
                    _LOG.warning("Überspringe Beispiel, kein Funktionsaufruf: %s", call_src)
                    continue
                arg_vals = [self._eval_ast(a, env) for a in call_ast.body.args]

                # Verletzt das Beispiel eine Prä­kon­di­tion?
                if dsc.violates_preconditions(preconds, func, arg_vals, env):
                    _LOG.info("Beispiel %s ignoriert – Präcondition verletzt.", call_src)
                    continue

                # Testfall aufbauen 
                tc = TestCaseChromosome(DefaultTestCase(self._cluster))
                tc.prevent_minimization = True

                arg_refs = [
                    tc.test_case.add_variable_creating_statement(
                        PrimitiveFactory.make_primitive(v, tc.test_case)
                    )
                    for v in arg_vals
                ]
                first_idx = arg_refs[0].get_statement_position() if arg_refs else 0

                call_stmt = prim_factory.create_function_call(
                    func, first_arg_index=first_idx, arity=len(arg_vals)
                )
                tc.test_case.add_statement(call_stmt)

                if expected_src.strip():
                    exp_val = self._safe_eval(expected_src, env)
                    PrimitiveFactory.make_primitive(exp_val, tc.test_case)
                    ret_var = getattr(call_stmt, "ret_val", None) or getattr(
                        call_stmt, "get_return_variable", lambda: None
                    )()
                    call_stmt.add_assertion(ObjectAssertion(ret_var, exp_val))

                tc.is_docstring_seed = True  # Markierung für Docstring-Seeds
                tc._do_not_mutate = True     # Verhindert Mutation von Seeds

                self.population.append(tc)
                _LOG.debug("Docstring-Seed erzeugt: %s", tc.test_case)

            except Exception:
                _LOG.error(
                    "Fehler beim Bauen des Seeds für %s:\n%s",
                    call_src,
                    textwrap.indent(traceback.format_exc(), "    "),
                )

        _LOG.info(
            "→ Insgesamt %d Docstring-Seeds in der Startpopulation",
            sum(1 for ind in self.population if getattr(ind, "is_docstring_seed", False)),
        )

    @staticmethod
    def _eval_ast(node: ast.AST, env: dict[str, Any]) -> Any:
        return eval(compile(ast.Expression(node), "<docstring>", "eval"), env)  # noqa: S307

    @staticmethod
    def _safe_eval(src: str, env: dict[str, Any]) -> Any:
        try:
            return eval(src, env)  # noqa: S307
        except Exception:
            return src.strip()
