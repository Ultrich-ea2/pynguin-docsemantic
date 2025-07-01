# SPDX-License-Identifier: MIT
"""
Semantik-Erweiterungen für Pynguin.

*   Extrahiert Doctest-Beispiele aus Docstrings und legt sie als
    unveränderliche Seeds in der Startpopulation an.
*   Monkey-Patch: Chromosomen, die das Flag ``_do_not_mutate`` tragen,
    werden vom GA-Mutator übersprungen.
*   Automatische Registrierung des Observers, sobald ein GA-Algorithmus
    instanziiert wird – aktiviert durch ``Configuration.enable_seed_examples``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pynguin.ga.testcasechromosome import TestCaseChromosome as _TCC

from .ga_seeding import DocstringSeedInitialiser  # noqa: F401  (Re-Export)
from .docstring_seeding_observer import DocstringSeedingObserver  # noqa: F401
from .docstring_constraints import extract_examples, extract_preconditions, parse_docstring


__all__ = [
    # öffentliche API
    "DocstringSeedInitialiser",
    "DocstringSeedingObserver",
    "FunctionSemantics",
    "ParamSpec",
    "semantics_for",
]

# --------------------------------------------------------------------------- #
#   1.  Monkey-Patch: Seeds dürfen nicht mutieren
# --------------------------------------------------------------------------- #
_original_mutate = _TCC.mutate


def _seed_safe_mutate(self: _TCC, *args, **kwargs):  # noqa: D401
    """Überschreibt das GA-Mutator-Verhalten für Seed-Chromosomen."""
    if getattr(self, "_do_not_mutate", False):
        return
    _original_mutate(self, *args, **kwargs)


_TCC.mutate = _seed_safe_mutate  # type: ignore[attr-defined]

# --------------------------------------------------------------------------- #
#   2.  GA-Observer automatisch einschleusen
# --------------------------------------------------------------------------- #
if TYPE_CHECKING:  # Import nur für Typ-Checker
    from pynguin.configuration import Configuration
    from pynguin.ga.algorithms.generationalgorithm import GenerationalAlgorithm


def _auto_register_observer():
    """Patched‐Init für alle GA-Algorithmen.

    Fügt bei Bedarf den :class:`DocstringSeedingObserver` hinzu.
    """
    from pynguin.ga.algorithms.generationalgorithm import (
        GenerationAlgorithm as _GA,
    )

    if getattr(_GA, "_semantics_patch_done", False):
        return  # Patch nur einmal ausführen

    _GA._semantics_patch_done = True
    _orig_init = _GA.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)  # type: ignore[misc]

        cfg = getattr(self, "configuration", None)
        if cfg and getattr(cfg, "enable_seed_examples", False):
            observer = DocstringSeedingObserver(self, getattr(self, "test_cluster", None))
            # add_observer existiert bei allen Generation-Algorithmen
            self.add_search_observer(observer)

    _GA.__init__ = _patched_init  # type: ignore[assignment]

# --- Ergänzung, damit DocstringSeedInitialiser Module findet -----------------
import importlib
from types import ModuleType
from pynguin.analyses.module import ModuleTestCluster

def _gather_modules(self) -> list[ModuleType]:
    """Sammelt alle Python-Module, in denen die UUT-Aufrufe definiert sind."""
    mods: set[ModuleType] = set()
    for acc in self.accessible_objects_under_test:
        if hasattr(acc, "callable"):                   # nur Functions/Methoden
            mods.add(importlib.import_module(acc.callable.__module__))
    return list(mods)

# Property dynamisch einhängen (Monkey-Patch)
ModuleTestCluster.modules = property(_gather_modules)


_auto_register_observer()

# --------------------------------------------------------------------------- #
#   3.  Logging
# --------------------------------------------------------------------------- #
logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.getLogger("pynguin.semantics").setLevel(logging.DEBUG)
