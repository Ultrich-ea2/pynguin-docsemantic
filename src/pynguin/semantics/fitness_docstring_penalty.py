# SPDX-License-Identifier: MIT
"""
Docstring-Constraint-Penalty
============================
Bestraft Testfälle, die während der Ausführung eine in einem Docstring
definierte Prä- oder Post-Condition verletzen.

Die Fitness ist **0.0**, wenn alle Constraints erfüllt wurden, andernfalls
wird pro Verletzung ein fester Malus addiert (Standard: 1.0).
"""

from __future__ import annotations
from typing import Any, Iterable

from pynguin.ga.testcasechromosome import TestCaseChromosome
from pynguin.ga.fitness.fitnessfunction import TestFitnessFunction


class DocstringConstraintPenalty(TestFitnessFunction):
    """Einfacher additiver Malus für Constraint-Verletzungen."""

    #: Wie viel jede Verletzung kostet.
    penalty_per_violation: float = 1.0

    def compute_fitness(self, individual: TestCaseChromosome) -> list[float]:
        """Zählt pro Statement alle gespeicherten Constraint-Flags."""
        violations: int = 0
        for stmt in individual.test_case.statements:
            # Docstring-Seeding markiert verletzte Constraints mit diesem Attribut
            violations += getattr(stmt, "_constraint_violated", 0)
        return [violations * self.penalty_per_violation]

    # ------------------------------------------------------------------ #
    # Pflicht-Eigenschaften für Pynguin
    # ------------------------------------------------------------------ #
    @property
    def num_objectives(self) -> int:   # noqa: D401
        return 1

    def _copy(self) -> "DocstringConstraintPenalty":  # noqa: D401
        clone = DocstringConstraintPenalty()
        clone.penalty_per_violation = self.penalty_per_violation
        return clone
