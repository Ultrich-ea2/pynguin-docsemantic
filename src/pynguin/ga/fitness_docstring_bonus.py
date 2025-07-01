# SPDX-License-Identifier: MIT
"""
Kleiner Fitness-Boost für Chromosomen, die aus Docstring-Beispielen stammen.
Wenn ein Testfall (Chromosom) das Flag ``is_docstring_seed`` trägt, wird der
Fitness-Wert um einen konstanten Bonus reduziert (niedriger = besser).
"""

from __future__ import annotations

import logging

from pynguin.ga.computations import TestCaseFitnessFunction
from pynguin.ga.testcasechromosome import TestCaseChromosome

_LOG = logging.getLogger(__name__)


class DocstringSeedBonusFitness(TestCaseFitnessFunction):
    """Verleiht Docstring-Seeds einen (negativen) Bonus von ``BONUS``."""

    # Negativer Bonus – wird vom Gesamtfitnesswert abgezogen
    BONUS: float = 0.10


    def compute_fitness(self, chromosome: TestCaseChromosome) -> float:  # noqa: D401
        """Liefert -BONUS, falls das Chromosom ein Docstring-Seed ist, sonst 0."""
        if getattr(chromosome, "is_docstring_seed", False):
            _LOG.debug("Docstring-Bonus für %s angewendet (-%s)", chromosome, self.BONUS)
            return -self.BONUS
        return 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(BONUS={self.BONUS})"
