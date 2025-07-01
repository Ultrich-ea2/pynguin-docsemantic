from __future__ import annotations
from typing import Callable, Any

import pynguin.ga.computations as ff
import pynguin.ga.testsuitechromosome as tsc

def _locals_after_last_stmt(chromo) -> dict[str, Any]:
    last_stmt = chromo.test_case.statements[-1]
    return getattr(last_stmt, "locals", {})

class ParamConstraintViolationFitness(ff.TestSuiteFitnessFunction):
    """One instance per predicate – counts 1 if violated anywhere in the suite."""

    identifier: str = "ParamConstraintViolation"
    optimizes_to: str = "MIN"

    def __init__(self, executor, predicate: Callable[..., bool]) -> None:
        super().__init__(executor)
        self._predicate = predicate

    def compute_fitness(self, suite: tsc.TestSuiteChromosome) -> float:
        for case in suite.test_case_chromosomes:
            try:
                if self._predicate(**_locals_after_last_stmt(case)):
                    return 0.0
            except Exception:
                pass
        return 1.0
