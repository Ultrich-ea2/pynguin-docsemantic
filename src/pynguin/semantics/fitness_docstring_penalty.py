# SPDX-License-Identifier: MIT
"""Penalises violations of pre/post-conditions that were parsed from docstrings."""

from __future__ import annotations
from typing import TYPE_CHECKING

import pynguin.ga.computations as ff
import pynguin.ga.testcasechromosome as tsc  # <-- für Typ-Check / Autovervollständigung
from math import inf
import logging

if TYPE_CHECKING:
    from pynguin.testcase.testcase import TestCase

_LOG = logging.getLogger(__name__)    

# --------------------------------------------------------------------------- #
class DocstringPenaltyFitness(ff.TestCaseFitnessFunction):
    """Returns the total number of violated predicates inside the whole test-suite."""

    identifier: str = "DocstringPenalty"
    optimizes_to: str = "MIN"

    # executor wird von Pynguin reingereicht
    def __init__(self, executor) -> None:
        super().__init__(executor, code_object_id=None)

    # -------- core API ------------------------------------------------------ #
    def compute_fitness(self, suite: tsc.TestCaseChromosome) -> float:  # noqa: D401
        any_violation = False

        for case_chromo in suite.test_case_chromosomes:
            violations = self._violations(case_chromo.test_case)
            if violations > 0:
                any_violation = True
                _LOG.debug(
                    "[Penalty] %s -> %d violations -> penalty=%s",
                    case_chromo.test_case.get_name(),
                    violations,
                    "inf" ,
                )
                case_chromo.set_fitness(self.identifier, inf)
        return inf if any_violation else 0.0            

    # -------- helper -------------------------------------------------------- #
    @staticmethod
    def _violations(tc: "TestCase") -> int:
        preds_map = getattr(tc.test_cluster, "constraint_preds", {})
        count = 0
        for stmt in tc.statements:
            acc = stmt.accessible_object()
            if acc is None:
                continue

            target = getattr(acc, "original_callable", acc)
            if target not in preds_map:
                continue
            locs = _locals_after(stmt, tc)
            for pred in preds_map[target]:
                try:
                    if not pred(**locs):
                        count += 1
                except Exception:
                    count += 1
        return count


def _locals_after(stmt, tc):
    locs = {name: tc.get_object(var) for name, var in getattr(stmt, "args", {}).items()}
    if getattr(stmt, "ret_val", None):
        locs["_ret"] = tc.get_object(stmt.ret_val)
    return locs
