#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2025 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
from pynguin.semantics.docstring_extractor import semantics_for

def test_constraints_and_examples():
    def foo(x: int, y: int):
        """Demo.

        Args:
            x: must be > 0. Example: foo(3, 0)
            y: any int
        """
    sem = semantics_for(foo)
    assert sem is not None
    assert sem.params["x"].constraint == "x > 0"
    assert 3 in sem.params["x"].example_values
