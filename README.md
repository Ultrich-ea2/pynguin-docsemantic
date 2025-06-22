<!--
  ~ This file is part of Pynguin.
  ~
  ~ SPDX-FileCopyrightText: 2019–2025 Pynguin Contributors
  ~
  ~ SPDX-License-Identifier: MIT
  -->
# Pynguin Docstring Extension

## Topic

Automated Test Case Generation

## Overview

This project extends Pynguin, an automated unit test generation tool for Python, by integrating docstring-parser. The extension enables Pynguin to leverage semantic information from Python docstrings to generate more meaningful and effective tests.

## Tool

[Pynguin](https://github.com/se2p/pynguin)

## Research Question

Can the integration of semantic information from Python docstrings into Pynguin significantly improve the quality of generated tests regarding fault detection, mutation score, and branch coverage?

## Motivation

In recent years, automated test case generation has made significant progress. Tools like Pynguin analyze source code both statically and dynamically to achieve structural test goals such as statement or branch coverage. However, the semantic layer of code — information documented in natural language within docstrings — has been largely neglected. Docstrings often contain precise specifications about expected input values and constraints (e.g., “parameter x must be > 0”), hints about possible exceptions, side effects, semantic contracts (e.g., “x must not be null”), or examples of typical valid usage.

If these semantic details are systematically integrated into the testing process, test cases can be deliberately constructed to cover important semantic scenarios rather than relying solely on random or structural analysis.

The central hypothesis of this project is that semantic information from docstrings can improve the quality of automatically generated tests in Pynguin. Quality here refers to:

- **Fault Detection Potential:** Can semantically informed tests uncover subtler or domain-specific faults, such as precondition violations or unexpected exceptions?

- **Mutation Score:** Does including pre- and post-conditions from docstrings lead to higher mutation coverage, measured by the proportion of killed mutants among all generated mutants?

- **Test Coverage:** Are additional semantic details enabling coverage of paths or value ranges that pure structural analysis would miss?

For example, if a docstring specifies that a parameter must be an integer greater than zero, test generation can explicitly target boundary cases (e.g., 0, 1, -1) instead of finding them randomly, helping to identify faults like division by zero or invalid indices more reliably.

Many Python projects already include extensive PEP 257-compliant docstrings containing important conceptual guidelines. Thus, integrating this information is not only scientifically interesting but also offers practical value for developers by bringing test generation closer to domain logic.
