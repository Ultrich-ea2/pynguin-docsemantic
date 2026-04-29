<!--
SPDX-FileCopyrightText: 2019–2026 Pynguin Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Pynguin Semantic Extension

Pynguin Semantic Extension adds document aware guidance to Pynguin test generation.
It reads natural language constraints from Python docstrings and converts them into useful signals for search based test generation.
The extension is designed for stable behavior, clear configuration, and repeatable results.

## Purpose

This extension helps Pynguin discover more meaningful tests when source code includes functional expectations in docstrings.
The semantic layer can improve search guidance in areas where structural coverage alone is not enough.

## Core Capabilities

1. Docstring constraint extraction.
2. Semantic seeding for the initial population.
3. Fitness guidance with semantic penalties and bonuses.
4. Integration with the existing search workflow.
5. Predictable execution with testable modules and dedicated unit tests.

## Architecture Overview

The semantic extension is implemented in the `src/pynguin/semantics` package.

| Module | Responsibility |
| :--- | :--- |
| `docstring_constraints.py` | Parses and represents constraints from docstrings |
| `docstring_seeding_observer.py` | Connects semantic extraction to runtime seeding events |
| `ga_seeding.py` | Supplies semantic aware seeds for evolutionary search |
| `fitness_docstring_penalty.py` | Applies penalties when generated behavior violates semantic expectations |
| `helpers.py` | Shared semantic utilities |

## Requirements

1. Python 3.10.
2. A supported Pynguin installation from this repository.
3. A module under test that includes meaningful docstrings.

## Installation

Install project dependencies with Poetry.

```bash
poetry install
```

Run Pynguin inside the Poetry environment.

```bash
poetry run pynguin --help
```

## Quick Start

Use the same project path and module parameters as standard Pynguin.
The semantic extension is loaded through the repository codebase and participates in the generation process.

```bash
poetry run pynguin \
  --project-path /path/to/project \
  --output-path /path/to/generated_tests \
  --module-name package.module
```

## Stability Notes

This extension was implemented to support deterministic workflows as far as the underlying test generation strategy allows.
For best stability, keep a fixed Python version, fixed dependencies, and fixed Pynguin configuration.

## Validation

Run focused tests for semantic behavior.

```bash
poetry run pytest src/pynguin/tests/semantics/test_docstring_extractor.py
```

Run the complete test suite when preparing integration changes.

```bash
poetry run pytest
```

## Safety

Pynguin executes the module under test.
Only run it on trusted code and prefer isolated environments such as containers.

## Documentation

User documentation is available in the `docs` directory and at Read the Docs.
Start with `docs/user/quickstart.rst` for end to end usage.

## Contributing

1. Install dependencies with Poetry.
2. Implement your change with tests.
3. Run quality checks.
4. Open a pull request with a clear summary.

## License

This project is licensed under the MIT License.
See `LICENSE.rst` for details.
