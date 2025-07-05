from __future__ import annotations
from typing import List, Protocol

import pynguin.ga.chromosome as chrom


class Initialiser(Protocol):
    """Basis-Interface für Initialisierer (Population Seeding)."""

    def __call__(self, size: int) -> list[chrom.Chromosome]: ...
