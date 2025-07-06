# helpers.py
"""Kleine Hilfs-Klassen/Funktionen, auf die mehrere Module zugreifen können."""

from collections import defaultdict
from typing import Any, Dict, Iterator, Tuple

class _InfinityDefaultDict(defaultdict):
    """Defaultdict, das für unbekannte Keys +∞ liefert.

    Das verhindert KeyError-Probleme, wenn Pynguin während
    der Suche Fitness-Werte nachschlägt, die (noch) nie
    berechnet wurden.
    """

    def __init__(self, backing: Dict[Any, float] | None = None) -> None:
        super().__init__(lambda: float("inf"))
        if backing:
            self.update(backing)

    # Optional hübschere Repräsentation – nur kosmetisch
    def __repr__(self) -> str:
        class_name = type(self).__name__
        items = ", ".join(f"{k}: {v}" for k, v in self.items())
        return f"{class_name}({{{items}}}, default=∞)"

    # Iterator & copy verhalten sich wie gewohnt
    def items(self) -> Iterator[Tuple[Any, float]]:
        return super().items()

    def copy(self) -> "_InfinityDefaultDict":  # type: ignore[override]
        return _InfinityDefaultDict(dict(self))
