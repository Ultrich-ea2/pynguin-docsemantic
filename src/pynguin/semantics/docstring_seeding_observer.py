# SPDX-License-Identifier: MIT
"""Observer, der nach dem Erzeugen der Initialpopulation Docstring-Seeds einfügt."""

from __future__ import annotations

import logging
from typing import Any, List

from pynguin.ga.searchobserver import SearchObserver
from pynguin.ga.algorithms.generationalgorithm import GenerationAlgorithm

from .ga_seeding import DocstringSeedInitialiser

_LOG = logging.getLogger(__name__)
_LOG.addHandler(logging.NullHandler())


class DocstringSeedingObserver(SearchObserver):
    """Fügt Docstring-Seeds **einmal** direkt nach Population-Erzeugung ein."""

    def __init__(self, algorithm: GenerationAlgorithm, test_cluster=None) -> None:
        self._algorithm = algorithm
        self._cluster = (
            test_cluster
            or getattr(algorithm, "test_cluster", None)
            or getattr(algorithm, "_test_cluster", None)
        )
        self._done = False


    def before_search_start(self, _):  # noqa: D401
        self._try_seeding()

    def before_first_search_iteration(self, _):  # noqa: D401
        self._try_seeding()

    def after_search_iteration(self, _):  # noqa: D401
        pass

    def after_search_finish(self, *_):  # noqa: D401
        if hasattr(self._algorithm, "_seed_counter"):
            _LOG.info("✱ %d Docstring-Seeds wurden übernommen.",
                      self._algorithm._seed_counter)


    def _try_seeding(self) -> None:
        if self._done:
            return

        population: List[Any] | None = getattr(self._algorithm, "population", None) or getattr(
            self._algorithm, "_population", None
        )
        if not population or not self._cluster:
            _LOG.debug("Seeding wartet … pop=%s cluster=%s", bool(population), bool(self._cluster))
            return

        _LOG.info("Docstring-Seeding für Population mit %d Individuen …", len(population))
        DocstringSeedInitialiser(population, self._cluster).initialise()

        seed_cnt = sum(1 for ind in population if getattr(ind, "is_docstring_seed", False))
        self._algorithm._seed_counter = seed_cnt
        _LOG.info("→ %d Seeds hinzugefügt", seed_cnt)

        self._done = True
