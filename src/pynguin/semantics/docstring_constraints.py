# SPDX-License-Identifier: MIT
"""
docstring_constraints
=====================

Hilfsfunktionen zum Parsen von Docstrings:

* Extract Doctest-Beispiele:           ``>>> add(3, 5)`` + erwarteter Wert  
* Extract *Preconditions*-Abschnitt:  einfache Zeilen wie ``x > 0``  
* Prüfen, ob gegebene Argumente eine der Prä­kon­di­tionen verletzen.

Bewusst ohne direkte Pynguin-Abhängigkeit gehalten.
"""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any

_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.DEBUG)

# Schlüsselwörter, die einen Preconditions-Block einleiten dürfen
_PRE_HEADERS = {"preconditions", "precondition", "requires", "require"}


# --------------------------------------------------------------------------- #
# Beispiel-Extraktion
# --------------------------------------------------------------------------- #
def extract_examples(doc: str) -> list[tuple[str, str]]:
    """Liefert alle Doctest-Beispiele ``(call_src, expected_src)``."""
    examples: list[tuple[str, str]] = []
    lines = doc.splitlines()

    for i, line in enumerate(lines):
        if line.strip().startswith(">>>"):
            call_src = line.strip()[4:]
            expected = ""
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt and not nxt.startswith(">>>") and not nxt.endswith(":"):
                    expected = nxt
            examples.append((call_src, expected))

    _LOG.debug("extract_examples: %s", examples)
    return examples


# --------------------------------------------------------------------------- #
# Preconditions-Extraktion
# --------------------------------------------------------------------------- #
def extract_preconditions(doc: str) -> list[str]:
    """Sucht einen »Preconditions«-Block und gibt dessen Zeilen zurück."""
    conds: list[str] = []
    lines = doc.splitlines()

    for idx, line in enumerate(lines):
        hdr = line.lower().strip().rstrip(":")
        if hdr in _PRE_HEADERS:
            for ln in lines[idx + 1 :]:
                clean = ln.strip()
                if not clean:
                    break
                # Markdown-Bullets o. Ä. abschneiden
                clean = re.sub(r"^[\-\*\•]\s*", "", clean)
                conds.append(clean)
            break

    _LOG.debug("extract_preconditions: %s", conds)
    return conds


def parse_docstring(doc: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Convenience-Wrapper: (examples, preconditions) zurückgeben."""
    return extract_examples(doc), extract_preconditions(doc)


# --------------------------------------------------------------------------- #
# Constraint-Auswertung
# --------------------------------------------------------------------------- #
def violates_preconditions(
    conds: list[str],
    func: Any,
    arg_values: list[Any],
    env: dict[str, Any] | None = None,
) -> bool:
    """Prüft, ob *arg_values* eine der *conds* verletzt.

    *conds* müssen valide Python-Ausdrücke sein (z.B. ``x > 0``).
    """
    if not conds:
        return False

    sig = inspect.signature(func)
    locals_ctx = dict(zip(sig.parameters, arg_values))
    globals_ctx = env or {}

    for cond in conds:
        try:
            ok = bool(eval(cond, globals_ctx, locals_ctx))  # noqa: S307
        except Exception as exc:  # pragma: no cover
            _LOG.warning("Kann Präcondition '%s' nicht auswerten (%s) – ignoriere sie.", cond, exc)
            continue
        if not ok:
            _LOG.debug("Präcondition '%s' verletzt (args=%s)", cond, locals_ctx)
            return True
    return False
