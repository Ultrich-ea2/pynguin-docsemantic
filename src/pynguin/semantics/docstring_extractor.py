from __future__ import annotations
import inspect, re, textwrap, ast
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from docstring_parser import parse as parse_doc  # poetry add docstring-parser

@dataclass
class ParamSpec:
    name: str
    constraint: str | None = None           # z. B. "x > 0"
    example_values: Sequence[Any] = ()

@dataclass
class FunctionSemantics:
    qual_name: str
    params: Mapping[str, ParamSpec]
    raises: Sequence[str]

_CONSTRAINT_RX = re.compile(r"(?P<var>\w+)\s*(==|!=|>=|<=|>|<)\s*([-+\w\.]+)")

def _extract_examples(text: str) -> list[Any]:
    """Sucht nach literalen Beispiel-Argumenten in der Description."""
    values = []
    for lit in re.findall(r"\((.*?)\)", text):
        try:
            values.append(ast.literal_eval(lit.split(",")[0]))
        except Exception:
            pass
    return values

def semantics_for(obj) -> FunctionSemantics | None:
    raw = inspect.getdoc(obj) or ""
    if not raw:
        return None
    parsed = parse_doc(textwrap.dedent(raw))
    params: dict[str, ParamSpec] = {}
    for p in parsed.params:
        desc = p.description or ""
        m = _CONSTRAINT_RX.search(desc)
        params[p.arg_name] = ParamSpec(
            name=p.arg_name,
            constraint=m.group(0) if m else None,
            example_values=_extract_examples(desc),
        )
    return FunctionSemantics(
        qual_name=f"{obj.__module__}.{obj.__qualname__}",
        params=params,
        raises=[r.type_name for r in parsed.raises],
    )