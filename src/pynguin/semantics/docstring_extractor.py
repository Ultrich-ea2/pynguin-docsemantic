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
class ReturnSpec:
    type_name: str | None = None
    description: str | None = None

@dataclass
class FunctionSemantics:
    qual_name: str
    params: Mapping[str, ParamSpec]
    raises: Sequence[str]
    returns: ReturnSpec | None = None

_CONSTRAINT_RX = re.compile(r"(==|!=|>=|<=|>|<)\s*([-+\w\.]+)")

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
        constraint: str | None = None
        m = _CONSTRAINT_RX.search(desc)
        if m:
            op, rhs = m.group(1), m.group(2).rstrip(".,;")
            constraint = f"{p.arg_name} {op} {rhs}"   # → „x > 0“

        params[p.arg_name] = ParamSpec(
            name=p.arg_name,
            constraint=constraint,
            example_values=_extract_examples(desc),
        )

    returns: ReturnSpec | None = None
    if parsed.returns:
        returns = ReturnSpec(
            type_name=parsed.returns.type_name,
            description=parsed.returns.description,
        )
        
    return FunctionSemantics(
        qual_name=f"{obj.__module__}.{obj.__qualname__}",
        params=params,
        raises=[r.type_name for r in parsed.raises],
    )