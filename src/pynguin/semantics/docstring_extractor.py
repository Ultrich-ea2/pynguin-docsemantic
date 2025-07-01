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
    type_name: str | None = None
    is_optional: bool = False
    default: Any = None

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
    constraints: Sequence[str] = ()
    description: str | None = None

# Regex patterns for parsing docstring elements
_CONSTRAINT_RX = re.compile(r"(==|!=|>=|<=|>|<)\s*([-+\w\.]+)")
_EXAMPLE_RX = re.compile(r"(?:e\.g\.|example|examples?|like|such as):?\s*([^,\n]+)")
_DEFAULT_RX = re.compile(r"(?:defaults? to|default:?)\s*([^,\n]+)")
_TYPE_RX = re.compile(r"(?:type|dtype):?\s*([^,\n]+)")
_RANGE_RX = re.compile(r"(?:range|between|from)\s+([^,\n]+?)\s+(?:to|and)\s+([^,\n]+)")

def _extract_examples(text: str) -> list[Any]:
    """Extract example values from parameter descriptions."""
    values = []
    
    # Extract examples from parentheses
    for lit in re.findall(r"\((.*?)\)", text):
        try:
            values.append(ast.literal_eval(lit.split(",")[0]))
        except Exception:
            pass
    
    # Extract examples from explicit markers
    for match in _EXAMPLE_RX.finditer(text):
        example_text = match.group(1).strip()
        try:
            for part in example_text.split(","):
                part = part.strip()
                if part:
                    values.append(ast.literal_eval(part))
        except Exception:
            values.append(example_text.strip("'\" "))
    
    return values

def _extract_constraints_from_description(text: str, param_name: str) -> str | None:
    """Extract constraints from parameter descriptions."""
    constraints = []
    
    # Extract comparison operators
    for match in _CONSTRAINT_RX.finditer(text):
        op, rhs = match.group(1), match.group(2).rstrip(".,;")
        constraints.append(f"{param_name} {op} {rhs}")
    
    # Extract range constraints
    for match in _RANGE_RX.finditer(text):
        lower, upper = match.group(1).strip(), match.group(2).strip()
        try:
            lower_val = ast.literal_eval(lower)
            upper_val = ast.literal_eval(upper)
            constraints.append(f"{lower_val} <= {param_name} <= {upper_val}")
        except Exception:
            constraints.append(f"{param_name} in range [{lower}, {upper}]")
    
    # Extract keyword-based constraints
    text_lower = text.lower()
    if "non-negative" in text_lower or ">= 0" in text:
        constraints.append(f"{param_name} >= 0")
    if "positive" in text_lower or "> 0" in text:
        constraints.append(f"{param_name} > 0")
    if "non-zero" in text_lower or "!= 0" in text:
        constraints.append(f"{param_name} != 0")
    if "integer" in text_lower or "int" in text_lower:
        constraints.append(f"isinstance({param_name}, int)")
    if "string" in text_lower or "str" in text_lower:
        constraints.append(f"isinstance({param_name}, str)")
    if "float" in text_lower or "double" in text_lower:        # python does not have double as a data type, but for docstrings we check both
        constraints.append(f"isinstance({param_name}, float)")
    return "; ".join(constraints) if constraints else None

def _extract_default_value(text: str) -> Any:
    """Extract default values from parameter descriptions."""
    for match in _DEFAULT_RX.finditer(text):
        default_text = match.group(1).strip()
        try:
            return ast.literal_eval(default_text)
        except Exception:
            return default_text.strip("'\" ")
    return None

def _extract_type_from_description(text: str) -> str | None:
    """Extract type information from parameter descriptions."""
    for match in _TYPE_RX.finditer(text):
        type_text = match.group(1).strip()
        return type_text.strip("'\" ")
    return None

def _extract_constraints_from_section(doc: str) -> list[str]:
    """Extract constraints from explicit sections in docstrings."""
    constraints = []
    
    constraint_patterns = [
        r"(?:Constraints|Preconditions|Requirements|Assumptions):\s*([\s\S]+?)(?:\n\S|$)",
        r"(?:Invariants|Conditions):\s*([\s\S]+?)(?:\n\S|$)",
        r"(?:Must|Should|Required):\s*([\s\S]+?)(?:\n\S|$)"
    ]
    
    for pattern in constraint_patterns:
        for match in re.finditer(pattern, doc, re.IGNORECASE):
            block = match.group(1)
            for line in block.splitlines():
                line = line.strip()
                if line and not line.startswith(":") and not line.startswith("-"):
                    line = re.sub(r"^\s*[-*•]\s*", "", line)
                    constraints.append(line)
    
    return constraints

def _extract_global_constraints(doc: str) -> list[str]:
    """Extract global constraints for the entire function."""
    global_constraints = []
    
    patterns = [
        r"(?:Global constraints|Function constraints|Overall constraints):\s*([\s\S]+?)(?:\n\S|$)",
        r"(?:All parameters|Parameters must|Inputs must):\s*([\s\S]+?)(?:\n\S|$)"
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, doc, re.IGNORECASE):
            block = match.group(1)
            for line in block.splitlines():
                line = line.strip()
                if line and not line.startswith(":") and not line.startswith("-"):
                    global_constraints.append(line)
    
    return global_constraints

def _map_global_constraints_to_parameters(global_constraints: list[str], param_names: list[str]) -> dict[str, str]:
    """Map global constraints to specific parameters."""
    param_constraints = {}
    
    for constraint in global_constraints:
        for param_name in param_names:
            import re
            pattern = r'\b' + re.escape(param_name) + r'\b'
            if re.search(pattern, constraint):
                if param_name not in param_constraints:
                    param_constraints[param_name] = []
                param_constraints[param_name].append(constraint)
    
    return {param: "; ".join(constraints) for param, constraints in param_constraints.items()}

def semantics_for(obj) -> FunctionSemantics | None:
    """Extract semantic information from docstrings."""
    raw = inspect.getdoc(obj) or ""
    if not raw:
        return None
    
    parsed = parse_doc(textwrap.dedent(raw))
    params: dict[str, ParamSpec] = {}
    
    param_names = [p.arg_name for p in parsed.params]
    
    global_constraints = _extract_constraints_from_section(raw)
    global_constraints.extend(_extract_global_constraints(raw))
    
    param_constraint_map = _map_global_constraints_to_parameters(global_constraints, param_names)
    
    for p in parsed.params:
        desc = p.description or ""
        
        constraint = _extract_constraints_from_description(desc, p.arg_name)
        
        if not constraint and p.arg_name in param_constraint_map:
            constraint = param_constraint_map[p.arg_name]
        
        example_values = _extract_examples(desc)
        default_value = _extract_default_value(desc)
        type_name = p.type_name or _extract_type_from_description(desc)
        
        params[p.arg_name] = ParamSpec(
            name=p.arg_name,
            constraint=constraint,
            example_values=example_values,
            type_name=type_name,
            is_optional=p.is_optional if hasattr(p, 'is_optional') else False,
            default=default_value,
        )

    returns: ReturnSpec | None = None
    if parsed.returns:
        returns = ReturnSpec(
            type_name=parsed.returns.type_name,
            description=parsed.returns.description,
        )
    
    result = FunctionSemantics(
        qual_name=f"{obj.__module__}.{obj.__qualname__}",
        params=params,
        raises=[r.type_name for r in parsed.raises],
        returns=returns,
        constraints=global_constraints,
        description=parsed.short_description,
    )
    
    return result
