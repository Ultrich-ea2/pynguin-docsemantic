"""
PrimitiveFactory
----------------
Erstellt PrimitiveStatement-Objekte (Konstanten/Literale) für das
Docstring-Seeding.  Minimal-Implementierung für int, float, str, bool,
None.  Reicht für erste Tests – später erweitern.
"""

from __future__ import annotations
from typing import Any

from pynguin.testcase.statement import PrimitiveStatement
from pynguin.testcase.variablereference import VariableReference
from pynguin.testcase.defaulttestcase import DefaultTestCase


class PrimitiveFactory:
    """Erzeugt PrimitiveStatements für die GA-Population."""

    # ------------------------------------------------------------------ #
    # Kern-API
    # ------------------------------------------------------------------ #
    def create_primitive(self, value: Any) -> PrimitiveStatement:  # noqa: D401
        """Gibt ein neues PrimitiveStatement mit dem übergebenen Wert zurück."""
        test = DefaultTestCase()
        lhs = VariableReference(type(value), "val")
        stmt = PrimitiveStatement(lhs, value)
        test.add_statement(stmt)
        return stmt
    
        # ------------------------------------------------------------------ #
    # Funktions-Aufrufe (für Docstring-Seeding)
    # ------------------------------------------------------------------ #
    def create_function_call(self, func: callable, first_arg_index: int, arity: int):
        """        Baut ein FunctionStatement für ``func``.
        ``first_arg_index`` = Index der ersten Argument‐Variable
        ``arity``           = Anzahl der Argumente
        """
        from pynguin.testcase.statement import FunctionStatement  # local import

        lhs = VariableReference(func.__annotations__.get("return", object), "ret")
        args =[
            VariableReference(object, f"arg{i}") for i in range(first_arg_index, first_arg_index + arity)
        ]
        return FunctionStatement(lhs, func, args)


    # Convenience-Shortcuts -------------------------------------------- #
    def create_int(self, value: int) -> PrimitiveStatement:
        return self.create_primitive(value)

    def create_float(self, value: float) -> PrimitiveStatement:
        return self.create_primitive(value)

    def create_str(self, value: str) -> PrimitiveStatement:
        return self.create_primitive(value)

    def create_bool(self, value: bool) -> PrimitiveStatement:
        return self.create_primitive(value)

    def create_none(self) -> PrimitiveStatement:
        return self.create_primitive(None)
    
    @staticmethod
    def _value_for_constraint(op: str, rhs: str | int | float):
        """Gibt den Wert zurück, der für die Constraint-Erfüllung nötig ist."""
        rhs = int(rhs)
        if op in (">", ">="):
            return rhs + 1
        if op in ("<", "<="):
            return rhs - 1
        return rhs

    # ------------------------------------------------------------------ #
    # Universelle Helfer-Methode (neu)
    # ------------------------------------------------------------------ #
    @staticmethod
    def make_primitive(value: Any, tc: DefaultTestCase):
        """Erstellt das passende PrimitiveStatement für *value* in *tc*."""
        from pynguin.testcase import statement as stmt  # lokaler Import

        try:  # neuere Pynguin-Version
            return PrimitiveFactory().create_primitive(value, test_case=tc)  # type: ignore[arg-type]
        except TypeError:
            # Fallback für ältere API-Versionen
            if value is None:
                return stmt.NoneStatement(tc)
            if isinstance(value, bool):
                return stmt.BooleanPrimitiveStatement(tc, value)
            if isinstance(value, int):
                return stmt.IntPrimitiveStatement(tc, value)
            if isinstance(value, float):
                return stmt.FloatPrimitiveStatement(tc, value)
            if isinstance(value, bytes):
                return stmt.BytesPrimitiveStatement(tc, value)
            if isinstance(value, str):
                return stmt.StringPrimitiveStatement(tc, value)

            prim = PrimitiveFactory().create_primitive(value)
            prim.test_case = tc  # type: ignore[attr-defined]
            return prim
