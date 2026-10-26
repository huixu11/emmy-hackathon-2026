"""Walking the Kernel IR tree: every node under a statement, the names a subtree refers to,
and the statements of a body in order."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import TreeHalve
from emmy.compiler.ir.stmt.blocks import Cond
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Let, Load, Select, SelectBranch


def walk(node) -> Iterator:
    """Every IR node (statement or expression) under `node`, parents before children."""
    stack = [node]
    while stack:
        item = stack.pop()
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            yield item
            stack.extend(reversed([getattr(item, f.name) for f in dataclasses.fields(item)]))
        elif isinstance(item, (tuple, list)):
            stack.extend(reversed(item))


def string_leaves(node) -> Iterator[str]:
    """The string fields of one IR node, looking inside tuples (names, args, values) and the
    records they hold (a Select's branches)."""
    for f in dataclasses.fields(node):
        value = getattr(node, f.name)
        if isinstance(value, str):
            yield value
        elif isinstance(value, tuple):
            for v in value:
                if isinstance(v, str):
                    yield v
                elif isinstance(v, SelectBranch):
                    yield from string_leaves(v)


def names_used(node) -> set[str]:
    """Every name an IR subtree refers to: its Vars, plus TreeHalve's tid variable (a string)."""
    found: set[str] = set()
    for item in walk(node):
        if isinstance(item, Var):
            found.add(item.name)
        elif isinstance(item, TreeHalve):
            found.add(item.tid_var)
    return found


def statements(body) -> Iterator:
    """Every statement of a body, those in nested branches and loops included."""
    for stmt in body:
        yield stmt
        for field in ("body", "else_body"):
            yield from statements(getattr(stmt, field, None) or ())


def names_defined(stmt) -> tuple[str, ...]:
    """The names a plain statement defines."""
    if isinstance(stmt, Load):
        return tuple(stmt.names)
    if isinstance(stmt, (Assign, Accum, Let, Select)):
        return (stmt.name,)
    return ()


def straight_line(body) -> Iterator:
    """The statements of a body, descending into branches but not into nested loops."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, Cond):
            yield from straight_line(stmt.body)
            yield from straight_line(stmt.else_body)
