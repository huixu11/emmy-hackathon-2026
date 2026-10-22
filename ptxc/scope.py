"""Lexical scopes for IR names and for already-computed values.

Names resolve outward, like C block scoping: a loop or branch body opens a scope, names first
defined inside it disappear when it closes, and assignments to an outer name (an accumulator,
a reduction state) write that name's existing register.

The value cache is scoped the same way. It maps a pure computation (an index multiply, an
address, a read-only load) to the register that already holds its result, so repeating it
costs nothing. A value lives in the scope where it was computed, so once a loop or branch
body ends its values are never reused.

A scope opened with a `hoist` mark lets pure integer instructions move out of it
(IndexCompiler.derive): one whose operands all come from enclosing scopes is placed at the
mark, a point in the enclosing code that every later instruction of that code comes after
(a loop's preheader, the spot before a branch), and is cached there, so it is computed once
instead of once per trip or copy.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager

from .errors import CompileError
from .ptx import Mark


class Scope:
    def __init__(self, parent: Scope | None, hoist: Mark | None = None) -> None:
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.hoist = hoist
        self.names: dict[str, object] = {}
        self.values: dict[Hashable, object] = {}


class Scopes:
    def __init__(self) -> None:
        self.current = Scope(None)

    @contextmanager
    def nested(self, hoist: Mark | None = None) -> Iterator[None]:
        self.current = Scope(self.current, hoist)
        try:
            yield
        finally:
            self.current = self.current.parent

    def define(self, name: str, value: object) -> None:
        self.current.names[name] = value

    def find(self, name: str) -> object | None:
        scope = self.current
        while scope is not None:
            if name in scope.names:
                return scope.names[name]
            scope = scope.parent
        return None

    def lookup(self, name: str) -> object:
        value = self.find(name)
        if value is None:
            raise CompileError(f"IR name {name!r} is used before it is defined")
        return value

    def cached(self, key: Hashable) -> object | None:
        scope = self.current
        while scope is not None:
            if key in scope.values:
                return scope.values[key]
            scope = scope.parent
        return None

    def memo(self, key: Hashable, compute: Callable[[], object]) -> object:
        """The cached result of `key`, or `compute()` (which emits it) cached in this scope."""
        value = self.cached(key)
        if value is None:
            value = compute()
            self.current.values[key] = value
        return value

    def placement(self, depth: int) -> tuple[Scope, Mark | None]:
        """Where a value needing only scopes up to `depth` can go: the outermost scope reachable
        through hoisting scopes, and the mark to emit it at (None: right here)."""
        scope, at = self.current, None
        while scope.depth > depth and scope.hoist is not None:
            at, scope = scope.hoist, scope.parent
        return scope, at
