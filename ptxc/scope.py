"""Lexical scopes for IR names and for already-computed values.

Names resolve outward, like C block scoping: a loop or branch body opens a scope, names first
defined inside it disappear when it closes, and assignments to an outer name (an accumulator,
a reduction state) write that name's existing register.

The value cache is scoped the same way. It maps a pure computation (an index multiply, an
address, a read-only load) to the register that already holds its result, so repeating it
costs nothing. Because the cache closes with its scope, a value computed inside a loop or
branch body is never reused once that body has ended.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager

from .errors import CompileError


class Scope:
    def __init__(self, parent: Scope | None) -> None:
        self.parent = parent
        self.names: dict[str, object] = {}
        self.values: dict[Hashable, object] = {}


class Scopes:
    def __init__(self) -> None:
        self.current = Scope(None)

    @contextmanager
    def nested(self) -> Iterator[None]:
        self.current = Scope(self.current)
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

    def memo(self, key: Hashable, compute: Callable[[], object]) -> object:
        """The cached result of `key`, or `compute()` (which emits it) cached in this scope."""
        scope = self.current
        while scope is not None:
            if key in scope.values:
                return scope.values[key]
            scope = scope.parent
        value = compute()
        self.current.values[key] = value
        return value
