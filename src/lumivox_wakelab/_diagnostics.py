"""Bounded processor-owned diagnostic event collection."""

from collections import deque

from .diagnostics import DiagnosticDrain, DiagnosticEvent


class DiagnosticCollector:
    def __init__(self, capacity: int, /) -> None:
        self._capacity = capacity
        self._events: deque[DiagnosticEvent] = deque()
        self._dropped = 0

    def append(self, event: DiagnosticEvent, /) -> None:
        if len(self._events) == self._capacity:
            self._events.popleft()
            self._dropped += 1
        self._events.append(event)

    def drain(self) -> DiagnosticDrain:
        result = DiagnosticDrain(tuple(self._events), self._dropped)
        self._events.clear()
        self._dropped = 0
        return result
