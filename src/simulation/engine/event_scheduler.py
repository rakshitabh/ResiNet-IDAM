"""Chronological event scheduler managing due queues and event states."""

from __future__ import annotations

from typing import Any
from src.simulation.models.simulation_event import EventState, SimulationEvent


class EventScheduler:
    """Manages ordered event scheduling and state transitions."""

    def __init__(self, events: list[SimulationEvent]) -> None:
        # Strictly order by offset ascending, preserving row number tie-breaker
        self._events: list[SimulationEvent] = sorted(events, key=lambda e: (e.offset_seconds, e.row_number))
        self._cursor: int = 0

    @property
    def events(self) -> list[SimulationEvent]:
        return self._events

    @property
    def total_count(self) -> int:
        return len(self._events)

    def peek_next_offset(self) -> float | None:
        if self._cursor < len(self._events):
            return self._events[self._cursor].offset_seconds
        return None

    def get_due_events(self, current_sim_time: float) -> list[SimulationEvent]:
        """Return all events whose scheduled offset has arrived and are still PENDING."""
        due: list[SimulationEvent] = []
        while self._cursor < len(self._events):
            candidate = self._events[self._cursor]
            if candidate.offset_seconds <= current_sim_time:
                if candidate.state == EventState.PENDING:
                    due.append(candidate)
                self._cursor += 1
            else:
                break
        return due

    def cancel_pending(self) -> list[SimulationEvent]:
        """Cancel all events that are still PENDING when simulation stops."""
        cancelled = []
        for e in self._events:
            if e.state == EventState.PENDING:
                e.state = EventState.CANCELLED
                cancelled.append(e)
        return cancelled

    def is_finished(self) -> bool:
        return all(e.state in (EventState.COMPLETED, EventState.FAILED, EventState.CANCELLED) for e in self._events)

