"""Monotonic simulation clock with pause/resume preservation and playback speed multipliers."""

from __future__ import annotations

from datetime import datetime, timezone
import time


class SimulationClock:
    """Accurate monotonic simulation clock independent of system wall-clock changes or UI render cycles."""

    def __init__(self, speed_multiplier: float = 1.0) -> None:
        self.speed_multiplier = float(speed_multiplier)
        self._wall_start_iso: str | None = None
        self._start_mono: float | None = None
        self._paused_mono: float | None = None
        self._accumulated_sim_seconds: float = 0.0
        self._accumulated_wall_seconds: float = 0.0
        self._is_running: bool = False
        self._is_paused: bool = False

    def start(self) -> None:
        now_mono = time.monotonic()
        self._wall_start_iso = datetime.now(timezone.utc).isoformat()
        self._start_mono = now_mono
        self._paused_mono = None
        self._accumulated_sim_seconds = 0.0
        self._accumulated_wall_seconds = 0.0
        self._is_running = True
        self._is_paused = False

    def pause(self) -> None:
        if self._is_running and not self._is_paused and self._start_mono is not None:
            now_mono = time.monotonic()
            elapsed_wall = now_mono - self._start_mono
            self._accumulated_wall_seconds += elapsed_wall
            self._accumulated_sim_seconds += elapsed_wall * self.speed_multiplier
            self._paused_mono = now_mono
            self._is_paused = True

    def resume(self) -> None:
        if self._is_running and self._is_paused:
            now_mono = time.monotonic()
            self._start_mono = now_mono
            self._paused_mono = None
            self._is_paused = False

    def stop(self) -> None:
        if self._is_running:
            if not self._is_paused and self._start_mono is not None:
                now_mono = time.monotonic()
                elapsed_wall = now_mono - self._start_mono
                self._accumulated_wall_seconds += elapsed_wall
                self._accumulated_sim_seconds += elapsed_wall * self.speed_multiplier
            self._is_running = False
            self._is_paused = False

    def reset(self) -> None:
        self._wall_start_iso = None
        self._start_mono = None
        self._paused_mono = None
        self._accumulated_sim_seconds = 0.0
        self._accumulated_wall_seconds = 0.0
        self._is_running = False
        self._is_paused = False

    def set_speed(self, multiplier: float) -> None:
        # Re-anchor accumulated time before switching speed multiplier
        if self._is_running and not self._is_paused and self._start_mono is not None:
            now_mono = time.monotonic()
            elapsed_wall = now_mono - self._start_mono
            self._accumulated_wall_seconds += elapsed_wall
            self._accumulated_sim_seconds += elapsed_wall * self.speed_multiplier
            self._start_mono = now_mono
        self.speed_multiplier = float(multiplier)

    def elapsed_sim_seconds(self) -> float:
        if not self._is_running:
            return self._accumulated_sim_seconds
        if self._is_paused:
            return self._accumulated_sim_seconds
        if self._start_mono is None:
            return 0.0
        now_mono = time.monotonic()
        elapsed_wall = now_mono - self._start_mono
        return self._accumulated_sim_seconds + (elapsed_wall * self.speed_multiplier)

    def active_wall_seconds(self) -> float:
        if not self._is_running:
            return self._accumulated_wall_seconds
        if self._is_paused:
            return self._accumulated_wall_seconds
        if self._start_mono is None:
            return 0.0
        now_mono = time.monotonic()
        elapsed_wall = now_mono - self._start_mono
        return self._accumulated_wall_seconds + elapsed_wall

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def wall_start_iso(self) -> str | None:
        return self._wall_start_iso

    def format_time(self, seconds: float | None = None) -> str:
        s = self.elapsed_sim_seconds() if seconds is None else seconds
        total_seconds = int(max(0.0, s))
        hrs = total_seconds // 3600
        mins = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"

