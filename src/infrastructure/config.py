"""Environment-aware application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    root: Path
    environment: str
    database_path: Path
    models_path: Path
    data_path: Path
    reports_path: Path

    @classmethod
    def from_environment(cls, root: Path | None = None) -> "AppConfig":
        project_root = root or Path(__file__).resolve().parents[2]
        return cls(
            root=project_root,
            environment=os.getenv("RESINET_ENVIRONMENT", "local"),
            database_path=Path(
                os.getenv(
                    "RESINET_DATABASE_PATH",
                    str(project_root / "data" / "resinet-events.sqlite3"),
                )
            ),
            models_path=Path(
                os.getenv("RESINET_MODELS_PATH", str(project_root / "models"))
            ),
            data_path=Path(
                os.getenv("RESINET_DATA_PATH", str(project_root / "data"))
            ),
            reports_path=Path(
                os.getenv("RESINET_REPORTS_PATH", str(project_root / "reports"))
            ),
        )
