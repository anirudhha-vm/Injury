"""Validate tables against the schema and load them into the research database.

Invalid rows are reported with their row number and reason; they are never silently dropped
or coerced. Loading refuses to proceed while any row is invalid, and refuses to mix synthetic
and real rows in one database.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from youthtwin.io.schema import TABLES, Row
from youthtwin.utils.config import repo_root

DDL_PATH = Path("data") / "schema" / "schema.sql"


@dataclass(frozen=True)
class RowError:
    table: str
    row: int
    message: str


@dataclass
class ValidationReport:
    n_rows: dict[str, int] = field(default_factory=dict)
    errors: list[RowError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _clean(record: dict[str, Any]) -> dict[str, Any]:
    """Turn pandas missing values into None so optional fields validate."""
    return {
        key: None if (not isinstance(value, (list, dict)) and pd.isna(value)) else value
        for key, value in record.items()
    }


def validate_table(name: str, frame: pd.DataFrame) -> tuple[list[Row], list[RowError]]:
    """Validate every row of ``frame`` against the model for table ``name``."""
    if name not in TABLES:
        raise KeyError(f"Unknown table '{name}'. Known: {sorted(TABLES)}")
    model = TABLES[name]
    rows: list[Row] = []
    errors: list[RowError] = []
    for number, record in enumerate(frame.to_dict(orient="records")):
        try:
            rows.append(model.model_validate(_clean({str(k): v for k, v in record.items()})))
        except ValidationError as exc:
            reasons = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'row'}: {e['msg']}" for e in exc.errors()
            )
            errors.append(RowError(name, number, reasons))
    return rows, errors


def validate_tables(tables: dict[str, pd.DataFrame]) -> ValidationReport:
    report = ValidationReport()
    for name, frame in tables.items():
        _, errors = validate_table(name, frame)
        report.n_rows[name] = len(frame)
        report.errors.extend(errors)
    return report


def create_database(path: Path | str) -> sqlite3.Connection:
    """Open (or create) a SQLite research database with the schema applied."""
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript((repo_root() / DDL_PATH).read_text(encoding="utf-8"))
    return connection


def _to_sql_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


def load_tables(
    connection: sqlite3.Connection, tables: dict[str, pd.DataFrame]
) -> ValidationReport:
    """Validate, then insert all tables in dependency order inside one transaction."""
    validated: dict[str, list[Row]] = {}
    report = ValidationReport()
    for name, frame in tables.items():
        rows, errors = validate_table(name, frame)
        validated[name] = rows
        report.n_rows[name] = len(frame)
        report.errors.extend(errors)
    if not report.ok:
        raise ValueError(
            f"{len(report.errors)} invalid row(s); nothing was loaded. First: "
            f"{report.errors[0].table}[{report.errors[0].row}] {report.errors[0].message}"
        )
    flags = {row.synthetic for rows in validated.values() for row in rows}
    existing = connection.execute("SELECT DISTINCT synthetic FROM athletes").fetchall()
    flags |= {bool(value) for (value,) in existing}
    if len(flags) > 1:
        raise ValueError("Refusing to mix synthetic and real rows in one database")
    with connection:
        for name in TABLES:  # TABLES is in foreign-key dependency order
            rows = validated.get(name, [])
            if not rows:
                continue
            columns = list(TABLES[name].model_fields)
            statement = (
                f"INSERT INTO {name} ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})"
            )
            connection.executemany(
                statement,
                [tuple(_to_sql_value(getattr(row, c)) for c in columns) for row in rows],
            )
    return report


COHORT_TABLES = (
    "athletes",
    "anthropometrics",
    "screening_sessions",
    "task_features",
    "exposure_weekly",
    "injuries",
)
REQUIRED_COHORT_TABLES = ("athletes", "anthropometrics", "exposure_weekly", "injuries")


def read_cohort_dir(directory: Path) -> dict[str, pd.DataFrame]:
    """Read a cohort stored as one parquet file per table and validate it against the schema.

    This is the entry point for real data: the same layout that ``make sim`` writes. Raises if a
    required table is missing, if any row is invalid (listing the first problems), or if
    synthetic and real rows are mixed.
    """
    absent = [t for t in REQUIRED_COHORT_TABLES if not (directory / f"{t}.parquet").is_file()]
    if absent:
        raise FileNotFoundError(f"{directory} is missing required tables: {absent}")
    tables = {
        name: pd.read_parquet(directory / f"{name}.parquet")
        for name in COHORT_TABLES
        if (directory / f"{name}.parquet").is_file()
    }
    report = validate_tables(tables)
    if not report.ok:
        shown = "; ".join(f"{e.table}[{e.row}] {e.message}" for e in report.errors[:5])
        raise ValueError(f"{len(report.errors)} invalid row(s) in {directory}. First: {shown}")
    flags = {bool(v) for table in tables.values() for v in table["synthetic"].unique()}
    if len(flags) > 1:
        raise ValueError("The cohort mixes synthetic and real rows")
    return tables
