from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg

from lib import load_config


@contextmanager
def connect(autocommit: bool = False) -> Iterator[psycopg.Connection]:
    cfg = load_config()
    conn = psycopg.connect(cfg["database_url"], autocommit=autocommit)
    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


def run_migrations() -> None:
    from pathlib import Path

    mig_dir = Path(__file__).resolve().parent / "migrations"
    with connect() as conn:
        for path in sorted(mig_dir.glob("*.sql")):
            conn.execute(path.read_text())
            print(f"applied {path.name}", flush=True)
