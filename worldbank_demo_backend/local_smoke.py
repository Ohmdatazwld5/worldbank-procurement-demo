"""Local Delta Lake/Hive catalog smoke test for the Docker Compose stack."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .ingestion import upsert_delta_rows
from .pipeline import create_delta_spark


def run_delta_hive_smoke(root: str | Path | None = None) -> dict[str, Any]:
    """Write, merge, register, and read a tiny Delta table through Hive."""
    catalog_mode = os.getenv("CATALOG_MODE", "hive").casefold()
    if catalog_mode != "hive":
        raise ValueError("The local Delta catalog smoke test requires CATALOG_MODE=hive.")

    data_root = Path(root or os.getenv("DELTA_LAKE_ROOT", "./data/local_lakehouse"))
    data_root.mkdir(parents=True, exist_ok=True)
    table_path = Path(tempfile.mkdtemp(prefix="delta-smoke-", dir=data_root))
    table_name = f"delta_smoke_{uuid.uuid4().hex[:12]}"
    spark = None
    table_registered = False
    try:
        spark = create_delta_spark()
        upsert_delta_rows(
            [
                {"record_id": "one", "value": "before"},
                {"record_id": "two", "value": "keep"},
            ],
            delta_path=str(table_path),
            key_columns=("record_id",),
            spark=spark,
        )
        upsert_delta_rows(
            [{"record_id": "one", "value": "after"}, {"record_id": "three", "value": "new"}],
            delta_path=str(table_path),
            key_columns=("record_id",),
            spark=spark,
        )
        spark.sql(
            f"CREATE TABLE default.`{table_name}` USING DELTA "
            f"LOCATION '{str(table_path).replace(chr(39), chr(39) * 2)}'"
        )
        table_registered = True
        rows = {
            row["record_id"]: row["value"]
            for row in spark.table(f"default.`{table_name}`").collect()
        }
        expected = {"one": "after", "two": "keep", "three": "new"}
        if rows != expected:
            raise RuntimeError(f"Delta/Hive read-back mismatch: expected {expected}, got {rows}.")
        return {
            "status": "passed",
            "catalog": "hive",
            "storage_format": "delta",
            "row_count_after_merge": len(rows),
            "merged_value": rows["one"],
        }
    finally:
        try:
            if spark is not None and table_registered:
                spark.sql(f"DROP TABLE IF EXISTS default.`{table_name}`")
        finally:
            try:
                if spark is not None:
                    spark.stop()
            finally:
                shutil.rmtree(table_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Temporary Delta table root.")
    args = parser.parse_args()
    result = run_delta_hive_smoke(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
