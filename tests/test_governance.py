from __future__ import annotations

import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from worldbank_demo_backend.governance import (
    build_nessie_spark_config,
    prepare_delta_spark,
)
from worldbank_demo_backend.pipeline import create_delta_spark


class FakeSparkBuilder:
    def __init__(self) -> None:
        self.settings: dict[str, str] = {}

    def config(self, key: str, value: str) -> "FakeSparkBuilder":
        self.settings[key] = value
        return self


class HiveSparkBuilder(FakeSparkBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.hive_support_enabled = False

    def appName(self, _name: str) -> "HiveSparkBuilder":
        return self

    def enableHiveSupport(self) -> "HiveSparkBuilder":
        self.hive_support_enabled = True
        return self

    def getOrCreate(self) -> "HiveSparkBuilder":
        return self


class GovernanceTests(unittest.TestCase):
    def test_hive_catalog_configures_delta_and_optional_metastore(self) -> None:
        builder = FakeSparkBuilder()
        delta_module = ModuleType("delta")
        delta_module.configure_spark_with_delta_pip = lambda value: value
        with (
            patch.dict(
                os.environ,
                {
                    "CATALOG_MODE": "hive",
                    "HIVE_METASTORE_URIS": "thrift://hive:9083",
                },
            ),
            patch.dict(sys.modules, {"delta": delta_module}),
        ):
            result = prepare_delta_spark(builder)
        self.assertIs(result, builder)
        self.assertEqual(
            builder.settings["spark.sql.catalog.spark_catalog"],
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        self.assertEqual(builder.settings["spark.sql.catalogImplementation"], "hive")
        self.assertEqual(
            builder.settings["spark.hadoop.hive.metastore.uris"],
            "thrift://hive:9083",
        )

    def test_nessie_catalog_requires_explicit_spark_packages(self) -> None:
        with patch.dict(
            os.environ,
            {
                "NESSIE_URI": "http://nessie:19120/api/v2",
                "NESSIE_REF": "main",
                "NESSIE_WAREHOUSE": "file:///tmp/warehouse",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "NESSIE_SPARK_PACKAGES"):
                build_nessie_spark_config()

    def test_hive_embedded_metastore_uses_configured_derby_database(self) -> None:
        builder = FakeSparkBuilder()
        delta_module = ModuleType("delta")
        delta_module.configure_spark_with_delta_pip = lambda value: value
        with (
            patch.dict(
                os.environ,
                {
                    "CATALOG_MODE": "hive",
                    "HIVE_METASTORE_DB_PATH": "/workspace/data/hive-metastore",
                },
                clear=True,
            ),
            patch.dict(sys.modules, {"delta": delta_module}),
        ):
            result = prepare_delta_spark(builder)
        self.assertIs(result, builder)
        self.assertEqual(
            builder.settings["spark.hadoop.javax.jdo.option.ConnectionURL"],
            "jdbc:derby:/workspace/data/hive-metastore;create=true",
        )

    def test_pipeline_enables_hive_support_for_hive_catalog_mode(self) -> None:
        builder = HiveSparkBuilder()
        spark_module = ModuleType("pyspark")
        sql_module = ModuleType("pyspark.sql")
        sql_module.SparkSession = SimpleNamespace(builder=builder)
        delta_module = ModuleType("delta")
        delta_module.configure_spark_with_delta_pip = lambda value: value
        with (
            patch.dict(os.environ, {"CATALOG_MODE": "hive"}, clear=True),
            patch.dict(
                sys.modules,
                {
                    "pyspark": spark_module,
                    "pyspark.sql": sql_module,
                    "delta": delta_module,
                },
            ),
        ):
            result = create_delta_spark()
        self.assertIs(result, builder)
        self.assertTrue(builder.hive_support_enabled)


if __name__ == "__main__":
    unittest.main()
