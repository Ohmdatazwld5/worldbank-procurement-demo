"""Open-stack catalog configuration and demo access-control policies."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Principal:
    subject: str
    roles: frozenset[str]
    project_ids: frozenset[str]


def build_nessie_spark_config() -> dict[str, str]:
    """Return Nessie/Iceberg catalog settings; Delta data stays path-addressed."""
    required = ("NESSIE_URI", "NESSIE_REF", "NESSIE_WAREHOUSE")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Nessie catalog configuration is incomplete: {', '.join(missing)}")
    catalog = os.getenv("NESSIE_CATALOG_NAME", "nessie")
    packages = os.getenv("NESSIE_SPARK_PACKAGES")
    if not packages:
        raise RuntimeError(
            "NESSIE_SPARK_PACKAGES must match the installed Spark/Scala versions."
        )
    return {
        "spark.sql.catalogImplementation": "in-memory",
        "spark.jars.packages": packages,
        "spark.sql.extensions": (
            "io.delta.sql.DeltaSparkSessionExtension,"
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"
        ),
        f"spark.sql.catalog.{catalog}": "org.apache.iceberg.spark.SparkCatalog",
        f"spark.sql.catalog.{catalog}.catalog-impl": "org.apache.iceberg.nessie.NessieCatalog",
        f"spark.sql.catalog.{catalog}.uri": os.environ["NESSIE_URI"],
        f"spark.sql.catalog.{catalog}.ref": os.environ["NESSIE_REF"],
        f"spark.sql.catalog.{catalog}.warehouse": os.environ["NESSIE_WAREHOUSE"],
        f"spark.sql.catalog.{catalog}.authentication.type": os.getenv(
            "NESSIE_AUTH_TYPE", "NONE"
        ),
    }


def load_demo_principals() -> dict[str, Principal]:
    """Load bearer-token principals from a secret environment variable.

    Example value is documented in .env.example. For deployment, use an OIDC
    gateway/secret manager rather than static demo tokens.
    """
    raw = os.getenv("API_TOKENS_JSON")
    if not raw:
        raise RuntimeError("API_TOKENS_JSON must be configured; anonymous access is disabled.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("API_TOKENS_JSON must contain a JSON object.") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError("API_TOKENS_JSON must map tokens to principal objects.")
    principals: dict[str, Principal] = {}
    for token, value in payload.items():
        if not isinstance(token, str) or not token or not isinstance(value, dict):
            raise ValueError("Each API token must map to a principal object.")
        subject = str(value.get("subject") or "").strip()
        roles = value.get("roles", [])
        project_ids = value.get("project_ids", [])
        if not subject or not isinstance(roles, list) or not isinstance(project_ids, list):
            raise ValueError("Each principal needs subject, roles[], and project_ids[].")
        principals[token] = Principal(
            subject=subject,
            roles=frozenset(str(role) for role in roles),
            project_ids=frozenset(str(project) for project in project_ids),
        )
    return principals


def authorize_project(principal: Principal, project_id: str) -> None:
    if "admin" in principal.roles or "*" in principal.project_ids or project_id in principal.project_ids:
        return
    raise PermissionError(f"Principal {principal.subject!r} may not access project {project_id!r}.")


def authorize_analytics(principal: Principal) -> None:
    if not principal.roles.intersection({"analyst", "admin"}):
        raise PermissionError(f"Principal {principal.subject!r} is not permitted to query analytics.")


def prepare_delta_spark(spark_builder: Any) -> Any:
    """Enable Delta Lake OSS on an existing SparkSession builder."""
    from delta import configure_spark_with_delta_pip

    catalog_mode = os.getenv("CATALOG_MODE", "hive").casefold()
    if catalog_mode not in {"hive", "nessie"}:
        raise ValueError(
            "The OSS backend supports CATALOG_MODE=hive or nessie; "
            "Unity Catalog is configured in the Databricks notebook/runtime."
        )
    builder = (
        spark_builder.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    if catalog_mode == "nessie":
        for key, value in build_nessie_spark_config().items():
            builder = builder.config(key, value)
    else:
        builder = builder.config("spark.sql.catalogImplementation", "hive")
        metastore_uris = os.getenv("HIVE_METASTORE_URIS")
        if metastore_uris:
            builder = builder.config("spark.hadoop.hive.metastore.uris", metastore_uris)
        else:
            metastore_db_path = os.getenv("HIVE_METASTORE_DB_PATH")
            if metastore_db_path:
                builder = builder.config(
                    "spark.hadoop.javax.jdo.option.ConnectionURL",
                    f"jdbc:derby:{metastore_db_path};create=true",
                )
    return configure_spark_with_delta_pip(builder)
