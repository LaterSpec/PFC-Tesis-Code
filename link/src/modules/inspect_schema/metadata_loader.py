"""Utilities to load database metadata from BigQuery or cached JSON."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.cloud import bigquery
from google.oauth2 import service_account

from .types import ColumnMetadata, DatabaseMetadata, TableMetadata


def load_db_metadata(db_config: Dict[str, Any]) -> List[DatabaseMetadata]:
    """Load database metadata from BigQuery, Snowflake, or cached JSON."""
    engine = db_config.get("engine", "bigquery")
    use_cache = db_config.get("use_cache", False)
    cache_path = db_config.get("cache_path", "link/cache/schema_cache.json")

    # Try loading from cache first
    if use_cache:
        cached = load_cached_metadata(cache_path)
        if cached:
            return cached

    # Load from live connection
    if engine == "bigquery":
        metadata = _load_from_bigquery(db_config)
        # Save to cache for future use
        if use_cache:
            save_cached_metadata(metadata, cache_path)
        return metadata
    else:
        raise NotImplementedError(f"Engine '{engine}' not supported yet")


def _load_from_bigquery(db_config: Dict[str, Any]) -> List[DatabaseMetadata]:
    """Query BigQuery INFORMATION_SCHEMA to extract table and column metadata."""
    project_id = db_config.get("project_id", "bigquery-public-data")
    datasets = db_config.get("datasets", ["usa_names"])
    credential_path = db_config.get("credential_path", "src/cred/clean_node.json")

    # Authenticate with service account
    credentials = service_account.Credentials.from_service_account_file(credential_path)
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    result = []
    for dataset_name in datasets:
        db_id = f"{project_id}.{dataset_name}"
        tables_list = []

        # Query INFORMATION_SCHEMA.TABLES to get all tables in the dataset
        tables_query = f"""
            SELECT table_name
            FROM `{project_id}.{dataset_name}.INFORMATION_SCHEMA.TABLES`
            WHERE table_type = 'BASE TABLE'
        """
        try:
            table_rows = client.query(tables_query).result()
            table_names = [row.table_name for row in table_rows]
        except Exception as e:
            # If INFORMATION_SCHEMA fails, fallback to listing tables via API
            dataset_ref = client.dataset(dataset_name, project=project_id)
            tables = list(client.list_tables(dataset_ref))
            table_names = [t.table_id for t in tables]

        # For each table, get column metadata
        for table_name in table_names:
            columns_query = f"""
                SELECT column_name, data_type, is_nullable
                FROM `{project_id}.{dataset_name}.INFORMATION_SCHEMA.COLUMNS`
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """
            try:
                col_rows = client.query(columns_query).result()
                columns = [
                    ColumnMetadata(
                        name=row.column_name,
                        type=row.data_type,
                        description=None,
                        extra={"nullable": row.is_nullable == "YES"}
                    )
                    for row in col_rows
                ]
            except Exception:
                # Fallback: get schema from table object
                table_ref = client.dataset(dataset_name, project=project_id).table(table_name)
                table_obj = client.get_table(table_ref)
                columns = [
                    ColumnMetadata(
                        name=field.name,
                        type=field.field_type,
                        description=field.description,
                        extra={"mode": field.mode}
                    )
                    for field in table_obj.schema
                ]

            full_name = f"`{project_id}.{dataset_name}.{table_name}`"
            tables_list.append(
                TableMetadata(
                    table_name=table_name,
                    full_name=full_name,
                    columns=columns,
                    description=None
                )
            )

        result.append(
            DatabaseMetadata(
                db_id=db_id,
                engine="bigquery",
                tables=tables_list,
                extra={"project_id": project_id, "dataset": dataset_name}
            )
        )

    return result


def load_cached_metadata(cache_file: str) -> Optional[List[DatabaseMetadata]]:
    """Attempt to load database metadata from a JSON cache file."""
    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = []
        for db_dict in data.get("databases", []):
            tables = []
            for tbl_dict in db_dict.get("tables", []):
                columns = [
                    ColumnMetadata(
                        name=col["name"],
                        type=col["type"],
                        description=col.get("description"),
                        extra=col.get("extra", {})
                    )
                    for col in tbl_dict.get("columns", [])
                ]
                tables.append(
                    TableMetadata(
                        table_name=tbl_dict["table_name"],
                        full_name=tbl_dict["full_name"],
                        columns=columns,
                        description=tbl_dict.get("description"),
                        extra=tbl_dict.get("extra", {})
                    )
                )
            result.append(
                DatabaseMetadata(
                    db_id=db_dict["db_id"],
                    engine=db_dict["engine"],
                    tables=tables,
                    extra=db_dict.get("extra", {})
                )
            )
        return result
    except Exception:
        return None


def save_cached_metadata(metadata: List[DatabaseMetadata], cache_file: str) -> None:
    """Persist database metadata to a JSON cache file."""
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)

    data = {
        "databases": [
            {
                "db_id": db.db_id,
                "engine": db.engine,
                "tables": [
                    {
                        "table_name": tbl.table_name,
                        "full_name": tbl.full_name,
                        "description": tbl.description,
                        "columns": [
                            {
                                "name": col.name,
                                "type": col.type,
                                "description": col.description,
                                "extra": col.extra
                            }
                            for col in tbl.columns
                        ],
                        "extra": tbl.extra
                    }
                    for tbl in db.tables
                ],
                "extra": db.extra
            }
            for db in metadata
        ]
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
