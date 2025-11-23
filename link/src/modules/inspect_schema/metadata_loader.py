"""Utilities to load database metadata from BigQuery or cached JSON."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.cloud import bigquery
from google.oauth2 import service_account
from snowflake import connector

from .types import ColumnMetadata, DatabaseMetadata, TableMetadata


def _metadata_has_samples(metadata: List[DatabaseMetadata]) -> bool:
    """Return True if cached metadata includes any sample_values."""
    for db in metadata:
        for table in getattr(db, "tables", []):
            for column in getattr(table, "columns", []):
                extra = getattr(column, "extra", {})
                sample_values = extra.get("sample_values") if isinstance(extra, dict) else None
                if sample_values:
                    return True
    return False


def _get_sample_values_snowflake(cursor, full_table_name: str, column_name: str, limit: int = 50) -> list:
    """
    Ejecuta query para obtener valores únicos de una columna en Snowflake.
    Devuelve lista de strings con valores sample (máximo `limit` únicos).
    """
    try:
        query = f'SELECT DISTINCT "{column_name}" FROM {full_table_name} WHERE "{column_name}" IS NOT NULL LIMIT {limit}'
        cursor.execute(query)
        rows = cursor.fetchall()
        samples = [str(row[0]) for row in rows if row[0] is not None]
        if not samples:
            print(f"[metadata_loader] ⚠️  No samples found for {column_name} in {full_table_name}")
        return samples
    except Exception as e:
        print(f"[metadata_loader] ⚠️  Could not load samples for {column_name} in {full_table_name}: {e}")
        return []


def _get_sample_values_bigquery(client, full_table_name: str, column_name: str, limit: int = 50) -> list:
    """
    Ejecuta query para obtener valores únicos de una columna en BigQuery.
    """
    try:
        query = f"""
            SELECT DISTINCT {column_name}
            FROM {full_table_name}
            WHERE {column_name} IS NOT NULL
            LIMIT {limit}
        """
        result = client.query(query).result()
        return [str(row[0]) for row in result if row[0] is not None]
    except Exception as e:
        print(f"⚠️  Could not load samples for {column_name} in {full_table_name}: {e}")
        return []


def _classify_column_profile(col_type: str, sample_values: list, num_distinct: int = None) -> dict:
    """
    Determina el perfil semántico de una columna basado en tipo y muestras.
    
    Returns:
        dict con: semantic_role, safe_for_enum_constraints, safe_for_repair_mapping
    """
    col_type_upper = col_type.upper()
    
    # Tipos numéricos / fechas no son enums
    if any(t in col_type_upper for t in ['INT', 'FLOAT', 'DECIMAL', 'NUMBER', 'DATE', 'TIMESTAMP']):
        return {
            "semantic_role": "measure" if 'INT' in col_type_upper or 'NUMBER' in col_type_upper else "temporal",
            "safe_for_enum_constraints": False,
            "safe_for_repair_mapping": False
        }
    
    # Texto con pocas muestras → candidato a enum
    if any(t in col_type_upper for t in ['TEXT', 'VARCHAR', 'CHAR', 'STRING']):
        num_samples = len(sample_values)
        
        if num_samples == 0:
            return {
                "semantic_role": "free_text",
                "safe_for_enum_constraints": False,
                "safe_for_repair_mapping": False
            }
        
        # Si tiene <= 50 valores únicos, probablemente es enum
        if num_samples <= 50:
            return {
                "semantic_role": "enum",
                "safe_for_enum_constraints": True,
                "safe_for_repair_mapping": True
            }
        else:
            return {
                "semantic_role": "free_text",
                "safe_for_enum_constraints": False,
                "safe_for_repair_mapping": False
            }
    
    # Fallback
    return {
        "semantic_role": "unknown",
        "safe_for_enum_constraints": False,
        "safe_for_repair_mapping": False
    }



def load_db_metadata(db_config: Dict[str, Any]) -> List[DatabaseMetadata]:
    engine = db_config.get("engine", "bigquery")
    use_cache = db_config.get("use_cache", False)
    cache_path = db_config.get("cache_path", "link/cache/schema_cache.json")
    enable_sampling = db_config.get("enable_column_sampling", False)

    if use_cache:
        cached = load_cached_metadata(cache_path)
        if cached:
            if enable_sampling and not _metadata_has_samples(cached):
                print("[inspect_schema] Cached metadata missing sample_values; refreshing cache...")
            else:
                return cached

    if engine == "bigquery":
        metadata = _load_from_bigquery(db_config)
    elif engine == "snowflake":
        metadata = _load_from_snowflake(db_config)
    else:
        raise NotImplementedError(f"Engine '{engine}' not supported yet")

    if use_cache:
        save_cached_metadata(metadata, cache_path)
    return metadata


def _load_from_bigquery(db_config: Dict[str, Any]) -> List[DatabaseMetadata]:
    """Query BigQuery INFORMATION_SCHEMA to extract table and column metadata."""
    project_id = db_config.get("project_id", "bigquery-public-data")
    datasets = db_config.get("datasets", ["usa_names"])
    credential_path = db_config.get("credential_path", "src/cred/clean_node.json")
    enable_sampling = db_config.get("enable_column_sampling", False)
    sample_limit = db_config.get("sample_limit", 50)

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
                columns = []
                
                for row in col_rows:
                    col_name = row.column_name
                    col_type = row.data_type
                    
                    # Sample values si está habilitado
                    sample_values = []
                    if enable_sampling:
                        full_table_name = f"`{project_id}.{dataset_name}.{table_name}`"
                        sample_values = _get_sample_values_bigquery(
                            client, full_table_name, col_name, sample_limit
                        )
                    
                    # Clasificar perfil
                    profile = _classify_column_profile(col_type, sample_values)
                    
                    columns.append(
                        ColumnMetadata(
                            name=col_name,
                            type=col_type,
                            description=None,
                            extra={
                                "nullable": row.is_nullable == "YES",
                                "sample_values": sample_values,
                                "profile": profile
                            }
                        )
                    )
                
            except Exception:
                # Fallback: get schema from table object
                table_ref = client.dataset(dataset_name, project=project_id).table(table_name)
                table_obj = client.get_table(table_ref)
                columns = [
                    ColumnMetadata(
                        name=field.name,
                        type=field.field_type,
                        description=field.description,
                        extra={
                            "mode": field.mode,
                            "sample_values": [],
                            "profile": _classify_column_profile(field.field_type, [])
                        }
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


def _load_from_snowflake(db_config: Dict[str, Any]) -> List[DatabaseMetadata]:
    """
    Query Snowflake INFORMATION_SCHEMA to extract table and column metadata.
    Uses credentials from JSON file or direct config parameters.
    """
    import snowflake.connector

    # Load credentials from JSON file if provided
    credential_path = db_config.get("credential_path")
    if credential_path:
        with open(credential_path, "r", encoding="utf-8") as f:
            cred = json.load(f)
        account = cred["account"]
        user = cred["username"]
        password = cred["password"]
        warehouse = cred.get("warehouse")
        role = cred.get("role")
    else:
        # Fallback to direct config parameters
        account = db_config["account"]
        user = db_config.get("user") or db_config.get("username")
        password = db_config["password"]
        warehouse = db_config.get("warehouse")
        role = db_config.get("role")

    # Lista de databases y schemas que quieres inspeccionar
    databases = db_config.get("databases", ["USA_NAMES"])
    schemas = db_config.get("schemas", ["USA_NAMES"])
    enable_sampling = db_config.get("enable_column_sampling", False)
    sample_limit = db_config.get("sample_limit", 50)

    conn = snowflake.connector.connect(
        account=account,
        user=user,
        password=password,
        warehouse=warehouse,
        role=role,
    )
    cur = conn.cursor()

    result: List[DatabaseMetadata] = []

    try:
        for database in databases:
            tables_list: List[TableMetadata] = []

            for schema in schemas:
                # 1) Obtener tablas BASE TABLE en ese schema
                tables_query = f"""
                    SELECT table_name
                    FROM {database}.INFORMATION_SCHEMA.TABLES
                    WHERE table_schema = '{schema}'
                      AND table_type = 'BASE TABLE'
                """
                cur.execute(tables_query)
                table_names = [row[0] for row in cur.fetchall()]

                # 2) Para cada tabla, obtener sus columnas
                for table_name in table_names:
                    columns_query = f"""
                        SELECT column_name, data_type, is_nullable
                        FROM {database}.INFORMATION_SCHEMA.COLUMNS
                        WHERE table_schema = '{schema}'
                          AND table_name = '{table_name}'
                        ORDER BY ordinal_position
                    """
                    cur.execute(columns_query)
                    col_rows = cur.fetchall()

                    columns = []
                    for row in col_rows:
                        col_name = row[0]
                        col_type = row[1]
                        is_nullable = row[2]
                        
                        # Sample values si está habilitado
                        sample_values = []
                        if enable_sampling:
                            full_table_name = f'{database}.{schema}.{table_name}'
                            sample_values = _get_sample_values_snowflake(
                                cur, full_table_name, col_name, sample_limit
                            )
                        
                        # Clasificar perfil
                        profile = _classify_column_profile(col_type, sample_values)
                        
                        columns.append(
                            ColumnMetadata(
                                name=col_name,
                                type=col_type,
                                description=None,
                                extra={
                                    "nullable": is_nullable == "YES",
                                    "sample_values": sample_values,
                                    "profile": profile
                                }
                            )
                        )

                    # full_name en formato Snowflake
                    # puedes usar comillas dobles si luego quieres ser súper estricto
                    full_name = f'{database}.{schema}.{table_name}'

                    tables_list.append(
                        TableMetadata(
                            table_name=table_name,
                            full_name=full_name,
                            columns=columns,
                            description=None,
                        )
                    )

            db_id = f"{database}.{','.join(schemas)}"
            result.append(
                DatabaseMetadata(
                    db_id=db_id,
                    engine="snowflake",
                    tables=tables_list,
                    extra={"databases": databases, "schemas": schemas},
                )
            )
    finally:
        cur.close()
        conn.close()

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
