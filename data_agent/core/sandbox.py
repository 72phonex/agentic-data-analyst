"""
Sandboxed SQL execution engine for tabular data analysis.
Executes safe, read-only DuckDB SQL queries over an in-memory DataFrame.
Strictly disallows filesystem, network, metadata manipulation, or state modification.
"""

from __future__ import annotations
import re
from typing import Any
import duckdb
import pandas as pd


DANGEROUS_KEYWORDS = {
    "DROP", "INSERT", "UPDATE", "DELETE", "ALTER", "CREATE", "ATTACH", "DETACH",
    "COPY", "EXPORT", "IMPORT", "INSTALL", "LOAD", "PRAGMA", "SET", "RESET",
    "EXEC", "EXECUTE", "CALL", "VACUUM", "CHECKPOINT"
}

DANGEROUS_FUNCTIONS = {
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
    "scan_csv", "scan_parquet", "glob", "read_blob", "read_text", "httpfs",
    "sqlite_scan", "postgres_scan"
}


class SandboxSecurityError(PermissionError):
    """Raised when an unsafe or forbidden SQL operation is attempted."""
    pass


def validate_sql_query(query: str) -> None:
    """
    Validate that an SQL query is strictly read-only and free of sandbox-escape vectors.
    """
    cleaned = re.sub(r"--.*?\n", " ", query)
    cleaned = re.sub(r"/\*.*?\*/", " ", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()

    if not cleaned:
        raise SandboxSecurityError("Empty query provided.")

    # Disallow multiple semicolons / chained statements
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    if len(statements) > 1:
        raise SandboxSecurityError("Chained queries or multiple statements are not permitted.")

    # Match tokens
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", cleaned)
    tokens_upper = {t.upper() for t in tokens}
    tokens_lower = {t.lower() for t in tokens}

    # Block dangerous keywords
    for kw in DANGEROUS_KEYWORDS:
        if kw in tokens_upper:
            raise SandboxSecurityError(f"Prohibited SQL keyword detected: '{kw}'. Queries must be read-only.")

    # Block external I/O functions
    for fn in DANGEROUS_FUNCTIONS:
        if fn in tokens_lower:
            raise SandboxSecurityError(f"Prohibited filesystem/network function detected: '{fn}'.")

    # Enforce SELECT or WITH
    first_token = tokens[0].upper() if tokens else ""
    if first_token not in ("SELECT", "WITH", "DESCRIBE", "EXPLAIN"):
        raise SandboxSecurityError(f"Query must begin with SELECT, WITH, or DESCRIBE. Received: '{first_token}'.")


def run_sandboxed_sql(
    df: pd.DataFrame,
    query: str,
    table_name: str = "df",
    max_rows: int = 200,
) -> dict[str, Any]:
    """
    Execute a sandboxed SQL query against an in-memory pandas DataFrame.
    """
    validate_sql_query(query)

    # Replace case-insensitive references to df or table_name
    conn = duckdb.connect(database=":memory:", read_only=False)
    try:
        # Register in-memory table
        conn.register(table_name, df)
        if table_name != "df":
            conn.register("df", df)

        result = conn.execute(query).fetchdf()

        total_rows = len(result)
        truncated = False
        if total_rows > max_rows:
            result = result.iloc[:max_rows]
            truncated = True

        return {
            "status": "success",
            "columns": list(result.columns),
            "row_count": total_rows,
            "returned_rows": len(result),
            "truncated": truncated,
            "data": result.to_dict(orient="records"),
            "dataframe": result,
        }
    except Exception as e:
        if isinstance(e, SandboxSecurityError):
            raise
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e),
        }
    finally:
        conn.close()
