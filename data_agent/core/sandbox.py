"""
Sandboxed SQL execution engine for tabular data analysis.
Executes safe, read-only DuckDB SQL queries over an in-memory DataFrame.
Strictly disallows filesystem, network, metadata manipulation, or state modification.
"""

from __future__ import annotations
import concurrent.futures
import re
from typing import Any, List, Optional
import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp


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

FORBIDDEN_AST_NODES = (
    exp.Drop, exp.Insert, exp.Delete, exp.Update, exp.Alter, exp.Create,
    exp.Command, exp.Set, exp.Pragma, exp.Copy
)


class SandboxSecurityError(PermissionError):
    """Raised when an unsafe or forbidden SQL operation is attempted."""
    pass


def split_statements_outside_quotes(query: str) -> List[str]:
    """Split query on semicolons that are not inside single quotes."""
    statements = []
    current = []
    in_single_quote = False
    i = 0
    while i < len(query):
        ch = query[i]
        if ch == "'":
            if in_single_quote and i + 1 < len(query) and query[i+1] == "'":
                current.append("''")
                i += 2
                continue
            in_single_quote = not in_single_quote
            current.append(ch)
        elif ch == ";" and not in_single_quote:
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
        else:
            current.append(ch)
        i += 1
    stmt = "".join(current).strip()
    if stmt:
        statements.append(stmt)
    return statements


def mask_literals_and_identifiers(query: str) -> str:
    """Mask string literals and quoted identifiers so keywords inside them do not trigger false positives."""
    masked = re.sub(r"'([^']|'')*'", "'__STR__'", query)
    masked = re.sub(r'"[^"]*"', '"__COL__"', masked)
    masked = re.sub(r'`[^`]*`', '`__COL__`', masked)
    masked = re.sub(r'\[[^\]]*\]', '[__COL__]', masked)
    return masked


def validate_sql_query(query: str) -> None:
    """
    Validate that an SQL query is strictly read-only and free of sandbox-escape vectors.
    Uses AST parsing with sqlglot and secondary lexical token verification.
    """
    cleaned = re.sub(r"--.*?\n", "\n", query)
    cleaned = re.sub(r"/\*.*?\*/", " ", cleaned, flags=re.DOTALL).strip()

    if not cleaned:
        raise SandboxSecurityError("Empty query provided.")

    # 1. Disallow chained statements outside string literals
    statements = split_statements_outside_quotes(cleaned)
    if len(statements) > 1:
        raise SandboxSecurityError("Chained queries or multiple statements are not permitted.")

    # 2. Block direct file paths in FROM clause
    if re.search(r"\bFROM\s+['\"][^'\"]+['\"]", cleaned, re.IGNORECASE):
        raise SandboxSecurityError("Direct file path syntax in queries is forbidden.")

    # 3. AST validation via sqlglot
    try:
        parsed = sqlglot.parse(cleaned, read="duckdb")
        if not parsed or len(parsed) != 1:
            raise SandboxSecurityError("Query must consist of exactly one single statement.")
        stmt = parsed[0]
        if not (isinstance(stmt, exp.Select) or (hasattr(stmt, "find") and stmt.find(exp.Select))):
            raise SandboxSecurityError(f"Query must be a SELECT or WITH statement. Got: {type(stmt).__name__}")
        for node in stmt.walk():
            if isinstance(node, FORBIDDEN_AST_NODES):
                raise SandboxSecurityError(f"Prohibited SQL statement type detected via AST: {type(node).__name__}")
    except Exception as e:
        if isinstance(e, SandboxSecurityError):
            raise

    # 4. Token validation over masked query
    masked = mask_literals_and_identifiers(cleaned)
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", masked)
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
    timeout: float = 5.0,
) -> dict[str, Any]:
    """
    Execute a sandboxed SQL query against an in-memory pandas DataFrame.
    Enforces disabled external access, locked configuration, execution timeout, and row capping.
    """
    validate_sql_query(query)

    conn = duckdb.connect(
        database=":memory:",
        config={"enable_external_access": False}
    )
    try:
        conn.execute("SET lock_configuration=true;")
    except Exception:
        pass

    try:
        # Register in-memory table
        conn.register(table_name, df)
        if table_name != "df":
            conn.register("df", df)

        def _exec():
            return conn.execute(query).fetchdf()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_exec)
            try:
                result = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                try:
                    conn.interrupt()
                except Exception:
                    pass
                raise SandboxSecurityError(f"Query execution timed out ({timeout}s limit exceeded). Possible Cartesian product or runaway computation.")

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
        err_msg = str(e)
        if "Cannot access file" in err_msg or "file system operations are disabled" in err_msg:
            raise SandboxSecurityError(f"Filesystem access blocked by sandbox configuration: {err_msg}")
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e),
        }
    finally:
        conn.close()
