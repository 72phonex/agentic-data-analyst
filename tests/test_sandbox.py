import pytest
import pandas as pd
from data_agent.core.sandbox import run_sandboxed_sql, validate_sql_query, SandboxSecurityError

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "David"],
        "score": [85, 92, 78, 95],
        "status": ["Active", "Set", "Drop", "Pending"],
        "export": [10, 20, 30, 40]
    })

def test_safe_select(sample_df):
    res = run_sandboxed_sql(sample_df, "SELECT name, score FROM df WHERE score > 80")
    assert res["status"] == "success"
    assert res["row_count"] == 3
    assert len(res["data"]) == 3

def test_aggregate_sql(sample_df):
    res = run_sandboxed_sql(sample_df, "SELECT AVG(score) as avg_score FROM df")
    assert res["status"] == "success"
    assert round(res["data"][0]["avg_score"], 1) == 87.5

def test_block_drop(sample_df):
    with pytest.raises(SandboxSecurityError):
        run_sandboxed_sql(sample_df, "DROP TABLE df")

def test_block_insert(sample_df):
    with pytest.raises(SandboxSecurityError):
        run_sandboxed_sql(sample_df, "INSERT INTO df VALUES (5, 'Eve', 100, 'Active', 50)")

def test_block_attach(sample_df):
    with pytest.raises(SandboxSecurityError):
        run_sandboxed_sql(sample_df, "ATTACH 'data/test.db' AS test_db")

def test_block_pragma(sample_df):
    with pytest.raises(SandboxSecurityError):
        run_sandboxed_sql(sample_df, "PRAGMA version")

def test_block_copy(sample_df):
    with pytest.raises(SandboxSecurityError):
        run_sandboxed_sql(sample_df, "COPY df TO 'out.csv'")

def test_block_direct_path_etc_passwd(sample_df):
    with pytest.raises(SandboxSecurityError):
        run_sandboxed_sql(sample_df, "SELECT * FROM '/etc/passwd'")

def test_block_direct_path_csv(sample_df):
    with pytest.raises(SandboxSecurityError):
        run_sandboxed_sql(sample_df, "SELECT * FROM 'data/sales_transactions.csv'")

def test_block_read_csv(sample_df):
    with pytest.raises(SandboxSecurityError):
        run_sandboxed_sql(sample_df, "SELECT * FROM read_csv('data/sales_transactions.csv')")

def test_block_glob(sample_df):
    with pytest.raises(SandboxSecurityError):
        run_sandboxed_sql(sample_df, "SELECT * FROM glob('*.csv')")

def test_block_multiple_statements(sample_df):
    with pytest.raises(SandboxSecurityError):
        run_sandboxed_sql(sample_df, "SELECT * FROM df; SELECT * FROM df")

def test_block_comments_hiding_keywords(sample_df):
    with pytest.raises(SandboxSecurityError):
        run_sandboxed_sql(sample_df, "SELECT * FROM df; -- comment\n DROP TABLE df")

def test_block_mixed_case(sample_df):
    with pytest.raises(SandboxSecurityError):
        run_sandboxed_sql(sample_df, "sElEcT * fRoM dF; dRoP tAbLe dF")

def test_allow_keyword_set_inside_string_literal(sample_df):
    res = run_sandboxed_sql(sample_df, "SELECT * FROM df WHERE status = 'Set'")
    assert res["status"] == "success"
    assert res["row_count"] == 1
    assert res["data"][0]["status"] == "Set"

def test_allow_keyword_drop_inside_string_literal(sample_df):
    res = run_sandboxed_sql(sample_df, "SELECT * FROM df WHERE status = 'Drop'")
    assert res["status"] == "success"
    assert res["row_count"] == 1
    assert res["data"][0]["status"] == "Drop"

def test_allow_quoted_export_column_name(sample_df):
    res = run_sandboxed_sql(sample_df, "SELECT \"export\", name FROM df WHERE \"export\" > 15")
    assert res["status"] == "success"
    assert res["row_count"] == 3
    assert "export" in res["columns"]

def test_row_limit_truncation(sample_df):
    res = run_sandboxed_sql(sample_df, "SELECT * FROM df", max_rows=2)
    assert res["status"] == "success"
    assert res["returned_rows"] == 2
    assert res["truncated"] is True

def test_execution_timeout_cartesian_product():
    large_df = pd.DataFrame({"x": range(50000)})
    with pytest.raises(SandboxSecurityError) as exc_info:
        run_sandboxed_sql(
            large_df,
            "SELECT COUNT(*) FROM df a, df b, df c, df d",
            timeout=0.1
        )
    assert "timed out" in str(exc_info.value).lower()
