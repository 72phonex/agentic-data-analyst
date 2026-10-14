import pytest
import pandas as pd
from data_agent.core.sandbox import run_sandboxed_sql, validate_sql_query, SandboxSecurityError

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "David"],
        "score": [85, 92, 78, 95]
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
        run_sandboxed_sql(sample_df, "INSERT INTO df VALUES (5, 'Eve', 100)")

def test_block_pragma(sample_df):
    with pytest.raises(SandboxSecurityError):
        run_sandboxed_sql(sample_df, "PRAGMA version")

def test_block_filesystem_read(sample_df):
    with pytest.raises(SandboxSecurityError):
        run_sandboxed_sql(sample_df, "SELECT * FROM read_csv('data/sales_transactions.csv')")

def test_block_multiple_statements(sample_df):
    with pytest.raises(SandboxSecurityError):
        run_sandboxed_sql(sample_df, "SELECT * FROM df; SELECT * FROM df")

def test_row_limit_truncation(sample_df):
    res = run_sandboxed_sql(sample_df, "SELECT * FROM df", max_rows=2)
    assert res["status"] == "success"
    assert res["returned_rows"] == 2
    assert res["truncated"] is True
