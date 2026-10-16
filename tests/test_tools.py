import os, pytest
import pandas as pd
from tools import describe_dataset, run_sql, group_stats, correlation, find_outliers, statistical_test, make_chart

@pytest.fixture
def sample_df():
    """Create a simple test dataframe with numeric and categorical data."""
    return pd.DataFrame({
        "status": ["Set", "Set", "Unset", "Set", "Unset"],
        "category": ["A", "A", "B", "B", "A"],
        "value": [10.0, 20.0, 15.0, 30.0, 25.0],
        "score": [100.0, 200.0, 150.0, 300.0, 250.0],
    })

def test_describe_dataset(sample_df):
    """Verify high-level dataset metrics and types."""
    res = describe_dataset(sample_df)
    assert res["row_count"] == 5
    assert res["column_count"] == 4
    assert "value" in res["columns"]
    assert "value" in res["numeric_summary"]

def test_run_sql_allowed(sample_df):
    """Verify normal SELECT works, including WHERE condition matching status = 'Set'."""
    res = run_sql(sample_df, "SELECT status, AVG(value) AS avg_v FROM df WHERE status = 'Set' GROUP BY status")
    assert "error" not in res
    assert len(res["data"]) == 1
    assert round(res["data"][0]["avg_v"], 1) == 20.0

def test_run_sql_blocks_drop(sample_df):
    """Verify queries starting with DROP are blocked."""
    res = run_sql(sample_df, "DROP TABLE df")
    assert "error" in res
    assert "rejected" in res["error"].lower()

def test_run_sql_blocks_chained_queries(sample_df):
    """Verify semicolon-chained statements are blocked."""
    res = run_sql(sample_df, "SELECT * FROM df; DROP TABLE df")
    assert "error" in res
    assert "rejected" in res["error"].lower()

def test_run_sql_blocks_external_file_access(sample_df):
    """Verify duckdb cannot access external filesystem files."""
    res = run_sql(sample_df, "SELECT * FROM 'tests/fixtures/dirty_types.csv'")
    assert "error" in res

def test_group_stats(sample_df):
    """Verify group_stats computes count, mean, sum, min, max."""
    res = group_stats(sample_df, group_by="status", target_column="value")
    assert "error" not in res
    assert len(res["results"]) == 2

def test_correlation(sample_df):
    """Verify correlation matrix and strongest pairs."""
    res = correlation(sample_df)
    assert "error" not in res
    assert len(res["strongest_pairs"]) > 0
    assert abs(res["strongest_pairs"][0]["correlation"]) > 0.9

def test_find_outliers():
    """Verify IQR outlier detection."""
    df = pd.DataFrame({"val": [10, 12, 11, 13, 12, 14, 11, 100]})
    res = find_outliers(df, "val")
    assert "error" not in res
    assert res["outlier_count"] == 1
    assert 100 in res["outlier_samples"]

def test_statistical_test(sample_df):
    """Verify t-test and ANOVA return valid test statistics and conclusions."""
    ttest_res = statistical_test(sample_df, "ttest", "value", "status")
    assert "statistic" in ttest_res
    assert "conclusion" in ttest_res

    anova_res = statistical_test(sample_df, "anova", "value", "category")
    assert "statistic" in anova_res
    assert "conclusion" in anova_res

def test_make_chart(sample_df):
    """Verify chart generation saves a PNG image file."""
    res = make_chart(sample_df, chart_type="bar", x="category", y="value")
    assert "error" not in res
    assert os.path.exists(res["chart_path"])
    # Clean up generated test chart
    if os.path.exists(res["chart_path"]):
        os.remove(res["chart_path"])
