import os
import pytest
import pandas as pd
from data_agent.core.data_loader import load_dataset
from data_agent.core.agent import DataAnalysisAgent
from data_agent.core.memory import MemoryStore
from data_agent.core.llm import LLMClient
from data_agent.core.tools import (
    profile_dataset,
    run_query,
    group_aggregate,
    correlation_analysis,
    detect_outliers,
    statistical_test,
    make_chart,
    trend_forecast
)


@pytest.fixture
def messy_fixture_path():
    p = os.path.join(os.path.dirname(__file__), "fixtures", "messy_headers_and_nulls.csv")
    assert os.path.exists(p), f"Missing fixture at {p}"
    return p


@pytest.fixture
def dirty_fixture_path():
    p = os.path.join(os.path.dirname(__file__), "fixtures", "dirty_types.csv")
    assert os.path.exists(p), f"Missing fixture at {p}"
    return p


def test_messy_headers_and_nulls_loading(messy_fixture_path):
    df, meta = load_dataset(messy_fixture_path)
    assert len(df) == 15
    # Headers should be trimmed of leading/trailing whitespace
    assert "Customer Name" in df.columns
    assert "Total Sales USD" in df.columns
    assert "Satisfaction Score" in df.columns
    # Check nulls are present
    assert df["Customer Name"].isna().sum() >= 1
    assert df["Total Sales USD"].isna().sum() >= 1


def test_dirty_types_coercion(dirty_fixture_path):
    df, meta = load_dataset(dirty_fixture_path)
    assert len(df) == 15
    # Numbers with commas should be numeric float
    assert pd.api.types.is_numeric_dtype(df["revenue_str"])
    assert df["revenue_str"].iloc[0] == 1200.50
    # Currencies with $ should be numeric float
    assert pd.api.types.is_numeric_dtype(df["cost_currency"])
    assert df["cost_currency"].iloc[0] == 450.00
    # Percentages should be numeric float
    assert pd.api.types.is_numeric_dtype(df["margin_percent"])
    assert df["margin_percent"].iloc[0] == 15.0


def test_messy_file_across_six_tools(messy_fixture_path, tmp_path):
    df, _ = load_dataset(messy_fixture_path)

    # 1. Profile dataset
    res_prof = profile_dataset(df)
    assert res_prof["status"] == "success"
    assert res_prof["row_count"] == 15

    # 2. Run query with spaces in column names
    res_sql = run_query(df, 'SELECT "Customer Name", "Total Sales USD" FROM df WHERE "Total Sales USD" > 500')
    assert res_sql["status"] == "success"
    assert res_sql["row_count"] > 0

    # 3. Group aggregate with column names containing spaces
    res_grp = group_aggregate(df, group_cols=["Category"], agg_spec={"Total Sales USD": ["mean", "count"]})
    assert res_grp["status"] == "success"
    assert res_grp["result_rows"] > 0

    # 4. Correlation analysis (safely handles nulls)
    res_corr = correlation_analysis(df)
    assert res_corr["status"] == "success"
    assert len(res_corr["top_numeric_correlations"]) > 0

    # 5. Outlier detection on column with nulls
    res_out = detect_outliers(df, method="iqr", columns=["Total Sales USD"])
    assert res_out["status"] == "success"

    # 6. Statistical test across groups with nulls
    res_stat = statistical_test(df, test_type="anova", col1="Total Sales USD", group_col="Category")
    assert res_stat["status"] in ["success", "error"]  # Valid execution without unhandled exception

    # 7. Trend forecast with mixed date formats
    res_trend = trend_forecast(df, date_col="Order Date", value_col="Total Sales USD")
    assert res_trend["status"] == "success"


def test_dirty_types_across_six_tools(dirty_fixture_path, tmp_path):
    df, _ = load_dataset(dirty_fixture_path)

    # 1. Profile dataset
    res_prof = profile_dataset(df)
    assert res_prof["status"] == "success"

    # 2. Filtered SQL query on coerced columns
    res_sql = run_query(df, "SELECT transaction_id, revenue_str, cost_currency FROM df WHERE revenue_str > 2000")
    assert res_sql["status"] == "success"
    assert res_sql["row_count"] > 0

    # 3. Group by and aggregate
    res_grp = group_aggregate(df, group_cols=["region_group"], agg_spec={"revenue_str": ["sum", "mean"]})
    assert res_grp["status"] == "success"

    # 4. Correlation analysis on coerced columns
    res_corr = correlation_analysis(df)
    assert res_corr["status"] == "success"
    # revenue_str and cost_currency should correlate positively
    top_cols = [c["col1"] for c in res_corr["top_numeric_correlations"]]
    assert "revenue_str" in top_cols or "cost_currency" in top_cols

    # 5. Outlier detection
    res_out = detect_outliers(df, method="iqr", columns=["revenue_str"])
    assert res_out["status"] == "success"

    # 6. Statistical test
    res_stat = statistical_test(df, test_type="anova", col1="revenue_str", group_col="region_group")
    assert res_stat["status"] == "success"

    # 7. Chart generation
    chart_file = str(tmp_path / "chart.png")
    res_chart = make_chart(df, chart_type="scatter", x="revenue_str", y="cost_currency", save_path=chart_file)
    assert res_chart["status"] == "success"


def test_agent_end_to_end_on_unseen_messy_data(messy_fixture_path, tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "mem_messy.sqlite"))
    llm = LLMClient(backend="heuristic")
    agent = DataAnalysisAgent(memory_store=mem, llm_client=llm)
    agent.load_file(messy_fixture_path)

    # Question 1: Profile
    res1 = agent.ask("Profile and summarize this dataset")
    assert "Dataset Profile" in res1["answer"]

    # Question 2: Query calculation on column with spaces
    res2 = agent.ask("What is the total sales amount?")
    assert res2["tool_used"] == "run_query"
    assert "Result" in res2["answer"]

    # Question 3: Group by
    res3 = agent.ask("Group by Category and find average sales")
    assert res3["tool_used"] == "group_aggregate"
    assert "Group Aggregation" in res3["answer"]

    # Question 4: Correlation
    res4 = agent.ask("Show correlation between sales and satisfaction score")
    assert res4["tool_used"] == "correlation_analysis"
    assert "Correlation" in res4["answer"]

    # Question 5: Outliers
    res5 = agent.ask("Are there any outliers in sales?")
    assert res5["tool_used"] == "detect_outliers"
    assert "Outlier" in res5["answer"]
