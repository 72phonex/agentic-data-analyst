import pytest
import os
import pandas as pd
import numpy as np
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
def df():
    return pd.DataFrame({
        "category": ["A", "A", "B", "B", "C", "C", "A", "B"],
        "region": ["North", "South", "North", "South", "North", "South", "North", "South"],
        "amount": [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 105.0, 210.0],
        "quantity": [2, 3, 4, 5, 6, 7, 2, 4],
        "date": pd.date_range("2024-01-01", periods=8, freq="D")
    })

def test_profile_dataset(df):
    res = profile_dataset(df)
    assert res["status"] == "success"
    assert res["row_count"] == 8
    assert res["col_count"] == 5
    assert "amount" in res["numeric_stats"]
    assert "category" in res["categorical_stats"]

def test_run_query(df):
    res = run_query(df, "SELECT category, SUM(amount) AS total FROM df GROUP BY category ORDER BY total DESC")
    assert res["status"] == "success"
    assert len(res["data"]) == 3
    assert res["data"][0]["category"] == "B"

def test_group_aggregate(df):
    res = group_aggregate(
        df,
        group_cols=["category"],
        agg_spec={"amount": ["mean", "sum"], "quantity": ["sum"]}
    )
    assert res["status"] == "success"
    assert res["result_rows"] == 3
    assert "amount_mean" in res["data"][0]

def test_correlation_analysis(df):
    res = correlation_analysis(df, method="pearson")
    assert res["status"] == "success"
    assert len(res["top_numeric_correlations"]) > 0
    # Amount and quantity are perfectly positively correlated
    assert res["top_numeric_correlations"][0]["correlation"] > 0.9

def test_detect_outliers(df):
    # Add an extreme outlier
    df_outlier = df.copy()
    df_outlier.loc[len(df_outlier)] = ["A", "North", 50000.0, 50, pd.Timestamp("2024-01-09")]

    res_iqr = detect_outliers(df_outlier, columns=["amount"], method="iqr")
    assert res_iqr["status"] == "success"
    assert res_iqr["total_anomalous_rows"] >= 1

    res_iso = detect_outliers(df_outlier, columns=["amount", "quantity"], method="isolation_forest")
    assert res_iso["status"] == "success"
    assert res_iso["total_anomalous_rows"] >= 1

def test_statistical_test(df):
    # Two-sample t-test
    res_ttest = statistical_test(df, test_type="ttest_ind", col1="amount", group_col="region")
    assert res_ttest["status"] == "success"
    assert "p_value" in res_ttest
    assert "interpretation" in res_ttest

    # ANOVA across 3 categories
    res_anova = statistical_test(df, test_type="anova", col1="amount", group_col="category")
    assert res_anova["status"] == "success"
    assert res_anova["num_groups"] == 3

    # Chi-square between category and region
    res_chi = statistical_test(df, test_type="chi_square", col1="category", col2="region")
    assert res_chi["status"] == "success"
    assert "degrees_of_freedom" in res_chi

def test_make_chart(df):
    os.makedirs("charts", exist_ok=True)
    res = make_chart(df, chart_type="bar", x="category", y="amount")
    assert res["status"] == "success"
    assert os.path.exists(res["saved_image_path"])

def test_trend_forecast(df):
    res = trend_forecast(df, date_col="date", value_col="amount", freq="D", forecast_periods=3)
    assert res["status"] == "success"
    assert len(res["forecast_projections"]) == 3
    assert res["trend_direction"] in ["Upward", "Downward", "Flat"]
