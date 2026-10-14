"""
Comprehensive tool registry for tabular data analysis.
Contains 8 distinct, fully-typed tools with detailed schemas, docstrings,
and defensive error handling.
"""

from __future__ import annotations
import os
import math
import uuid
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_agent.core.sandbox import run_sandboxed_sql


TOOL_SCHEMAS = [
    {
        "name": "profile_dataset",
        "description": "Examine overall dataset structure: row/column counts, data types, nulls, duplicates, and summary statistics for numerical and categorical features.",
        "parameters": {
            "type": "object",
            "properties": {
                "sample_rows": {
                    "type": "integer",
                    "description": "Number of sample preview rows to include (default 3)",
                    "default": 3
                }
            },
            "required": []
        }
    },
    {
        "name": "run_query",
        "description": "Execute a safe, read-only SQL (DuckDB) query against the dataset table 'df'. Use this for custom filtering, WHERE clauses, selecting specific columns, ordering, or complex calculations.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Read-only SQL query over table 'df' (e.g. \"SELECT category, SUM(total_amount) FROM df WHERE status = 'Active' GROUP BY category\")"
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum number of rows to return (default 50)",
                    "default": 50
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "group_aggregate",
        "description": "Perform group-by operations with aggregations (sum, mean, count, min, max, std, median) across one or more columns, with optional pivoting and sorting.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_cols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to group by (e.g. ['category', 'region'])"
                },
                "agg_spec": {
                    "type": "object",
                    "description": "Mapping from column name to list of aggregation functions, e.g. {'unit_price': ['mean', 'max'], 'quantity': ['sum']}"
                },
                "sort_by": {
                    "type": "string",
                    "description": "Column to sort results by",
                    "default": None
                },
                "ascending": {
                    "type": "boolean",
                    "description": "Sort direction (default False)",
                    "default": False
                },
                "pivot_col": {
                    "type": "string",
                    "description": "Optional column to pivot into columns",
                    "default": None
                }
            },
            "required": ["group_cols", "agg_spec"]
        }
    },
    {
        "name": "correlation_analysis",
        "description": "Analyze linear or monotonic correlations among numerical columns (Pearson/Spearman) and categorical-categorical / numerical-categorical associations (Cramér's V, Correlation Ratio).",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["pearson", "spearman"],
                    "description": "Correlation method for numerical columns (default 'pearson')",
                    "default": "pearson"
                },
                "target_col": {
                    "type": "string",
                    "description": "Optional focus column to rank correlations against",
                    "default": None
                },
                "include_categorical": {
                    "type": "boolean",
                    "description": "Whether to compute Cramér's V associations for categorical features",
                    "default": True
                }
            },
            "required": []
        }
    },
    {
        "name": "detect_outliers",
        "description": "Detect anomalous or outlier rows across specified numerical columns using IQR (Interquartile Range), Z-Score, or Isolation Forest.",
        "parameters": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Numerical columns to test for outliers (if omitted, tests all numeric columns)"
                },
                "method": {
                    "type": "string",
                    "enum": ["iqr", "zscore", "isolation_forest"],
                    "description": "Outlier detection method (default 'iqr')",
                    "default": "iqr"
                },
                "threshold": {
                    "type": "number",
                    "description": "Threshold factor (default 1.5 for IQR, 3.0 for Z-Score, 0.05 contamination for Isolation Forest)",
                    "default": 1.5
                }
            },
            "required": []
        }
    },
    {
        "name": "statistical_test",
        "description": "Perform hypothesis testing (t-test, ANOVA, chi-square, Mann-Whitney U) and receive test statistics, p-values, and a plain-English conclusion.",
        "parameters": {
            "type": "object",
            "properties": {
                "test_type": {
                    "type": "string",
                    "enum": ["ttest_ind", "ttest_paired", "anova", "chi_square", "mann_whitney"],
                    "description": "Statistical test to perform"
                },
                "col1": {
                    "type": "string",
                    "description": "Primary numerical or categorical variable"
                },
                "col2": {
                    "type": "string",
                    "description": "Second variable (for paired test, two-sample test, or chi-square)",
                    "default": None
                },
                "group_col": {
                    "type": "string",
                    "description": "Grouping column for two-sample t-test or one-way ANOVA",
                    "default": None
                },
                "alpha": {
                    "type": "number",
                    "description": "Significance level (default 0.05)",
                    "default": 0.05
                }
            },
            "required": ["test_type", "col1"]
        }
    },
    {
        "name": "make_chart",
        "description": "Generate and save a visualization (bar, line, scatter, histogram, box, heatmap) to visually illustrate findings.",
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "scatter", "histogram", "box", "heatmap"],
                    "description": "Type of visualization"
                },
                "x": {
                    "type": "string",
                    "description": "Column name for X axis"
                },
                "y": {
                    "type": "string",
                    "description": "Column name for Y axis (optional for histogram)",
                    "default": None
                },
                "hue": {
                    "type": "string",
                    "description": "Optional column for color grouping/legend",
                    "default": None
                },
                "title": {
                    "type": "string",
                    "description": "Title of the plot",
                    "default": None
                }
            },
            "required": ["chart_type", "x"]
        }
    },
    {
        "name": "trend_forecast",
        "description": "Analyze time-series trends (moving average, growth rate, linear trend slope) and project future values.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_col": {
                    "type": "string",
                    "description": "Column containing dates or timestamps"
                },
                "value_col": {
                    "type": "string",
                    "description": "Numerical column to analyze and forecast"
                },
                "freq": {
                    "type": "string",
                    "description": "Resampling frequency ('D'=Daily, 'W'=Weekly, 'M'=Monthly, default 'D')",
                    "default": "D"
                },
                "forecast_periods": {
                    "type": "integer",
                    "description": "Number of future time steps to project (default 7)",
                    "default": 7
                }
            },
            "required": ["date_col", "value_col"]
        }
    }
]


# ==============================================================================
# TOOL IMPLEMENTATIONS
# ==============================================================================

def profile_dataset(df: pd.DataFrame, sample_rows: int = 3) -> Dict[str, Any]:
    """Examine dataset dimensions, data types, null counts, and summary statistics."""
    if df.empty:
        return {"status": "error", "message": "Dataset is empty."}

    num_rows, num_cols = df.shape
    duplicate_count = int(df.duplicated().sum())

    columns_info = []
    for col in df.columns:
        s = df[col]
        null_cnt = int(s.isna().sum())
        columns_info.append({
            "name": col,
            "dtype": str(s.dtype),
            "null_count": null_cnt,
            "null_pct": round(null_cnt / num_rows * 100, 2),
            "unique_count": int(s.nunique(dropna=True))
        })

    # Numeric summaries
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_stats = {}
    if not numeric_df.empty:
        desc = numeric_df.describe().T
        for col, row in desc.iterrows():
            numeric_stats[col] = {
                "count": int(row["count"]),
                "mean": round(float(row["mean"]), 3) if not np.isnan(row["mean"]) else None,
                "std": round(float(row["std"]), 3) if not np.isnan(row["std"]) else None,
                "min": round(float(row["min"]), 3) if not np.isnan(row["min"]) else None,
                "25%": round(float(row["25%"]), 3) if not np.isnan(row["25%"]) else None,
                "50%": round(float(row["50%"]), 3) if not np.isnan(row["50%"]) else None,
                "75%": round(float(row["75%"]), 3) if not np.isnan(row["75%"]) else None,
                "max": round(float(row["max"]), 3) if not np.isnan(row["max"]) else None,
            }

    # Categorical summaries
    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    cat_stats = {}
    for col in cat_df.columns:
        top_vals = df[col].value_counts().head(3).to_dict()
        cat_stats[col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in top_vals.items()}
        }

    return {
        "status": "success",
        "row_count": num_rows,
        "col_count": num_cols,
        "duplicate_rows": duplicate_count,
        "columns": columns_info,
        "numeric_stats": numeric_stats,
        "categorical_stats": cat_stats,
        "sample_preview": df.head(sample_rows).to_dict(orient="records")
    }


def run_query(df: pd.DataFrame, query: str, max_rows: int = 50) -> Dict[str, Any]:
    """Execute sandboxed SQL query on table 'df'."""
    res = run_sandboxed_sql(df, query, table_name="df", max_rows=max_rows)
    if "dataframe" in res:
        del res["dataframe"]
    return res


def group_aggregate(
    df: pd.DataFrame,
    group_cols: List[str],
    agg_spec: Dict[str, Union[str, List[str]]],
    sort_by: Optional[str] = None,
    ascending: bool = False,
    pivot_col: Optional[str] = None
) -> Dict[str, Any]:
    """Group by specified columns and compute aggregations."""
    for col in group_cols:
        if col not in df.columns:
            return {"status": "error", "message": f"Group column '{col}' not in dataset columns."}

    # Normalize agg_spec
    clean_agg = {}
    for col, aggs in agg_spec.items():
        if col not in df.columns:
            return {"status": "error", "message": f"Aggregation column '{col}' not in dataset columns."}
        clean_agg[col] = [aggs] if isinstance(aggs, str) else list(aggs)

    try:
        grouped = df.groupby(group_cols).agg(clean_agg)

        # Flatten multi-level columns
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [f"{col}_{fn}" for col, fn in grouped.columns]

        grouped = grouped.reset_index()

        if pivot_col and pivot_col in grouped.columns:
            val_cols = [c for c in grouped.columns if c not in group_cols]
            if val_cols:
                index_cols = [c for c in group_cols if c != pivot_col]
                if index_cols:
                    grouped = grouped.pivot(index=index_cols, columns=pivot_col, values=val_cols[0]).reset_index()

        if sort_by:
            if sort_by in grouped.columns:
                grouped = grouped.sort_values(by=sort_by, ascending=ascending)
            else:
                matches = [c for c in grouped.columns if sort_by in c]
                if matches:
                    grouped = grouped.sort_values(by=matches[0], ascending=ascending)

        records = grouped.to_dict(orient="records")
        return {
            "status": "success",
            "group_cols": group_cols,
            "result_rows": len(records),
            "data": records[:100],
            "truncated": len(records) > 100
        }
    except Exception as e:
        return {"status": "error", "message": f"Group aggregate failed: {str(e)}"}


def correlation_analysis(
    df: pd.DataFrame,
    method: str = "pearson",
    target_col: Optional[str] = None,
    include_categorical: bool = True
) -> Dict[str, Any]:
    """Calculate numerical correlations and categorical associations."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {"status": "error", "message": "No numerical columns available for correlation analysis."}

    corr_matrix = numeric_df.corr(method=method).round(4)
    corr_dict = corr_matrix.to_dict()

    top_correlations = []
    cols = list(corr_matrix.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val):
                top_correlations.append({
                    "col1": cols[i],
                    "col2": cols[j],
                    "correlation": float(val),
                    "abs_correlation": abs(float(val))
                })

    top_correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)

    target_ranking = None
    if target_col and target_col in corr_matrix.columns:
        s = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
        target_ranking = [{"column": k, "correlation": round(float(v), 4)} for k, v in s.items()]

    categorical_associations = {}
    if include_categorical:
        cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                c1, c2 = cat_cols[i], cat_cols[j]
                cont_table = pd.crosstab(df[c1], df[c2])
                if cont_table.size > 0 and cont_table.sum().sum() > 0:
                    chi2 = stats.chi2_contingency(cont_table)[0]
                    n = cont_table.sum().sum()
                    phi2 = chi2 / n
                    r, k = cont_table.shape
                    cramers_v = np.sqrt(phi2 / min(k - 1, r - 1)) if min(k - 1, r - 1) > 0 else 0.0
                    categorical_associations[f"{c1} ~ {c2}"] = round(float(cramers_v), 4)

    return {
        "status": "success",
        "method": method,
        "numeric_columns_count": len(cols),
        "top_numeric_correlations": top_correlations[:15],
        "target_ranking": target_ranking,
        "categorical_associations": categorical_associations
    }


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5
) -> Dict[str, Any]:
    """Detect outliers via IQR, Z-Score, or Isolation Forest."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {"status": "error", "message": "No numerical columns available for outlier detection."}

    target_cols = [c for c in columns if c in numeric_df.columns] if columns else list(numeric_df.columns)
    if not target_cols:
        return {"status": "error", "message": f"None of the requested columns {columns} are numerical."}

    outlier_summary = {}
    total_outlier_indices = set()

    if method == "iqr":
        for col in target_cols:
            s = df[col].dropna()
            q25, q75 = s.quantile(0.25), s.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - (threshold * iqr)
            upper_bound = q75 + (threshold * iqr)
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            total_outlier_indices.update(outliers.index.tolist())
            outlier_summary[col] = {
                "lower_bound": round(float(lower_bound), 3),
                "upper_bound": round(float(upper_bound), 3),
                "outlier_count": len(outliers),
                "outlier_percentage": round(len(outliers) / len(df) * 100, 2),
                "sample_values": [round(float(v), 2) for v in outliers[col].head(5).tolist()]
            }

    elif method == "zscore":
        for col in target_cols:
            s = df[col].dropna()
            mean, std = s.mean(), s.std()
            if std == 0 or np.isnan(std):
                continue
            z_scores = np.abs((s - mean) / std)
            outliers = df.loc[s.index[z_scores > threshold]]
            total_outlier_indices.update(outliers.index.tolist())
            outlier_summary[col] = {
                "mean": round(float(mean), 3),
                "std": round(float(std), 3),
                "threshold_z": threshold,
                "outlier_count": len(outliers),
                "outlier_percentage": round(len(outliers) / len(df) * 100, 2),
            }

    elif method == "isolation_forest":
        clean_subset = df[target_cols].dropna()
        if len(clean_subset) < 5:
            return {"status": "error", "message": "Not enough rows for Isolation Forest analysis (requires at least 5 rows)."}
        contamination = min(0.2, max(0.01, threshold if threshold < 0.5 else 0.05))
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(clean_subset)
        outlier_idx = clean_subset.index[preds == -1].tolist()
        total_outlier_indices.update(outlier_idx)
        outlier_summary["multivariate_isolation_forest"] = {
            "tested_columns": target_cols,
            "contamination_rate": contamination,
            "outlier_count": len(outlier_idx),
            "outlier_percentage": round(len(outlier_idx) / len(clean_subset) * 100, 2)
        }
    else:
        return {"status": "error", "message": f"Unknown outlier method '{method}'. Use 'iqr', 'zscore', or 'isolation_forest'."}

    sample_outlier_rows = df.loc[list(total_outlier_indices)[:5]].to_dict(orient="records")

    return {
        "status": "success",
        "method": method,
        "total_anomalous_rows": len(total_outlier_indices),
        "total_rows": len(df),
        "column_breakdown": outlier_summary,
        "sample_outlier_rows": sample_outlier_rows
    }


def statistical_test(
    df: pd.DataFrame,
    test_type: str,
    col1: str,
    col2: Optional[str] = None,
    group_col: Optional[str] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Execute hypothesis tests with plain-English interpretations."""
    if col1 not in df.columns:
        return {"status": "error", "message": f"Column '{col1}' not found in dataset."}

    try:
        if test_type == "ttest_ind":
            # Two-sample independent t-test
            if group_col and group_col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[group_col]):
                    col1, group_col = group_col, col1
                groups = df[group_col].dropna().unique()
                if len(groups) != 2:
                    return {"status": "error", "message": f"Group column '{group_col}' must contain exactly 2 unique categories for independent t-test. Found: {list(groups)}"}
                g1 = df[df[group_col] == groups[0]][col1].dropna()
                g2 = df[df[group_col] == groups[1]][col1].dropna()
                stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                g1_name, g2_name = str(groups[0]), str(groups[1])
            elif col2 and col2 in df.columns:
                g1 = df[col1].dropna()
                g2 = df[col2].dropna()
                stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                g1_name, g2_name = col1, col2
            else:
                return {"status": "error", "message": "ttest_ind requires either a second column 'col2' or a 2-class 'group_col'."}

            sig = p_val < alpha
            interpretation = (
                f"Statistically SIGNIFICANT difference detected between {g1_name} (mean={g1.mean():.2f}) "
                f"and {g2_name} (mean={g2.mean():.2f}) (t = {stat:.3f}, p = {p_val:.4e} < {alpha}). "
                f"We reject the null hypothesis of equal means."
                if sig else
                f"NO statistically significant difference detected between {g1_name} (mean={g1.mean():.2f}) "
                f"and {g2_name} (mean={g2.mean():.2f}) (t = {stat:.3f}, p = {p_val:.4f} >= {alpha}). "
                f"We fail to reject the null hypothesis."
            )
            return {
                "status": "success",
                "test": "Welch's Independent Two-Sample T-Test",
                "statistic": round(float(stat), 4),
                "p_value": float(p_val),
                "alpha": alpha,
                "significant": sig,
                "group1": {"name": g1_name, "mean": round(float(g1.mean()), 3), "n": len(g1)},
                "group2": {"name": g2_name, "mean": round(float(g2.mean()), 3), "n": len(g2)},
                "interpretation": interpretation
            }

        elif test_type == "ttest_paired":
            if not col2 or col2 not in df.columns:
                return {"status": "error", "message": "Paired t-test requires both 'col1' and 'col2'."}
            valid = df[[col1, col2]].dropna()
            stat, p_val = stats.ttest_rel(valid[col1], valid[col2])
            sig = p_val < alpha
            interpretation = (
                f"Statistically SIGNIFICANT difference between paired observations of {col1} and {col2} "
                f"(t = {stat:.3f}, p = {p_val:.4e} < {alpha})."
                if sig else
                f"NO statistically significant paired difference between {col1} and {col2} "
                f"(t = {stat:.3f}, p = {p_val:.4f} >= {alpha})."
            )
            return {
                "status": "success",
                "test": "Paired Samples T-Test",
                "statistic": round(float(stat), 4),
                "p_value": float(p_val),
                "significant": sig,
                "interpretation": interpretation
            }

        elif test_type == "anova":
            if not group_col or group_col not in df.columns:
                return {"status": "error", "message": "One-Way ANOVA requires 'group_col' specifying the categorical groups."}
            # Auto-swap if user or LLM passed categorical in col1 and numeric in group_col
            if not pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[group_col]):
                col1, group_col = group_col, col1
            elif not pd.api.types.is_numeric_dtype(df[col1]):
                return {"status": "error", "message": f"Metric column '{col1}' must be numerical for ANOVA."}
            groups = [group[col1].dropna().values for _, group in df.groupby(group_col)]
            if len(groups) < 2:
                return {"status": "error", "message": "ANOVA requires at least 2 distinct groups."}
            stat, p_val = stats.f_oneway(*groups)
            sig = p_val < alpha
            interpretation = (
                f"Statistically SIGNIFICANT difference across groups in '{group_col}' for metric '{col1}' "
                f"(F = {stat:.3f}, p = {p_val:.4e} < {alpha}). At least one group mean significantly differs."
                if sig else
                f"NO statistically significant difference across groups in '{group_col}' for metric '{col1}' "
                f"(F = {stat:.3f}, p = {p_val:.4f} >= {alpha}). Group means appear statistically comparable."
            )
            return {
                "status": "success",
                "test": "One-Way Analysis of Variance (ANOVA)",
                "statistic": round(float(stat), 4),
                "p_value": float(p_val),
                "significant": sig,
                "group_column": group_col,
                "num_groups": len(groups),
                "interpretation": interpretation
            }

        elif test_type == "chi_square":
            if not col2 or col2 not in df.columns:
                return {"status": "error", "message": "Chi-Square test of independence requires both 'col1' and 'col2'."}
            contingency = pd.crosstab(df[col1], df[col2])
            stat, p_val, dof, _ = stats.chi2_contingency(contingency)
            sig = p_val < alpha
            interpretation = (
                f"Statistically SIGNIFICANT association between '{col1}' and '{col2}' "
                f"(Chi2 = {stat:.3f}, dof = {dof}, p = {p_val:.4e} < {alpha}). Variables are dependent."
                if sig else
                f"NO statistically significant association between '{col1}' and '{col2}' "
                f"(Chi2 = {stat:.3f}, dof = {dof}, p = {p_val:.4f} >= {alpha}). Variables appear independent."
            )
            return {
                "status": "success",
                "test": "Pearson's Chi-Square Test of Independence",
                "statistic": round(float(stat), 4),
                "p_value": float(p_val),
                "degrees_of_freedom": int(dof),
                "significant": sig,
                "interpretation": interpretation
            }

        elif test_type == "mann_whitney":
            if not group_col or group_col not in df.columns:
                return {"status": "error", "message": "Mann-Whitney U test requires 'group_col'."}
            groups = df[group_col].dropna().unique()
            if len(groups) != 2:
                return {"status": "error", "message": f"Mann-Whitney U requires exactly 2 groups in '{group_col}'."}
            g1 = df[df[group_col] == groups[0]][col1].dropna()
            g2 = df[df[group_col] == groups[1]][col1].dropna()
            stat, p_val = stats.mannwhitneyu(g1, g2)
            sig = p_val < alpha
            return {
                "status": "success",
                "test": "Mann-Whitney U Test (Non-Parametric)",
                "statistic": round(float(stat), 4),
                "p_value": float(p_val),
                "significant": sig,
                "interpretation": f"{'Significant' if sig else 'No significant'} distribution shift between {groups[0]} and {groups[1]} (U={stat:.1f}, p={p_val:.4f})."
            }
        else:
            return {"status": "error", "message": f"Unsupported test type '{test_type}'."}
    except Exception as e:
        return {"status": "error", "message": f"Statistical test execution failed: {str(e)}"}


def make_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Render and save an exploratory data visualization."""
    os.makedirs("charts", exist_ok=True)
    if x not in df.columns:
        return {"status": "error", "message": f"Column '{x}' not found in dataset."}
    if y and y not in df.columns:
        return {"status": "error", "message": f"Column '{y}' not found in dataset."}
    if hue and hue not in df.columns:
        hue = None

    if not save_path:
        filename = f"{chart_type}_{x}_{uuid.uuid4().hex[:6]}.png"
        save_path = os.path.join("charts", filename)

    plt.figure(figsize=(9, 5.5))
    sns.set_theme(style="whitegrid")

    try:
        if chart_type == "bar":
            if y:
                sns.barplot(data=df, x=x, y=y, hue=hue, errorbar=None, palette="viridis")
            else:
                sns.countplot(data=df, x=x, hue=hue, palette="viridis")
            plt.xticks(rotation=35, ha="right")

        elif chart_type == "line":
            sns.lineplot(data=df, x=x, y=y, hue=hue, marker="o", palette="tab10")
            plt.xticks(rotation=35, ha="right")

        elif chart_type == "scatter":
            sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="deep", alpha=0.75, s=60)

        elif chart_type == "histogram":
            sns.histplot(data=df, x=x, hue=hue, kde=True, palette="muted", bins=20)

        elif chart_type == "box":
            sns.boxplot(data=df, x=x, y=y, hue=hue, palette="Set2")
            plt.xticks(rotation=35, ha="right")

        elif chart_type == "heatmap":
            num_cols = list(df.select_dtypes(include=[np.number]).columns)
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)

        else:
            plt.close()
            return {"status": "error", "message": f"Unsupported chart type '{chart_type}'."}

        chart_title = title or f"{chart_type.title()} Plot of {x}" + (f" vs {y}" if y else "")
        plt.title(chart_title, fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()

        return {
            "status": "success",
            "chart_type": chart_type,
            "saved_image_path": save_path,
            "title": chart_title
        }
    except Exception as e:
        plt.close()
        return {"status": "error", "message": f"Chart rendering failed: {str(e)}"}


def trend_forecast(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "D",
    forecast_periods: int = 7
) -> Dict[str, Any]:
    """Compute time series moving averages, linear trend, and simple future projections."""
    if date_col not in df.columns or value_col not in df.columns:
        return {"status": "error", "message": f"Columns '{date_col}' or '{value_col}' missing from dataset."}

    try:
        temp = df[[date_col, value_col]].dropna().copy()
        temp[date_col] = pd.to_datetime(temp[date_col], format="mixed", errors="coerce")
        temp = temp.dropna(subset=[date_col])
        temp = temp.sort_values(by=date_col)

        # Resample
        ts = temp.set_index(date_col)[value_col].resample(freq).sum().fillna(0)
        if len(ts) < 3:
            return {"status": "error", "message": "Not enough historical timestamps to establish a trend."}

        n = len(ts)
        x_numeric = np.arange(n)
        y_values = ts.values

        # Linear regression trend
        slope, intercept, r_val, p_val, std_err = stats.linregress(x_numeric, y_values)

        # 3-period moving average
        ma = ts.rolling(window=min(3, n), min_periods=1).mean().round(2).tolist()

        # Forecast future points
        future_x = np.arange(n, n + forecast_periods)
        future_y = [max(0.0, round(float(slope * fx + intercept), 2)) for fx in future_x]

        last_date = ts.index[-1]
        future_dates = [
            (last_date + pd.Timedelta(days=(i + 1) * (7 if freq == "W" else 1))).strftime("%Y-%m-%d")
            for i in range(forecast_periods)
        ]

        projections = [{"date": d, "projected_value": val} for d, val in zip(future_dates, future_y)]

        return {
            "status": "success",
            "historical_periods": n,
            "trend_direction": "Upward" if slope > 0 else ("Downward" if slope < 0 else "Flat"),
            "daily_trend_slope": round(float(slope), 3),
            "r_squared": round(float(r_val**2), 3),
            "p_value": float(p_val),
            "recent_actuals": [{"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)} for d, v in ts.tail(5).items()],
            "forecast_projections": projections
        }
    except Exception as e:
        return {"status": "error", "message": f"Trend forecast failed: {str(e)}"}


# Map tool names to functions
TOOL_REGISTRY = {
    "profile_dataset": profile_dataset,
    "run_query": run_query,
    "group_aggregate": group_aggregate,
    "correlation_analysis": correlation_analysis,
    "detect_outliers": detect_outliers,
    "statistical_test": statistical_test,
    "make_chart": make_chart,
    "trend_forecast": trend_forecast
}
