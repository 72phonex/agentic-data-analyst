import os, time, duckdb
import numpy as np, pandas as pd, scipy.stats as stats
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for servers
import matplotlib.pyplot as plt

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from headers and safely coerce numeric-looking strings into float."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna().astype(str).str.strip().str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            converted = pd.to_numeric(s, errors="coerce")
            if len(s) > 0 and (converted.notna().sum() / len(s)) >= 0.7:
                full_s = df[col].astype(str).str.strip().str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
                df[col] = pd.to_numeric(full_s, errors="coerce")
    return df

def describe_dataset(df: pd.DataFrame) -> dict:
    """Return high-level summary: row count, columns, data types, nulls, and stats."""
    num_df = df.select_dtypes(include=[np.number])
    stats_dict = num_df.describe().round(2).to_dict() if not num_df.empty else {}
    return {
        "row_count": int(len(df)), "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "data_types": {c: str(d) for c, d in df.dtypes.items()},
        "missing_values": {c: int(n) for c, n in df.isnull().sum().items()},
        "numeric_summary": stats_dict,
    }

def run_sql(df: pd.DataFrame, query: str) -> dict:
    """Run a safe, read-only SQL query against the table 'df' using DuckDB."""
    clean_q = query.strip().rstrip(";")
    upper_q = clean_q.upper()
    if not (upper_q.startswith("SELECT") or upper_q.startswith("WITH")):
        return {"error": "Query rejected: Only read-only SELECT or WITH statements are allowed."}
    if ";" in clean_q:
        return {"error": "Query rejected: Semicolon-chained statements are not allowed."}
    try:
        con = duckdb.connect(config={"enable_external_access": False})
        con.register("df", df)
        result = con.execute(clean_q).fetch_df().head(200)
        return {"columns": list(result.columns), "row_count": len(result), "data": result.to_dict(orient="records")}
    except Exception as e:
        return {"error": f"SQL execution error: {str(e)}"}

def group_stats(df: pd.DataFrame, group_by: str = None, target_column: str = None) -> dict:
    """Group rows by a column and compute aggregate stats (sum, mean, count, min, max).
    If `group_by` is omitted, compute overall aggregates for `target_column`.
    Guardrail: if the grouping column has >50% unique values or >50 distinct groups, return an error.
    Results are truncated to the first 30 rows with a notice.
    """
    # Validate target column
    if not target_column:
        return {"error": "Missing required argument 'target_column'."}
    if target_column not in df.columns:
        return {"error": f"Column '{target_column}' not found. Available columns: {list(df.columns)}"}

    # No grouping requested – overall aggregate
    if not group_by:
        series = pd.to_numeric(df[target_column], errors='coerce')
        agg = {
            "count": int(series.count()),
            "mean": round(series.mean(), 2),
            "sum": round(series.sum(), 2),
            "min": round(series.min(), 2),
            "max": round(series.max(), 2),
        }
        return {"group_by": None, "target_column": target_column, "results": [agg]}

    # Validate grouping column
    if group_by not in df.columns:
        return {"error": f"Column '{group_by}' not found. Available columns: {list(df.columns)}"}
    unique_vals = df[group_by].nunique()
    total_rows = len(df)
    if unique_vals > 0.5 * total_rows or unique_vals > 50:
        return {"error": f"Column '{group_by}' has {unique_vals} unique values out of {total_rows} rows, so grouping by it gives one row per record. Pick a category column with few distinct values, or use run_sql for an overall total."}
    # Perform grouping
    res = df.groupby(group_by)[target_column].agg(["count", "mean", "sum", "min", "max"]).round(2).reset_index()
    truncated = False
    if len(res) > 30:
        res = res.head(30)
        truncated = True
    out = {"group_by": group_by, "target_column": target_column, "results": res.to_dict(orient='records')}
    if truncated:
        out["truncated"] = "Only first 30 groups shown."
    return out

def correlation(df: pd.DataFrame, method: str = "pearson", column1: str = None, column2: str = None, **kwargs) -> dict:
    """Calculate correlation between numeric columns and rank the strongest pairs."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return {"error": "Need at least two numeric columns for correlation."}
    corr = num_df.corr(method=method).round(3)
    if column1 and column2 and column1 in corr.columns and column2 in corr.columns:
        return {"col1": column1, "col2": column2, "correlation": float(corr.loc[column1, column2]), "matrix": corr.to_dict()}
    cols = list(num_df.columns)
    pairs = [{"col1": cols[i], "col2": cols[j], "correlation": float(corr.iloc[i, j])}
             for i in range(len(cols)) for j in range(i + 1, len(cols)) if not np.isnan(corr.iloc[i, j])]
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return {"method": method, "matrix": corr.to_dict(), "strongest_pairs": pairs[:5]}

def find_outliers(df: pd.DataFrame, column: str) -> dict:
    """Identify outliers in a numeric column using the Interquartile Range (IQR) rule."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return {"error": f"Column '{column}' contains no numeric data."}
    q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < low) | (series > high)]
    return {"column": column, "q1": round(q1, 2), "q3": round(q3, 2), "iqr": round(iqr, 2),
            "lower_bound": round(low, 2), "upper_bound": round(high, 2),
            "outlier_count": len(outliers), "outlier_samples": [round(float(v), 2) for v in outliers.head(10)]}

def statistical_test(df: pd.DataFrame, test_type: str, column1: str, column2: str = None) -> dict:
    """Run t-test, ANOVA, or chi-square test with a plain-English conclusion."""
    t = test_type.lower().replace("-", "_")
    try:
        if t in ("ttest", "anova", "one_way_anova"):
            if not column2: return {"error": "Test requires column1 (numeric) and column2 (groups)."}
            if not pd.api.types.is_numeric_dtype(df[column2]):
                groups = [pd.to_numeric(grp[column1], errors="coerce").dropna().values for _, grp in df.groupby(column2) if len(grp) > 0]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) == 2 and t == "ttest":
                    stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False); used = "t-test"
                elif len(groups) >= 2:
                    stat, p_val = stats.f_oneway(*groups); used = "one-way ANOVA"
                else: return {"error": "Need at least 2 distinct groups with numeric data."}
            else:
                s1, s2 = pd.to_numeric(df[column1], errors="coerce").dropna(), pd.to_numeric(df[column2], errors="coerce").dropna()
                stat, p_val = stats.ttest_ind(s1, s2, equal_var=False); used = "t-test"
            is_sig = bool(p_val < 0.05)
            conclusion = f"Statistically significant ({used}, p={p_val:.4f} < 0.05)" if is_sig else f"Not statistically significant ({used}, p={p_val:.4f} >= 0.05)"
            return {"test": used, "statistic": round(float(stat), 4), "p_value": round(float(p_val), 5), "significant": is_sig, "conclusion": conclusion}
        elif t in ("chi_square", "chisquare"):
            if not column2: return {"error": "Chi-square requires two categorical columns."}
            stat, p_val, _, _ = stats.chi2_contingency(pd.crosstab(df[column1], df[column2]))
            is_sig = bool(p_val < 0.05)
            conclusion = f"Statistically significant (Chi-square, p={p_val:.4f} < 0.05)" if is_sig else f"Not statistically significant (p={p_val:.4f} >= 0.05)"
            return {"test": "chi_square", "statistic": round(float(stat), 4), "p_value": round(float(p_val), 5), "significant": is_sig, "conclusion": conclusion}
        return {"error": f"Unsupported test '{test_type}'. Use 'ttest', 'anova', or 'chi_square'."}
    except Exception as e:
        return {"error": f"Statistical test error: {str(e)}"}

def make_chart(df: pd.DataFrame, chart_type: str, x: str, y: str = None, title: str = None) -> dict:
    """Create and save a matplotlib chart (bar, line, scatter, histogram, box)."""
    os.makedirs("charts", exist_ok=True)
    path = os.path.join("charts", f"chart_{int(time.time() * 1000)}.png")
    plt.figure(figsize=(7, 4.2))
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    try:
        c = chart_type.lower()
        if c == "bar":
            val_col = y if (y and y in df.columns) else (num_cols[0] if num_cols else None)
            agg = "sum" if (y and str(y).lower() in ("sum", "total")) else "mean"
            if val_col and val_col in df.columns and x in df.columns:
                df.groupby(x)[val_col].agg(agg).plot(kind="bar", color="#3b82f6")
            elif x in df.columns:
                df[x].value_counts().head(10).plot(kind="bar", color="#3b82f6")
        elif c == "line":
            val_col = y if (y and y in df.columns) else (num_cols[0] if num_cols else None)
            df.plot(x=x, y=val_col, kind="line", color="#10b981", ax=plt.gca())
        elif c == "scatter":
            val_col = y if (y and y in df.columns) else (num_cols[1] if len(num_cols) > 1 else num_cols[0])
            plt.scatter(df[x], df[val_col], alpha=0.7, color="#8b5cf6")
            plt.xlabel(x); plt.ylabel(val_col)
        elif c in ("histogram", "hist"):
            col = x if x in df.columns else (num_cols[0] if num_cols else x)
            pd.to_numeric(df[col], errors="coerce").dropna().plot(kind="hist", bins=15, color="#f59e0b", edgecolor="black", ax=plt.gca())
        elif c in ("box", "boxplot"):
            col = y if (y and y in df.columns) else (x if x in df.columns else num_cols[0])
            df.boxplot(column=col, by=(x if (y and x in df.columns and x != col) else None), ax=plt.gca())
        else:
            plt.close(); return {"error": f"Unsupported chart_type '{chart_type}'."}
        plt.title(title or f"{chart_type.title()} of {x}" + (f" vs {y}" if y else ""))
        plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
        return {"chart_path": path, "chart_type": chart_type, "status": "saved"}
    except Exception as e:
        plt.close(); return {"error": f"Chart error: {str(e)}"}
