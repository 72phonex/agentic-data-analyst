import os, time, duckdb
import numpy as np, pandas as pd, scipy.stats as stats
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for servers
import matplotlib.pyplot as plt

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric-looking strings (e.g. '1,200' or '$50') into float/int safely."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            cleaned = df[col].dropna().astype(str).str.strip().str.replace("$", "", regex=False).str.replace(",", "", regex=False)
            converted = pd.to_numeric(cleaned, errors="coerce")
            if len(cleaned) > 0 and (converted.notna().sum() / len(cleaned)) >= 0.8:
                df[col] = pd.to_numeric(df[col].astype(str).str.strip().str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
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
    # Safety: Only allow SELECT or WITH queries to prevent destructive changes
    if not (upper_q.startswith("SELECT") or upper_q.startswith("WITH")):
        return {"error": "Query rejected: Only read-only SELECT or WITH statements are allowed."}
    # Safety: Block semicolon chaining to prevent multi-statement injection
    if ";" in clean_q:
        return {"error": "Query rejected: Semicolon-chained statements are not allowed."}
    try:
        # enable_external_access=False prevents reading filesystem files or network from SQL
        con = duckdb.connect(config={"enable_external_access": False})
        con.register("df", df)
        result = con.execute(clean_q).fetch_df().head(200)  # Cap at 200 rows for memory safety
        return {"columns": list(result.columns), "row_count": len(result), "data": result.to_dict(orient="records")}
    except Exception as e:
        return {"error": f"SQL execution error: {str(e)}"}

def group_stats(df: pd.DataFrame, group_by: str, target_column: str) -> dict:
    """Group rows by one column and compute aggregate stats (sum, mean, count, min, max)."""
    if group_by not in df.columns or target_column not in df.columns:
        return {"error": f"Columns not found. Available columns: {list(df.columns)}"}
    res = df.groupby(group_by)[target_column].agg(["count", "mean", "sum", "min", "max"]).round(2).reset_index()
    return {"group_by": group_by, "target_column": target_column, "results": res.to_dict(orient="records")}

def correlation(df: pd.DataFrame, method: str = "pearson") -> dict:
    """Calculate correlation between numeric columns and rank the strongest pairs."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return {"error": "Need at least two numeric columns for correlation."}
    corr = num_df.corr(method=method).round(3)
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
        if t == "ttest":
            if column2 and df[column2].nunique() == 2:
                g = [grp[column1].dropna().values for _, grp in df.groupby(column2)]
                stat, p_val = stats.ttest_ind(g[0], g[1], equal_var=False)
            elif column2:
                stat, p_val = stats.ttest_ind(df[column1].dropna(), df[column2].dropna(), equal_var=False)
            else:
                return {"error": "t-test requires column1 (numeric) and column2 (2 groups or 2nd numeric column)."}
        elif t in ("anova", "one_way_anova"):
            if not column2: return {"error": "ANOVA requires column1 (numeric) and column2 (group categorical)."}
            groups = [grp[column1].dropna().values for _, grp in df.groupby(column2) if len(grp[column1].dropna()) > 0]
            stat, p_val = stats.f_oneway(*groups)
        elif t in ("chi_square", "chisquare"):
            if not column2: return {"error": "Chi-square requires two categorical columns."}
            stat, p_val, _, _ = stats.chi2_contingency(pd.crosstab(df[column1], df[column2]))
        else:
            return {"error": f"Unsupported test '{test_type}'. Use 'ttest', 'anova', or 'chi_square'."}
        is_sig = bool(p_val < 0.05)
        conclusion = f"Statistically significant (p={p_val:.4f} < 0.05)" if is_sig else f"Not statistically significant (p={p_val:.4f} >= 0.05)"
        return {"test": t, "statistic": round(float(stat), 4), "p_value": round(float(p_val), 5), "significant": is_sig, "conclusion": conclusion}
    except Exception as e:
        return {"error": f"Statistical test error: {str(e)}"}

def make_chart(df: pd.DataFrame, chart_type: str, x: str, y: str = None, title: str = None) -> dict:
    """Create and save a matplotlib chart (bar, line, scatter, histogram, box)."""
    os.makedirs("charts", exist_ok=True)
    path = os.path.join("charts", f"chart_{int(time.time() * 1000)}.png")
    plt.figure(figsize=(7, 4.2))
    try:
        c = chart_type.lower()
        if c == "bar":
            (df.groupby(x)[y].mean() if y else df[x].value_counts().head(10)).plot(kind="bar", color="#3b82f6")
        elif c == "line":
            df.plot(x=x, y=y, kind="line", color="#10b981", ax=plt.gca())
        elif c == "scatter":
            plt.scatter(df[x], df[y], alpha=0.7, color="#8b5cf6")
            plt.xlabel(x); plt.ylabel(y)
        elif c in ("histogram", "hist"):
            df[x].plot(kind="hist", bins=15, color="#f59e0b", edgecolor="black", ax=plt.gca())
        elif c in ("box", "boxplot"):
            df.boxplot(column=y if y else x, by=x if y else None, ax=plt.gca())
        else:
            plt.close()
            return {"error": f"Unsupported chart_type '{chart_type}'. Choose bar, line, scatter, histogram, box."}
        plt.title(title or f"{chart_type.title()} of {x}" + (f" vs {y}" if y else ""))
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        return {"chart_path": path, "chart_type": chart_type, "status": "saved"}
    except Exception as e:
        plt.close()
        return {"error": f"Chart error: {str(e)}"}
