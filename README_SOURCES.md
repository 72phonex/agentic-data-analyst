# External Sources & Open-Source Libraries: Agentic Data Analyst

This document details all third-party libraries, execution engines, and optional LLM backends utilized within the Agentic Data Analyst system.

---

## 1. Core Data Processing & Sandboxed Execution

- **DuckDB (`duckdb`)**:
  - Source: DuckDB Foundation (`https://duckdb.org`)
  - License: MIT
  - Role: High-performance embedded in-process analytical SQL database. Used as the core query execution engine with `enable_external_access=False` to strictly forbid file and network access from SQL.
- **SQLGlot (`sqlglot`)**:
  - Source: Toby Mao / SQLGlot Developers (`https://github.com/tobymao/sqlglot`)
  - License: MIT
  - Role: AST SQL parser used to validate that user queries are strictly single read-only `SELECT` or `WITH` statements, preventing token-evasion and chained attack vectors.
- **Pandas (`pandas`)**:
  - Source: NumFOCUS / PyData (`https://pandas.pydata.org`)
  - License: BSD-3-Clause
  - Role: Tabular data representation, schema profiling, group aggregations, missing value analysis, and data preprocessing.
- **OpenPyXL (`openpyxl`)**:
  - Source: OpenPyXL Developers (`https://openpyxl.readthedocs.io`)
  - License: MIT
  - Role: Excel spreadsheet (`.xlsx`) parsing engine for `pandas.read_excel`.
- **NumPy (`numpy`)**:
  - Source: NumPy Developers (`https://numpy.org`)
  - License: BSD-3-Clause
  - Role: Fast numerical array operations and mathematical primitives.

---

## 2. Statistical Analysis & Machine Learning

- **Scikit-Learn (`scikit-learn`)**:
  - Source: Scikit-learn developers (`https://scikit-learn.org`)
  - License: BSD-3-Clause
  - Role: `IsolationForest` algorithm for multidimensional anomaly and outlier detection; `LinearRegression` for baseline trend estimation.
- **SciPy (`scipy`)**:
  - Source: SciPy community (`https://scipy.org`)
  - License: BSD-3-Clause
  - Role: Hypothesis testing in `statistical_test` tool: Two-sample Student's t-test (`scipy.stats.ttest_ind`), Chi-square test of independence (`scipy.stats.chi2_contingency`), and One-way Analysis of Variance (`scipy.stats.f_oneway`).

---

## 3. Visualization & User Interface

- **Matplotlib (`matplotlib`) & Seaborn (`seaborn`)**:
  - Source: Matplotlib Development Team (`https://matplotlib.org`) / Michael Waskom (`https://seaborn.pydata.org`)
  - License: BSD-compatible
  - Role: Headless rendering and export of high-resolution statistical PNG charts (bar, scatter, line, histogram, box).
- **Streamlit (`streamlit`)**:
  - Source: Snowflake Inc. (`https://streamlit.io`)
  - License: Apache 2.0
  - Role: Interactive web user interface supporting drag-and-drop CSV uploads, live conversation transcripts, interactive memory inspection, and dynamic chart rendering.

---

## 4. LLM Routing & Multi-Backend Architecture

- **Heuristic Rule-Based Planner (Default Fallback)**:
  - Custom deterministic regex-guided tool planner that requires zero external API keys and guarantees zero operational cost while correctly routing queries to appropriate tools.
- **Google GenAI SDK (`google-genai`)**:
  - Source: Google LLC (`https://github.com/googleapis/python-genai`)
  - License: Apache 2.0
  - Role: Official modern Google GenAI SDK for Gemini 2.5 / 2.0 / 1.5 models via the `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variable.
- **Requests (`requests`)**:
  - Source: Python Software Foundation (`https://requests.readthedocs.io`)
  - License: Apache 2.0
  - Role: HTTP communication with local Ollama daemon (`http://localhost:11434/api/chat`).
