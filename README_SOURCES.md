# External Sources & Open-Source Libraries: Agentic Data Analyst

This document details all third-party libraries, execution engines, and optional LLM backends utilized within the Agentic Data Analyst system.

---

## 1. Core Data Processing & Sandboxed Execution

- **DuckDB (`duckdb`)**:
  - Source: DuckDB Foundation (`https://duckdb.org`)
  - License: MIT
  - Role: High-performance embedded in-process analytical SQL database. Used as the core query execution engine with strict query sanitization and blocklisting (blocking DROP, ALTER, INSERT, UPDATE, DELETE, multi-statement chaining, and filesystem access functions like `read_csv` and `read_parquet`).
- **Pandas (`pandas`)**:
  - Source: NumFOCUS / PyData (`https://pandas.pydata.org`)
  - License: BSD-3-Clause
  - Role: Tabular data representation, schema profiling, group aggregations, missing value analysis, and data preprocessing.
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

- **Matplotlib (`matplotlib`)**:
  - Source: Matplotlib Development Team (`https://matplotlib.org`)
  - License: PSF / BSD-compatible
  - Role: Headless rendering and export of high-resolution publication-quality PNG charts.
- **Plotly (`plotly`)**:
  - Source: Plotly Technologies Inc. (`https://plotly.com`)
  - License: MIT
  - Role: Interactive web-ready data visualisations.
- **Streamlit (`streamlit`)**:
  - Source: Snowflake Inc. (`https://streamlit.io`)
  - License: Apache 2.0
  - Role: Interactive web user interface supporting drag-and-drop CSV uploads, live conversation transcripts, interactive memory inspection, and dynamic chart rendering.

---

## 4. LLM Routing & Multi-Backend Architecture

- **Heuristic Rule-Based Planner (Default Fallback)**:
  - Custom deterministic regex-guided tool planner that requires zero external API keys and guarantees zero operational cost while correctly routing queries to appropriate tools.
- **Google Generative AI SDK (`google-generativeai`)**:
  - Source: Google LLC (`https://github.com/google-gemini/generative-ai-python`)
  - License: Apache 2.0
  - Role: Optional backend supporting Gemini 1.5 Pro / Flash models via the `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variable.
- **Ollama REST API (`requests`)**:
  - Source: Ollama (`https://ollama.com`)
  - Role: Optional local LLM backend (e.g., Llama 3, Qwen 2.5) running completely locally via standard HTTP endpoints (`http://localhost:11434/api/generate`).
