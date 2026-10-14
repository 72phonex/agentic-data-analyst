# Engineering Limitations & Design Considerations: Agentic Data Analyst

---

## 1. Synthetic Demo Datasets & Statistical Interpretation

The demonstration datasets included with this project (`data/sales_transactions.csv`, `data/student_performance.csv`, `data/health_survey.csv`, `data/employee_attrition.csv`, and `data/real_estate_listings.csv`) were synthetically generated using `scripts/generate_datasets.py` and `scripts/generate_unrelated_datasets.py` to benchmark system components (schema profiling, memory injection, SQL query generation, outlier detection, and chart generation).

### Key Implications:
- **Weak or Artificially Controlled Statistical Associations**: Unlike real enterprise operational data, synthetic data generated via pseudorandom distributions often displays lower correlation coefficients (e.g., $|r| < 0.3$) or non-significant p-values in hypothesis testing ($p > 0.05$).
- **No Domain-Specific Noise**: Synthetic data lacks real-world messiness such as corrupted character encodings, inconsistent date formats across decades, or extreme real-world multivariate clustering.
- **Agent Behavior**: The agent strictly computes and reports the genuine empirical calculations on the loaded dataset without hallucinating significance or faking positive trends.

---

## 2. Hardened SQL Sandbox Architecture & Capabilities

The DuckDB sandboxing module (`data_agent/core/sandbox.py`) has been hardened with multi-layered defense-in-depth security:

### What the Sandbox Now Does:
1. **Engine-Level Access Isolation**:
   - Every execution connects via `duckdb.connect(database=":memory:", config={"enable_external_access": False})`.
   - File system reads/writes and external network extensions are completely disabled at the C++ kernel level.
   - Configuration is locked via `SET lock_configuration=true;` so queries cannot re-enable access at runtime.
2. **Direct File Path & Function Blocking**:
   - Directly catches and blocks file paths in `FROM` clauses (e.g., `SELECT * FROM '/etc/passwd'` or `SELECT * FROM 'data/sales.csv'`).
   - Forbids all I/O functions (`read_csv`, `read_parquet`, `glob`, `httpfs`, `sqlite_scan`, etc.).
3. **AST-Level Verification via `sqlglot`**:
   - Parses statements into abstract syntax trees to guarantee that every submitted query is strictly a single, read-only `SELECT` or `WITH` statement.
   - Forbids mutation AST nodes (`Drop`, `Insert`, `Delete`, `Update`, `Alter`, `Create`, `Command`, `Pragma`, `Copy`).
4. **False Positive Reduction via Literal & Identifier Masking**:
   - Semicolons are split only outside string literals, allowing queries like `WHERE val = 'a;b'`.
   - Quoted string literals (`'...'`) and quoted identifiers (`"..."`) are masked prior to keyword checking. Legitimate queries containing keywords like `WHERE status = 'Set'`, `WHERE status = 'Drop'`, or `SELECT "export"` are permitted without false-positive blocks.
5. **Execution Timeout & Runaway Protection**:
   - Queries are executed in isolated worker threads with a strict execution timeout (default: 5.0 seconds).
   - If a runaway Cartesian product (e.g., multi-table cross join) exceeds the timeout, `conn.interrupt()` is dispatched and a `SandboxSecurityError` is raised.
6. **Result Row Capping**:
   - Truncates query results exceeding `max_rows` (default: 200) to prevent memory exhaustion and UI lockups.

### Remaining Honest Limitations:
- **Transient Memory Spikes Before Timeout**: While execution timeouts terminate runaway Cartesian products after $N$ seconds, a query producing millions of rows per second can cause a brief RAM spike before the thread interrupt takes effect. True bulletproof multi-tenant isolation requires OS-level memory cgroups or microVM containerization (e.g., Docker, gVisor).
- **In-Memory Table Scope Only**: Queries operate exclusively on in-memory registered DataFrames. No table persistence across queries is permitted.
- **Dialect Edge Cases**: Advanced DuckDB-specific functions or procedural macros not recognized by standard ANSI SQL AST parsers may fall back to lexical validation.
