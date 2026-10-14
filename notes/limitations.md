# Engineering Limitations & Design Considerations: Agentic Data Analyst

---

## 1. Synthetic Demo Datasets & Statistical Interpretation

The demonstration datasets included with this project (`data/sales_transactions.csv`, `data/student_performance.csv`, `data/health_survey.csv`, `data/employee_attrition.csv`, and `data/real_estate_listings.csv`) were synthetically generated to test and benchmark system components (schema profiling, memory injection, SQL query generation, outlier detection, and chart generation).

### Key Implications:
- **Weak or Artificially Controlled Statistical Associations**: Unlike real enterprise operational data, synthetic data generated via pseudorandom distributions often displays lower correlation coefficients (e.g., $|r| < 0.3$) or non-significant p-values in hypothesis testing ($p > 0.05$).
- **No Domain-Specific Noise**: Synthetic data lacks real-world messiness such as corrupted character encodings, inconsistent date formats across decades, or extreme real-world multivariate clustering.
- **Agent Behavior**: The agent strictly computes and reports the genuine empirical calculations on the loaded dataset without hallucinating significance or faking positive trends.

---

## 2. Security Limits of Keyword & Token Blocklists in SQL Sandboxes

The DuckDB sandboxing module (`data_agent/core/sandbox.py`) enforces safety via token extraction and regex-based blocklists:
1. **Statement Count Enforcer**: Blocks semicolon-chained queries (`split(";")`).
2. **Keyword Blocklist**: Forbids schema mutation keywords (`DROP`, `ALTER`, `CREATE`, `INSERT`, `UPDATE`, `DELETE`, `PRAGMA`, `ATTACH`, `DETACH`, etc.).
3. **Function Blocklist**: Forbids filesystem and network functions (`read_csv`, `read_parquet`, `glob`, `httpfs`, etc.).
4. **Statement Whitelist**: Enforces that queries must start with `SELECT`, `WITH`, or `DESCRIBE`.

### Inherent Limitations of This Approach:
- **Lexical Evasion & Obfuscation**: Advanced SQL injection techniques in complex database dialects may use encoding, comments inside identifier names, or nested dialect-specific functions to bypass simple token scanners.
- **Resource Exhaustion & Denial of Service (DoS)**: Keyword checking does not prevent resource-intensive queries that are syntactically valid `SELECT` statements (e.g., unindexed Cartesian products like `SELECT * FROM df CROSS JOIN df CROSS JOIN df`, or pathological regular expressions). These can cause high CPU utilization or memory spikes.
- **Information Leakage via Error Messages**: If an attacker attempts to infer schema or table details, raw database error strings might reveal in-memory structure details.

### Recommended Production Mitigations:
1. **Process-Level Isolation**: Run the analytical engine in an unprivileged, isolated microVM or container (e.g., Docker with drop-all capabilities, gVisor sandbox).
2. **Strict Timeouts & Memory Cgroups**: Enforce execution limits (e.g., 5-second query timeout, 512MB RAM cap per session).
3. **AST-Level SQL Parser**: Replace token scanning with a formal SQL AST parser (e.g., `sqlglot`) to inspect the abstract syntax tree and forbid non-SELECT AST nodes regardless of string formatting.
