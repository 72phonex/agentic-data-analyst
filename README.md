# Agentic Data Analyst

An autonomous, general-purpose tabular data analysis system supporting CSV and Excel files. Features 8 analysis tools, sandboxed SQL query execution via DuckDB, persistent cross-session memory in SQLite, an interactive Streamlit UI, and a command-line interface.

---

## LLM Backend Configuration

The agent supports three LLM execution modes, configurable via the `LLM_BACKEND` environment variable.

### Default: Heuristic Fallback (Testing / Offline / CI Only)
By default, `LLM_BACKEND` is set to `"heuristic"`.
- **Honest Note**: The heuristic mode is a rule-based deterministic fallback for testing, CI, and offline evaluation without API keys. It is **not an LLM**; it uses regex pattern matching and query templates, and only handles simple, canned question forms.
- To use the full general-purpose agentic capabilities with dynamic reasoning and multi-step tool chaining on unseen questions, configure a real LLM backend below (Gemini or Ollama).

### Optional Alternative Backends
1. **Google Gemini API**:
   ```bash
   export LLM_BACKEND="gemini"
   export GEMINI_API_KEY="your-gemini-api-key"
   ```
2. **Local Ollama Model**:
   ```bash
   export LLM_BACKEND="ollama"
   export OLLAMA_MODEL="qwen2.5:7b"
   ```

### Memory Database Configuration
The SQLite persistent memory database path is configurable via the `MEMORY_DB_PATH` environment variable:
```bash
export MEMORY_DB_PATH="data_agent_memory.sqlite"
```
Defaults to `data_agent_memory.sqlite`. Setting `MEMORY_DB_PATH` (or using the `--memory-db` CLI flag) enables running tests, demonstrations, and concurrent evaluations against isolated database files without stomping persistent state.

---

## Setup & Installation

### Requirements
- Python 3.10+
- Virtual environment recommended

### Installation
```bash
pip install -r requirements.txt
```

---

## Execution Commands

### 1. Interactive Streamlit Web UI
```bash
streamlit run app.py
```
Provides a browser interface with drag-and-drop CSV upload, interactive chat, memory inspector, and automatic chart rendering.

### 2. Interactive Command-Line Interface (CLI)
```bash
python cli.py
```
Supports interactive analysis and special slash commands:
- `/help`: Display available tools and commands
- `/memory`: Inspect all stored definitions in SQLite
- `/forget <id>`: Delete a stored memory
- `/edit <id> <val>`: Update a stored memory with audit tracking
- `/clear`: Reset current session context
- `/exit`: Terminate session

### 3. Scripted Memory Lifecycle Demonstration
```bash
python demo_memory_flow.py
```
Runs a 2-session demonstration verifying persistent memory definition, cross-session restart, definition injection into SQL queries, and audit history tracking.

### 4. Test Suite Execution
```bash
pytest tests/ -v
```

### 5. Independent Verification & Benchmark Scripts
```bash
# Independently verify demo revenue figures via plain pandas
python scripts/verify_figures.py

# Verify SQL injection & filesystem blocking in DuckDB sandbox
python scripts/test_sandbox_attacks.py

# Verify end-to-end SQLite memory lifecycle (persistence, audit, forget, relevance)
python scripts/test_memory_lifecycle.py

# Run agent across 2 new unrelated CSV datasets (5 varied questions each)
python scripts/test_unrelated_datasets.py
```

---

## Verification Results & Benchmark Table

| Test Suite / Verification Step | Target / Invariant | Real Output / Empirical Result | Status |
|---|---|---|---|
| **Pytest Test Suite** | 46 automated unit tests | 46 passed, 0 failed in 6.52s | **PASS** |
| **Total Revenue Verification** | Match demo claim | $92,477.23 (sum of `total_amount`) | **PASS** |
| **Active Customer Revenue** | `status = 'Active'` & `cancelled_flag = 0` | $37,023.46 (across 91 rows) | **PASS** |
| **DuckDB Security Sandbox** | Block DROP, INSERT, read_csv, PRAGMA, chained ';', Cartesian DoS | 13 of 13 attack vectors blocked, 4/4 legitimate queries allowed | **PASS** |
| **Memory Persistence** | Persist across process restart | 1 memory entry recovered from SQLite | **PASS** |
| **Memory Audit Trail** | Record changes in `memory_history` | 1 audit change record logged with timestamp | **PASS** |
| **Memory Relevance Filter** | No irrelevant memories injected | 0 memories injected for unrelated query, 1 for related query | **PASS** |
| **Memory Deletion** | `/forget <id>` command | Memory removed from SQLite (0 remaining) | **PASS** |
| **Unrelated Dataset 1** | Employee Attrition (250 rows, 9 cols) | 5/5 questions executed across analysis tools | **PASS** |
| **Unrelated Dataset 2** | Real Estate Listings (250 rows, 9 cols) | 5/5 questions executed across analysis tools | **PASS** |

---

## Implemented Analysis Tools

| # | Tool Name | Description | Technology |
|---|---|---|---|
| 1 | `profile_dataset` | Schema inspection, data types, missing values, duplicates, summary stats | Pandas |
| 2 | `run_query` | Read-only SQL queries with automatic row limit truncation | Sandboxed DuckDB |
| 3 | `group_aggregate` | Group-by aggregations (`sum`, `mean`, `count`, `min`, `max`) and pivots | Pandas |
| 4 | `correlation_analysis` | Pearson / Spearman correlation matrices and strong association discovery | SciPy / Pandas |
| 5 | `detect_outliers` | Statistical outlier detection via IQR, Z-Score, and Isolation Forest | Scikit-Learn / SciPy |
| 6 | `statistical_test` | Hypothesis testing: Two-sample t-test, Chi-square independence, One-way ANOVA | SciPy Stats |
| 7 | `make_chart` | Headless generation of bar, scatter, line, histogram, and box charts | Matplotlib / Seaborn |
| 8 | `trend_forecast` | Date-based time series linear regression trend projection | Scikit-Learn |

---


### Datasets
> **Note**: All datasets in this repository are synthetic, generated for testing and demonstration purposes.

## Repository Structure & File Map

```
agentic-data-analyst/
├── app.py                        # Streamlit web user interface
├── cli.py                        # Terminal command-line interface
├── demo_memory_flow.py           # Cross-session persistent memory demonstration
├── data_agent/
│   └── core/
│       ├── agent.py              # Main execution loop and retry logic
│       ├── data_loader.py        # CSV/Excel parser and metadata extractor
│       ├── llm.py                # Multi-backend router (heuristic, gemini, ollama)
│       ├── memory.py             # Persistent SQLite memory engine and audit logger
│       ├── sandbox.py            # DuckDB SQL validator and execution sandbox
│       └── tools.py              # 8 core data analysis tools
├── data/
│   ├── sales_transactions.csv    # 200 retail sales records, 11 columns
│   ├── student_performance.csv   # 150 academic performance records, 10 columns
│   ├── health_survey.csv         # 180 wellness survey records, 10 columns
│   ├── employee_attrition.csv    # 250 HR compensation records, 9 columns
│   └── real_estate_listings.csv  # 250 property market records, 9 columns
├── scripts/
│   ├── generate_datasets.py      # Demo dataset generation script
│   ├── test_all_datasets.py      # Batch evaluation runner
│   ├── verify_figures.py         # Independent plain pandas revenue verification
│   ├── test_sandbox_attacks.py   # SQL injection and sandbox escape verification
│   ├── test_memory_lifecycle.py  # End-to-end memory lifecycle verification
│   └── test_unrelated_datasets.py# Verification on 2 new datasets (10 questions)
├── tests/
│   ├── test_agent.py             # Agent interaction tests
│   ├── test_memory.py            # SQLite memory and audit tests
│   ├── test_sandbox.py           # DuckDB security sandbox tests
│   └── test_tools.py             # Unit tests for all 8 analysis tools
├── notes/
│   └── limitations.md            # Synthetic data limitations and sandbox security notes
├── requirements.txt              # Project package dependencies
├── README_SOURCES.md             # Third-party libraries and licenses
└── README.md                     # Project documentation
```

---

## Approach
<!-- TODO: write in my own words -->

---

## What Worked
<!-- TODO: write in my own words -->

---

## What Failed
<!-- TODO: write in my own words -->

---

## What I Would Try Next
<!-- TODO: write in my own words -->

---

## Limitations
<!-- TODO: write in my own words -->

### Known Technical Limitations:
- **SQLite Concurrency**: SQLite file-based locking restricts high-throughput concurrent writes in distributed multi-user deployments.
- **DuckDB In-Memory Scope**: Queries execute in isolated, ephemeral in-memory DuckDB instances (`:memory:`) with external access disabled, preventing disk persistence or cross-session table creation.
- **Single-Dataset-at-a-Time Scope**: The core orchestrator binds to one active DataFrame at a time; cross-file SQL joins across disparate CSVs require merging datasets prior to loading.
- **Context Windowing**: Multi-turn chat sessions currently accumulate full history; very long sessions without periodic truncation could exceed LLM token context limits.
- **Heuristic Fallback Scope**: When running without API keys or Ollama, the deterministic heuristic router relies on regex intent patterns and query templates; it does not perform general semantic reasoning or dynamic multi-step SQL synthesis.

---

## Personal Reflections
<!-- TODO: write in my own words -->
