# 📊 Agentic Data Analyst

An autonomous, local tabular data analysis chatbot powered by **local LLMs via Ollama**. It performs end-to-end data exploration, read-only SQL queries via DuckDB, statistical hypothesis testing, outlier detection, and dynamic chart generation—**completely local with zero cloud API keys or external API subscriptions required**.

---

## 🌟 Key Highlights

- **🔒 100% Local & Private**: Runs entirely on your machine using [Ollama](https://ollama.com). No OpenAI, Gemini, or third-party cloud API keys are used. Your data never leaves your computer.
- **🤖 Autonomous Tool-Calling Agent**: Employs an agent loop that dynamically selects and executes specialized Python and SQL tools to inspect and analyze datasets.
- **🛡️ Anti-Hallucination Guardrails**: The agent is strictly instructed to run tools before answering numerical questions and quotes figures directly from tool outputs.
- **💾 Persistent Business Rule Memory**: Users can define domain-specific business rules (e.g., `"Active customers means status is Active and cancelled_flag is 0"`), persisted across sessions in `memory.json` and dynamically injected into prompts.
- **📈 Interactive Web Interface**: A clean Streamlit application with drag-and-drop CSV/TSV/Excel support, synthetic sample data loaders, visual chart displays, and inspectable tool execution traces.

---

## 🛠️ Implemented Analysis Tools

The agent can autonomously call 7 specialized analysis tools implemented in `tools.py`:

| # | Tool | Description | Engine |
|---|---|---|---|
| 1 | `describe_dataset` | Column names, data types, missing value counts, and summary statistics | Pandas |
| 2 | `run_sql` | Read-only SQL queries (`SELECT` / `WITH`) over table `df` with external access disabled | DuckDB |
| 3 | `group_stats` | Group-by aggregations (`sum`, `mean`, `count`, `min`, `max`) with guardrails and truncation | Pandas |
| 4 | `correlation` | Pearson and Spearman correlation matrices and pairwise coefficients | Pandas / NumPy |
| 5 | `find_outliers` | Statistical outlier detection using the Interquartile Range (IQR) method | NumPy / Pandas |
| 6 | `statistical_test` | Hypothesis testing: Two-sample t-test, one-way ANOVA, and Chi-square with plain-English conclusions | SciPy Stats |
| 7 | `make_chart` | Headless chart creation (bar, line, scatter, histogram, box plot) saved to `charts/` | Matplotlib |

---

## 📁 Repository Structure

```
agentic-data-analyst/
├── app.py              # Streamlit web user interface
├── agent.py            # Multi-turn agent loop with tool execution
├── llm.py              # Local Ollama client & tool definitions schema
├── memory.py           # Persistent business rules storage (memory.json)
├── tools.py            # Core data analysis tools (DuckDB SQL, stats, charts)
├── run_app.bat         # 1-click Windows launcher
├── sample_data/        # Synthetic sample datasets (sales, student, health)
│   ├── health_survey.csv
│   ├── sales_transactions.csv
│   └── student_performance.csv
├── tests/              # Pytest test suite (agent, memory, tools)
│   ├── test_agent.py
│   ├── test_memory.py
│   └── test_tools.py
├── requirements.txt    # Project dependencies
├── pytest.ini          # Pytest configuration
└── README.md           # Project documentation
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- **Python 3.10+**
- **Ollama**: Download and install from [ollama.com](https://ollama.com).

### 2. Pull Local Model
Start Ollama and pull the recommended lightweight tool-calling model (default: `qwen2.5:3b`):
```bash
ollama pull qwen2.5:3b
```
*(You can also use `qwen2.5:7b`, `llama3.1`, or any tool-compatible model by setting `OLLAMA_MODEL` environment variable).*

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Application

### Option A: Windows 1-Click Launcher
Double-click `run_app.bat` or run:
```bat
run_app.bat
```

### Option B: Terminal Command
```bash
python -m streamlit run app.py
```

Once running, navigate to `http://localhost:8501` in your browser.

---

## 🧠 Using Business Rules Memory

You can teach the agent custom definitions directly from the chat:
1. Type `remember: <your rule>` in the chat input (e.g. `remember: High value orders are orders where total_amount > 100`).
2. Click **Save Rule** in the confirmation prompt.
3. The rule is saved to `memory.json` and automatically injected whenever relevant questions are asked.
4. Rules can be viewed and deleted anytime from the sidebar.

---

## 🧪 Running Unit Tests

The test suite runs completely offline without network dependencies or active Ollama servers:
```bash
python -m pytest
```

All 15 automated unit tests cover:
- Agent tool calling and answer synthesis with mocked model responses
- Streamlit application headless initialization (`AppTest`)
- Memory persistence, keyword retrieval, and deletion lifecycle
- Clean DataFrame sanitization and numeric string coercion
- Safe SQL execution and security rejection of write/destructive statements
- Group aggregation, outlier detection, correlations, and hypothesis tests
