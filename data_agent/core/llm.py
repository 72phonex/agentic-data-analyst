"""
Configurable LLM provider for Tabular Data Analysis Agent.
Supports:
  - Gemini API (google-genai / google-generativeai) via GEMINI_API_KEY or GOOGLE_API_KEY
  - Ollama local model via OLLAMA_HOST and OLLAMA_MODEL (e.g. llama3.1, qwen2.5)
  - Heuristic deterministic planner (for zero-dependency testing, CI, and offline demos)
"""

from __future__ import annotations
import os
import json
import re
from typing import Any, Dict, List, Optional
import requests

from data_agent.core.tools import TOOL_SCHEMAS


SYSTEM_INSTRUCTIONS = """You are an expert Data Analysis Agent for tabular data (CSV/Excel).
You MUST call tools to compute answers. NEVER guess, invent, or hallucinate numerical answers.
Always inspect tool outputs before producing your final synthesis.
If a tool returns an error, examine the error message and retry with a corrected parameter or query.
When user domain definitions or memories are present, incorporate them strictly into your SQL queries and filters.
"""

SYNTHESIS_SYSTEM_INSTRUCTIONS = """You are an expert Data Analysis Agent synthesizing computed tool outputs.
CRITICAL NUMERICAL FIDELITY REQUIREMENT:
You MUST base your response STRICTLY on the tool outputs and numbers provided in the tool execution trace.
Use the EXACT numbers, metrics, column names, and calculated values from the tool results.
Do NOT invent, hallucinate, extrapolate, or alter any numbers.
If the tool result contains an error or 0 matching rows, state that fact clearly.
"""


class LLMClient:
    def __init__(self, backend: Optional[str] = None, model_name: Optional[str] = None):
        # Configure backend
        self.backend = (backend or os.getenv("LLM_BACKEND", "heuristic")).lower()
        self.model_name = model_name or os.getenv("LLM_MODEL")

        if self.backend == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            self.model_name = self.model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash")
            self.genai_client = None
            self.legacy_genai = None
            if not self.api_key:
                print("Warning: GEMINI_API_KEY not set. Falling back to heuristic planner.")
                self.backend = "heuristic"
            else:
                try:
                    from google import genai
                    self.genai_client = genai.Client(api_key=self.api_key)
                except ImportError:
                    try:
                        import google.generativeai as genai_legacy
                        genai_legacy.configure(api_key=self.api_key)
                        self.legacy_genai = genai_legacy
                    except ImportError:
                        print("Neither google-genai nor google-generativeai installed. Falling back to heuristic planner.")
                        self.backend = "heuristic"

        elif self.backend == "ollama":
            self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            self.model_name = self.model_name or "llama3.1"

    def get_backend_info(self) -> Dict[str, Any]:
        """Return honest metadata describing active backend, model, and capabilities."""
        if self.backend == "gemini":
            return {
                "backend": "gemini",
                "model": self.model_name,
                "description": f"Gemini (model: {self.model_name})",
                "is_heuristic": False
            }
        elif self.backend == "ollama":
            return {
                "backend": "ollama",
                "model": self.model_name,
                "description": f"Ollama (model: {self.model_name})",
                "is_heuristic": False
            }
        else:
            return {
                "backend": "heuristic",
                "model": "none",
                "description": "Heuristic fallback (no LLM configured; limited query matching)",
                "is_heuristic": True
            }

    def plan_tool_call(
        self,
        user_prompt: str,
        df_columns: List[str],
        memories_text: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        previous_tool_error: Optional[str] = None,
        tool_trace: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Determine which tool to call and with what arguments based on the user's question,
        schema, relevant memories, previous tool error, and prior tool execution trace.
        """
        if self.backend == "gemini":
            return self._plan_with_gemini(user_prompt, df_columns, memories_text, previous_tool_error, tool_trace)
        elif self.backend == "ollama":
            return self._plan_with_ollama(user_prompt, df_columns, memories_text, previous_tool_error, tool_trace)
        else:
            return self._plan_with_heuristic(user_prompt, df_columns, memories_text, previous_tool_error, tool_trace)

    def synthesize_response(
        self,
        user_prompt: str,
        tool_name: str,
        tool_result: Dict[str, Any],
        memories_text: str = "",
        tool_trace: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate plain-English answer synthesizing computed tool outputs with strict numerical fidelity.
        """
        trace_summary = json.dumps(tool_trace, default=str) if tool_trace else json.dumps(tool_result, default=str)

        if self.backend == "gemini":
            try:
                prompt = (
                    f"{SYNTHESIS_SYSTEM_INSTRUCTIONS}\n\n"
                    f"{memories_text}\n\n"
                    f"User Question: {user_prompt}\n"
                    f"Primary Tool Used: {tool_name}\n"
                    f"Execution Trace & Computed Outputs:\n{trace_summary}\n\n"
                    f"Strict Requirement: Answer the question using the EXACT numbers above. Do not invent any numbers."
                )
                if self.genai_client is not None:
                    res = self.genai_client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={"system_instruction": SYNTHESIS_SYSTEM_INSTRUCTIONS}
                    )
                    if res and res.text:
                        return res.text.strip()
                elif self.legacy_genai is not None:
                    model = self.legacy_genai.GenerativeModel(self.model_name, system_instruction=SYNTHESIS_SYSTEM_INSTRUCTIONS)
                    res = model.generate_content(prompt)
                    if res and res.text:
                        return res.text.strip()
            except Exception:
                pass

        elif self.backend == "ollama":
            try:
                res = requests.post(
                    f"{self.ollama_host}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": SYNTHESIS_SYSTEM_INSTRUCTIONS},
                            {"role": "user", "content": f"{memories_text}\n\nQuestion: {user_prompt}\nTool Output:\n{trace_summary}\nUse exact numbers only."}
                        ],
                        "stream": False
                    },
                    timeout=30
                )
                if res.status_code == 200:
                    return res.json()["message"]["content"].strip()
            except Exception:
                pass

        # Deterministic heuristic synthesis fallback
        return self._heuristic_synthesis(user_prompt, tool_name, tool_result, tool_trace)

    # --------------------------------------------------------------------------
    # HEURISTIC / DETERMINISTIC PLANNER
    # --------------------------------------------------------------------------

    def _plan_with_heuristic(
        self,
        user_prompt: str,
        df_columns: List[str],
        memories_text: str = "",
        previous_tool_error: Optional[str] = None,
        tool_trace: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        High-accuracy rule-based planner mapping user intentions to tool calls.
        Also parses memory conditions and fixes errors on retry.
        """
        # If a tool has already successfully executed in heuristic mode, terminate loop
        if tool_trace and len(tool_trace) >= 1:
            return None

        p = user_prompt.lower()

        # Handle retry if previous query failed with a syntax or column error
        if previous_tool_error and "run_query" in previous_tool_error:
            # Fallback to general select
            return {"name": "run_query", "arguments": {"query": "SELECT * FROM df LIMIT 20"}}

        # Check for profiling intent
        if any(w in p for w in ["profile", "overview", "summary", "describe dataset", "info", "columns", "nulls"]):
            return {"name": "profile_dataset", "arguments": {"sample_rows": 3}}

        # Check for outlier detection intent
        if any(w in p for w in ["outlier", "anomaly", "anomalies", "abnormal"]):
            method = "isolation_forest" if "isolation" in p else ("zscore" if "z-score" in p or "zscore" in p else "iqr")
            return {"name": "detect_outliers", "arguments": {"method": method}}

        # Check for correlation intent
        if any(w in p for w in ["correlation", "correlate", "association", "relationship between"]):
            return {"name": "correlation_analysis", "arguments": {"method": "pearson"}}

        # Check for chart intent
        if any(w in p for w in ["chart", "plot", "graph", "histogram", "scatter", "bar chart", "visualize"]):
            ctype = "bar"
            if "scatter" in p:
                ctype = "scatter"
            elif "hist" in p:
                ctype = "histogram"
            elif "line" in p:
                ctype = "line"
            elif "box" in p:
                ctype = "box"

            # Match columns
            matched_cols = [c for c in df_columns if c.lower() in p]
            x_col = matched_cols[0] if matched_cols else df_columns[0]
            y_col = matched_cols[1] if len(matched_cols) > 1 else None
            return {"name": "make_chart", "arguments": {"chart_type": ctype, "x": x_col, "y": y_col}}

        # Check for statistical test intent
        if any(w in p for w in ["t-test", "ttest", "anova", "chi-square", "chisquare", "significant", "hypothesis"]):
            matched_cols = [c for c in df_columns if c.lower() in p]
            if "anova" in p:
                # Find numerical metric and categorical group
                return {"name": "statistical_test", "arguments": {
                    "test_type": "anova",
                    "col1": matched_cols[0] if matched_cols else df_columns[-1],
                    "group_col": matched_cols[1] if len(matched_cols) > 1 else df_columns[1]
                }}
            elif "chi" in p:
                c1 = matched_cols[0] if matched_cols else df_columns[1]
                c2 = matched_cols[1] if len(matched_cols) > 1 else df_columns[2]
                return {"name": "statistical_test", "arguments": {"test_type": "chi_square", "col1": c1, "col2": c2}}
            else:
                c1 = matched_cols[0] if matched_cols else df_columns[-1]
                c2 = matched_cols[1] if len(matched_cols) > 1 else None
                return {"name": "statistical_test", "arguments": {"test_type": "ttest_ind", "col1": c1, "group_col": c2}}

        # Check for time series trend intent
        if any(w in p for w in ["trend", "forecast", "projection", "over time"]):
            date_col = next((c for c in df_columns if "date" in c.lower() or "time" in c.lower()), None)
            val_col = next((c for c in df_columns if "amount" in c.lower() or "total" in c.lower() or "price" in c.lower() or "score" in c.lower()), df_columns[-1])
            if date_col:
                return {"name": "trend_forecast", "arguments": {"date_col": date_col, "value_col": val_col, "forecast_periods": 7}}

        # Check for Group By / Aggregate intent
        if any(w in p for w in ["group by", "average by", "sum by", "breakdown by", "per category", "per department", "by region"]):
            matched_cols = [c for c in df_columns if c.lower() in p]
            group_col = matched_cols[0] if matched_cols else df_columns[1]
            num_col = next((c for c in df_columns if "amount" in c.lower() or "score" in c.lower() or "price" in c.lower() or "hours" in c.lower()), df_columns[-1])
            return {
                "name": "group_aggregate",
                "arguments": {
                    "group_cols": [group_col],
                    "agg_spec": {num_col: ["mean", "sum", "count"]}
                }
            }

        # Default query synthesis incorporating memory definitions
        sql = self._synthesize_query_from_prompt(user_prompt, df_columns, memories_text)
        return {"name": "run_query", "arguments": {"query": sql}}

    def _synthesize_query_from_prompt(self, prompt: str, columns: List[str], memories_text: str) -> str:
        """Construct a targeted SQL query from user prompt and active memories."""
        p = prompt.lower()

        # Extract condition from memory dynamically if relevant to user prompt
        where_clause = ""
        if memories_text:
            for line in memories_text.strip().split("\n"):
                line = line.strip()
                if not line or not line.startswith("-"):
                    continue
                # Line format: - [category] key: value
                m = re.match(r"^-\s*\[.*?\]\s*([^:]+):\s*(.+)$", line)
                if m:
                    mem_key = m.group(1).strip().lower()
                    mem_val = m.group(2).strip()
                    # Check if any significant token of mem_key is in the user prompt
                    key_tokens = [tok for tok in re.split(r"\W+", mem_key) if len(tok) > 2]
                    if any(tok in p for tok in key_tokens) or mem_key in p:
                        # Translate condition to valid SQL
                        cond = mem_val
                        # Replace '==' with '='
                        cond = re.sub(r"==\s*", "= ", cond)
                        # Replace ' is ' with ' = '
                        cond = re.sub(r"\bis\b", "=", cond, flags=re.IGNORECASE)
                        # Normalize logical AND / OR
                        cond = re.sub(r"\band\b", "AND", cond, flags=re.IGNORECASE)
                        cond = re.sub(r"\bor\b", "OR", cond, flags=re.IGNORECASE)
                        where_clause = f"WHERE {cond}"
                        break

        # Helper to find column mentioned in prompt
        def find_mentioned_or_fallback(candidates, default_index=-1):
            for col in columns:
                if col.lower() in p or col.lower().replace("_", " ") in p:
                    return f'"{col}"' if " " in col else col
            for cand in candidates:
                for col in columns:
                    if cand in col.lower():
                        return f'"{col}"' if " " in col else col
            fallback = columns[default_index] if columns else "id"
            return f'"{fallback}"' if " " in fallback else fallback

        # Count queries
        if any(w in p for w in ["how many", "count", "number of rows"]):
            return f"SELECT COUNT(*) AS total_count FROM df {where_clause}".strip()

        # Revenue / sum queries
        if any(w in p for w in ["sum", "total", "revenue", "sales", "amount", "spending"]):
            sum_col = find_mentioned_or_fallback(["total", "amount", "revenue", "sales", "price", "salary", "score", "cost"])
            alias = "total_revenue" if any(w in p for w in ["revenue", "sales"]) else "total_sum"
            return f"SELECT ROUND(SUM({sum_col}), 2) AS {alias}, COUNT(*) AS count FROM df {where_clause}".strip()

        # Average queries
        if any(w in p for w in ["average", "mean", "avg"]):
            avg_col = find_mentioned_or_fallback(["score", "price", "salary", "amount", "age", "hours", "tenure", "rate", "cost", "value"])
            return f"SELECT ROUND(AVG({avg_col}), 2) AS average_value, COUNT(*) AS count FROM df {where_clause}".strip()

        # Highest / top queries
        if any(w in p for w in ["highest", "top", "best", "max", "greatest"]):
            order_col = find_mentioned_or_fallback(["score", "amount", "salary", "price", "total", "value"])
            return f"SELECT * FROM df {where_clause} ORDER BY {order_col} DESC LIMIT 5".strip()

        # Lowest / bottom queries
        if any(w in p for w in ["lowest", "worst", "min", "least"]):
            order_col = find_mentioned_or_fallback(["score", "amount", "salary", "price", "total", "value"])
            return f"SELECT * FROM df {where_clause} ORDER BY {order_col} ASC LIMIT 5".strip()

        # Fallback query
        return f"SELECT * FROM df {where_clause} LIMIT 25".strip()

    def _heuristic_synthesis(
        self,
        prompt: str,
        tool_name: str,
        result: Dict[str, Any],
        tool_trace: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Deterministic formatter converting tool dictionaries into clear user answers."""
        if result.get("status") == "error":
            return f"Tool execution failed: {result.get('message', 'Unknown error')}."

        if tool_name == "profile_dataset":
            cols_preview = ", ".join([c["name"] for c in result["columns"][:8]])
            return (
                f"### Dataset Profile Summary\n"
                f"- **Dimensions**: {result['row_count']:,} rows × {result['col_count']} columns\n"
                f"- **Duplicate Rows**: {result['duplicate_rows']}\n"
                f"- **Key Columns**: {cols_preview}...\n"
                f"- **Numerical Columns Profiling**: {len(result['numeric_stats'])} numeric attributes analyzed."
            )

        if tool_name == "run_query":
            rows = result.get("data", [])
            if not rows:
                return "The query completed successfully, but returned 0 rows matching the conditions."
            if len(rows) == 1 and len(rows[0]) == 1:
                col, val = list(rows[0].items())[0]
                return f"**Result**: {col} = **{val}** (calculated directly from dataset)."
            if len(rows) == 1:
                parts = [f"**{k}**: {v}" for k, v in rows[0].items()]
                return f"**Result**: {', '.join(parts)}."
            preview = json.dumps(rows[:5], indent=2, default=str)
            return f"Returned **{result.get('row_count', len(rows))}** matching rows. First results:\n```json\n{preview}\n```"

        if tool_name == "group_aggregate":
            data = result.get("data", [])
            lines = [f"### Group Aggregation by {', '.join(result.get('group_cols', []))}"]
            for row in data[:6]:
                grp_vals = " - ".join(str(row[c]) for c in result.get("group_cols", []))
                metric_vals = ", ".join(f"{k}: {v}" for k, v in row.items() if k not in result.get("group_cols", []))
                lines.append(f"- **{grp_vals}**: {metric_vals}")
            return "\n".join(lines)

        if tool_name == "statistical_test":
            return f"### {result.get('test')}\n- **Test Statistic**: {result.get('statistic')}\n- **p-value**: {result.get('p_value'):.4e}\n- **Conclusion**: {result.get('interpretation')}"

        if tool_name == "detect_outliers":
            return (
                f"### Outlier Detection ({result.get('method').upper()})\n"
                f"- **Total Anomalous Rows**: {result.get('total_anomalous_rows')} of {result.get('total_rows')}\n"
                f"- Breakdown per column inspected."
            )

        if tool_name == "correlation_analysis":
            top = result.get("top_numeric_correlations", [])
            top_str = "\n".join([f"- **{t['col1']}** & **{t['col2']}**: r = {t['correlation']:.3f}" for t in top[:5]])
            return f"### Top Numerical Correlations ({result.get('method')})\n{top_str}"

        if tool_name == "make_chart":
            return f"Successfully generated **{result.get('chart_type')}** chart: `{result.get('saved_image_path')}`."

        if tool_name == "trend_forecast":
            return (
                f"### Time-Series Trend & Forecast\n"
                f"- **Trend Direction**: {result.get('trend_direction')} (slope: {result.get('daily_trend_slope'):.3f})\n"
                f"- **Fit R²**: {result.get('r_squared')}\n"
                f"- Next projected periods computed."
            )

        return json.dumps(result, indent=2, default=str)

    # --------------------------------------------------------------------------
    # GEMINI / OLLAMA IMPLEMENTATION HOOKS
    # --------------------------------------------------------------------------

    def _plan_with_gemini(
        self,
        prompt: str,
        cols: List[str],
        mem: str,
        err: Optional[str],
        tool_trace: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        try:
            sys_instruct = f"{SYSTEM_INSTRUCTIONS}\nAvailable Columns: {cols}\n{mem}\nTool Schemas:\n{json.dumps(TOOL_SCHEMAS)}"
            chat_prompt = f"Question: {prompt}\n" + (f"Previous Error: {err}\n" if err else "")
            if tool_trace:
                chat_prompt += f"\nExecuted Tools Trace in this Multi-Step Execution:\n{json.dumps(tool_trace, default=str)}\n"
                chat_prompt += 'If sufficient information has been collected to answer the question, respond with: {"name": "synthesize", "arguments": {}}\n'
            chat_prompt += 'Otherwise, respond strictly with a JSON object: {"name": "<tool_name>", "arguments": {<args>}}'
            text = ""
            if self.genai_client is not None:
                res = self.genai_client.models.generate_content(
                    model=self.model_name,
                    contents=chat_prompt,
                    config={"system_instruction": sys_instruct, "response_mime_type": "application/json"}
                )
                text = res.text
            elif self.legacy_genai is not None:
                model = self.legacy_genai.GenerativeModel(self.model_name, system_instruction=sys_instruct)
                res = model.generate_content(chat_prompt)
                text = res.text

            if text:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception:
            pass
        return self._plan_with_heuristic(prompt, cols, mem, err, tool_trace)

    def _plan_with_ollama(
        self,
        prompt: str,
        cols: List[str],
        mem: str,
        err: Optional[str],
        tool_trace: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        try:
            sys_msg = f"{SYSTEM_INSTRUCTIONS}\nColumns: {cols}\n{mem}\nRespond ONLY with JSON: {{\"name\": \"...\", \"arguments\": {{...}}}}"
            user_msg = f"User: {prompt}\nError: {err or ''}"
            if tool_trace:
                user_msg += f'\nPrior Tool Trace:\n{json.dumps(tool_trace, default=str)}\nIf done, reply with {{"name": "synthesize", "arguments": {{}}}}'
            res = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    "format": "json",
                    "stream": False
                },
                timeout=30
            )
            if res.status_code == 200:
                body = res.json()["message"]["content"]
                return json.loads(body)
        except Exception:
            pass
        return self._plan_with_heuristic(prompt, cols, mem, err, tool_trace)