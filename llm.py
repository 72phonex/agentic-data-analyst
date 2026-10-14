import os, re, json, requests
import pandas as pd

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "describe_dataset", "description": "Get summary stats, types, row count, nulls.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "run_sql", "description": "Execute read-only SQL query against table 'df'.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "SQL SELECT query over df"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "group_stats", "description": "Group by one column and compute sum/mean/count/min/max of another.", "parameters": {"type": "object", "properties": {"group_by": {"type": "string"}, "target_column": {"type": "string"}}, "required": ["group_by", "target_column"]}}},
    {"type": "function", "function": {"name": "correlation", "description": "Compute correlation between numeric columns.", "parameters": {"type": "object", "properties": {"method": {"type": "string", "enum": ["pearson", "spearman"]}}}}},
    {"type": "function", "function": {"name": "find_outliers", "description": "Detect outliers in a numeric column using IQR method.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}}},
    {"type": "function", "function": {"name": "statistical_test", "description": "Run hypothesis test: ttest, anova, or chi_square.", "parameters": {"type": "object", "properties": {"test_type": {"type": "string", "enum": ["ttest", "anova", "chi_square"]}, "column1": {"type": "string"}, "column2": {"type": "string"}}, "required": ["test_type", "column1"]}}},
    {"type": "function", "function": {"name": "make_chart", "description": "Generate and save a chart (bar, line, scatter, histogram, box).", "parameters": {"type": "object", "properties": {"chart_type": {"type": "string", "enum": ["bar", "line", "scatter", "histogram", "box"]}, "x": {"type": "string"}, "y": {"type": "string"}, "title": {"type": "string"}}, "required": ["chart_type", "x"]}}},
]

def clean_model_reply(text: str) -> str:
    """Strip reasoning/thinking tags (e.g. <think>...</think>) from Qwen/DeepSeek replies."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()

def build_system_prompt(df: pd.DataFrame, memories: list = None) -> str:
    """Create system instructions with column names, sample rows, and rules."""
    memory_text = ("\nRelevant business rules:\n" + "\n".join(f"- {m}" for m in memories) + "\n") if memories else ""
    return (
        "You are a helpful data analysis assistant. You analyze datasets by picking the right python tools.\n"
        f"The active dataset has columns: {list(df.columns)}\n"
        f"Sample rows:\n{json.dumps(df.head(3).to_dict(orient='records'), default=str)}\n{memory_text}"
        "CRITICAL RULES:\n"
        "1. You must call a tool to inspect the data before answering numerical questions.\n"
        "2. The final answer MUST use numbers directly from tool results. Do not invent numbers.\n"
        "3. Use exact column names as shown above. Quote column names that contain spaces.\n"
        "4. Be concise and explain your finding clearly."
    )

def call_ollama(messages: list, tools: list = None, model: str = None) -> dict:
    """Send chat request to Ollama and handle connection/model errors gracefully."""
    active_model = model or os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)
    payload = {"model": active_model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if resp.status_code == 404 or "not found" in resp.text.lower():
            return {"error": f"Model '{active_model}' is not pulled.\nRun this command to install it:\n\nollama pull {active_model}"}
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {})
        if "content" in message:
            message["content"] = clean_model_reply(message["content"])
        return {"message": message}
    except requests.exceptions.ConnectionError:
        return {"error": "Ollama is not running.\nStart Ollama by running this command in a terminal:\n\nollama serve"}
    except Exception as e:
        return {"error": f"Error communicating with Ollama: {str(e)}"}
