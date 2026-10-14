import json
import pandas as pd
from tools import describe_dataset, run_sql, group_stats, correlation, find_outliers, statistical_test, make_chart
from llm import TOOLS_SCHEMA, build_system_prompt, call_ollama

TOOL_MAP = {
    "describe_dataset": describe_dataset, "run_sql": run_sql, "group_stats": group_stats,
    "correlation": correlation, "find_outliers": find_outliers, "statistical_test": statistical_test, "make_chart": make_chart
}

def ask_agent(df: pd.DataFrame, question: str, chat_history: list = None, memories: list = None, model: str = None) -> dict:
    """Agent loop: sends question to model, runs picked tools (up to 3 times), and returns final explanation."""
    messages = [{"role": "system", "content": build_system_prompt(df, memories)}]
    if chat_history:
        for m in chat_history:
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": question})

    traces, chart_path = [], None

    for _ in range(3):  # Limit tool calling loop to at most 3 turns
        resp = call_ollama(messages, tools=TOOLS_SCHEMA, model=model)
        if "error" in resp:
            return {"answer": resp["error"], "traces": traces, "chart_path": chart_path}
        msg = resp.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:  # Model provided its final response text
            return {"answer": msg.get("content", ""), "traces": traces, "chart_path": chart_path}

        messages.append(msg)
        for tc in tool_calls:
            name = tc.get("function", {}).get("name", "")
            raw_args = tc.get("function", {}).get("arguments", {})
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            tool_func = TOOL_MAP.get(name)
            result = tool_func(df, **args) if tool_func else {"error": f"Unknown tool '{name}'"}
            traces.append({"tool": name, "args": args, "result": result})
            if isinstance(result, dict) and result.get("chart_path"):
                chart_path = result["chart_path"]
            # Feed tool output back to the model so it can interpret results or fix errors
            messages.append({"role": "tool", "content": json.dumps(result, default=str)})

    # If maximum tool iterations reached, ask model for a final synthesis of findings
    final_resp = call_ollama(messages, tools=None, model=model)
    answer = final_resp.get("message", {}).get("content", "Analysis completed.") if "message" in final_resp else final_resp.get("error", "")
    return {"answer": answer, "traces": traces, "chart_path": chart_path}
