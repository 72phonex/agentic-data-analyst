"""
Core orchestrator for the Tabular Data Analysis Agent.
Coordinates LLM tool-calling, multi-step execution loop, dynamic memory retrieval,
sandboxed query execution, error handling with retries, and result synthesis.
"""

from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional
import pandas as pd

from data_agent.core.tools import TOOL_REGISTRY
from data_agent.core.memory import MemoryStore
from data_agent.core.llm import LLMClient
from data_agent.core.data_loader import load_dataset


class DataAnalysisAgent:
    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        llm_client: Optional[LLMClient] = None,
        max_retries: int = 2,
        max_steps: int = 3,
        df: Optional[pd.DataFrame] = None,
        dataset_name: Optional[str] = None
    ):
        self.memory = memory_store or MemoryStore()
        self.llm = llm_client or LLMClient()
        self.max_retries = max_retries
        self.max_steps = max_steps
        self.df = df
        self.dataset_name = os.path.basename(dataset_name) if dataset_name else "dataset"
        self.execution_history: List[Dict[str, Any]] = []

    def set_dataset(self, df: pd.DataFrame, dataset_name: str = "dataset"):
        """Attach a pandas DataFrame to the agent."""
        self.df = df
        self.dataset_name = os.path.basename(dataset_name)

    def load_file(self, filepath: str) -> Dict[str, Any]:
        """Load a CSV or Excel file into the agent."""
        df, meta = load_dataset(filepath)
        self.set_dataset(df, meta["filename"])
        return meta

    # --------------------------------------------------------------------------
    # COMMAND DISPATCHER (Slash Commands)
    # --------------------------------------------------------------------------

    def handle_command(self, user_input: str) -> Optional[str]:
        """Check if user entered a special slash command (/memory, /forget, /edit, /add)."""
        cmd = user_input.strip()
        if not cmd.startswith("/"):
            return None

        tokens = cmd.split()
        base = tokens[0].lower()

        if base == "/memory":
            # List memories
            mems = self.memory.list_memories(dataset_name=self.dataset_name)
            if not mems:
                return "No persistent memories found."
            lines = ["### Stored Domain Memories & Definitions:"]
            for m in mems:
                ds = f" (dataset: {m['dataset_name']})" if m['dataset_name'] else ""
                lines.append(f"- **#{m['id']} [{m['category']}] `{m['key']}`** -> `{m['value']}`{ds}")
            return "\n".join(lines)

        elif base == "/forget":
            # /forget <id>
            if len(tokens) < 2:
                return "Usage: `/forget <memory_id>`"
            try:
                mem_id = int(tokens[1])
                success = self.memory.delete_memory(mem_id)
                return f"Memory #{mem_id} has been forgotten." if success else f"Memory #{mem_id} not found."
            except ValueError:
                return "Error: Memory ID must be an integer."

        elif base == "/edit":
            # /edit <id> <new_val>
            if len(tokens) < 3:
                return "Usage: `/edit <memory_id> <new_value>`"
            try:
                mem_id = int(tokens[1])
                new_val = " ".join(tokens[2:])
                mems = self.memory.list_memories()
                target = next((m for m in mems if m["id"] == mem_id), None)
                if target:
                    res = self.memory.save_memory(
                        category=target["category"],
                        key=target["key"],
                        value=new_val,
                        dataset_name=target["dataset_name"]
                    )
                    return f"Updated memory #{mem_id} -> '{new_val}'"
                return f"Memory #{mem_id} not found."
            except ValueError:
                return "Error: Memory ID must be an integer."

        elif base == "/add":
            # /add metric_definition active customers = status == 'Active'
            rest = cmd[4:].strip()
            if "=" in rest:
                cat_key, val = rest.split("=", 1)
                tokens = cat_key.strip().split(maxsplit=1)
                category = tokens[0] if len(tokens) > 1 and tokens[0] in [
                    "column_meaning", "metric_definition", "user_correction", "preference", "filter"
                ] else "metric_definition"
                key = tokens[1] if len(tokens) > 1 and tokens[0] in [
                    "column_meaning", "metric_definition", "user_correction", "preference", "filter"
                ] else cat_key.strip()

                prop = self.memory.propose_memory(category, key.strip(), val.strip(), dataset_name=self.dataset_name)
                # Auto-confirm manual add command
                res = self.memory.confirm_proposal(prop["proposal_id"])
                return f"{res['message']} (Key: '{key.strip()}', Value: '{val.strip()}')"
            return "Usage: `/add <category> <key> = <value>`"

        return f"Unknown command: '{base}'. Available commands: `/memory`, `/forget <id>`, `/edit <id> <new_val>`, `/add`"

    # --------------------------------------------------------------------------
    # MAIN AGENT CHAT LOOP WITH MULTI-STEP EXECUTION
    # --------------------------------------------------------------------------

    def ask(self, question: str, confirm_pending_proposals: bool = False) -> Dict[str, Any]:
        """
        Process a user question:
        1. Check for command
        2. Detect and propose new candidate memories (ask user to confirm)
        3. Retrieve relevant persistent memories
        4. Multi-step tool execution loop (up to max_steps steps: plan -> execute -> observe -> plan next or finish)
        5. Synthesize final answer using strict numerical fidelity
        """
        # 1. Check for command
        cmd_response = self.handle_command(question)
        if cmd_response is not None:
            return {
                "answer": cmd_response,
                "tool_used": None,
                "tool_arguments": None,
                "intermediate_result": None,
                "tool_trace": [],
                "memories_used": [],
                "pending_proposal": None
            }

        # 2. Check for memory confirmation if user typed 'yes' / 'confirm'
        if self.memory.pending_proposals and question.strip().lower() in ["yes", "confirm", "save"]:
            latest_id = list(self.memory.pending_proposals.keys())[-1]
            res = self.memory.confirm_proposal(latest_id)
            return {
                "answer": f"Confirmed! {res['message']}",
                "tool_used": None,
                "tool_arguments": None,
                "intermediate_result": res,
                "tool_trace": [],
                "memories_used": [],
                "pending_proposal": None
            }

        if self.memory.pending_proposals and question.strip().lower() in ["no", "cancel"]:
            latest_id = list(self.memory.pending_proposals.keys())[-1]
            self.memory.cancel_proposal(latest_id)
            return {
                "answer": "Cancelled. The definition was not saved.",
                "tool_used": None,
                "tool_arguments": None,
                "intermediate_result": None,
                "tool_trace": [],
                "memories_used": [],
                "pending_proposal": None
            }

        # 3. Detect candidate memory facts in user input
        fact_candidate = self.memory.detect_fact_candidate(question)
        pending_proposal = None
        if fact_candidate:
            prop = self.memory.propose_memory(
                category=fact_candidate["category"],
                key=fact_candidate["key"],
                value=fact_candidate["value"],
                dataset_name=self.dataset_name
            )
            if confirm_pending_proposals:
                save_res = self.memory.confirm_proposal(prop["proposal_id"])
                proposal_msg = f"{save_res['message']} (Key: '{prop['key']}' -> '{prop['value']}')."
            else:
                pending_proposal = prop
                proposal_msg = (
                    f"I noticed a new {prop['category']}:\n"
                    f"**Key**: `{prop['key']}`\n"
                    f"**Value**: `{prop['value']}`\n"
                    f"Would you like me to save this to persistent memory? (Reply 'yes' or 'confirm' to save, or 'no' to ignore)."
                )
                return {
                    "answer": proposal_msg,
                    "tool_used": None,
                    "tool_arguments": None,
                    "intermediate_result": prop,
                    "tool_trace": [],
                    "memories_used": [],
                    "pending_proposal": prop
                }

        # Ensure dataset is available
        if self.df is None:
            return {
                "answer": "Please load a CSV or Excel dataset first before asking data questions.",
                "tool_used": None,
                "tool_arguments": None,
                "intermediate_result": None,
                "tool_trace": [],
                "memories_used": [],
                "pending_proposal": None
            }

        # 4. Retrieve relevant memories for this question
        relevant_mems = self.memory.retrieve_relevant(question, dataset_name=self.dataset_name)
        memories_prompt = self.memory.format_memories_prompt(relevant_mems)

        columns = list(self.df.columns)

        # 5. Multi-step Tool Execution Loop (plan -> execute -> observe -> next step or finish)
        tool_trace: List[Dict[str, Any]] = []
        last_error: Optional[str] = None
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            # Step planning
            tool_call = None
            for retry in range(self.max_retries + 1):
                tool_call = self.llm.plan_tool_call(
                    user_prompt=question,
                    df_columns=columns,
                    memories_text=memories_prompt,
                    previous_tool_error=last_error,
                    tool_trace=tool_trace
                )
                if tool_call and "name" in tool_call:
                    break

            # If no tool planned or model indicated completion
            if not tool_call or "name" not in tool_call or tool_call["name"] in ["synthesize", "finish", "none"]:
                break

            tool_name = tool_call["name"]
            tool_args = tool_call.get("arguments", {})
            tool_func = TOOL_REGISTRY.get(tool_name)

            if not tool_func:
                last_error = f"Tool '{tool_name}' is not recognized."
                continue

            try:
                # Execute tool
                tool_result = tool_func(self.df, **tool_args)

                tool_step_record = {
                    "step": len(tool_trace) + 1,
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": tool_result
                }
                tool_trace.append(tool_step_record)

                if tool_result.get("status") == "error":
                    last_error = f"Tool '{tool_name}' returned error: {tool_result.get('message')}"
                    continue
                else:
                    last_error = None
            except Exception as e:
                last_error = f"Exception in '{tool_name}': {str(e)}"
                continue

        if not tool_trace:
            tool_name = "error_handler"
            tool_result = {"status": "error", "message": last_error or "Unable to determine tool to execute."}
        else:
            tool_name = tool_trace[-1]["tool"]
            tool_result = tool_trace[-1]["result"]

        # 6. Final synthesis with numerical fidelity
        final_answer = self.llm.synthesize_response(
            user_prompt=question,
            tool_name=tool_name,
            tool_result=tool_result,
            memories_text=memories_prompt,
            tool_trace=tool_trace
        )

        record = {
            "question": question,
            "tool_used": tool_name if tool_trace else None,
            "tool_arguments": tool_trace[-1]["arguments"] if tool_trace else {},
            "intermediate_result": tool_result,
            "tool_trace": tool_trace,
            "memories_used": [m["key"] for m in relevant_mems],
            "answer": final_answer,
            "pending_proposal": pending_proposal
        }
        self.execution_history.append(record)
        return record
