"""
Data Analysis Agent execution engine.
Coordinates persistent memory retrieval, tool execution, error recovery,
and final answer synthesis.
"""

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from data_agent.core.tools import TOOL_REGISTRY, TOOL_SCHEMAS
from data_agent.core.memory import MemoryStore, DEFAULT_DB_PATH
from data_agent.core.llm import LLMClient
from data_agent.core.data_loader import load_dataset


class DataAnalysisAgent:
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        dataset_name: str = "global",
        memory_store: Optional[MemoryStore] = None,
        llm_client: Optional[LLMClient] = None,
        max_tool_retries: int = 2
    ):
        self.df = df
        self.dataset_name = dataset_name
        self.memory = memory_store or MemoryStore()
        self.llm = llm_client or LLMClient()
        self.max_retries = max_tool_retries
        self.execution_history: List[Dict[str, Any]] = []

    def set_dataset(self, df: pd.DataFrame, dataset_name: str = "dataset") -> None:
        """Set or switch the active dataset."""
        self.df = df
        self.dataset_name = dataset_name

    def load_file(self, file_path_or_buffer: Any) -> Dict[str, Any]:
        """Load dataset from path or buffer."""
        df, meta = load_dataset(file_path_or_buffer)
        self.set_dataset(df, meta["filename"])
        return meta

    # --------------------------------------------------------------------------
    # COMMAND DISPATCHER (/memory, /forget, /edit, /add)
    # --------------------------------------------------------------------------

    def handle_command(self, user_input: str) -> Optional[str]:
        """Check for and handle CLI / chat commands starting with '/'."""
        cmd = user_input.strip()
        if not cmd.startswith("/"):
            return None

        parts = cmd.split(maxsplit=2)
        base = parts[0].lower()

        if base == "/memory":
            mems = self.memory.list_memories(self.dataset_name)
            if not mems:
                return "No persistent memories stored yet. Tell me definitions or rules (e.g. \"Active customers means status == 'Active'\") to add some."
            lines = ["### Persistent Memories Stored:"]
            for m in mems:
                lines.append(f"- **#{m['id']}** `[{m['category']}]` **{m['key']}**: {m['value']} *(updated: {m['updated_at']})*")
            return "\n".join(lines)

        elif base == "/forget":
            if len(parts) < 2:
                return "Usage: `/forget <id>`"
            try:
                mem_id = int(parts[1])
                success = self.memory.delete_memory(mem_id)
                return f"Memory #{mem_id} forgotten." if success else f"Memory #{mem_id} not found."
            except ValueError:
                return "Error: Memory ID must be an integer."

        elif base == "/edit":
            if len(parts) < 3:
                return "Usage: `/edit <id> <new_value>`"
            try:
                mem_id = int(parts[1])
                new_val = parts[2].strip()
                res = self.memory.edit_memory(mem_id, new_val)
                if res:
                    return f"Memory #{mem_id} ('{res['key']}') updated to: \"{new_val}\". (Previous value archived in audit history)."
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
    # MAIN AGENT CHAT LOOP
    # --------------------------------------------------------------------------

    def ask(self, question: str, confirm_pending_proposals: bool = False) -> Dict[str, Any]:
        """
        Process a user question:
        1. Check for command
        2. Detect and propose new candidate memories (ask user to confirm)
        3. Retrieve relevant persistent memories
        4. Call tools, handle errors, retry if necessary
        5. Synthesize final answer
        """
        # 1. Check for command
        cmd_response = self.handle_command(question)
        if cmd_response is not None:
            return {
                "answer": cmd_response,
                "tool_used": None,
                "tool_arguments": None,
                "intermediate_result": None,
                "memories_used": [],
                "pending_proposal": None
            }

        # 2. Check for memory confirmation if user typed 'yes' / 'confirm'
        if self.memory.pending_proposals and question.strip().lower() in ["yes", "confirm", "save"]:
            # Confirm latest proposal
            latest_id = list(self.memory.pending_proposals.keys())[-1]
            res = self.memory.confirm_proposal(latest_id)
            return {
                "answer": f"Confirmed! {res['message']}",
                "tool_used": None,
                "tool_arguments": None,
                "intermediate_result": res,
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
                # Auto-confirm for non-interactive or batch runs
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
                "memories_used": [],
                "pending_proposal": None
            }

        # 4. Retrieve relevant memories for this question
        relevant_mems = self.memory.retrieve_relevant(question, dataset_name=self.dataset_name)
        memories_prompt = self.memory.format_memories_prompt(relevant_mems)

        columns = list(self.df.columns)

        # 5. Tool planning and execution loop with retry
        tool_call = None
        tool_result = None
        last_error = None
        tools_executed = []

        for attempt in range(self.max_retries + 1):
            tool_call = self.llm.plan_tool_call(
                user_prompt=question,
                df_columns=columns,
                memories_text=memories_prompt,
                previous_tool_error=last_error
            )

            if not tool_call or "name" not in tool_call:
                break

            tool_name = tool_call["name"]
            tool_args = tool_call.get("arguments", {})
            tools_executed.append((tool_name, tool_args))

            tool_func = TOOL_REGISTRY.get(tool_name)
            if not tool_func:
                last_error = f"Tool '{tool_name}' is not recognized."
                continue

            try:
                # Execute tool
                tool_result = tool_func(self.df, **tool_args)

                # Check if tool execution returned an error
                if tool_result.get("status") == "error":
                    last_error = f"Tool '{tool_name}' returned error: {tool_result.get('message')}"
                    continue  # Retry
                else:
                    # Success
                    break
            except Exception as e:
                last_error = f"Exception in '{tool_name}': {str(e)}"
                continue  # Retry

        if not tool_result:
            tool_name = "error_handler"
            tool_result = {"status": "error", "message": last_error or "Unable to determine tool to execute."}

        # 6. Final synthesis
        final_answer = self.llm.synthesize_response(
            user_prompt=question,
            tool_name=tool_name,
            tool_result=tool_result,
            memories_text=memories_prompt
        )

        record = {
            "question": question,
            "tool_used": tool_name,
            "tool_arguments": tool_args if tool_call else {},
            "intermediate_result": tool_result,
            "memories_used": [m["key"] for m in relevant_mems],
            "answer": final_answer,
            "pending_proposal": pending_proposal
        }
        self.execution_history.append(record)
        return record
