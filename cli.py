#!/usr/bin/env python3
"""
Command-Line Interface (CLI) for the General-Purpose Tabular Data Analysis Agent.
"""

from __future__ import annotations
import sys
import os
import argparse
import json
from pathlib import Path

# Add project root to path
repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from data_agent.core.agent import DataAnalysisAgent
from data_agent.core.memory import MemoryStore
from data_agent.core.llm import LLMClient


def format_json_preview(data, max_len: int = 500) -> str:
    s = json.dumps(data, indent=2, default=str)
    if len(s) > max_len:
        return s[:max_len] + f"\n... [truncated, total {len(s)} chars]"
    return s


def run_cli():
    parser = argparse.ArgumentParser(description="Tabular Data Analysis Agent CLI")
    parser.add_argument("--data", "-d", type=str, default="data/sales_transactions.csv", help="Path to CSV or Excel dataset")
    parser.add_argument("--memory-db", type=str, default="data_agent_memory.sqlite", help="Path to SQLite memory database")
    parser.add_argument("--backend", "-b", type=str, default="heuristic", choices=["heuristic", "gemini", "ollama"], help="LLM backend")
    args = parser.parse_args()

    print("=" * 70)
    print("🤖 TABULAR DATA ANALYSIS AGENT (CLI Mode)")
    print(f"Backend: {args.backend.upper()} | Memory DB: {args.memory_db}")
    print("=" * 70)

    mem_store = MemoryStore(db_path=args.memory_db)
    llm = LLMClient(backend=args.backend)
    agent = DataAnalysisAgent(memory_store=mem_store, llm_client=llm)

    # Load dataset
    if os.path.exists(args.data):
        meta = agent.load_file(args.data)
        print(f"Loaded: '{meta['filename']}' ({meta['rows']:,} rows, {len(meta['columns'])} columns)")
        print(f"Columns: {', '.join(meta['columns'][:10])}{'...' if len(meta['columns']) > 10 else ''}")
    else:
        print(f"Warning: File '{args.data}' not found. Please load one using a question or specify --data.")

    print("\nSpecial Commands:")
    print("  /memory               List stored persistent memories")
    print("  /forget <id>          Delete a memory by ID")
    print("  /edit <id> <new_val>  Edit memory value")
    print("  /add <cat> <k> = <v>  Manually add a memory fact")
    print("  exit / quit           Exit the program\n")

    while True:
        try:
            user_input = input("User ❯ ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            response = agent.ask(user_input)

            # Display tool trace if tool was used
            if response.get("tool_used"):
                print("\n" + "─" * 60)
                print(f"🛠️  TOOL EXECUTED: [{response['tool_used']}]")
                if response.get("memories_used"):
                    print(f"🧠 MEMORIES APPLIED: {', '.join(response['memories_used'])}")
                print(f"⚙️  ARGUMENTS: {json.dumps(response.get('tool_arguments', {}), default=str)}")
                print("📊 INTERMEDIATE COMPUTATION RESULT:")
                print(format_json_preview(response.get("intermediate_result", {})))
                print("─" * 60)

            print(f"\nAgent ❯ {response['answer']}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    run_cli()
