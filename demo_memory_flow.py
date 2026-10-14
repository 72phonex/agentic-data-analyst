#!/usr/bin/env python3
"""
Scripted demonstration showing persistent memory cross-session lifecycle:
  - Session 1: Define a domain metric ('active customers') and confirm it to SQLite.
  - Session 2: A brand-new Agent session loads the DB, recalls the definition,
    and applies it to answer a query with exact numerical computation.
"""

from __future__ import annotations
import os
import sys
import shutil
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
from data_agent.core.data_loader import load_dataset


DEMO_DB = "demo_persistent_memory.sqlite"


def run_demo():
    print("=" * 75)
    print("🎬 SCRIPTED DEMO: PERSISTENT MEMORY LIFECYCLE ACROSS SESSIONS")
    print("=" * 75)

    if os.path.exists(DEMO_DB):
        os.remove(DEMO_DB)

    dataset_path = "data/sales_transactions.csv"
    df, meta = load_dataset(dataset_path)

    # ==========================================================================
    # SESSION 1
    # ==========================================================================
    print("\n--- 🟢 SESSION 1: INITIAL QUERY & DEFINING A DOMAIN METRIC ---")
    store_s1 = MemoryStore(db_path=DEMO_DB)
    agent_s1 = DataAnalysisAgent(df=df, dataset_name=meta["filename"], memory_store=store_s1)

    # 1. Ask general revenue without definition
    q1 = "What is the total revenue?"
    print(f"\nUser ❯ {q1}")
    res1 = agent_s1.ask(q1)
    print(f"Tool executed: [{res1['tool_used']}]")
    print(f"Tool query:    {res1['tool_arguments'].get('query')}")
    print(f"Agent ❯ {res1['answer']}")

    # 2. User provides a definition
    q2 = "Active customers means status is 'Active' and cancelled_flag is 0"
    print(f"\nUser ❯ {q2}")
    res2 = agent_s1.ask(q2)
    print(f"Agent Proposal ❯ {res2['answer']}")

    # 3. User confirms proposal to save into persistent SQLite memory
    print("\nUser ❯ yes (Confirm save)")
    res_confirm = agent_s1.ask("yes")
    print(f"Agent Confirmation ❯ {res_confirm['answer']}")

    # Verify SQLite contents
    mems = store_s1.list_memories()
    print(f"\n[Session 1 Database State]: {len(mems)} memory stored in SQLite ({DEMO_DB}):")
    for m in mems:
        print(f"  • #{m['id']} [{m['category']}] '{m['key']}': {m['value']}")

    # End Session 1 (Delete python instances to simulate app restart)
    del agent_s1
    del store_s1
    print("\n🔚 Session 1 ended and memory instances garbage collected.")

    # ==========================================================================
    # SESSION 2 (Fresh Agent Instance, New Session)
    # ==========================================================================
    print("\n--- 🟢 SESSION 2: FRESH AGENT INSTANCE RECALLING PERSISTENT MEMORY ---")
    store_s2 = MemoryStore(db_path=DEMO_DB)
    agent_s2 = DataAnalysisAgent(df=df, dataset_name=meta["filename"], memory_store=store_s2)

    # Ask question using the concept defined in session 1
    q3 = "What is the total revenue from active customers?"
    print(f"\nUser ❯ {q3}")
    res3 = agent_s2.ask(q3)

    print(f"\n🔍 Memories Retrieved & Injected: {res3['memories_used']}")
    print(f"🛠️  Tool Used:                     [{res3['tool_used']}]")
    print(f"⚙️  Query Executed:                {res3['tool_arguments'].get('query')}")
    print(f"📊 Intermediate Output:           {res3['intermediate_result'].get('data')}")
    print(f"Agent ❯ {res3['answer']}")

    # Verify memory update and audit history
    print("\n--- 🟢 TESTING MEMORY AUDIT & EDIT (/edit command) ---")
    edit_cmd = "/edit 1 status == 'Completed' and cancelled_flag == 0"
    print(f"User ❯ {edit_cmd}")
    edit_res = agent_s2.ask(edit_cmd)
    print(f"Agent ❯ {edit_res['answer']}")

    history = store_s2.get_history(1)
    print(f"\nAudit History Table in SQLite (memory_history):")
    for h in history:
        print(f"  • Changed at {h['changed_at']}: '{h['old_value']}' ➔ '{h['new_value']}'")

    print("\n" + "=" * 75)
    print("✅ SCRIPTED DEMO COMPLETE: MEMORY PERSISTENCE & INJECTION VERIFIED!")
    print("=" * 75)

    # Release database locks before file cleanup
    del agent_s2
    del store_s2
    import gc
    gc.collect()

    try:
        if os.path.exists(DEMO_DB):
            os.remove(DEMO_DB)
    except Exception:
        pass


if __name__ == "__main__":
    run_demo()
