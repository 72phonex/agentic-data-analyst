import os
import sys
import gc
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data_agent.core.memory import MemoryStore
from data_agent.core.agent import DataAnalysisAgent
import pandas as pd

TEST_DB = "test_memory_lifecycle.db"

def test_full_memory_lifecycle():
    print("=" * 75)
    print("TESTING FULL MEMORY LIFECYCLE: PERSISTENCE, AUDIT, FORGET, RELEVANCE")
    print("=" * 75)

    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass

    dummy_df = pd.DataFrame({
        "id": [1, 2, 3],
        "job_role": ["Associate", "Senior", "Lead"],
        "years_at_company": [1.5, 6.0, 9.0],
        "city": ["Chicago", "Austin", "Seattle"]
    })

    # -------------------------------------------------------------
    # Step 1: Save a definition
    # -------------------------------------------------------------
    print("\n--- STEP 1: Save Definition to SQLite ---")
    store1 = MemoryStore(db_path=TEST_DB)
    prop = store1.propose_memory(
        category="metric_definition",
        key="senior engineers",
        value="job_role is 'Senior' and years_at_company >= 5",
        dataset_name="dummy_df"
    )
    conf = store1.confirm_proposal(prop["proposal_id"])
    print(f"Proposed: key='{prop['key']}' -> value='{prop['value']}'")
    print(f"Confirmed: {conf['message']}")
    mem_id = conf["memory_id"]

    # -------------------------------------------------------------
    # Step 2: Restart & Confirm Persistence
    # -------------------------------------------------------------
    print("\n--- STEP 2: Restart Session & Confirm Persistence ---")
    del store1
    gc.collect()

    store2 = MemoryStore(db_path=TEST_DB)
    mems = store2.list_memories()
    print(f"Loaded {len(mems)} memory entries from {TEST_DB} after restart:")
    for m in mems:
        print(f"  • ID {m['id']}: [{m['category']}] '{m['key']}' = \"{m['value']}\"")
    assert len(mems) == 1, "Persistence check failed!"
    assert mems[0]["key"] == "senior engineers"
    print("[PASS] Definition successfully persisted across session restart.")

    # -------------------------------------------------------------
    # Step 3: Contradict it & Show Old Value in memory_history
    # -------------------------------------------------------------
    print("\n--- STEP 3: Contradict Definition & Verify Audit History ---")
    update_res = store2.save_memory(
        category="metric_definition",
        key="senior engineers",
        value="job_role is 'Lead' and years_at_company >= 8",
        dataset_name="dummy_df"
    )
    print(f"Update Result: {update_res['message']}")
    history = store2.get_history(mem_id)
    print(f"Audit History Table in SQLite (memory_history): {len(history)} change records:")
    for h in history:
        print(f"  • Changed at {h['changed_at']}:")
        print(f"    Old Value: '{h['old_value']}'")
        print(f"    New Value: '{h['new_value']}'")
    assert len(history) >= 1, "Audit history was not recorded!"
    assert history[0]["old_value"] == "job_role is 'Senior' and years_at_company >= 5"
    print("[PASS] Audit history successfully tracked old value.")

    # -------------------------------------------------------------
    # Step 4: Show Unrelated Question Does NOT Inject Irrelevant Memories
    # -------------------------------------------------------------
    print("\n--- STEP 4: Relevance Filtering (No Irrelevant Injections) ---")
    unrelated_query = "What is the total population of Chicago?"
    relevant = store2.retrieve_relevant(unrelated_query, dataset_name="dummy_df")
    print(f"Query: '{unrelated_query}'")
    print(f"Retrieved memories for unrelated query: {relevant}")
    assert len(relevant) == 0, f"Expected 0 relevant memories, but got: {relevant}"
    print("[PASS] Irrelevant memories correctly filtered out.")

    # Related query verification
    related_query = "List all senior engineers in the company"
    rel_mem = store2.retrieve_relevant(related_query, dataset_name="dummy_df")
    print(f"\nQuery: '{related_query}'")
    print(f"Retrieved memories for related query: {len(rel_mem)} matches: {[m['key'] for m in rel_mem]}")
    assert len(rel_mem) == 1, "Expected 1 relevant memory for related query!"

    # -------------------------------------------------------------
    # Step 5: Test /forget command
    # -------------------------------------------------------------
    print("\n--- STEP 5: Test /forget Command ---")
    agent = DataAnalysisAgent(df=dummy_df, dataset_name="dummy_df", memory_store=store2)
    forget_res = agent.ask(f"/forget {mem_id}")
    print(f"Agent response to '/forget {mem_id}': {forget_res['answer']}")
    remaining_mems = store2.list_memories()
    print(f"Remaining memories in database: {len(remaining_mems)}")
    assert len(remaining_mems) == 0, "Memory was not forgotten!"
    print("[PASS] /forget command successfully removed memory from SQLite.")

    # Cleanup
    del agent
    del store2
    gc.collect()
    try:
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
    except Exception:
        pass

    print("\n" + "=" * 75)
    print("ALL 5 MEMORY LIFECYCLE CHECKS PASSED WITH REAL EVIDENCE!")
    print("=" * 75)

if __name__ == "__main__":
    test_full_memory_lifecycle()
