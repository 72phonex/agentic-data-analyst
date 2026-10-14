import pytest
import os
import sqlite3
from data_agent.core.memory import MemoryStore

@pytest.fixture
def memory_store(tmp_path):
    db_file = str(tmp_path / "test_memory.sqlite")
    store = MemoryStore(db_path=db_file)
    return store

def test_propose_and_confirm(memory_store):
    prop = memory_store.propose_memory(
        category="metric_definition",
        key="active customers",
        value="status == 'Active' and cancelled == 0"
    )
    assert prop["proposal_id"] in memory_store.pending_proposals
    assert prop["action"] == "add"

    res = memory_store.confirm_proposal(prop["proposal_id"])
    assert res["status"] == "created"
    assert memory_store.get_by_id(res["memory_id"]) is not None

def test_deduplication(memory_store):
    memory_store.save_memory("metric_definition", "churn rate", "cancelled / total")
    res_dup = memory_store.save_memory("metric_definition", "churn rate", "cancelled / total")
    assert res_dup["status"] == "deduplicated"
    assert len(memory_store.list_memories()) == 1

def test_update_and_audit_history(memory_store):
    init = memory_store.save_memory("metric_definition", "high_value", "spend > 100")
    mem_id = init["memory_id"]

    updated = memory_store.save_memory("metric_definition", "high_value", "spend > 500")
    assert updated["status"] == "updated"

    # Check history table
    history = memory_store.get_history(mem_id)
    assert len(history) == 1
    assert history[0]["old_value"] == "spend > 100"
    assert history[0]["new_value"] == "spend > 500"

def test_retrieve_relevant(memory_store):
    memory_store.save_memory("metric_definition", "active customers", "status == 'Active'")
    memory_store.save_memory("column_meaning", "tenure", "number of subscription months")
    memory_store.save_memory("filter", "west coast", "region in ('CA', 'WA', 'OR')")

    # Ask about active customers
    retrieved = memory_store.retrieve_relevant("How many active customers are there?")
    assert len(retrieved) >= 1
    assert retrieved[0]["key"] == "active customers"

    # Ask about tenure
    retrieved_tenure = memory_store.retrieve_relevant("What is the average tenure?")
    assert len(retrieved_tenure) >= 1
    assert retrieved_tenure[0]["key"] == "tenure"

def test_forget_command(memory_store):
    saved = memory_store.save_memory("preference", "date_format", "weekly")
    mem_id = saved["memory_id"]

    deleted = memory_store.delete_memory(mem_id)
    assert deleted is True
    assert memory_store.get_by_id(mem_id) is None

def test_edit_command(memory_store):
    saved = memory_store.save_memory("column_meaning", "gpa", "Grade Point Average out of 4.0")
    mem_id = saved["memory_id"]

    edited = memory_store.edit_memory(mem_id, "Grade Point Average out of 5.0")
    assert edited["new_value"] == "Grade Point Average out of 5.0"
    assert len(memory_store.get_history(mem_id)) == 1

def test_fact_detection(memory_store):
    cand1 = memory_store.detect_fact_candidate("Active customers means status is 'Active' and cancelled is 0")
    assert cand1 is not None
    assert cand1["key"].lower() == "active customers"

    cand2 = memory_store.detect_fact_candidate("Note that tenure refers to number of months subscribed")
    assert cand2 is not None
    assert cand2["key"].lower() == "tenure"
