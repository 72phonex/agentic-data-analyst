import os
from memory import load_memories, save_memory, delete_memory, find_relevant_memories

TEST_MEM_FILE = "test_memory.json"

def setup_function():
    """Ensure clean slate before each memory test."""
    if os.path.exists(TEST_MEM_FILE):
        os.remove(TEST_MEM_FILE)

def teardown_function():
    """Clean up test memory file after each test."""
    if os.path.exists(TEST_MEM_FILE):
        os.remove(TEST_MEM_FILE)

def test_memory_save_and_load():
    """Verify saving a rule writes to file and loads correctly."""
    item = save_memory("active customers means status is Active and cancelled_flag is 0", filepath=TEST_MEM_FILE)
    assert item["text"].startswith("active customers")

    loaded = load_memories(filepath=TEST_MEM_FILE)
    assert len(loaded) == 1
    assert loaded[0]["id"] == item["id"]

def test_memory_delete():
    """Verify deleting a rule by ID."""
    item1 = save_memory("rule one for testing", filepath=TEST_MEM_FILE)
    item2 = save_memory("rule two for testing", filepath=TEST_MEM_FILE)
    assert len(load_memories(filepath=TEST_MEM_FILE)) == 2

    deleted = delete_memory(item1["id"], filepath=TEST_MEM_FILE)
    assert deleted is True
    remaining = load_memories(filepath=TEST_MEM_FILE)
    assert len(remaining) == 1
    assert remaining[0]["id"] == item2["id"]

def test_memory_keyword_retrieval():
    """Verify relevant rules are retrieved only when they share keywords with the question."""
    save_memory("revenue means total price minus tax and shipping", filepath=TEST_MEM_FILE)
    save_memory("active customers have status Active", filepath=TEST_MEM_FILE)

    # Should match first rule (contains 'revenue')
    results1 = find_relevant_memories("Can you calculate the total revenue?", filepath=TEST_MEM_FILE)
    assert len(results1) == 1
    assert "revenue means total price" in results1[0]

    # Should match second rule (contains 'customers')
    results2 = find_relevant_memories("How many customers do we have?", filepath=TEST_MEM_FILE)
    assert len(results2) == 1
    assert "active customers" in results2[0]

    # Unrelated question should match nothing
    results3 = find_relevant_memories("What is the weather outside today?", filepath=TEST_MEM_FILE)
    assert len(results3) == 0
