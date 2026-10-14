import pytest
import os
import pandas as pd
from data_agent.core.agent import DataAnalysisAgent
from data_agent.core.memory import MemoryStore
from data_agent.core.llm import LLMClient

@pytest.fixture
def agent(tmp_path):
    db_file = str(tmp_path / "test_agent_memory.sqlite")
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "category": ["Electronics", "Books", "Electronics", "Apparel", "Books"],
        "price": [100.0, 20.0, 150.0, 50.0, 30.0],
        "quantity": [1, 2, 1, 3, 1],
        "total_amount": [100.0, 40.0, 150.0, 150.0, 30.0],
        "status": ["Active", "Cancelled", "Active", "Active", "Cancelled"],
        "cancelled_flag": [0, 1, 0, 0, 1]
    })
    mem = MemoryStore(db_path=db_file)
    llm = LLMClient(backend="heuristic")
    return DataAnalysisAgent(df=df, dataset_name="orders.csv", memory_store=mem, llm_client=llm)

def test_agent_profile_query(agent):
    res = agent.ask("Give me an overview profile of the dataset")
    assert res["tool_used"] == "profile_dataset"
    assert "row_count" in res["intermediate_result"]
    assert res["intermediate_result"]["row_count"] == 5

def test_agent_numeric_calculation_without_guesswork(agent):
    res = agent.ask("What is the total revenue?")
    assert res["tool_used"] == "run_query"
    assert "total_revenue" in str(res["intermediate_result"])
    # 100 + 40 + 150 + 150 + 30 = 470.0
    assert "470" in res["answer"]

def test_agent_commands(agent):
    # Add memory via command
    res_add = agent.ask("/add metric_definition active_user = status == 'Active'")
    assert "Saved new memory" in res_add["answer"]

    # List memories
    res_list = agent.ask("/memory")
    assert "active_user" in res_list["answer"]

    # Edit memory
    res_edit = agent.ask("/edit 1 status == 'Active' and cancelled_flag == 0")
    assert "updated" in res_edit["answer"].lower()

    # Forget memory
    res_forget = agent.ask("/forget 1")
    assert "forgotten" in res_forget["answer"].lower()

def test_agent_memory_injection_changes_query(agent):
    # 1. Total revenue without memory filter = 470.0
    res1 = agent.ask("What is the total revenue?")
    assert "470" in res1["answer"]

    # 2. Add definition for active customers
    agent.ask("Active customers means status is 'Active' and cancelled_flag is 0")
    agent.ask("yes") # confirm

    # 3. Query active customers -> filter applied: active customers total is 100 + 150 + 150 = 400.0
    res2 = agent.ask("What is the total revenue from active customers?")
    assert "Active customers" in res2["memories_used"]
    assert "400" in res2["answer"]

def test_agent_error_retry(agent):
    # Ask something triggering a tool with fallback
    res = agent.ask("Show me correlation")
    assert res["tool_used"] == "correlation_analysis"
    assert res["intermediate_result"]["status"] == "success"
