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

def test_multistep_tool_chaining_and_numerical_fidelity(tmp_path):
    df = pd.DataFrame({
        "product_id": [101, 102, 103],
        "category": ["A", "B", "A"],
        "price": [150.0, 300.0, 200.0]
    })
    mem = MemoryStore(db_path=str(tmp_path / "mem.sqlite"))
    mock_llm = LLMClient(backend="heuristic")

    # Step 1: query for max price
    # Step 2: query for product with that price
    # Step 3: finish / synthesize
    def mock_plan(user_prompt, df_columns, memories_text="", previous_tool_error=None, tool_trace=None):
        if not tool_trace or len(tool_trace) == 0:
            return {"name": "run_query", "arguments": {"query": "SELECT MAX(price) AS max_price FROM df"}}
        elif len(tool_trace) == 1:
            max_val = tool_trace[0]["result"]["data"][0]["max_price"]
            return {"name": "run_query", "arguments": {"query": f"SELECT * FROM df WHERE price = {max_val}"}}
        else:
            return {"name": "synthesize", "arguments": {}}

    synthesis_captured = {}
    def mock_synthesize(user_prompt, tool_name, tool_result, memories_text="", tool_trace=None):
        synthesis_captured["tool_trace"] = tool_trace
        synthesis_captured["tool_result"] = tool_result
        matched_price = tool_result["data"][0]["price"]
        prod_id = tool_result["data"][0]["product_id"]
        return f"The highest priced product is #{prod_id} with price {matched_price}."

    mock_llm.plan_tool_call = mock_plan
    mock_llm.synthesize_response = mock_synthesize

    agent = DataAnalysisAgent(df=df, dataset_name="products.csv", memory_store=mem, llm_client=mock_llm, max_steps=3)
    response = agent.ask("Find the highest price and which product has it")

    # Verify 2 steps were executed in sequence
    assert len(response["tool_trace"]) == 2
    assert response["tool_trace"][0]["tool"] == "run_query"
    assert response["tool_trace"][0]["result"]["data"][0]["max_price"] == 300.0
    assert response["tool_trace"][1]["tool"] == "run_query"
    assert response["tool_trace"][1]["result"]["data"][0]["product_id"] == 102

    # Verify synthesis received trace and final answer includes exact tool number
    assert len(synthesis_captured["tool_trace"]) == 2
    assert "300.0" in response["answer"]
    assert "102" in response["answer"]

def test_backend_info_and_honesty():
    client_h = LLMClient(backend="heuristic")
    info_h = client_h.get_backend_info()
    assert info_h["backend"] == "heuristic"
    assert info_h["is_heuristic"] is True
    assert "heuristic fallback" in info_h["description"].lower()

    client_o = LLMClient(backend="ollama")
    info_o = client_o.get_backend_info()
    assert info_o["backend"] == "ollama"
    assert info_o["is_heuristic"] is False
    assert "ollama" in info_o["description"].lower()

