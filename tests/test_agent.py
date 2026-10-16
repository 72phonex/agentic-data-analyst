from unittest.mock import patch
import pandas as pd
from agent import ask_agent

def test_agent_loop_with_mocked_model():
    """Verify agent loop runs a tool on step 1 and the final answer includes the tool's computed number."""
    df = pd.DataFrame({
        "employee": ["Alice", "Bob", "Charlie"],
        "salary": [50000, 60000, 70000]
    })

    # Step 1 response from model: model requests group_stats tool call
    step1_response = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "run_sql",
                        "arguments": {"query": "SELECT AVG(salary) AS avg_sal FROM df"}
                    }
                }
            ]
        }
    }

    # Step 2 response from model: after receiving tool result (avg_sal = 60000), model generates answer
    step2_response = {
        "message": {
            "role": "assistant",
            "content": "The average salary computed from the dataset is 60000.0."
        }
    }

    # Mock call_ollama so the test runs completely offline without network
    with patch("agent.call_ollama", side_effect=[step1_response, step2_response]) as mock_call:
        res = ask_agent(df, "What is the average salary?")
        assert mock_call.call_count == 2
        assert len(res["traces"]) == 1
        assert res["traces"][0]["tool"] == "run_sql"
        assert res["traces"][0]["result"]["data"][0]["avg_sal"] == 60000.0
        # Prove the final answer explicitly contains the computed number 60000
        assert "60000" in res["answer"]

def test_streamlit_app_renders():
    """Headless check proving app.py imports and renders cleanly via Streamlit AppTest."""
    from streamlit.testing.v1 import AppTest
    import os
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
    at = AppTest.from_file(app_path)
    at.run(timeout=15)
    assert len(at.exception) == 0
