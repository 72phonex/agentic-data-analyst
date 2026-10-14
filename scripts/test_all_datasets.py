"""
End-to-end verification script testing the Data Analysis Agent on all 3 datasets
and verifying all 8 tools, memory retrieval, commands, and chart generation.
"""

import sys
import os
import json
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
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


def run_e2e_tests():
    print("=" * 80)
    print("🔬 COMPREHENSIVE END-TO-END TEST SUITE")
    print("=" * 80)

    db_path = "e2e_test_memory.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)

    memory_store = MemoryStore(db_path=db_path)
    agent = DataAnalysisAgent(memory_store=memory_store)

    # --------------------------------------------------------------------------
    # DATASET 1: Sales Transactions
    # --------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("📌 [TEST 1] SALES TRANSACTIONS DATASET")
    print("#" * 80)
    meta1 = agent.load_file("data/sales_transactions.csv")
    print(f"Loaded: {meta1['filename']} ({meta1['rows']} rows, {len(meta1['columns'])} cols)")

    # 1. Profile Dataset
    print("\n--- 1. Testing 'profile_dataset' ---")
    r1 = agent.ask("Give me an overview profile of the dataset")
    print(f"Tool: {r1['tool_used']}")
    print(f"Answer:\n{r1['answer']}")
    assert r1['tool_used'] == "profile_dataset"

    # 2. Group Aggregate
    print("\n--- 2. Testing 'group_aggregate' ---")
    r2 = agent.ask("Breakdown total amount and quantity by region")
    print(f"Tool: {r2['tool_used']}")
    print(f"Args: {r2['tool_arguments']}")
    print(f"Answer:\n{r2['answer']}")
    assert r2['tool_used'] == "group_aggregate"

    # 3. Correlation Analysis
    print("\n--- 3. Testing 'correlation_analysis' ---")
    r3 = agent.ask("What is the correlation between unit_price and total_amount?")
    print(f"Tool: {r3['tool_used']}")
    print(f"Answer:\n{r3['answer']}")
    assert r3['tool_used'] == "correlation_analysis"

    # 4. Make Chart
    print("\n--- 4. Testing 'make_chart' ---")
    r4 = agent.ask("Create a bar chart of total_amount by category")
    print(f"Tool: {r4['tool_used']}")
    print(f"Chart path: {r4['intermediate_result'].get('saved_image_path')}")
    print(f"Answer:\n{r4['answer']}")
    assert r4['tool_used'] == "make_chart"
    assert os.path.exists(r4['intermediate_result']['saved_image_path'])

    # 5. Trend Forecast
    print("\n--- 5. Testing 'trend_forecast' ---")
    r5 = agent.ask("What is the trend and forecast of total_amount over time?")
    print(f"Tool: {r5['tool_used']}")
    print(f"Answer:\n{r5['answer']}")
    assert r5['tool_used'] == "trend_forecast"

    # 6. Persistent Memory Test on Sales
    print("\n--- 6. Testing Memory Definition & Cross-Query Injection ---")
    r6_prop = agent.ask("Active customers means status is 'Active' and cancelled_flag is 0")
    print(f"Proposal: {r6_prop['answer']}")
    r6_conf = agent.ask("yes")
    print(f"Confirmation: {r6_conf['answer']}")
    assert "Confirmed" in r6_conf['answer']

    r6_query = agent.ask("What is the total revenue from active customers?")
    print(f"Memories used: {r6_query['memories_used']}")
    print(f"Query: {r6_query['tool_arguments'].get('query')}")
    print(f"Answer:\n{r6_query['answer']}")
    assert "Active customers" in r6_query['memories_used']
    assert "WHERE status = 'Active' AND cancelled_flag = 0" in r6_query['tool_arguments']['query']

    # --------------------------------------------------------------------------
    # DATASET 2: Student Performance
    # --------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("📌 [TEST 2] STUDENT PERFORMANCE DATASET")
    print("#" * 80)
    meta2 = agent.load_file("data/student_performance.csv")
    print(f"Loaded: {meta2['filename']} ({meta2['rows']} rows, {len(meta2['columns'])} cols)")

    # 7. Statistical Test (ANOVA)
    print("\n--- 7. Testing 'statistical_test' (ANOVA across departments) ---")
    r7 = agent.ask("Run an ANOVA test to see if total_score significantly differs by department")
    print(f"Tool: {r7['tool_used']}")
    print(f"Args: {r7['tool_arguments']}")
    print(f"Answer:\n{r7['answer']}")
    assert r7['tool_used'] == "statistical_test"
    assert "ANOVA" in str(r7['intermediate_result'])

    # 8. Detect Outliers (IQR / Isolation Forest)
    print("\n--- 8. Testing 'detect_outliers' (study_hours_per_week) ---")
    r8 = agent.ask("Check for outliers in study_hours_per_week using iqr")
    print(f"Tool: {r8['tool_used']}")
    print(f"Outlier count: {r8['intermediate_result'].get('total_anomalous_rows')}")
    print(f"Answer:\n{r8['answer']}")
    assert r8['tool_used'] == "detect_outliers"

    # --------------------------------------------------------------------------
    # DATASET 3: Community Health Survey
    # --------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("📌 [TEST 3] COMMUNITY HEALTH SURVEY DATASET")
    print("#" * 80)
    meta3 = agent.load_file("data/health_survey.csv")
    print(f"Loaded: {meta3['filename']} ({meta3['rows']} rows, {len(meta3['columns'])} cols)")

    # 9. Statistical Test (Chi-Square)
    print("\n--- 9. Testing 'statistical_test' (Chi-Square independence) ---")
    r9 = agent.ask("Run a chi-square test between smoker_status and diabetes_risk")
    print(f"Tool: {r9['tool_used']}")
    print(f"Answer:\n{r9['answer']}")
    assert r9['tool_used'] == "statistical_test"

    # 10. Memory Commands (/memory, /edit, /forget)
    print("\n--- 10. Testing Commands: /memory, /edit, /forget ---")
    r_list = agent.ask("/memory")
    print(f"/memory output:\n{r_list['answer']}")

    r_edit = agent.ask("/edit 1 status == 'Active' and cancelled_flag == 0 and unit_price > 50")
    print(f"/edit output:\n{r_edit['answer']}")

    r_forget = agent.ask("/forget 1")
    print(f"/forget output:\n{r_forget['answer']}")
    assert "forgotten" in r_forget['answer']

    # Cleanup
    del agent
    del memory_store
    import gc; gc.collect()
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("🎉 ALL END-TO-END TESTS PASSED SUCCESSFULLY ACROSS ALL 3 DATASETS!")
    print("=" * 80)


if __name__ == "__main__":
    run_e2e_tests()
