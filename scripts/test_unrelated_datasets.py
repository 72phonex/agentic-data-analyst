import sys
from pathlib import Path
import pandas as pd

# Ensure data_agent can be imported
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data_agent.core.agent import DataAnalysisAgent
from data_agent.core.memory import MemoryStore

def run_unrelated_tests():
    print("=" * 80)
    print("AGENTIC EVALUATION ON 2 UNRELATED DATASETS (5 QUESTIONS EACH)")
    print("=" * 80)

    # -------------------------------------------------------------
    # Dataset 1: Employee Attrition & Compensation
    # -------------------------------------------------------------
    emp_path = "data/employee_attrition.csv"
    df_emp = pd.read_csv(emp_path)
    mem_emp = MemoryStore(db_path=":memory:")
    agent_emp = DataAnalysisAgent(df=df_emp, dataset_name="employee_attrition.csv", memory_store=mem_emp)

    emp_questions = [
        "Profile this employee dataset",
        "What is the average annual salary by department?",
        "Analyze the correlation between years_at_company, annual_salary, and job_satisfaction",
        "Detect outliers in annual_salary",
        "Is there a statistically significant difference in annual_salary across department using anova?"
    ]

    print("\n" + "#" * 80)
    print(f"DATASET 1: {emp_path} (Shape: {df_emp.shape[0]} rows, {df_emp.shape[1]} columns)")
    print("#" * 80)

    for i, q in enumerate(emp_questions, 1):
        print(f"\n[Q{i}] User > {q}")
        res = agent_emp.ask(q)
        print(f"Tool Used:    [{res['tool_used']}]")
        print(f"Tool Args:    {res['tool_arguments']}")
        print(f"Agent Output: {res['answer'][:200]}...")

    # -------------------------------------------------------------
    # Dataset 2: Real Estate Listings
    # -------------------------------------------------------------
    re_path = "data/real_estate_listings.csv"
    df_re = pd.read_csv(re_path)
    mem_re = MemoryStore(db_path=":memory:")
    agent_re = DataAnalysisAgent(df=df_re, dataset_name="real_estate_listings.csv", memory_store=mem_re)

    re_questions = [
        "Summarize and profile the real estate dataset",
        "What is the average listing_price across all properties?",
        "Breakdown average listing_price and count by city",
        "Show the relationship and correlation between square_feet and listing_price",
        "Create a scatter chart comparing square_feet and listing_price"
    ]

    print("\n" + "#" * 80)
    print(f"DATASET 2: {re_path} (Shape: {df_re.shape[0]} rows, {df_re.shape[1]} columns)")
    print("#" * 80)

    for i, q in enumerate(re_questions, 1):
        print(f"\n[Q{i}] User > {q}")
        res = agent_re.ask(q)
        print(f"Tool Used:    [{res['tool_used']}]")
        print(f"Tool Args:    {res['tool_arguments']}")
        print(f"Agent Output: {res['answer'][:200]}...")

    print("\n" + "=" * 80)
    print("[PASS] Successfully executed all 10 questions across 2 completely new datasets!")
    print("=" * 80)

if __name__ == "__main__":
    run_unrelated_tests()
