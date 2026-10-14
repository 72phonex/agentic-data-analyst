import sys
import pandas as pd
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data_agent.core.sandbox import run_sandboxed_sql, SandboxSecurityError

def test_security_sandbox():
    print("=" * 75)
    print("COMPREHENSIVE DUCKDB SQL SANDBOX SECURITY VERIFICATION")
    print("=" * 75)

    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "status": ["Active", "Set", "Drop"],
        "export": [100, 250, 500]
    })
    large_df = pd.DataFrame({"x": range(50000)})

    # 1. Attack Vectors (Must be blocked)
    attack_payloads = [
        ("Direct Path Exfiltration (/etc/passwd)", df, "SELECT * FROM '/etc/passwd';"),
        ("Direct Path CSV Access", df, "SELECT * FROM 'data/sales_transactions.csv';"),
        ("Filesystem Function (read_csv)", df, "SELECT * FROM read_csv('data/sales_transactions.csv');"),
        ("Filesystem Function (glob)", df, "SELECT * FROM glob('*.csv');"),
        ("DROP Table Injection", df, "DROP TABLE df;"),
        ("INSERT Data Tampering", df, "INSERT INTO df VALUES (4, 'Attacker', 'Active', 999);"),
        ("ATTACH Database Escape", df, "ATTACH 'compromise.db' AS evil;"),
        ("PRAGMA Command Execution", df, "PRAGMA database_list;"),
        ("COPY Exfiltration", df, "COPY df TO 'exfil.csv';"),
        ("Chained Semicolon Multi-Statement", df, "SELECT * FROM df; SELECT * FROM df;"),
        ("Comments Hiding Keywords", df, "SELECT * FROM df; -- harmless comment\n DROP TABLE df;"),
        ("Mixed-Case Keyword Evasion", df, "sElEcT * fRoM dF; dRoP tAbLe dF;"),
        ("Cartesian Product Runaway DoS", large_df, "SELECT COUNT(*) FROM df a, df b, df c, df d;"),
    ]

    blocked_count = 0
    print("\n--- 1. ATTACK VECTORS (ALL MUST BE BLOCKED) ---")
    for name, test_df, attack_sql in attack_payloads:
        print(f"\n[ATTACK] {name}")
        print(f"Payload:  {attack_sql.strip()}")
        try:
            res = run_sandboxed_sql(test_df, attack_sql, timeout=0.2)
            print(f"Result:   FAILED TO BLOCK! Output: {res}")
        except (SandboxSecurityError, Exception) as e:
            print(f"Result:   BLOCKED [Expected]")
            print(f"Reason:   {type(e).__name__}: {e}")
            blocked_count += 1

    print(f"\nBlocked: {blocked_count} of {len(attack_payloads)} attack vectors.")
    assert blocked_count == len(attack_payloads), "One or more attack vectors bypassed the sandbox!"

    # 2. Legitimate Queries With False-Positive Triggers (Must be ALLOWED)
    print("\n--- 2. FALSE POSITIVE REDUCTION (ALL MUST BE ALLOWED) ---")
    legitimate_queries = [
        ("Keyword 'Set' inside string literal", "SELECT * FROM df WHERE status = 'Set'"),
        ("Keyword 'Drop' inside string literal", "SELECT * FROM df WHERE status = 'Drop'"),
        ("Quoted column name matching keyword 'export'", "SELECT \"export\", name FROM df WHERE \"export\" > 150"),
        ("Semicolon inside string literal", "SELECT * FROM df WHERE status = 'Active; DROP TABLE df;'")
    ]

    allowed_count = 0
    for name, query_sql in legitimate_queries:
        print(f"\n[LEGITIMATE] {name}")
        print(f"Query:    {query_sql}")
        try:
            res = run_sandboxed_sql(df, query_sql)
            if res.get("status") == "success":
                print(f"Result:   ALLOWED [Expected] -> Returned {res['row_count']} rows")
                allowed_count += 1
            else:
                print(f"Result:   ERRORED: {res.get('message')}")
        except Exception as e:
            print(f"Result:   FALSE POSITIVE BLOCK: {type(e).__name__}: {e}")

    print(f"\nAllowed: {allowed_count} of {len(legitimate_queries)} legitimate queries.")
    assert allowed_count == len(legitimate_queries), "One or more legitimate queries were falsely blocked!"

    print("\n" + "=" * 75)
    print("[PASS] COMPREHENSIVE SECURITY SANDBOX VERIFICATION COMPLETE!")
    print(f"Summary: {blocked_count}/{len(attack_payloads)} attacks blocked, {allowed_count}/{len(legitimate_queries)} legitimate queries permitted.")
    print("=" * 75)

if __name__ == "__main__":
    test_security_sandbox()
