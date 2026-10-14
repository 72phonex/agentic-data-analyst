import sys
import pandas as pd
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data_agent.core.sandbox import run_sandboxed_sql, SandboxSecurityError

def test_security_sandbox():
    print("=" * 70)
    print("SECURITY SANDBOX ATTACK INJECTION VERIFICATION")
    print("=" * 70)

    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "secret_balance": [1000, 2500, 5000]
    })

    attack_payloads = [
        ("DROP Table Injection", "DROP TABLE df; SELECT * FROM df;"),
        ("INSERT Data Tampering", "INSERT INTO df VALUES (4, 'Attacker', 999999);"),
        ("Filesystem Exfiltration (read_csv)", "SELECT * FROM read_csv('/etc/passwd');"),
        ("Filesystem Windows Exfiltration (read_csv)", "SELECT * FROM read_csv('C:/Windows/win.ini');"),
        ("Chained Semicolon Multi-Statement", "SELECT * FROM df; SELECT * FROM df;"),
        ("PRAGMA Command Execution", "PRAGMA database_list;"),
        ("DELETE Command Execution", "DELETE FROM df WHERE id = 1;")
    ]

    blocked_count = 0

    for name, attack_sql in attack_payloads:
        print(f"\n[ATTACK VECTOR] {name}")
        print(f"Payload:  {attack_sql}")
        try:
            res = run_sandboxed_sql(df, attack_sql)
            print(f"Result:   FAILED TO BLOCK! Output: {res}")
        except (SandboxSecurityError, Exception) as e:
            print(f"Result:   BLOCKED [Expected]")
            print(f"Reason:   {type(e).__name__}: {e}")
            blocked_count += 1

    print("\n" + "=" * 70)
    print(f"SUMMARY: {blocked_count} of {len(attack_payloads)} attack vectors successfully blocked.")
    print("=" * 70)
    assert blocked_count == len(attack_payloads), "One or more attack vectors were not blocked!"
    print("[PASS] Security sandbox passed all injection & filesystem protection checks.")

if __name__ == "__main__":
    test_security_sandbox()
