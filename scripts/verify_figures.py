import pandas as pd

df = pd.read_csv("data/sales_transactions.csv")
total_rev = df["total_amount"].sum()
active_df = df[(df["status"] == "Active") & (df["cancelled_flag"] == 0)]
active_rev = active_df["total_amount"].sum()

print("=" * 60)
print("INDEPENDENT PLAIN PANDAS VERIFICATION")
print("=" * 60)
print(f"Total rows in dataset:                 {len(df)}")
print(f"1. Total revenue (sum of total_amount): ${total_rev:,.2f} ({total_rev})")
print(f"2. Active customer matching rows:       {len(active_df)}")
print(f"   Active customer revenue:             ${active_rev:,.2f} ({active_rev})")
print("=" * 60)
assert round(total_rev, 2) == 92477.23, f"Mismatch on total revenue: {total_rev}"
assert round(active_rev, 2) == 37023.46, f"Mismatch on active revenue: {active_rev}"
print("ASSERTIONS PASSED: Exactly matches $92,477.23 and $37,023.46!")
