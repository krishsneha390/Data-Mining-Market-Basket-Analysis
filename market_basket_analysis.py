"""
Market Basket Analysis in Python with Apriori

Business question:
Which products are purchased together more often than we would reasonably
expect from their individual popularity?

Dataset:
Groceries Dataset
https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Groceries_dataset.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD THE DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDATA PREVIEW")
print(df.head())

print("\nRAW DATA SHAPE")
print(df.shape)

print("\nMISSING VALUES")
print(df.isna().sum())


# ============================================================
# 3. DEFINE A TRANSACTION
#    Transaction = Member_number + Date
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y"
)

df["transaction_id"] = (
    df["Member_number"].astype(str)
    + "_"
    + df["Date"].dt.strftime("%Y-%m-%d")
)


# ============================================================
# 4. REMOVE DUPLICATE TRANSACTION-PRODUCT COMBINATIONS
# ============================================================

df = df.drop_duplicates(
    subset=["transaction_id", "itemDescription"]
)


# ============================================================
# 5. SUMMARISE THE DATASET
# ============================================================

summary = pd.Series({
    "purchase_rows_after_deduplication": len(df),
    "customers": df["Member_number"].nunique(),
    "transactions": df["transaction_id"].nunique(),
    "unique_items": df["itemDescription"].nunique()
})

print("\nDATASET SUMMARY")
print(summary)


# ============================================================
# 6. PRODUCT FREQUENCY
# ============================================================

item_frequency = (
    df.groupby("itemDescription")["transaction_id"]
      .nunique()
      .sort_values(ascending=False)
)

print("\nTOP 10 PRODUCTS BY TRANSACTION FREQUENCY")
print(item_frequency.head(10))


# ============================================================
# 7. VISUALISE THE MOST FREQUENT PRODUCTS
# ============================================================

top_items = item_frequency.head(15).sort_values()

plt.figure(figsize=(9, 6))
top_items.plot(kind="barh")
plt.xlabel("Number of transactions")
plt.ylabel("Product")
plt.title("Most Frequently Purchased Products")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "top_products.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 8. BUILD THE BOOLEAN BASKET MATRIX
# ============================================================

basket = pd.crosstab(
    df["transaction_id"],
    df["itemDescription"]
).gt(0)

print("\nBASKET MATRIX SHAPE")
print(basket.shape)


# ============================================================
# 9. FIND FREQUENT ITEMSETS WITH APRIORI
# ============================================================

MIN_SUPPORT = 0.001

frequent_itemsets = apriori(
    basket,
    min_support=MIN_SUPPORT,
    use_colnames=True,
    max_len=3
)

frequent_itemsets["length"] = (
    frequent_itemsets["itemsets"].apply(len)
)

print("\nMOST FREQUENT MULTI-PRODUCT ITEMSETS")
print(
    frequent_itemsets[
        frequent_itemsets["length"] >= 2
    ]
    .sort_values("support", ascending=False)
    .head(10)
)


# ============================================================
# 10. GENERATE ASSOCIATION RULES
# ============================================================

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.05,
    num_itemsets=len(basket)
)

rules = rules[
    [
        "antecedents",
        "consequents",
        "antecedent support",
        "consequent support",
        "support",
        "confidence",
        "lift"
    ]
].copy()


# ============================================================
# 11. FORMAT ITEMSETS FOR READABLE OUTPUT
# ============================================================

def format_itemset(itemset):
    """Convert a frozenset itemset into a readable comma-separated string."""
    return ", ".join(sorted(itemset))


rules["antecedents"] = rules["antecedents"].apply(format_itemset)
rules["consequents"] = rules["consequents"].apply(format_itemset)


# ============================================================
# 12. ADD SUPPORT COUNTS
# ============================================================

rules["support_count"] = (
    rules["support"] * len(basket)
).round().astype(int)


# ============================================================
# 13. INSPECT AND SAVE THE STRONGEST RULES
# ============================================================

top_rules = rules.sort_values(
    ["lift", "confidence"],
    ascending=False
)

top_rule_columns = [
    "antecedents",
    "consequents",
    "support",
    "support_count",
    "confidence",
    "lift"
]

print("\nTOP ASSOCIATION RULES BY LIFT")
print(top_rules[top_rule_columns].head(20))

top_rules[top_rule_columns].head(50).to_csv(
    TABLE_DIR / "top_rules.csv",
    index=False
)


# ============================================================
# 14. APPLY BUSINESS-ORIENTED RULE FILTERS
# ============================================================

candidate_rules = rules[
    (rules["support"] >= 0.002)
    & (rules["confidence"] >= 0.05)
    & (rules["lift"] > 1.20)
].copy()

candidate_rules = candidate_rules.sort_values(
    ["lift", "support", "confidence"],
    ascending=False
)

print("\nCANDIDATE BUSINESS RULES")
print(candidate_rules[top_rule_columns].head(20))

candidate_rules[top_rule_columns].to_csv(
    TABLE_DIR / "candidate_rules.csv",
    index=False
)


# ============================================================
# 15. VISUALISE SUPPORT, CONFIDENCE, LIFT, AND SCALE
# ============================================================

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    rules["support"],
    rules["confidence"],
    c=rules["lift"],
    s=rules["support_count"] * 2,
    alpha=0.6
)

plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Association Rules: Support vs Confidence")
plt.colorbar(scatter, label="Lift")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "association_rules_scatter.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 16. TARGET-PRODUCT ANALYSIS: SAUSAGE
# ============================================================

sausage_rules = (
    rules[
        rules["consequents"] == "sausage"
    ]
    .sort_values(
        "lift",
        ascending=False
    )
)

print("\nRULES WITH SAUSAGE AS THE CONSEQUENT")
print(sausage_rules[top_rule_columns].head(10))

sausage_rules[top_rule_columns].to_csv(
    TABLE_DIR / "sausage_rules.csv",
    index=False
)


# ============================================================
# 17. MANUALLY VERIFY:
#     whole milk + yogurt -> sausage
# ============================================================

TOTAL_TRANSACTIONS = len(basket)

antecedent_mask = basket[["whole milk", "yogurt"]].all(axis=1)
consequent_mask = basket["sausage"]
both_mask = antecedent_mask & consequent_mask

antecedent_count = int(antecedent_mask.sum())
consequent_count = int(consequent_mask.sum())
both_count = int(both_mask.sum())

support = both_count / TOTAL_TRANSACTIONS
antecedent_support = antecedent_count / TOTAL_TRANSACTIONS
consequent_support = consequent_count / TOTAL_TRANSACTIONS
confidence = support / antecedent_support
lift = confidence / consequent_support

manual_rule = pd.Series({
    "total_transactions": TOTAL_TRANSACTIONS,
    "antecedent_count": antecedent_count,
    "consequent_count": consequent_count,
    "both_count": both_count,
    "support": support,
    "confidence": confidence,
    "lift": lift
})

print("\nMANUAL RULE VERIFICATION")
print(manual_rule)


# ============================================================
# 18. HELPER FUNCTIONS FOR TIME-BASED VALIDATION
# ============================================================

def build_basket(data):
    """Build a Boolean transaction-item basket matrix."""
    temp = data.copy()

    temp["transaction_id"] = (
        temp["Member_number"].astype(str)
        + "_"
        + temp["Date"].dt.strftime("%Y-%m-%d")
    )

    temp = temp.drop_duplicates(
        subset=["transaction_id", "itemDescription"]
    )

    return pd.crosstab(
        temp["transaction_id"],
        temp["itemDescription"]
    ).gt(0)


def evaluate_rule(basket_matrix, antecedents, consequents):
    """Evaluate support, confidence, lift, and count for one rule."""
    lhs = basket_matrix[list(antecedents)].all(axis=1)
    rhs = basket_matrix[list(consequents)].all(axis=1)
    both = lhs & rhs

    support_lhs = lhs.mean()
    support_rhs = rhs.mean()
    support_both = both.mean()

    confidence_value = (
        support_both / support_lhs
        if support_lhs > 0
        else 0
    )

    lift_value = (
        confidence_value / support_rhs
        if support_rhs > 0
        else 0
    )

    return {
        "support": support_both,
        "confidence": confidence_value,
        "lift": lift_value,
        "count": int(both.sum())
    }


# ============================================================
# 19. VALIDATE THE RULE ACROSS 2014 AND 2015
# ============================================================

df_2014 = df[df["Date"].dt.year == 2014]
df_2015 = df[df["Date"].dt.year == 2015]

basket_2014 = build_basket(df_2014)
basket_2015 = build_basket(df_2015)

rule_2014 = evaluate_rule(
    basket_2014,
    antecedents={"whole milk", "yogurt"},
    consequents={"sausage"}
)

rule_2015 = evaluate_rule(
    basket_2015,
    antecedents={"whole milk", "yogurt"},
    consequents={"sausage"}
)

validation = pd.DataFrame(
    [
        {"year": 2014, **rule_2014},
        {"year": 2015, **rule_2015}
    ]
)

print("\nTIME-BASED RULE VALIDATION")
print(validation)

validation.to_csv(
    TABLE_DIR / "rule_validation_2014_2015.csv",
    index=False
)


# ============================================================
# 20. FINAL OUTPUT LOCATIONS
# ============================================================

print("\nANALYSIS COMPLETE")
print(f"Figures saved to: {FIGURE_DIR}")
print(f"Tables saved to:  {TABLE_DIR}")
