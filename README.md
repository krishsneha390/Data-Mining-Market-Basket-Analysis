## **What Do Customers Buy Together? Market Basket Analysis in Python**

A practical data analytics project using **market basket analysis** and the **Apriori algorithm** to discover products that are purchased together more often than expected from their individual popularity.

The project uses the **Groceries Dataset** from Kaggle and is built around one business question:

> Which products are purchased together more often than we would reasonably expect from their individual popularity?

Rather than treating Apriori as a library function to run and sort, the project develops the analysis from the business problem through transaction construction, product frequency, support, confidence, lift, association rules, business-oriented filtering, visualisation, and time-based validation.

---

## Project Overview

Retail transaction data can tell us which products are popular, but popularity alone does not tell us whether two products have a meaningful relationship.

A common product such as whole milk will naturally appear beside many other items. If milk and yogurt are frequently purchased together, that does not automatically mean yogurt buyers have an unusual preference for milk.

This project separates simple co-occurrence from meaningful association using three core metrics:

- **Support** — how frequently an item or itemset occurs across all baskets.
- **Confidence** — how often the consequent appears when the antecedent is present.
- **Lift** — how much more likely the consequent is when the antecedent is present compared with the consequent's normal baseline probability.

The analysis then uses the **Apriori algorithm** to search efficiently for frequent itemsets and generate directional association rules.

---

## Business Question

**Which products are purchased together more often than we would reasonably expect from their individual popularity?**

Possible business applications include:

- product recommendations
- cross-selling
- product bundling
- merchandising research
- promotion design
- customer behaviour analysis

The project treats association rules as **hypothesis generators**, not causal evidence. A high-lift rule does not prove that bundling, recommending, or repositioning products will increase sales.

---

## Dataset

**Dataset:** Groceries Dataset  
**Source:** Kaggle  
**Author:** Heeral Dedhia  
**Link:** https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset

The raw dataset contains approximately:

- **38,765 purchase rows**
- **3 columns**
- **3,898 customers**
- **14,963 reconstructed transactions**
- **167 unique products**

### Columns

| Column | Description |
|---|---|
| `Member_number` | Customer identifier |
| `Date` | Purchase date |
| `itemDescription` | Product purchased |

One raw row represents one purchased product, not one complete basket.

For this project, a transaction is reconstructed using:

```text
Member_number + Date
```

This assumes that all products bought by the same member on the same date belong to one basket.

---

## Analytical Workflow

```text
Business problem
        ↓
Define the analytical question
        ↓
Understand what one raw row represents
        ↓
Reconstruct customer-date baskets
        ↓
Remove duplicate transaction-product rows
        ↓
Measure individual product popularity
        ↓
Convert transactions to a one-hot basket matrix
        ↓
Choose a minimum support threshold
        ↓
Find frequent itemsets with Apriori
        ↓
Generate association rules
        ↓
Interpret support, confidence, and lift
        ↓
Add supporting transaction counts
        ↓
Filter weak or very small rules
        ↓
Visualise support, confidence, lift, and scale
        ↓
Investigate target products
        ↓
Validate important rules across time
        ↓
Translate stable associations into testable business hypotheses
```

---

## Key Concepts

### Support

For products `A` and `B`:

```text
Support(A ∪ B) = transactions containing A and B / total transactions
```

Support measures how common a combination is across the full transaction database.

### Confidence

```text
Confidence(A → B) = Support(A ∪ B) / Support(A)
```

Confidence is directional. It estimates the proportion of baskets containing `A` that also contain `B`.

### Lift

```text
Lift(A → B) = Confidence(A → B) / Support(B)
```

Lift compares the observed conditional probability with the baseline probability of the consequent.

| Lift | Meaning |
|---:|---|
| `> 1` | Positive association |
| `= 1` | Consistent with independence |
| `< 1` | Negative association |

---

## Example Finding

One notable rule in the dataset is:

```text
whole milk + yogurt -> sausage
```

Approximate metrics:

| Metric | Value |
|---|---:|
| Support | 0.00147 |
| Confidence | 0.1317 |
| Lift | 2.1829 |
| Supporting transactions | 22 |

Sausage appears in roughly **6.03%** of all baskets but approximately **13.17%** of baskets containing both whole milk and yogurt.

The lift of approximately **2.18** means sausage is about **2.18 times as likely to appear in those baskets as its normal transaction frequency would suggest**.

This is an association, not a causal effect.

---

## Project Structure

```text
market-basket-analysis-python/
│
├── README.md

├── .gitignore
│
├── data/
│   ├── README.md
│   └── Groceries_dataset.csv
│

│   └── market_basket_analysis.py
│
├
│   ├── figures/
│   │   ├── top_products.png
│   │   └── association_rules_scatter.png
│   │
│   └── tables/
│       ├── top_rules.csv
│       ├── candidate_rules.csv
│       ├── sausage_rules.csv
│       └── rule_validation_2014_2015.csv
│
└── tutorial/
    └── market-basket-analysis.md
```

### Dataset note

Do **not** commit the Kaggle CSV unless its licence clearly permits redistribution. The included `.gitignore` excludes `data/Groceries_dataset.csv` by default.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/market-basket-analysis-python.git
cd market-basket-analysis-python
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download `Groceries_dataset.csv` from Kaggle and place it inside:

```text
data/
```

Run the project:

```bash
python src/market_basket_analysis.py
```

---

## Requirements

The analysis uses:

```text
pandas
matplotlib
mlxtend
```

---

## Outputs

Running the script creates:

### Figures

- `outputs/figures/top_products.png`
- `outputs/figures/association_rules_scatter.png`

### Tables

- `outputs/tables/top_rules.csv`
- `outputs/tables/candidate_rules.csv`
- `outputs/tables/sausage_rules.csv`
- `outputs/tables/rule_validation_2014_2015.csv`

---

## Important Analytical Choices

### Transaction definition

The dataset does not provide a unique basket ID.

This project defines one transaction as:

```text
Member_number + Date
```

If the same member visited the store twice on the same day, those visits cannot be separated.

### Minimum support

The exploratory Apriori run uses:

```python
min_support = 0.001
```

With 14,963 baskets, this corresponds to roughly 15 transactions. This is intentionally permissive for exploration and should not be interpreted as a universal business threshold.

### Candidate business rules

The project also applies a stricter filter:

```python
support >= 0.002
confidence >= 0.05
lift > 1.20
```

This removes many high-lift relationships supported by very few baskets.

---

## Why Lift Matters

Confidence can be misleading when the consequent is already very common.

If 90% of all customers buy milk and 90% of bread buyers also buy milk:

```text
Confidence(Bread -> Milk) = 90%
```

That sounds strong, but bread has not made milk more likely than its existing baseline. Lift corrects for this by comparing conditional probability with baseline popularity.

---

## Time-Based Validation

A rule that appears in the complete dataset may not be stable.

The script therefore evaluates an important rule separately in **2014** and **2015**.

This does not prove causality, but it provides a simple robustness check:

> Does the relationship remain visible when the data moves forward in time?

---

## Limitations

- There is no true basket ID.
- Quantity information is not used.
- Prices and profit margins are unavailable.
- Promotion history is unavailable.
- Store layout is unavailable.
- Inventory constraints are unavailable.
- Recommendation exposure is unavailable.
- Rich customer characteristics are unavailable.
- Association does not establish causation.
- Product relationships can change over time.

These limitations determine what the analysis can reasonably claim.

---

## Business Interpretation

Market basket analysis is useful for identifying where a business may want to investigate further.

```text
transactions
    ↓
patterns
    ↓
robust associations
    ↓
business hypotheses
    ↓
controlled experiments
    ↓
evidence-based decisions
```

For example, a stable association such as:

```text
whole milk + yogurt -> sausage
```

could motivate an A/B test in which eligible customers are randomly assigned to a normal shopping experience or a sausage recommendation when whole milk and yogurt are both in the basket.

That experiment would answer a causal question association rules alone cannot answer:

> Does showing the recommendation actually increase sausage purchases?

---

## Skills Demonstrated

- transaction-level data preparation
- exploratory data analysis
- one-hot transaction encoding
- association rule mining
- Apriori
- support
- confidence
- lift
- business-oriented threshold selection
- rule filtering
- data visualisation
- temporal robustness checks
- association versus causation
- translating analytics into testable business hypotheses

---

## Possible Extensions

- Apriori versus FP-Growth performance comparison
- seasonal association analysis
- monthly rule stability
- customer segmentation
- recommendation-system integration
- rule network visualisation
- leverage and conviction
- statistical uncertainty for low-support rules
- product hierarchy analysis
- A/B test simulation for recommendation strategies

---


