import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng(42)
n = 2000

orders = pd.DataFrame({
    "order_date": (
        pd.Timestamp("2025-01-01")
        + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    ),
    "customer_id": rng.integers(1, 301, n),
    "category": rng.choice(
        ["Electronics", "Fashion", "Home", "Books"],
        n,
        p=[0.30, 0.35, 0.20, 0.15]
    ),
    "amount": rng.gamma(2.0, 600, n).round(2)
})

orders["order_date"] = pd.to_datetime(
    orders["order_date"], errors="coerce"
)

orders = orders.drop_duplicates()

orders = orders.dropna(
    subset=["order_date", "customer_id", "category", "amount"]
).copy()

orders = orders[orders["amount"] > 0].copy()

dataset_shape = orders.shape
data_types = orders.dtypes
missing_values = orders.isnull().sum()
duplicate_count = orders.duplicated().sum()

orders["month"] = orders["order_date"].dt.to_period("M")
orders["month_name"] = orders["order_date"].dt.month_name()
orders["weekday"] = orders["order_date"].dt.day_name()
orders["quarter"] = orders["order_date"].dt.quarter
orders["year"] = orders["order_date"].dt.year


monthly = orders.groupby("month")["amount"].sum()

monthly = monthly.reindex(
    pd.period_range("2025-01", "2025-12", freq="M"),
    fill_value=0
)

monthly_growth = monthly.pct_change() * 100

monthly_analysis = pd.DataFrame({
    "Revenue": monthly,
    "Growth_Percentage": monthly_growth
})

best_month = monthly.idxmax()
worst_month = monthly.idxmin()
average_monthly_revenue = monthly.mean()
total_revenue = orders["amount"].sum()

print(monthly_analysis.round(2))
plt.figure(figsize=(12, 6))

plt.plot(
    monthly.index.astype(str),
    monthly.values,
    marker="o",
    linewidth=2
)

plt.scatter(
    str(best_month),
    monthly.max(),
    s=100,
    label="Best Month"
)

plt.scatter(
    str(worst_month),
    monthly.min(),
    s=100,
    label="Lowest Month"
)

plt.title("Monthly E-Commerce Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

by_category = (
    orders.groupby("category")["amount"]
    .agg(
        Total_Revenue="sum",
        Average_Order_Value="mean",
        Number_of_Orders="count"
    )
    .sort_values("Total_Revenue", ascending=False)
)

best_category = by_category.index[0]

print(by_category.round(2))


plt.figure(figsize=(9, 5))

plt.bar(
    by_category.index,
    by_category["Total_Revenue"]
)

plt.title("Category-Wise Revenue Analysis")
plt.xlabel("Category")
plt.ylabel("Total Revenue")
plt.tight_layout()
plt.show()

pivot_table = pd.pivot_table(
    orders,
    values="amount",
    index="month",
    columns="category",
    aggfunc="sum",
    fill_value=0
)

print(pivot_table.round(2))
customer_spending = (
    orders.groupby("customer_id")["amount"]
    .sum()
    .sort_values(ascending=False)
)

number_of_customers = len(customer_spending)

top_10_count = max(
    1,
    int(np.ceil(number_of_customers * 0.10))
)
top_customers = customer_spending.head(top_10_count)
top_customer_revenue = top_customers.sum()
top_share = (
    top_customer_revenue / total_revenue
) * 100
customer_analysis = customer_spending.reset_index()
customer_analysis.columns = [
    "Customer_ID",
    "Total_Spending"
]
customer_analysis["Customer_Rank"] = range(
    1, len(customer_analysis) + 1
)
customer_analysis["Revenue_Contribution_Percentage"] = (
    customer_analysis["Total_Spending"] / total_revenue
) * 100
print(customer_analysis.head(20).round(2))
customer_cumulative = customer_spending.cumsum()
customer_cumulative_percentage = (
    customer_cumulative / total_revenue
) * 100

pareto_data = pd.DataFrame({
    "Customer_Rank": range(1, number_of_customers + 1),
    "Cumulative_Revenue_Percentage":
        customer_cumulative_percentage.values
})

plt.figure(figsize=(10, 6))

plt.plot(
    pareto_data["Customer_Rank"],
    pareto_data["Cumulative_Revenue_Percentage"]
)

plt.axhline(
    y=80,
    linestyle="--",
    label="80% Revenue"
)

plt.axvline(
    x=top_10_count,
    linestyle="--",
    label="Top 10% Customers"
)

plt.title("Customer Revenue Pareto Analysis")
plt.xlabel("Customer Rank")
plt.ylabel("Cumulative Revenue (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
quarterly_sales = (
    orders.groupby("quarter")["amount"]
    .sum()
    .reset_index()
)

quarterly_sales.columns = [
    "Quarter",
    "Total_Revenue"
]

print(quarterly_sales.round(2))
weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_sales = (
    orders.groupby("weekday")["amount"]
    .sum()
    .reindex(weekday_order)
)

print(weekday_sales.to_frame("Total_Revenue").round(2))

plt.figure(figsize=(10, 5))

plt.bar(
    weekday_sales.index,
    weekday_sales.values
)

plt.title("Weekday-Wise Revenue Analysis")
plt.xlabel("Weekday")
plt.ylabel("Revenue")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
best_category_share = (
    by_category.loc[best_category, "Total_Revenue"]
    / total_revenue
) * 100

best_weekday = weekday_sales.idxmax()

recommendations = pd.DataFrame({
    "Area": [
        "Customer Retention",
        "Category Marketing",
        "Sales Improvement"
    ],
    "Action": [
        "Introduce loyalty rewards for high-value customers",
        f"Prioritise the {best_category} category",
        f"Improve sales during {worst_month}"
    ],
    "Supporting_Metric": [
        round(top_share, 2),
        round(best_category_share, 2),
        round(monthly.min(), 2)
    ],
    "Metric": [
        "Top 10% Revenue Contribution (%)",
        "Best Category Revenue Share (%)",
        "Lowest Monthly Revenue"
    ]
})
print(recommendations)
kpi_summary = pd.DataFrame({
    "Metric": [
        "Total Orders",
        "Total Customers",
        "Total Revenue",
        "Average Monthly Revenue",
        "Best Month",
        "Lowest Month",
        "Best Category",
        "Top 10% Customer Revenue",
        "Top 10% Revenue Contribution (%)",
        "Best Performing Weekday"
    ],
    "Value": [
        len(orders),
        number_of_customers,
        round(total_revenue, 2),
        round(average_monthly_revenue, 2),
        str(best_month),
        str(worst_month),
        best_category,
        round(top_customer_revenue, 2),
        round(top_share, 2),
        best_weekday
    ]
})

print(kpi_summary)

# 15. EXPORT RESULTS

output_dir = Path("Project_03_Results")
output_dir.mkdir(exist_ok=True)

orders.to_csv(
    output_dir / "cleaned_orders.csv",
    index=False
)

monthly_analysis.to_csv(
    output_dir / "monthly_sales_analysis.csv"
)

by_category.to_csv(
    output_dir / "category_analysis.csv"
)

pivot_table.to_csv(
    output_dir / "monthly_category_pivot.csv"
)
customer_analysis.to_csv(
    output_dir / "customer_analysis.csv",
    index=False
)
quarterly_sales.to_csv(
    output_dir / "quarterly_analysis.csv",
    index=False
)
pareto_data.to_csv(
    output_dir / "pareto_analysis.csv",
    index=False
)
recommendations.to_csv(
    output_dir / "business_recommendations.csv",
    index=False
)
kpi_summary.to_csv(
    output_dir / "kpi_summary.csv",
    index=False
)
