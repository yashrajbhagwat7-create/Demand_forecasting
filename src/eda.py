"""
Exploratory Data Analysis (EDA) for prepared_sales.csv.

Results (summary CSVs and PNG plots) are saved to:
    eda_results/
        plots/
        data/
"""

import os
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "prepared_sales.csv"
RESULTS_DIR = PROJECT_ROOT / "eda_results"
PLOTS_DIR = RESULTS_DIR / "plots"
DATA_OUT_DIR = RESULTS_DIR / "data"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Seaborn style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_numeric_tokens(cell: str) -> int:
    """Count numeric tokens inside a string representation of an array."""
    if pd.isna(cell):
        return 0
    return len(re.findall(r"\d+\.\d+|\d+", str(cell)))


def load_data() -> pd.DataFrame:
    """Load prepared sales CSV and add derived features."""
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Parse date
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

    # Add derived lengths for array-like columns
    df["hours_sale_len"] = df["hours_sale"].apply(count_numeric_tokens)
    df["stock_hour6_22_cnt_len"] = df["stock_hour6_22_cnt"].apply(count_numeric_tokens)

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def save_summary_stats(df: pd.DataFrame) -> None:
    """Save basic summary statistics to CSV."""
    print("\nGenerating summary statistics ...")

    # General info
    info = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null_count": df.count(),
            "unique_values": df.nunique(),
            "missing_values": df.isnull().sum(),
        }
    )
    info_path = DATA_OUT_DIR / "column_info.csv"
    info.to_csv(info_path)
    print(f"Saved column info to {info_path}")

    # Numeric descriptions
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc = df[numeric_cols].describe().T
    desc["missing"] = df[numeric_cols].isnull().sum()
    desc_path = DATA_OUT_DIR / "numeric_descriptions.csv"
    desc.to_csv(desc_path)
    print(f"Saved numeric descriptions to {desc_path}")


def plot_sales_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of sale_amount."""
    fig, ax = plt.subplots()
    sns.histplot(df["sale_amount"], bins=80, kde=True, ax=ax, color="steelblue")
    ax.set_title("Distribution of Sale Amount")
    ax.set_xlabel("Sale Amount")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sale_distribution.png")
    plt.close(fig)
    print("Saved plot: sale_distribution.png")


def plot_daily_sales(df: pd.DataFrame) -> None:
    """Line chart of total sales per day."""
    daily = df.groupby("dt")["sale_amount"].sum()
    fig, ax = plt.subplots()
    daily.plot(ax=ax, color="darkorange")
    ax.set_title("Total Sales per Day")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Sale Amount")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "daily_sales.png")
    plt.close(fig)
    print("Saved plot: daily_sales.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of correlations among numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 9},
    )
    ax.set_title("Correlation Heatmap (Numeric Features)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close(fig)
    print("Saved plot: correlation_heatmap.png")

    # Save correlation matrix as CSV
    corr.to_csv(DATA_OUT_DIR / "correlation_matrix.csv")
    print(f"Saved correlation matrix to {DATA_OUT_DIR / 'correlation_matrix.csv'}")


def plot_categorical_by_sales(df: pd.DataFrame) -> None:
    """Box plots of sale_amount by flags."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col in zip(axes, ["holiday_flag", "activity_flag", "discount"]):
        sns.boxplot(
            data=df,
            x=col,
            y="sale_amount",
            ax=ax,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
        )
        ax.set_title(f"Sale Amount by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Sale Amount")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sales_by_flags.png")
    plt.close(fig)
    print("Saved plot: sales_by_flags.png")


def plot_weather_vs_sales(df: pd.DataFrame) -> None:
    """Scatter plots of weather variables vs sale_amount."""
    weather_cols = ["precpt", "avg_temperature", "avg_humidity", "avg_wind_level"]
    fig, axes = plt.subplots(1, len(weather_cols), figsize=(20, 4))
    for ax, col in zip(axes, weather_cols):
        ax.scatter(df[col], df["sale_amount"], alpha=0.2, s=5, color="teal")
        ax.set_title(f"Sale Amount vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Sale Amount")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "weather_vs_sales.png")
    plt.close(fig)
    print("Saved plot: weather_vs_sales.png")


def plot_sales_by_category(df: pd.DataFrame) -> None:
    """Bar plot of total sales by first_category_id."""
    if "first_category_id" in df.columns:
        cat_sales = df.groupby("first_category_id")["sale_amount"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots()
        cat_sales.plot(kind="bar", ax=ax, color="salmon")
        ax.set_title("Total Sales by First Category")
        ax.set_xlabel("First Category ID")
        ax.set_ylabel("Total Sale Amount")
        plt.xticks(rotation=45)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "sales_by_category.png")
        plt.close(fig)
        print("Saved plot: sales_by_category.png")


def plot_stockout_vs_sales(df: pd.DataFrame) -> None:
    """Violin/swarm plot of stockout count vs sale_amount."""
    fig, ax = plt.subplots()
    sns.violinplot(
        data=df,
        x="stock_hour6_22_cnt",
        y="sale_amount",
        ax=ax,
        inner="quartile",
    )
    ax.set_title("Sale Amount by Stockout Count (stock_hour6_22_cnt)")
    ax.set_xlabel("Stockout Count")
    ax.set_ylabel("Sale Amount")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "stockout_vs_sales.png")
    plt.close(fig)
    print("Saved plot: stockout_vs_sales.png")


def plot_hours_sale_len_vs_sales(df: pd.DataFrame) -> None:
    """Scatter of hours_sale_len vs sale_amount."""
    fig, ax = plt.subplots()
    ax.scatter(df["hours_sale_len"], df["sale_amount"], alpha=0.2, s=5, color="purple")
    ax.set_title("Sale Amount vs Hours Sale Length")
    ax.set_xlabel("Hours Sale Length (count of intervals)")
    ax.set_ylabel("Sale Amount")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "hours_sale_len_vs_sales.png")
    plt.close(fig)
    print("Saved plot: hours_sale_len_vs_sales.png")


def plot_top_products(df: pd.DataFrame, n: int = 10) -> None:
    """Bar plot of total sales by top N products."""
    top = df.groupby("product_id")["sale_amount"].sum().sort_values(ascending=False).head(n)
    fig, ax = plt.subplots()
    top.plot(kind="bar", ax=ax, color="mediumseagreen")
    ax.set_title(f"Top {n} Products by Total Sales")
    ax.set_xlabel("Product ID")
    ax.set_ylabel("Total Sale Amount")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"top_{n}_products.png")
    plt.close(fig)
    print(f"Saved plot: top_{n}_products.png")


def plot_hourly_sales(df: pd.DataFrame) -> None:
    """Bar plot of average sales by hour of the day."""
    df["hour"] = df["dt"].dt.hour
    hourly_sales = df.groupby("hour")["sale_amount"].mean()
    fig, ax = plt.subplots()
    hourly_sales.plot(kind="bar", ax=ax, color="darkcyan")
    ax.set_title("Average Sales by Hour of Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Sale Amount")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "hourly_sales.png")
    plt.close(fig)
    print("Saved plot: hourly_sales.png")


def plot_day_of_week_sales(df: pd.DataFrame) -> None:
    """Bar plot of average sales by day of the week."""
    df["day_of_week"] = df["dt"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_of_week_sales = df.groupby("day_of_week")["sale_amount"].mean().reindex(day_order)
    fig, ax = plt.subplots()
    day_of_week_sales.plot(kind="bar", ax=ax, color="indigo")
    ax.set_title("Average Sales by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Sale Amount")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "day_of_week_sales.png")
    plt.close(fig)
    print("Saved plot: day_of_week_sales.png")


def plot_monthly_sales(df: pd.DataFrame) -> None:
    """Line plot of total sales by month."""
    df["month"] = df["dt"].dt.to_period("M")
    monthly_sales = df.groupby("month")["sale_amount"].sum()
    fig, ax = plt.subplots()
    monthly_sales.plot(kind="line", ax=ax, color="firebrick")
    ax.set_title("Total Sales by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sale Amount")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "monthly_sales.png")
    plt.close(fig)
    print("Saved plot: monthly_sales.png")


def plot_acf_pacf(df: pd.DataFrame) -> None:
    """Plot ACF and PACF for daily sales."""
    daily_sales = df.groupby("dt")["sale_amount"].sum()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(daily_sales, lags=30, ax=axes[0])
    axes[0].set_title("Autocorrelation Function (ACF) for Daily Sales")
    plot_pacf(daily_sales, lags=30, ax=axes[1])
    axes[1].set_title("Partial Autocorrelation Function (PACF) for Daily Sales")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "acf_pacf_daily_sales.png")
    plt.close(fig)
    print("Saved plot: acf_pacf_daily_sales.png")


def plot_missing_values(df: pd.DataFrame) -> None:
    """Heatmap of missing values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False, ax=ax)
    ax.set_title("Missing Values Heatmap")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "missing_values.png")
    plt.close(fig)
    print("Saved plot: missing_values.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("STARTING EDA")
    print("=" * 60)

    df = load_data()

    # Basic info
    print("\nDataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Date range:", df["dt"].min(), "to", df["dt"].max())

    save_summary_stats(df)

    print("\nGenerating plots ...")
    plot_sales_distribution(df)
    plot_daily_sales(df)
    plot_correlation_heatmap(df)
    plot_categorical_by_sales(df)
    plot_weather_vs_sales(df)
    plot_sales_by_category(df)
    plot_stockout_vs_sales(df)
    plot_hours_sale_len_vs_sales(df)
    plot_top_products(df, n=10)
    plot_hourly_sales(df)
    plot_day_of_week_sales(df)
    plot_monthly_sales(df)
    plot_acf_pacf(df)
    plot_missing_values(df)

    # Save a simple text report
    report_path = DATA_OUT_DIR / "eda_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("EDA REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset shape: {df.shape}\n")
        f.write(f"Date range: {df['dt'].min().date()} to {df['dt'].max().date()}\n\n")
        f.write("Missing values:\n")
        f.write(str(df.isnull().sum()) + "\n\n")
        f.write("Numeric summary:\n")
        f.write(str(df.describe()) + "\n")
    print(f"\nSaved text report to {report_path}")

    print("\n" + "=" * 60)
    print("EDA COMPLETED")
    print(f"All results saved in: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
