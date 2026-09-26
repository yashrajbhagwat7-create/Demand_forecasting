import pandas as pd
import numpy as np
from collections import OrderedDict

#=========================
#========configuration====

GROUP_COLUMNS = ["store_id", "product_id"]

LAG_WINDOWS = [1, 7, 14, 28]

ROLLING_WINDOWS = [7, 14, 28]

EXTERNAL_FEATURES = [
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
    "stock_hour6_22_cnt",
    "hours_sale",
    "hours_stock_status"]

#-----------------------------------------
# Calendar features
#-----------------------------------------

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar based features from the date column."""
    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"])

    df["day_of_week"] = df["dt"].dt.dayofweek
    df["day_of_month"] = df["dt"].dt.day
    df["week_of_year"] = (
        df["dt"]
        .dt.isocalendar()
        .week
        .astype(int)
    )
    df["month"] = df["dt"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    print("Calendar features created successfully")
    return df

#-----------------------------------------
# Lag & Rolling features
#-----------------------------------------

def create_lag_features(
    df: pd.DataFrame,
    group_cols: list = None,
    lag_windows: list = None,
    target: str = "sale_amount",
) -> pd.DataFrame:
    """Create lag features per store-product group."""
    if group_cols is None:
        group_cols = GROUP_COLUMNS
    if lag_windows is None:
        lag_windows = LAG_WINDOWS

    df = df.copy()
    df = df.sort_values(group_cols + ["dt"]).reset_index(drop=True)

    for lag in lag_windows:
        col_name = f"lag_{lag}"
        df[col_name] = df.groupby(group_cols)[target].shift(lag)

    print(f"Lag features created: {[f'lag_{w}' for w in lag_windows]}")
    return df


def create_rolling_features(
    df: pd.DataFrame,
    group_cols: list = None,
    rolling_windows: list = None,
    target: str = "sale_amount",
) -> pd.DataFrame:
    """Create rolling mean / std features per store-product group."""
    if group_cols is None:
        group_cols = GROUP_COLUMNS
    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS

    df = df.copy()
    df = df.sort_values(group_cols + ["dt"]).reset_index(drop=True)

    for window in rolling_windows:
        mean_col = f"rolling_mean_{window}"
        std_col = f"rolling_std_{window}"
        df[mean_col] = df.groupby(group_cols)[target].shift(1).transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[std_col] = df.groupby(group_cols)[target].shift(1).transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    print(f"Rolling features created: {[(f'rolling_mean_{w}', f'rolling_std_{w}') for w in rolling_windows]}")
    return df

#-----------------------------------------
# Category aggregation features
#-----------------------------------------

def create_category_features(
    df: pd.DataFrame,
    target: str = "sale_amount",
) -> pd.DataFrame:
    """Create category-level aggregation features."""
    df = df.copy()

    cat_groups = [
        "city_id",
        "store_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
    ]

    for col in cat_groups:
        if col in df.columns:
            stats = df.groupby(col)[target].agg(["mean", "sum", "std", "count"]).reset_index()
            stats.columns = [
                col,
                f"{col}_mean_sales",
                f"{col}_sum_sales",
                f"{col}_std_sales",
                f"{col}_count_sales",
            ]
            df = df.merge(stats, on=col, how="left")

    print(f"Category features created for: {cat_groups}")
    return df

#-----------------------------------------
# Interaction features
#-----------------------------------------

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple interaction features."""
    df = df.copy()

    # discount * holiday_flag
    if {"discount", "holiday_flag"}.issubset(df.columns):
        df["discount_x_holiday"] = df["discount"] * df["holiday_flag"]

    # discount * activity_flag
    if {"discount", "activity_flag"}.issubset(df.columns):
        df["discount_x_activity"] = df["discount"] * df["activity_flag"]

    # temperature * humidity
    if {"avg_temperature", "avg_humidity"}.issubset(df.columns):
        df["temp_x_humidity"] = df["avg_temperature"] * df["avg_humidity"]

    print("Interaction features created")
    return df

#-----------------------------------------
# Full feature engineering pipeline
#-----------------------------------------

def run_feature_engineering(
    df: pd.DataFrame,
    with_lag: bool = True,
    with_rolling: bool = True,
    with_category: bool = True,
    with_interaction: bool = True,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    df = create_calendar_features(df)
    if with_lag:
        df = create_lag_features(df)
    if with_rolling:
        df = create_rolling_features(df)
    if with_category:
        df = create_category_features(df)
    if with_interaction:
        df = create_interaction_features(df)

    print(f"\nFeature engineering completed.")
    print(f"Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, __file__.rsplit("\\", 1)[0])
    from prepare_data import load_data

    df = load_data()
    df = run_feature_engineering(df)