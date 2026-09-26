import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================

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
    "hours_stock_status",
]


# ============================================================
# Validation
# ============================================================

def validate_input_data(
    df: pd.DataFrame,
    required_columns: list = None,
) -> None:
    """Validate that required columns exist."""

    if required_columns is None:
        required_columns = [
            "dt",
            "sale_amount",
            *GROUP_COLUMNS,
        ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    print("Input data validation passed.")


# ============================================================
# Calendar Features
# ============================================================

def create_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create calendar features from the date column."""

    df = df.copy()

    df["dt"] = pd.to_datetime(
        df["dt"],
        errors="coerce"
    )

    if df["dt"].isna().any():
        raise ValueError(
            "Some values in 'dt' could not be converted to datetime."
        )

    df["day_of_week"] = df["dt"].dt.dayofweek

    df["day_of_month"] = df["dt"].dt.day

    df["week_of_year"] = (
        df["dt"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["month"] = df["dt"].dt.month

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    print("Calendar features created.")

    return df


# ============================================================
# Lag Features
# ============================================================

def create_lag_features(
    df: pd.DataFrame,
    group_cols: list = None,
    lag_windows: list = None,
    target: str = "sale_amount",
) -> pd.DataFrame:
    """
    Create past-sales lag features.

    Example:
        lag_1  -> previous observation
        lag_7  -> 7 observations ago
        lag_14 -> 14 observations ago
        lag_28 -> 28 observations ago
    """

    if group_cols is None:
        group_cols = GROUP_COLUMNS

    if lag_windows is None:
        lag_windows = LAG_WINDOWS

    df = df.copy()

    df = df.sort_values(
        group_cols + ["dt"]
    ).reset_index(drop=True)

    for lag in lag_windows:

        column_name = f"lag_{lag}"

        df[column_name] = (
            df.groupby(group_cols)[target]
            .shift(lag)
        )

    print(
        "Lag features created:",
        [f"lag_{lag}" for lag in lag_windows]
    )

    return df


# ============================================================
# Rolling Features
# ============================================================

def create_rolling_features(
    df: pd.DataFrame,
    group_cols: list = None,
    rolling_windows: list = None,
    target: str = "sale_amount",
) -> pd.DataFrame:
    """
    Create past-only rolling statistics.

    IMPORTANT:
    shift(1) is applied BEFORE rolling so that the current
    target value is never included in its own features.
    """

    if group_cols is None:
        group_cols = GROUP_COLUMNS

    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS

    df = df.copy()

    df = df.sort_values(
        group_cols + ["dt"]
    ).reset_index(drop=True)

    grouped_target = df.groupby(group_cols)[target]

    for window in rolling_windows:

        mean_column = f"rolling_mean_{window}"
        std_column = f"rolling_std_{window}"

        # Previous observations only
        shifted_target = grouped_target.shift(1)

        # Rolling mean
        df[mean_column] = (
            shifted_target
            .groupby(
                [df[col] for col in group_cols]
            )
            .transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=1
                ).mean()
            )
        )

        # Rolling standard deviation
        df[std_column] = (
            shifted_target
            .groupby(
                [df[col] for col in group_cols]
            )
            .transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=2
                ).std()
            )
        )

    print(
        "Rolling features created:",
        [
            (
                f"rolling_mean_{window}",
                f"rolling_std_{window}"
            )
            for window in rolling_windows
        ]
    )

    return df


# ============================================================
# Safe Historical Category Features
# ============================================================

def create_historical_category_features(
    df: pd.DataFrame,
    target: str = "sale_amount",
) -> pd.DataFrame:
    """
    Create historical category statistics without using
    the current/future target.

    These are expanding historical statistics.

    NOTE:
    This function is optional and is NOT enabled by default.
    """

    df = df.copy()

    df = df.sort_values("dt").reset_index(drop=True)

    category_columns = [
        "city_id",
        "store_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
    ]

    for column in category_columns:

        if column not in df.columns:
            continue

        group = df.groupby(column)[target]

        # Previous sales only
        previous_sum = group.transform(
            lambda x: x.shift(1).expanding().sum()
        )

        previous_count = group.transform(
            lambda x: x.shift(1).expanding().count()
        )

        previous_mean = (
            previous_sum /
            previous_count.replace(0, np.nan)
        )

        df[f"{column}_historical_mean"] = previous_mean

    print(
        "Historical category features created."
    )

    return df


# ============================================================
# Interaction Features
# ============================================================

def create_interaction_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create simple domain-based interaction features."""

    df = df.copy()

    if {
        "discount",
        "holiday_flag"
    }.issubset(df.columns):

        df["discount_x_holiday"] = (
            df["discount"]
            * df["holiday_flag"]
        )

    if {
        "discount",
        "activity_flag"
    }.issubset(df.columns):

        df["discount_x_activity"] = (
            df["discount"]
            * df["activity_flag"]
        )

    if {
        "avg_temperature",
        "avg_humidity"
    }.issubset(df.columns):

        df["temp_x_humidity"] = (
            df["avg_temperature"]
            * df["avg_humidity"]
        )

    print("Interaction features created.")

    return df


# ============================================================
# Remove Invalid Training Rows
# ============================================================

def remove_invalid_training_rows(
    df: pd.DataFrame,
    lag_windows: list = None,
    rolling_windows: list = None,
) -> pd.DataFrame:
    """
    Remove rows that do not have enough historical information
    for the requested lag/rolling features.
    """

    if lag_windows is None:
        lag_windows = LAG_WINDOWS

    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS

    df = df.copy()

    required_columns = []

    for lag in lag_windows:
        required_columns.append(
            f"lag_{lag}"
        )

    for window in rolling_windows:
        required_columns.append(
            f"rolling_mean_{window}"
        )

    before = len(df)

    df = df.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    removed = before - len(df)

    print(
        f"Removed {removed} rows without enough "
        f"historical information."
    )

    return df


# ============================================================
# Feature Report
# ============================================================

def feature_report(
    df: pd.DataFrame,
) -> None:
    """Print a simple feature-engineering report."""

    print("\n" + "=" * 60)
    print("FEATURE REPORT")
    print("=" * 60)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nFeatures:")

    for column in df.columns:
        print(f"  - {column}")

    print("\nMissing values:")

    missing = df.isna().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if missing.empty:
        print("  No missing values.")
    else:
        print(missing)


# ============================================================
# Full Feature Engineering Pipeline
# ============================================================

def run_feature_engineering(
    df: pd.DataFrame,
    with_lag: bool = True,
    with_rolling: bool = True,
    with_category: bool = False,
    with_interaction: bool = True,
    remove_invalid_rows: bool = False,
) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.

    Category target statistics are disabled by default because
    naive target aggregations can cause data leakage.

    Historical category features can be enabled later using
    the leakage-safe implementation.
    """

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_input_data(df)

    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------

    df = create_calendar_features(df)

    # --------------------------------------------------------
    # Lag
    # --------------------------------------------------------

    if with_lag:
        df = create_lag_features(df)

    # --------------------------------------------------------
    # Rolling
    # --------------------------------------------------------

    if with_rolling:
        df = create_rolling_features(df)

    # --------------------------------------------------------
    # Historical category features
    # --------------------------------------------------------

    if with_category:
        df = create_historical_category_features(df)

    # --------------------------------------------------------
    # Interaction
    # --------------------------------------------------------

    if with_interaction:
        df = create_interaction_features(df)

    # --------------------------------------------------------
    # Remove rows without historical information
    # --------------------------------------------------------

    if remove_invalid_rows:
        df = remove_invalid_training_rows(df)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\nFeature engineering completed.")

    feature_report(df)

    return df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    import sys

    sys.path.insert(
        0,
        __file__.rsplit("\\", 1)[0]
    )

    from prepare_data import load_data

    df = load_data()

    df = run_feature_engineering(
        df,
        with_lag=True,
        with_rolling=True,
        with_category=False,
        with_interaction=True,
        remove_invalid_rows=False,
    )