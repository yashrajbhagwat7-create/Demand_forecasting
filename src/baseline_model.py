"""
Baseline Model Pipeline
Feature Engineering -> Train/Test Split -> Feature Selection -> Training -> Evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import time

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
)

# Import YOUR feature engineering pipeline
from feature_engineering import run_feature_engineering


warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "model_results"
RESULTS_DIR.mkdir(exist_ok=True)

DATA_PATH = PROJECT_ROOT / "data" / "prepared_sales.csv"

TARGET = "sale_amount"

TEST_SIZE = 0.20

RANDOM_STATE = 42

N_ESTIMATORS = 100

TOP_K_FEATURES = 20


# ============================================================
# Load Data
# ============================================================

def load_data():
    """Load prepared sales data."""

    print("\n--- Loading Data ---")

    start_time = time.time()

    df = pd.read_csv(DATA_PATH)

    print(
        f"Loaded {len(df)} rows "
        f"in {time.time() - start_time:.2f}s"
    )

    print(f"Shape: {df.shape}")

    return df


# ============================================================
# Prepare X and y
# ============================================================

def prepare_xy(
    df: pd.DataFrame,
    target: str = TARGET,
):
    """
    Prepare feature matrix X and target y.

    Only numeric columns are used for the baseline models.
    """

    print("\n--- Preparing X and y ---")

    df = df.copy()

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found."
        )

    # Remove rows where target is missing
    df = df.dropna(
        subset=[target]
    )

    # Only numeric columns
    numeric_columns = (
        df.select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    feature_columns = [
        column
        for column in numeric_columns
        if column != target
    ]

    # Check missing values
    missing = (
        df[feature_columns]
        .isna()
        .sum()
    )

    missing = missing[
        missing > 0
    ]

    if not missing.empty:

        print("\nMissing values:")

        for column, count in missing.items():
            print(
                f"  {column}: {count}"
            )

        # For this baseline, remove rows
        # with incomplete feature history.
        df = df.dropna(
            subset=feature_columns
        )

    X = df[feature_columns].copy()

    y = df[target].copy()

    print(
        f"\nX shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    return X, y, feature_columns


# ============================================================
# Chronological Train/Test Split
# ============================================================

def chronological_split(
    X,
    y,
    test_size=TEST_SIZE,
):
    """
    Split data chronologically.

    Earlier observations -> training
    Later observations   -> testing
    """

    split_index = int(
        len(X) * (1 - test_size)
    )

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    print("\n--- Chronological Split ---")

    print(
        f"Train: {X_train.shape}"
    )

    print(
        f"Test:  {X_test.shape}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# Feature Selection
# ============================================================

def feature_selection(
    X_train,
    y_train,
    feature_names,
    k=TOP_K_FEATURES,
):
    """
    Perform feature selection using training data only.

    Two methods are used:
    1. F-test
    2. Mutual Information

    Their top features are combined.
    """

    print("\n--- Feature Selection ---")

    k = min(
        k,
        X_train.shape[1]
    )

    # --------------------------------------------------------
    # F-test
    # --------------------------------------------------------

    selector_f = SelectKBest(
        score_func=f_regression,
        k=k,
    )

    selector_f.fit(
        X_train,
        y_train,
    )

    f_scores = selector_f.scores_

    f_indices = np.argsort(
        f_scores
    )[::-1][:k]

    f_features = [
        feature_names[i]
        for i in f_indices
    ]

    print(
        f"\nTop {k} F-test features:"
    )

    for feature in f_features:
        print(
            f"  {feature}"
        )

    # --------------------------------------------------------
    # Mutual Information
    # --------------------------------------------------------

    selector_mi = SelectKBest(
        score_func=mutual_info_regression,
        k=k,
    )

    selector_mi.fit(
        X_train,
        y_train,
    )

    mi_scores = selector_mi.scores_

    mi_indices = np.argsort(
        mi_scores
    )[::-1][:k]

    mi_features = [
        feature_names[i]
        for i in mi_indices
    ]

    print(
        f"\nTop {k} Mutual Information features:"
    )

    for feature in mi_features:
        print(
            f"  {feature}"
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    selected_features = list(
        dict.fromkeys(
            f_features + mi_features
        )
    )

    print(
        f"\nCombined features: "
        f"{len(selected_features)}"
    )

    for feature in selected_features:
        print(
            f"  {feature}"
        )

    return (
        selected_features,
        f_features,
        mi_features,
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    y_pred,
):
    """Calculate regression metrics."""

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    print(
        f"   RMSE: {rmse:.4f}"
    )

    print(
        f"   MAE:  {mae:.4f}"
    )

    print(
        f"   R²:   {r2:.4f}"
    )

    return {
        "model_name": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "y_pred": y_pred,
    }


# ============================================================
# Train Linear Regression
# ============================================================

def train_linear_regression(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Train Linear Regression using all features."""

    print(
        "\n1. Linear Regression (all features)..."
    )

    start_time = time.time()

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(X_train)
    )

    X_test_scaled = (
        scaler.transform(X_test)
    )

    model = LinearRegression()

    model.fit(
        X_train_scaled,
        y_train,
    )

    predictions = model.predict(
        X_test_scaled
    )

    result = evaluate_model(
        "LinearRegression_All",
        y_test,
        predictions,
    )

    result["model"] = model

    result["scaler"] = scaler

    result["features"] = list(
        X_train.columns
    )

    result["training_time"] = (
        time.time() - start_time
    )

    return result


# ============================================================
# Train Linear Regression - Selected
# ============================================================

def train_linear_regression_selected(
    X_train,
    X_test,
    y_train,
    y_test,
    selected_features,
):
    """Train Linear Regression using selected features."""

    print(
        "\n2. Linear Regression (selected features)..."
    )

    start_time = time.time()

    X_train_selected = (
        X_train[selected_features]
    )

    X_test_selected = (
        X_test[selected_features]
    )

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train_selected
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test_selected
        )
    )

    model = LinearRegression()

    model.fit(
        X_train_scaled,
        y_train,
    )

    predictions = model.predict(
        X_test_scaled
    )

    result = evaluate_model(
        "LinearRegression_Selected",
        y_test,
        predictions,
    )

    result["model"] = model

    result["scaler"] = scaler

    result["features"] = selected_features

    result["training_time"] = (
        time.time() - start_time
    )

    return result


# ============================================================
# Train Random Forest
# ============================================================

def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
):
    """Train Random Forest using all features."""

    print(
        "\n3. Random Forest (all features)..."
    )

    start_time = time.time()

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    result = evaluate_model(
        "RandomForest",
        y_test,
        predictions,
    )

    result["model"] = model

    result["features"] = feature_names

    result["importances"] = (
        model.feature_importances_
    )

    result["training_time"] = (
        time.time() - start_time
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = (
        model.feature_importances_
    )

    indices = np.argsort(
        importance
    )[::-1][:15]

    print(
        "\n   Top 15 feature importances:"
    )

    for index in indices:

        print(
            f"      "
            f"{feature_names[index]:30s} "
            f"{importance[index]:.4f}"
        )

    return result


# ============================================================
# Train Random Forest - Selected
# ============================================================

def train_random_forest_selected(
    X_train,
    X_test,
    y_train,
    y_test,
    selected_features,
):
    """Train Random Forest using selected features."""

    print(
        "\n4. Random Forest (selected features)..."
    )

    start_time = time.time()

    X_train_selected = (
        X_train[selected_features]
    )

    X_test_selected = (
        X_test[selected_features]
    )

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train_selected,
        y_train,
    )

    predictions = model.predict(
        X_test_selected
    )

    result = evaluate_model(
        "RandomForest_Selected",
        y_test,
        predictions,
    )

    result["model"] = model

    result["features"] = selected_features

    result["importances"] = (
        model.feature_importances_
    )

    result["training_time"] = (
        time.time() - start_time
    )

    return result


# ============================================================
# Plot Results
# ============================================================

def plot_results(
    results,
    y_test,
):
    """Save model prediction plots."""

    if not results:
        return

    number_of_models = len(results)

    fig, axes = plt.subplots(
        1,
        number_of_models,
        figsize=(
            6 * number_of_models,
            5,
        ),
    )

    if number_of_models == 1:
        axes = [axes]

    for ax, (name, result) in zip(
        axes,
        results.items(),
    ):

        predictions = result["y_pred"]

        ax.scatter(
            y_test,
            predictions,
            alpha=0.4,
            s=10,
        )

        minimum = min(
            y_test.min(),
            predictions.min(),
        )

        maximum = max(
            y_test.max(),
            predictions.max(),
        )

        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            "r--",
            linewidth=2,
        )

        ax.set_xlabel(
            "Actual"
        )

        ax.set_ylabel(
            "Predicted"
        )

        ax.set_title(
            f"{name}\n"
            f"R² = {result['r2']:.3f}"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

    plt.tight_layout()

    path = (
        RESULTS_DIR /
        "predictions.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nSaved prediction plot: {path}"
    )


# ============================================================
# Feature Importance Plot
# ============================================================

def plot_feature_importance(
    results,
):
    """Save Random Forest feature importance plot."""

    if "RandomForest" not in results:
        return

    result = results[
        "RandomForest"
    ]

    importance = result.get(
        "importances"
    )

    names = result.get(
        "features"
    )

    if importance is None:
        return

    if names is None:
        return

    top_n = min(
        20,
        len(names),
    )

    indices = np.argsort(
        importance
    )[::-1][:top_n]

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        range(top_n),
        importance[indices][::-1],
    )

    plt.yticks(
        range(top_n),
        [
            names[i]
            for i in indices
        ][::-1],
    )

    plt.xlabel(
        "Importance"
    )

    plt.title(
        "Top Random Forest Feature Importances"
    )

    plt.tight_layout()

    path = (
        RESULTS_DIR /
        "feature_importance.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved feature importance plot: {path}"
    )


# ============================================================
# Save Results
# ============================================================

def save_results(
    results,
    selected_features,
):
    """Save model results to a text file."""

    path = (
        RESULTS_DIR /
        "results_summary.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "DEMAND FORECASTING - "
            "BASELINE MODEL RESULTS\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        for name, result in results.items():

            file.write(
                f"{name}:\n"
            )

            file.write(
                f"  RMSE: "
                f"{result['rmse']:.4f}\n"
            )

            file.write(
                f"  MAE:  "
                f"{result['mae']:.4f}\n"
            )

            file.write(
                f"  R²:   "
                f"{result['r2']:.4f}\n"
            )

            if "training_time" in result:

                file.write(
                    f"  Time: "
                    f"{result['training_time']:.2f}s\n"
                )

            file.write("\n")

        file.write(
            "SELECTED FEATURES\n"
        )

        file.write(
            "-" * 40 + "\n"
        )

        for feature in selected_features:

            file.write(
                f"{feature}\n"
            )

    print(
        f"Results saved to: {path}"
    )


# ============================================================
# Main Pipeline
# ============================================================

def main():

    print("=" * 60)

    print(
        "DEMAND FORECASTING - "
        "BASELINE MODEL PIPELINE"
    )

    print("=" * 60)

    start_time = time.time()

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Feature Engineering
    # --------------------------------------------------------

    print(
        "\n--- Running Feature Engineering Module ---"
    )

    df_fe = run_feature_engineering(
        df,

        with_lag=True,

        with_rolling=True,

        # Disabled by default because naive
        # target aggregation causes leakage.
        with_category=False,

        with_interaction=True,

        remove_invalid_rows=False,
    )

    # --------------------------------------------------------
    # 3. Prepare X and y
    # --------------------------------------------------------

    X, y, feature_names = prepare_xy(
        df_fe,
        target=TARGET,
    )

    # --------------------------------------------------------
    # 4. Chronological Split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = chronological_split(
        X,
        y,
    )

    # --------------------------------------------------------
    # 5. Feature Selection
    # --------------------------------------------------------

    (
        selected_features,
        f_top,
        mi_top,
    ) = feature_selection(
        X_train,
        y_train,
        feature_names,
        k=TOP_K_FEATURES,
    )

    # --------------------------------------------------------
    # 6. Train Models
    # --------------------------------------------------------

    results = {}

    # Linear Regression - all
    results[
        "LinearRegression_All"
    ] = train_linear_regression(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Linear Regression - selected
    results[
        "LinearRegression_Selected"
    ] = train_linear_regression_selected(
        X_train,
        X_test,
        y_train,
        y_test,
        selected_features,
    )

    # Random Forest - all
    results[
        "RandomForest"
    ] = train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    )

    # Random Forest - selected
    results[
        "RandomForest_Selected"
    ] = train_random_forest_selected(
        X_train,
        X_test,
        y_train,
        y_test,
        selected_features,
    )

    # --------------------------------------------------------
    # 7. Save plots
    # --------------------------------------------------------

    plot_results(
        results,
        y_test,
    )

    plot_feature_importance(
        results,
    )

    # --------------------------------------------------------
    # 8. Save results
    # --------------------------------------------------------

    save_results(
        results,
        selected_features,
    )

    # --------------------------------------------------------
    # 9. Finish
    # --------------------------------------------------------

    total_time = (
        time.time() - start_time
    )

    print("\n" + "=" * 60)

    print(
        f"Total pipeline time: "
        f"{total_time:.2f}s"
    )

    print(
        "PIPELINE COMPLETED"
    )

    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()