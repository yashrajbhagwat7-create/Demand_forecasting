import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest, 
    f_regression, 
    mutual_info_regression,
    RFE
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "prepared_sales.csv"
RESULTS_DIR = PROJECT_ROOT / "model_results"
RESULTS_DIR.mkdir(exist_ok=True)

def load_and_feature_engineer():
    """Load data and apply feature engineering."""
    print("Loading data and applying feature engineering...")
    
    # Load from cached CSV (created by prepare_data.py / eda.py)
    from feature_engineering import run_feature_engineering
    
    DATA_PATH = PROJECT_ROOT / "data" / "prepared_sales.csv"
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape}")
    
    # Drop rows with NaN
    df = df.dropna(subset=['sale_amount'])
    
    # Apply feature engineering
    df_featured = run_feature_engineering(
        df,
        with_lag=True,
        with_rolling=True,
        with_category=True,
        with_interaction=True
    )
    
    # Drop rows with NaN (from lag features)
    print(f"Shape before dropping NaN: {df_featured.shape}")
    df_featured = df_featured.dropna().reset_index(drop=True)
    print(f"Shape after dropping NaN: {df_featured.shape}")
    
    return df_featured

def prepare_features_target(df, target_col='sale_amount'):
    """Prepare features and target for modeling."""
    print("\nPreparing features and target...")
    
    # Drop columns that are not suitable as features
    # Keep only numeric columns for simplicity in baseline
    exclude_cols = [
        'dt',  # Will be handled by calendar features
        'hours_sale',  # Array/string column
        'hours_stock_status',  # Array/string column
        target_col
    ]
    
    # Also drop columns that are mostly NaN due to lag features
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    
    # Drop columns with too many NaN (more than 50%)
    threshold = len(numeric_df) * 0.5
    numeric_df = numeric_df.dropna(thresh=threshold, axis=1)
    
    # Remove target from features if present
    feature_cols = [col for col in numeric_df.columns if col != target_col]
    X = numeric_df[feature_cols]
    y = numeric_df[target_col] if target_col in numeric_df.columns else None
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape if y is not None else 'None'}")
    print(f"Number of features: {len(feature_cols)}")
    
    return X, y, feature_cols

def feature_selection_univariate(X, y, k=20):
    """Select top k features using univariate statistical tests."""
    print(f"\nPerforming univariate feature selection (top {k} features)...")
    
    # F-test for regression
    selector_f = SelectKBest(score_func=f_regression, k=k)
    X_f_selected = selector_f.fit_transform(X, y)
    f_scores = selector_f.scores_
    f_features = X.columns[selector_f.get_support()]
    
    # Mutual information
    selector_mi = SelectKBest(score_func=mutual_info_regression, k=k)
    X_mi_selected = selector_mi.fit_transform(X, y)
    mi_scores = selector_mi.scores_
    mi_features = X.columns[selector_mi.get_support()]
    
    print(f"Top {k} features by F-test: {list(f_features)}")
    print(f"Top {k} features by Mutual Info: {list(mi_features)}")
    
    return X_f_selected, f_features, f_scores, X_mi_selected, mi_features, mi_scores

def feature_selection_rfe(X, y, estimator=None, n_features_to_select=15):
    """Select features using Recursive Feature Elimination."""
    print(f"\nPerforming RFE feature selection (top {n_features_to_select} features)...")
    
    if estimator is None:
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
    
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.support_]
    ranking = selector.ranking_
    
    print(f"Selected features: {list(selected_features)}")
    return X_selected, selected_features, ranking

def train_baseline_models(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate baseline models."""
    print("\n" + "=" * 60)
    print("BASELINE MODEL TRAINING")
    print("=" * 60)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_pred_test': y_pred_test
        }
        
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE:  {test_rmse:.4f}")
        print(f"  Train MAE:  {train_mae:.4f}")
        print(f"  Test MAE:   {test_mae:.4f}")
        print(f"  Train R²:   {train_r2:.4f}")
        print(f"  Test R²:    {test_r2:.4f}")
        
        # Cross-validation for Random Forest
        if name == 'Random Forest':
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            print(f"  CV RMSE:    {cv_rmse:.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results, scaler

def plot_results(results, y_test, feature_names=None):
    """Plot prediction results."""
    plt.figure(figsize=(15, 5))
    
    # Prediction vs Actual plots
    for i, (name, result) in enumerate(results.items()):
        plt.subplot(1, len(results), i+1)
        plt.scatter(y_test, result['y_pred_test'], alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Sale Amount')
        plt.ylabel('Predicted Sale Amount')
        plt.title(f'{name}\nR² = {result["test_r2"]:.3f}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'prediction_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Feature importance for tree-based models
    if 'Random Forest' in results:
        model = results['Random Forest']['model']
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20
            
            plt.title('Top 20 Feature Importances (Random Forest)')
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.ylabel('Feature Importance')
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()

def save_results(results, feature_names=None):
    """Save model results to file."""
    results_path = RESULTS_DIR / 'baseline_model_results.txt'
    
    with open(results_path, 'w') as f:
        f.write("BASELINE MODEL RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        for name, result in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Train RMSE: {result['train_rmse']:.4f}\n")
            f.write(f"  Test RMSE:  {result['test_rmse']:.4f}\n")
            f.write(f"  Train MAE:  {result['train_mae']:.4f}\n")
            f.write(f"  Test MAE:   {result['test_mae']:.4f}\n")
            f.write(f"  Train R²:   {result['train_r2']:.4f}\n")
            f.write(f"  Test R²:    {result['test_r2']:.4f}\n")
            f.write("\n")
    
    print(f"\nResults saved to: {results_path}")

def main():
    """Main pipeline for feature engineering, selection, and baseline modeling."""
    print("=" * 60)
    print("DEMAND FORECASTING - BASELINE MODEL PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data and apply feature engineering
    df = load_and_feature_engineer()
    
    # Step 2: Prepare features and target
    X, y, feature_names = prepare_features_target(df, target_col='sale_amount')
    
    if X is None or y is None:
        print("Error: Could not prepare features and target")
        return
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # Time series - no shuffle
    )
    
    print(f"\nData split: Train={X_train.shape}, Test={X_test.shape}")
    
    # Step 4: Feature selection (optional - we can try with all features first)
    print("\nTrying with all features first...")
    X_train_selected, X_test_selected = X_train, X_test
    selected_features = feature_names
    
    # Uncomment below to use feature selection
    # X_train_selected, selected_features_f, f_scores, X_test_selected_mi, selected_features_mi, mi_scores = \
    #     feature_selection_univariate(X_train, y_train, k=20)
    # X_test_selected = X_test[selected_features_f]
    # selected_features = selected_features_f
    
    # Step 5: Train baseline models
    results, scaler = train_baseline_models(
        X_train_selected, X_test_selected, y_train, y_test, selected_features
    )
    
    # Step 6: Plot and save results
    plot_results(results, y_test, selected_features)
    save_results(results, selected_features)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Results saved in: {RESULTS_DIR}")

if __name__ == "__main__":
    main()