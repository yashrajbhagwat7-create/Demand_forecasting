"""
Baseline Model Pipeline: Feature Engineering + Feature Selection + Training
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

warnings.filterwarnings('ignore')
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "model_results"
RESULTS_DIR.mkdir(exist_ok=True)

def load_data():
    """Load prepared sales data."""
    print("Loading data...")
    t0 = time.time()
    df = pd.read_csv(PROJECT_ROOT / "data" / "prepared_sales.csv")
    print(f"Loaded {len(df)} rows in {time.time()-t0:.1f}s")
    print(f"Columns: {df.columns.tolist()}")
    return df

def engineer_features(df):
    """
    Feature engineering: calendar, lag, rolling, category, interaction features.
    """
    print("\n--- Feature Engineering ---")
    t0 = time.time()
    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"])

    # --- Calendar features ---
    df["day_of_week"] = df["dt"].dt.dayofweek
    df["day_of_month"] = df["dt"].dt.day
    df["month"] = df["dt"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    print(f"  Calendar features added ({time.time()-t0:.1f}s)")

    # --- Lag features ---
    t1 = time.time()
    df = df.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby(["store_id", "product_id"])["sale_amount"].shift(lag)
    print(f"  Lag features added ({time.time()-t1:.1f}s)")

    # --- Rolling mean features ---
    t2 = time.time()
    for win in [7, 14, 28]:
        df[f"rolling_mean_{win}"] = (
            df.groupby(["store_id", "product_id"])["sale_amount"]
            .shift(1)
            .transform(lambda x: x.rolling(win, min_periods=1).mean())
        )
        df[f"rolling_std_{win}"] = (
            df.groupby(["store_id", "product_id"])["sale_amount"]
            .shift(1)
            .transform(lambda x: x.rolling(win, min_periods=1).std())
        )
    print(f"  Rolling features added ({time.time()-t2:.1f}s)")

    # --- Category aggregation features ---
    t3 = time.time()
    for col in ["store_id", "product_id", "first_category_id", "second_category_id"]:
        grp = df.groupby(col)["sale_amount"].agg(["mean", "std"]).reset_index()
        grp.columns = [col, f"{col}_mean", f"{col}_std"]
        df = df.merge(grp, on=col, how="left")
    print(f"  Category features added ({time.time()-t3:.1f}s)")

    # --- Interaction features ---
    df["discount_x_holiday"] = df["discount"] * df["holiday_flag"]
    df["temp_x_humidity"] = df["avg_temperature"] * df["avg_humidity"]
    print("  Interaction features added")

    print(f"Feature engineering completed in {time.time()-t0:.1f}s")
    print(f"Final shape: {df.shape}")
    return df

def prepare_xy(df, target='sale_amount'):
    """Select numeric features and target, drop NaN rows."""
    df = df.dropna(subset=[target]).copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target from features
    feature_cols = [c for c in numeric_cols if c != target]
    
    # Check for NaN in features and report
    nan_info = df[feature_cols].isnull().sum()
    total_nan = nan_info.sum()
    print(f"NaN values in features: {total_nan}")
    if total_nan > 0:
        print(f"NaN by column: {nan_info[nan_info > 0].to_dict()}")
    
    # Drop rows with any NaN in features
    df_clean = df.dropna(subset=feature_cols)
    print(f"Rows before NaN drop: {len(df)}, after: {len(df_clean)}")
    
    X = df_clean[feature_cols].values
    y = df_clean[target].values
    print(f"Prepared X shape: {X.shape}, y shape: {y.shape}")
    return X, y, feature_cols

def feature_selection(X, y, feature_names, k=20):
    """Feature selection using univariate F-test and mutual info."""
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    
    print("\n--- Feature Selection ---")
    
    # Use all features for selection
    sel_f = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    sel_f.fit(X, y)
    f_scores = sel_f.scores_
    f_top = np.argsort(f_scores)[::-1][:k]
    f_features = [feature_names[i] for i in f_top]
    print(f"Top {k} by F-test: {f_features}")
    
    # Mutual info
    sel_mi = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
    sel_mi.fit(X, y)
    mi_scores = sel_mi.scores_
    mi_top = np.argsort(mi_scores)[::-1][:k]
    mi_features = [feature_names[i] for i in mi_top]
    print(f"Top {k} by Mutual Info: {mi_features}")
    
    # Union of top features from both methods
    all_selected = list(set(f_features + mi_features))
    print(f"Combined unique features ({len(all_selected)}): {all_selected}")
    
    return all_selected, f_features, mi_features

def train_models(X, y, X_selected=None, feature_names=None):
    """Train baseline models."""
    print("\n--- Model Training ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    scaler = StandardScaler()
    
    results = {}
    
    # 1. Linear Regression with all features
    print("\n1. Linear Regression (all features)...")
    t0 = time.time()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    lr = LinearRegression()
    lr.fit(X_tr_s, y_train)
    y_pred_lr_tr = lr.predict(X_tr_s)
    y_pred_lr_te = lr.predict(X_te_s)
    lr_time = time.time() - t0
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr_te))
    lr_mae = mean_absolute_error(y_test, y_pred_lr_te)
    lr_r2 = r2_score(y_test, y_pred_lr_te)
    print(f"   Time: {lr_time:.1f}s, RMSE: {lr_rmse:.4f}, MAE: {lr_mae:.4f}, R²: {lr_r2:.4f}")
    results['LinearRegression_All'] = {
        'model': lr, 'scaler': scaler, 'rmse': lr_rmse, 'mae': lr_mae, 'r2': lr_r2,
        'y_pred': y_pred_lr_te
    }
    
    # 2. Linear Regression with selected features
    if X_selected is not None and len(X_selected) > 0:
        print("\n2. Linear Regression (selected features)...")
        t0 = time.time()
        idx = [feature_names.index(f) for f in X_selected if f in feature_names]
        if idx:
            X_sel = X[:, idx]
            X_tr_sel, X_te_sel, y_tr_sel, y_te_sel = train_test_split(X_sel, y, test_size=0.2, random_state=42, shuffle=False)
            scaler_sel = StandardScaler()
            X_tr_sel_s = scaler_sel.fit_transform(X_tr_sel)
            X_te_sel_s = scaler_sel.transform(X_te_sel)
            lr_sel = LinearRegression()
            lr_sel.fit(X_tr_sel_s, y_tr_sel)
            y_pred_sel_te = lr_sel.predict(X_te_sel_s)
            lr_sel_time = time.time() - t0
            lr_sel_rmse = np.sqrt(mean_squared_error(y_te_sel, y_pred_sel_te))
            lr_sel_mae = mean_absolute_error(y_te_sel, y_pred_sel_te)
            lr_sel_r2 = r2_score(y_te_sel, y_pred_sel_te)
            print(f"   Time: {lr_sel_time:.1f}s, RMSE: {lr_sel_rmse:.4f}, MAE: {lr_sel_mae:.4f}, R²: {lr_sel_r2:.4f}")
            results['LinearRegression_Selected'] = {
                'model': lr_sel, 'scaler': scaler_sel, 'rmse': lr_sel_rmse, 'mae': lr_sel_mae, 'r2': lr_sel_r2,
                'y_pred': y_pred_sel_te, 'features': X_selected
            }
    
    # 3. Random Forest
    print("\n3. Random Forest...")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf_te = rf.predict(X_test)
    rf_time = time.time() - t0
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_te))
    rf_mae = mean_absolute_error(y_test, y_pred_rf_te)
    rf_r2 = r2_score(y_test, y_pred_rf_te)
    print(f"   Time: {rf_time:.1f}s, RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")
    
    # Feature importance
    if feature_names:
        imp = rf.feature_importances_
        top_idx = np.argsort(imp)[::-1][:15]
        top_feats = [(feature_names[i], imp[i]) for i in top_idx]
        print("   Top 15 feature importances:")
        for name, score in top_feats:
            print(f"      {name:30s} {score:.4f}")
    results['RandomForest'] = {
        'model': rf, 'rmse': rf_rmse, 'mae': rf_mae, 'r2': rf_r2,
        'y_pred': y_pred_rf_te, 'importances': rf.feature_importances_,
        'feature_names': feature_names
    }
    
    # 4. Random Forest with selected features
    if X_selected is not None and len(X_selected) > 0:
        print("\n4. Random Forest (selected features)...")
        t0 = time.time()
        idx = [feature_names.index(f) for f in X_selected if f in feature_names]
        if idx:
            X_sel = X[:, idx]
            X_tr_sel, X_te_sel, y_tr_sel, y_te_sel = train_test_split(
                X_sel, y, test_size=0.2, random_state=42, shuffle=False
            )
            rf_sel = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_sel.fit(X_tr_sel, y_tr_sel)
            y_pred_rf_sel = rf_sel.predict(X_te_sel)
            rf_sel_time = time.time() - t0
            rf_sel_rmse = np.sqrt(mean_squared_error(y_te_sel, y_pred_rf_sel))
            rf_sel_mae = mean_absolute_error(y_te_sel, y_pred_rf_sel)
            rf_sel_r2 = r2_score(y_te_sel, y_pred_rf_sel)
            print(f"   Time: {rf_sel_time:.1f}s, RMSE: {rf_sel_rmse:.4f}, MAE: {rf_sel_mae:.4f}, R²: {rf_sel_r2:.4f}")
            results['RandomForest_Selected'] = {
                'model': rf_sel, 'rmse': rf_sel_rmse, 'mae': rf_sel_mae, 'r2': rf_sel_r2,
                'y_pred': y_pred_rf_sel
            }
    
    return results, y_test, scaler

def plot_results(results, y_test, feature_names=None):
    """Save prediction plots and feature importance."""
    if not results:
        print("No results to plot.")
        return
    
    plt.figure(figsize=(15, 5))
    for i, (name, res) in enumerate(results.items()):
        plt.subplot(1, len(results), i+1)
        plt.scatter(y_test, res['y_pred'], alpha=0.4, s=10)
        lims = [min(y_test.min(), res['y_pred'].min()), max(y_test.max(), res['y_pred'].max())]
        plt.plot(lims, lims, 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{name}\nR²={res["r2"]:.3f}')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved predictions plot to {RESULTS_DIR / 'predictions.png'}")
    
    # Feature importance
    if 'RandomForest' in results and results['RandomForest'].get('importances') is not None:
        imp = results['RandomForest']['importances']
        names = results['RandomForest'].get('feature_names', [])
        if len(imp) == len(names) and len(names) > 0:
            top_idx = np.argsort(imp)[::-1][:20]
            
            plt.figure(figsize=(10, 6))
            plt.title('Top 20 Feature Importances (Random Forest)')
            plt.barh(range(len(top_idx)), imp[top_idx][::-1], align='center')
            plt.yticks(range(len(top_idx)), [names[i] for i in top_idx][::-1])
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved feature importance plot to {RESULTS_DIR / 'feature_importance.png'}")

def save_results(results, selected_features=None):
    """Write results summary to file."""
    path = RESULTS_DIR / 'results_summary.txt'
    with open(path, 'w') as f:
        f.write("DEMAND FORECASTING - BASELINE MODEL RESULTS\n")
        f.write("=" * 50 + "\n\n")
        for name, res in results.items():
            f.write(f"{name}:\n")
            f.write(f"  RMSE: {res['rmse']:.4f}\n")
            f.write(f"  MAE:  {res['mae']:.4f}\n")
            f.write(f"  R²:   {res['r2']:.4f}\n")
            f.write("\n")
        if selected_features:
            f.write("Selected features:\n")
            for feat in selected_features:
                f.write(f"  {feat}\n")
    print(f"Results saved to {path}")

def main():
    print("=" * 60)
    print("DEMAND FORECASTING - BASELINE MODEL PIPELINE")
    print("=" * 60)
    t_start = time.time()
    
    # 1. Load
    df = load_data()
    
    # 2. Feature Engineering
    df_fe = engineer_features(df)
    
    # 3. Prepare X, y
    X, y, feature_names = prepare_xy(df_fe, target='sale_amount')
    
    # 4. Feature Selection
    all_selected, f_top, mi_top = feature_selection(X, y, feature_names, k=20)
    
    # 5. Train
    results, y_test, scaler = train_models(X, y, all_selected, feature_names)
    
    # 6. Plot & save
    plot_results(results, y_test, feature_names)
    save_results(results, all_selected)
    
    print(f"\nTotal pipeline time: {time.time()-t_start:.1f}s")
    print("=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()