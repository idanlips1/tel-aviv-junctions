"""
Experiment script to find optimal model configuration.
Tests different sampling ratios and objectives.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_PATH = SCRIPT_DIR / '../training-data/tel_aviv_junctions_panel_labeled.csv'

# ==========================================
# EXPERIMENTS TO RUN
# ==========================================
EXPERIMENTS = [
    # (name, sampling_ratio, objective, extra_params)
    ("Baseline (Poisson, 2:1)", 2, "count:poisson", {}),
    ("Poisson 1:1", 1, "count:poisson", {}),
    ("Poisson 0.5:1", 0.5, "count:poisson", {}),
    ("Tweedie 2:1", 2, "reg:tweedie", {"tweedie_variance_power": 1.5}),
    ("Tweedie 1:1", 1, "reg:tweedie", {"tweedie_variance_power": 1.5}),
    ("Squared Error 1:1", 1, "reg:squarederror", {}),
    ("Gamma 1:1", 1, "reg:gamma", {}),  # For positive targets only
]

FEATURES = [
    'year', 'x_utm', 'y_utm', 
    'history_scaled', 'history_count', 'history_years',
    'road_count', 'total_lanes', 'max_speed',
    'highway_residential', 'highway_tertiary', 'highway_secondary',
    'has_oneway', 'has_cycleway'
]

# ==========================================
# DATA PREP
# ==========================================
def load_and_balance(df, sampling_ratio):
    """Apply negative sampling with given ratio."""
    # Cluster locations
    dbscan = DBSCAN(eps=15, min_samples=1, metric='euclidean', n_jobs=-1)
    coords = df[['x_utm', 'y_utm']].values
    df = df.copy()
    df['location_id'] = dbscan.fit_predict(coords)
    
    # Identify dangerous vs safe
    loc_stats = df.groupby('location_id')['accident_count'].sum()
    dangerous_ids = loc_stats[loc_stats > 0].index.values
    safe_ids = loc_stats[loc_stats == 0].index.values
    
    # Sample safe locations
    n_dangerous = len(dangerous_ids)
    n_safe_to_keep = int(n_dangerous * sampling_ratio)
    
    np.random.seed(42)
    if n_safe_to_keep < len(safe_ids):
        kept_safe_ids = np.random.choice(safe_ids, size=n_safe_to_keep, replace=False)
    else:
        kept_safe_ids = safe_ids
        
    final_ids = np.concatenate([dangerous_ids, kept_safe_ids])
    return df[df['location_id'].isin(final_ids)].copy()

def train_test_split(df, test_year=2024):
    """Split by time."""
    train = df[df['year'] < test_year]
    test = df[df['year'] >= test_year]
    
    X_train = train[FEATURES].fillna(0)
    y_train = train['accident_count']
    X_test = test[FEATURES].fillna(0)
    y_test = test['accident_count']
    
    return X_train, y_train, X_test, y_test

# ==========================================
# EVALUATION METRICS
# ==========================================
def evaluate(y_true, y_pred):
    """Calculate multiple metrics."""
    y_pred = np.maximum(y_pred, 0)  # No negative predictions
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Classification metrics (dangerous = any predicted accident)
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0.3).astype(int)  # Threshold for "dangerous"
    
    # Precision: Of predicted dangerous, how many actually had accidents?
    tp = ((y_pred_binary == 1) & (y_true_binary == 1)).sum()
    fp = ((y_pred_binary == 1) & (y_true_binary == 0)).sum()
    fn = ((y_pred_binary == 0) & (y_true_binary == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4)
    }

# ==========================================
# RUN EXPERIMENTS
# ==========================================
def run_experiments():
    print("=" * 70)
    print("ACCIDENT PREDICTION MODEL EXPERIMENTS")
    print("=" * 70)
    
    # Load raw data once
    print("\n>> Loading data...")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df_raw)} rows")
    
    results = []
    
    for name, ratio, objective, extra_params in EXPERIMENTS:
        print(f"\n{'─' * 70}")
        print(f"🧪 Experiment: {name}")
        print(f"   Sampling Ratio: {ratio}, Objective: {objective}")
        
        try:
            # Prepare data
            df = load_and_balance(df_raw, ratio)
            X_train, y_train, X_test, y_test = train_test_split(df)
            
            # For Gamma, need to avoid zeros in target
            if objective == "reg:gamma":
                # Add small epsilon to avoid log(0)
                y_train = y_train + 0.01
                y_test_orig = y_test.copy()
                y_test = y_test + 0.01
            else:
                y_test_orig = y_test
            
            # Train model
            params = {
                'objective': objective,
                'n_estimators': 300,  # Fewer for speed
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_jobs': -1,
                'random_state': 42,
                'verbosity': 0,
                **extra_params
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            
            # Predict & evaluate
            preds = model.predict(X_test)
            if objective == "reg:gamma":
                preds = preds - 0.01  # Remove epsilon
                y_test = y_test_orig
            
            metrics = evaluate(y_test, preds)
            metrics['Experiment'] = name
            results.append(metrics)
            
            print(f"MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | F1: {metrics['F1']:.4f}")
            
        except Exception as e:
            print(f"Failed: {e}")
            results.append({'Experiment': name, 'MAE': None, 'Error': str(e)})
    
    # Summary table
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    results_df = results_df[['Experiment', 'MAE', 'RMSE', 'Precision', 'Recall', 'F1']]
    print(results_df.to_string(index=False))
    
    # Best model
    best = results_df.loc[results_df['MAE'].idxmin()]
    print(f"\n🏆 Best by MAE: {best['Experiment']} (MAE={best['MAE']})")
    
    best_f1 = results_df.loc[results_df['F1'].idxmax()]
    print(f"🏆 Best by F1:  {best_f1['Experiment']} (F1={best_f1['F1']})")
    
    return results_df

if __name__ == "__main__":
    run_experiments()

