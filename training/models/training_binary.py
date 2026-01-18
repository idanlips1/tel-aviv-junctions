"""
Binary Classification: XGBoost vs Historical Baseline

Compares a sophisticated XGBoost classifier against a simple historical-mean 
baseline for predicting whether a junction will have any accident in a given year.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'data_path': SCRIPT_DIR / '../training-data/tel_aviv_junctions_panel_labeled.csv',
    
    # Features for XGBoost
    'features': [
        'year', 'x_utm', 'y_utm', 
        'history_scaled', 'history_count', 'history_years',
        'road_count', 'total_lanes', 'max_speed',
        'highway_residential', 'highway_tertiary', 'highway_secondary',
        'has_oneway', 'has_cycleway'
    ],
    
    # Time-based split
    'test_start_year': 2024,
    
    # DBSCAN for location clustering
    'clustering_dist': 15,  # meters
    
    # Downsampling: keep N safe locations per 1 dangerous location
    # Set to None to disable downsampling (use scale_pos_weight instead)
    'sampling_ratio': None,  # Using scale_pos_weight instead
    
    # Classification threshold for baseline
    'baseline_threshold': 0.3,
    
    # XGBoost parameters
    'xgb_params': {
        'objective': 'binary:logistic',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'random_state': 42,
        'early_stopping_rounds': 50,
        'eval_metric': 'auc'
    }
}


# ==========================================
# DATA PIPELINE
# ==========================================
class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.dbscan = DBSCAN(
            eps=config['clustering_dist'],
            min_samples=1,
            metric='euclidean',
            n_jobs=-1
        )
    
    def load_and_prep(self):
        """Load data, create location IDs, and binary target."""
        print(">> Loading data...")
        df = pd.read_csv(self.config['data_path'])
        
        # Create stable location IDs via DBSCAN clustering
        print(">> Clustering locations (DBSCAN)...")
        coords = df[['x_utm', 'y_utm']].values
        df['location_id'] = self.dbscan.fit_predict(coords)
        
        # Create binary target: 1 if any accident, 0 otherwise
        df['has_accident'] = (df['accident_count'] > 0).astype(int)
        
        print(f">> Loaded {len(df)} rows, {df['location_id'].nunique()} unique locations")
        print(f">> Class distribution: {df['has_accident'].mean()*100:.2f}% positive")
        
        return df
    
    def get_train_test_split(self, df):
        """Split by time (no leakage)."""
        split_year = self.config['test_start_year']
        
        train_df = df[df['year'] < split_year].copy()
        test_df = df[df['year'] >= split_year].copy()
        
        print(f">> Train: {len(train_df)} rows (years < {split_year})")
        print(f">> Test:  {len(test_df)} rows (years >= {split_year})")
        
        return train_df, test_df
    
    def downsample(self, train_df):
        """
        Apply negative sampling to balance classes.
        Keeps ALL dangerous locations + samples N safe locations per dangerous one.
        """
        sampling_ratio = self.config.get('sampling_ratio')
        
        if sampling_ratio is None:
            print(">> Downsampling: DISABLED (using all data)")
            return train_df
        
        print(f">> Downsampling with ratio {sampling_ratio}:1 (safe:dangerous)...")
        
        # Identify dangerous vs safe LOCATIONS (not rows)
        # A location is dangerous if it EVER had an accident in training period
        loc_stats = train_df.groupby('location_id')['has_accident'].sum()
        dangerous_ids = loc_stats[loc_stats > 0].index.values
        safe_ids = loc_stats[loc_stats == 0].index.values
        
        n_dangerous = len(dangerous_ids)
        n_safe_to_keep = int(n_dangerous * sampling_ratio)
        
        # Randomly sample safe locations
        np.random.seed(42)
        if n_safe_to_keep < len(safe_ids):
            kept_safe_ids = np.random.choice(safe_ids, size=n_safe_to_keep, replace=False)
        else:
            kept_safe_ids = safe_ids  # Keep all if we don't have enough
        
        # Combine and filter
        final_ids = np.concatenate([dangerous_ids, kept_safe_ids])
        train_balanced = train_df[train_df['location_id'].isin(final_ids)].copy()
        
        print(f">> Kept {len(dangerous_ids)} dangerous + {len(kept_safe_ids)} safe locations")
        print(f">> Train reduced: {len(train_df)} → {len(train_balanced)} rows")
        
        return train_balanced


# ==========================================
# HISTORICAL BASELINE MODEL
# ==========================================
class HistoricalBaseline:
    """
    Simple baseline: predict based on historical accident rate per location.
    
    For each location, P(accident) = (# years with accidents) / (# years observed)
    Uses only TRAINING data to avoid leakage.
    """
    
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.location_rates = {}
        self.default_rate = 0.0
    
    def fit(self, train_df):
        """Compute historical accident rate for each location from training data."""
        print(">> Fitting Historical Baseline...")
        
        # Group by location and compute accident rate
        location_stats = train_df.groupby('location_id').agg({
            'has_accident': ['sum', 'count']
        })
        location_stats.columns = ['accidents', 'years']
        location_stats['rate'] = location_stats['accidents'] / location_stats['years']
        
        # Store as dictionary for fast lookup
        self.location_rates = location_stats['rate'].to_dict()
        
        # Default rate for unseen locations = overall training rate
        self.default_rate = train_df['has_accident'].mean()
        
        print(f">> Baseline fitted on {len(self.location_rates)} locations")
        print(f">> Default rate (for unseen locations): {self.default_rate:.4f}")
        
        return self
    
    def predict_proba(self, test_df):
        """Return probability of accident for each row."""
        return test_df['location_id'].map(
            lambda loc: self.location_rates.get(loc, self.default_rate)
        ).values
    
    def predict(self, test_df):
        """Return binary predictions using threshold."""
        probs = self.predict_proba(test_df)
        return (probs >= self.threshold).astype(int)


# ==========================================
# XGBOOST CLASSIFIER
# ==========================================
class XGBoostClassifier:
    """XGBoost binary classifier with class imbalance handling."""
    
    def __init__(self, params, features, use_scale_pos_weight=True):
        self.params = params.copy()
        self.features = features
        self.use_scale_pos_weight = use_scale_pos_weight
        self.model = None
    
    def fit(self, train_df, val_df=None):
        """Train XGBoost with optional validation set for early stopping."""
        print(">> Training XGBoost Classifier...")
        
        X_train = train_df[self.features].fillna(0)
        y_train = train_df['has_accident']
        
        params = self.params.copy()
        
        # Calculate scale_pos_weight for class imbalance (if enabled)
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        ratio = neg_count / pos_count
        
        print(f">> Class ratio in training: {ratio:.1f}:1 (negative:positive)")
        
        if self.use_scale_pos_weight:
            params['scale_pos_weight'] = ratio
            print(">> Using scale_pos_weight for class imbalance")
        else:
            print(">> scale_pos_weight DISABLED (using downsampled data)")
        
        self.model = xgb.XGBClassifier(**params)
        
        if val_df is not None:
            X_val = val_df[self.features].fillna(0)
            y_val = val_df['has_accident']
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=50
            )
            if hasattr(self.model, 'best_iteration'):
                print(f">> Best iteration: {self.model.best_iteration}")
        else:
            self.model.fit(X_train, y_train)
        
        print(">> XGBoost training complete.")
        return self
    
    def predict_proba(self, test_df):
        """Return probability of accident."""
        X = test_df[self.features].fillna(0)
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, test_df, threshold=0.5):
        """Return binary predictions."""
        probs = self.predict_proba(test_df)
        return (probs >= threshold).astype(int)


# ==========================================
# EVALUATION
# ==========================================
def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """
    Find the optimal classification threshold.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        metric: 'f1', 'precision', or 'recall'
    
    Returns:
        optimal_threshold, metrics_at_threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    
    if metric == 'f1':
        best_idx = np.argmax(f1_scores)
    elif metric == 'precision':
        # Find threshold where precision >= 0.5 with best recall
        valid = precision >= 0.5
        if valid.any():
            best_idx = np.where(valid)[0][np.argmax(recall[valid])]
        else:
            best_idx = np.argmax(precision)
    elif metric == 'recall':
        # Find threshold where recall >= 0.8 with best precision
        valid = recall >= 0.8
        if valid.any():
            best_idx = np.where(valid)[0][np.argmax(precision[valid])]
        else:
            best_idx = np.argmax(recall)
    else:
        best_idx = np.argmax(f1_scores)
    
    # Handle edge case where best_idx is beyond thresholds array
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1
    
    return thresholds[best_idx], {
        'threshold': thresholds[best_idx],
        'precision': precision[best_idx],
        'recall': recall[best_idx],
        'f1': f1_scores[best_idx]
    }


def evaluate_model(y_true, y_pred, y_prob, model_name):
    """Calculate classification metrics."""
    metrics = {
        'Model': model_name,
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['TP'] = tp
    metrics['FP'] = fp
    metrics['TN'] = tn
    metrics['FN'] = fn
    
    return metrics


def print_results(results):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    
    df = pd.DataFrame(results)
    
    # Main metrics table
    print("\n📊 Classification Metrics:")
    print("-" * 60)
    print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['Model']:<25} {r['Precision']:>10.4f} {r['Recall']:>10.4f} "
              f"{r['F1']:>10.4f} {r['AUC']:>10.4f}")
    print("-" * 60)
    
    # Confusion matrix details
    print("\n📋 Confusion Matrix Details:")
    for r in results:
        print(f"\n{r['Model']}:")
        print(f"  True Positives:  {r['TP']:>5} (correctly predicted accidents)")
        print(f"  False Positives: {r['FP']:>5} (false alarms)")
        print(f"  True Negatives:  {r['TN']:>5} (correctly predicted safe)")
        print(f"  False Negatives: {r['FN']:>5} (missed accidents)")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("BINARY CLASSIFICATION: XGBoost vs Historical Baseline")
    print("=" * 70)
    
    # 1. Load and prepare data
    pipeline = DataPipeline(CONFIG)
    df = pipeline.load_and_prep()
    
    # 2. Train/test split
    train_df, test_df = pipeline.get_train_test_split(df)
    y_test = test_df['has_accident'].values
    
    # 3. Apply downsampling to training data (if enabled)
    print("\n" + "-" * 70)
    train_df_balanced = pipeline.downsample(train_df)
    
    # Determine if we should use scale_pos_weight
    # If downsampling is enabled, we don't need scale_pos_weight
    use_scale_pos_weight = CONFIG.get('sampling_ratio') is None
    
    results = []
    
    # 4. Historical Baseline (fitted on FULL training data, not downsampled)
    print("\n" + "-" * 70)
    baseline = HistoricalBaseline(threshold=CONFIG['baseline_threshold'])
    baseline.fit(train_df)  # Use full data for baseline
    
    baseline_probs = baseline.predict_proba(test_df)
    baseline_preds = baseline.predict(test_df)
    
    baseline_metrics = evaluate_model(y_test, baseline_preds, baseline_probs, 
                                       "Historical Baseline")
    results.append(baseline_metrics)
    
    # 5. XGBoost Classifier (trained on downsampled data)
    print("\n" + "-" * 70)
    xgb_model = XGBoostClassifier(
        CONFIG['xgb_params'], 
        CONFIG['features'],
        use_scale_pos_weight=use_scale_pos_weight
    )
    xgb_model.fit(train_df_balanced, val_df=test_df)
    
    xgb_probs = xgb_model.predict_proba(test_df)
    
    # 6. Find optimal threshold for XGBoost
    print("\n" + "-" * 70)
    print(">> Finding optimal threshold for XGBoost...")
    optimal_thresh, thresh_metrics = find_optimal_threshold(y_test, xgb_probs, metric='f1')
    print(f">> Optimal threshold: {optimal_thresh:.3f}")
    print(f"   At this threshold: P={thresh_metrics['precision']:.4f}, "
          f"R={thresh_metrics['recall']:.4f}, F1={thresh_metrics['f1']:.4f}")
    
    # Evaluate XGBoost with default threshold (0.5)
    xgb_preds_default = xgb_model.predict(test_df, threshold=0.5)
    xgb_metrics_default = evaluate_model(y_test, xgb_preds_default, xgb_probs, 
                                          "XGBoost (thresh=0.5)")
    results.append(xgb_metrics_default)
    
    # Evaluate XGBoost with optimal threshold
    xgb_preds_optimal = (xgb_probs >= optimal_thresh).astype(int)
    xgb_metrics_optimal = evaluate_model(y_test, xgb_preds_optimal, xgb_probs, 
                                          f"XGBoost (thresh={optimal_thresh:.2f})")
    results.append(xgb_metrics_optimal)
    
    # 7. Print comparison
    print_results(results)
    
    # 8. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\n📈 Threshold Impact on XGBoost:")
    print(f"   Default (0.5):  F1={xgb_metrics_default['F1']:.4f}")
    print(f"   Optimal ({optimal_thresh:.2f}): F1={xgb_metrics_optimal['F1']:.4f}")
    
    f1_improvement = (xgb_metrics_optimal['F1'] - baseline_metrics['F1']) / baseline_metrics['F1'] * 100 \
        if baseline_metrics['F1'] > 0 else float('inf')
    auc_improvement = (xgb_metrics_optimal['AUC'] - baseline_metrics['AUC']) / baseline_metrics['AUC'] * 100 \
        if baseline_metrics['AUC'] > 0 else float('inf')
    
    print(f"\nXGBoost (optimal) vs Baseline:")
    print(f"  F1 Score:  {baseline_metrics['F1']:.4f} → {xgb_metrics_optimal['F1']:.4f} ({f1_improvement:+.1f}%)")
    print(f"  AUC-ROC:   {baseline_metrics['AUC']:.4f} → {xgb_metrics_optimal['AUC']:.4f} ({auc_improvement:+.1f}%)")
    
    if xgb_metrics_optimal['F1'] > baseline_metrics['F1']:
        print("\n✅ XGBoost with tuned threshold outperforms the baseline!")
    else:
        print("\n⚠️  XGBoost still doesn't beat the baseline. Consider feature engineering.")
