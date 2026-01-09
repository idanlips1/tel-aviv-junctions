import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# ==========================================
# 1. CONFIGURATION (Edit this to change behavior)
# ==========================================
CONFIG = {
    # File settings
    'data_path': SCRIPT_DIR / '../training-data/tel_aviv_junctions_panel_labeled.csv',
    'model_save_path': SCRIPT_DIR / 'accident_model.pkl',

    # Features to use for training
    'features': [
        'year', 'x_utm', 'y_utm', 
        'history_scaled', 'history_count', 'history_years',
        'road_count', 'total_lanes', 'max_speed',
        'highway_residential', 'highway_tertiary', 'highway_secondary',
        'has_oneway', 'has_cycleway'
    ],
    'target': 'accident_count',

    # Splitting
    'test_start_year': 2024,  # Train on 2015-2023, Test on 2024+

    # Sampling Strategy
    'sampling_ratio': 2,      # Keep 2 Safe locations for every 1 Dangerous location
    'clustering_dist': 15,    # Meters for grouping junctions (DBSCAN)

    # Model Hyperparameters (Easy to tune)
    'xgb_params': {
        'objective': 'reg:tweedie',      # Better than Poisson for zero-inflated data
        'tweedie_variance_power': 1.5,   # 1=Poisson, 2=Gamma, 1.5=compound
        'n_estimators': 2000,            # High number, early stopping will find optimal
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'random_state': 42,
        'early_stopping_rounds': 100,    # Stop if no improvement for 100 rounds
        'eval_metric': 'mae'             # Use MAE for early stopping, not Tweedie loss
    }
}

# ==========================================
# 2. DATA PIPELINE (Cleaning & Sampling)
# ==========================================
class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.location_dbscan = DBSCAN(
            eps=config['clustering_dist'], 
            min_samples=1, 
            metric='euclidean', 
            n_jobs=-1
        )

    def load_and_prep(self):
        """Loads data, creates location_ids, and performs negative sampling."""
        print(">> Loading data...")
        df = pd.read_csv(self.config['data_path'])

        # 1. Create Location IDs (Spatial Clustering)
        # This fixes the issue of inconsistent IDs across years
        print(">> Clustering locations (DBSCAN)...")
        coords = df[['x_utm', 'y_utm']].values
        df['location_id'] = self.location_dbscan.fit_predict(coords)

        # 2. Identify Dangerous vs Safe Locations (History)
        loc_stats = df.groupby('location_id')[self.config['target']].sum()
        dangerous_ids = loc_stats[loc_stats > 0].index.values
        safe_ids = loc_stats[loc_stats == 0].index.values

        # 3. Apply Negative Sampling
        # Keep ALL dangerous locations, Sample specific SAFE locations
        n_dangerous = len(dangerous_ids)
        n_safe_to_keep = int(n_dangerous * self.config['sampling_ratio'])
        
        # Randomly choose safe IDs to keep
        np.random.seed(42)
        if n_safe_to_keep < len(safe_ids):
            kept_safe_ids = np.random.choice(safe_ids, size=n_safe_to_keep, replace=False)
        else:
            kept_safe_ids = safe_ids # Keep all if we don't have enough
            
        final_ids = np.concatenate([dangerous_ids, kept_safe_ids])
        
        # Filter the DataFrame
        df_balanced = df[df['location_id'].isin(final_ids)].copy()
        print(f">> Data Balanced: Kept {len(final_ids)} unique locations "
              f"({len(dangerous_ids)} dangerous, {len(kept_safe_ids)} safe).")
        
        return df_balanced

    def get_train_test_split(self, df):
        """Splits data by TIME, not randomly."""
        split_year = self.config['test_start_year']
        
        train_df = df[df['year'] < split_year]
        test_df = df[df['year'] >= split_year]

        X_train = train_df[self.config['features']]
        y_train = train_df[self.config['target']]
        
        X_test = test_df[self.config['features']]
        y_test = test_df[self.config['target']]

        return X_train, y_train, X_test, y_test

# ==========================================
# 3. MODEL ENGINE (Training & Inference)
# ==========================================
class AccidentModel:
    def __init__(self, model_type='xgboost', params=None):
        self.model_type = model_type
        
        if model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**params)
        elif model_type == 'random_forest':
            # Example of swapping models easily
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Unknown model type")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        print(f">> Training {self.model_type}...")
        
        if self.model_type == 'xgboost':
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=50  # Print progress every 50 rounds
            )
            # Report best iteration if early stopping was used
            if hasattr(self.model, 'best_iteration'):
                print(f">> Best iteration: {self.model.best_iteration}")
        else:
            self.model.fit(X_train, y_train)
            
        print(">> Training Complete.")

    def predict(self, X):
        preds = self.model.predict(X)
        # Ensure we don't predict negative accidents
        return np.maximum(preds, 0)
    
    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {'MAE': mae, 'RMSE': rmse}

    def save(self, path):
        joblib.dump(self.model, path)
        print(f">> Model saved to {path}")

# ==========================================
# 4. MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # A. Data Prep
    pipeline = DataPipeline(CONFIG)
    df = pipeline.load_and_prep()
    
    # B. Split
    X_train, y_train, X_test, y_test = pipeline.get_train_test_split(df)
    
    # C. Train
    # Initialize model (Change 'xgboost' to 'random_forest' here to swap!)
    engine = AccidentModel(model_type='xgboost', params=CONFIG['xgb_params'])
    engine.train(X_train, y_train, X_test, y_test)
    
    # D. Evaluate
    preds = engine.predict(X_test)
    metrics = engine.evaluate(y_test, preds)
    print(f"\n>> Test Results (2024-2025): {metrics}")

    # E. Save the model
    engine.save(CONFIG['model_save_path'])

    # F. Inference Example (Predicting for Next Year)
    # Let's say we want to predict risk for a specific junction in 2026.
    # We take its most recent known features (e.g., from 2025) and update the 'year'.
    
    print("\n>> Running Inference for a sample junction...")
    sample_junction = X_test.iloc[0:1].copy() # Take one real junction
    sample_junction['year'] = 2026            # Update year to future
    
    risk_prediction = engine.predict(sample_junction)[0]
    print(f"Predicted Accidents for Junction at {sample_junction['x_utm'].values[0]:.1f}, "
          f"{sample_junction['y_utm'].values[0]:.1f} in 2026: {risk_prediction:.4f}")