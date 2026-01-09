import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Suppress pandas warnings
pd.set_option('future.no_silent_downcasting', True)

SCRIPT_DIR = Path(__file__).parent.resolve()

# Load model and data
model = joblib.load(SCRIPT_DIR / 'accident_model.pkl')
df = pd.read_csv(SCRIPT_DIR / '../training-data/tel_aviv_junctions_panel_labeled.csv')

# Get features directly from the trained model
FEATURES = model.get_booster().feature_names

def find_nearest_junction(x, y, year=2024):
    """Find the nearest junction to given coordinates."""
    year_data = df[df['year'] == year].copy()
    
    # Calculate distance to all junctions
    year_data['distance'] = np.sqrt(
        (year_data['x_utm'] - x)**2 + (year_data['y_utm'] - y)**2
    )
    
    # Get nearest
    nearest = year_data.nsmallest(1, 'distance').iloc[0]
    return nearest

def predict_for_coordinates(x, y, year=2025):
    """Predict accident risk for given coordinates."""
    
    # Find nearest junction to get its features
    nearest = find_nearest_junction(x, y, year=min(year, 2024))
    distance = nearest['distance']
    
    # Prepare features as a dict, then create DataFrame with proper types
    feature_dict = {feat: [nearest[feat]] for feat in FEATURES}
    features = pd.DataFrame(feature_dict)
    features['year'] = year
    features['x_utm'] = x
    features['y_utm'] = y
    
    # Fill any missing values and ensure numeric types
    features = features.fillna(0)
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
    
    # Predict
    prediction = model.predict(features)[0]
    prediction = max(0, prediction)
    
    return {
        'x': x,
        'y': y,
        'year': year,
        'predicted_accidents': round(prediction, 4),
        'risk_level': 'HIGH' if prediction > 0.5 else 'MEDIUM' if prediction > 0.2 else 'LOW',
        'nearest_junction_distance_m': round(distance, 1),
        'nearest_junction_features': {
            'road_count': int(nearest['road_count']),
            'max_speed': nearest['max_speed'],
            'total_lanes': nearest['total_lanes'],
            'has_cycleway': bool(nearest['has_cycleway']),
            'history_count': int(nearest['history_count'])
        }
    }

def show_high_risk_junctions(year=2024, top_n=10):
    """Show junctions with highest predicted risk."""
    year_data = df[df['year'] == year].copy()
    
    # Predict for all junctions - ensure numeric types
    X = year_data[FEATURES].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
    year_data['predicted'] = model.predict(X)
    year_data['predicted'] = year_data['predicted'].clip(lower=0)
    
    # Sort by risk
    high_risk = year_data.nlargest(top_n, 'predicted')[
        ['x_utm', 'y_utm', 'predicted', 'accident_count', 'road_count', 'max_speed']
    ]
    
    return high_risk

# ==========================================
# INTERACTIVE TESTING
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("ACCIDENT RISK PREDICTION MODEL TESTER")
    print("=" * 50)
    
    # Example coordinates (central Tel Aviv)
    test_coords = [
        (667000, 3550000, "Central Tel Aviv"),
        (668000, 3548000, "South Tel Aviv"),
        (666900, 3551000, "North area"),
    ]
    
    print("\n>> Testing sample coordinates:\n")
    for x, y, name in test_coords:
        result = predict_for_coordinates(x, y, year=2025)
        print(f"📍 {name} ({x}, {y})")
        print(f"   Risk: {result['risk_level']} ({result['predicted_accidents']} predicted accidents)")
        print(f"   Nearest junction: {result['nearest_junction_distance_m']}m away")
        print(f"   Features: {result['nearest_junction_features']}")
        print()
    
    print("\n>> Top 10 highest risk junctions (2024):\n")
    high_risk = show_high_risk_junctions(year=2024, top_n=10)
    print(high_risk.to_string(index=False))
    
    # Interactive mode (only if running with --interactive flag)
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        print("\n" + "=" * 50)
        print("INTERACTIVE MODE - Enter coordinates to test")
        print("Coordinates are in UTM (Israel TM Grid)")
        print("Tel Aviv range: X ~666000-670000, Y ~3546000-3553000")
        print("Type 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nEnter X,Y coordinates (e.g., 667000,3550000): ")
                if user_input.lower() == 'quit':
                    break
                
                x, y = map(float, user_input.replace(' ', '').split(','))
                result = predict_for_coordinates(x, y, year=2025)
                
                print(f"\n🎯 Prediction for ({x:.0f}, {y:.0f}):")
                print(f"   Predicted accidents: {result['predicted_accidents']}")
                print(f"   Risk level: {result['risk_level']}")
                print(f"   (Based on junction {result['nearest_junction_distance_m']}m away)")
                
            except ValueError:
                print("Invalid input. Use format: X,Y (e.g., 667000,3550000)")
            except KeyboardInterrupt:
                break
        
        print("\nGoodbye!")
    else:
        print("\n💡 Tip: Run with --interactive flag to enter custom coordinates")