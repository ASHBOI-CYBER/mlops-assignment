import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train(data_path, n_estimators):
    # Enable MLflow Autologging
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        data = pd.read_csv(data_path)
        
        # Prepare features and target
        X = data.drop("median_house_value", axis=1)
        y = data["median_house_value"]

        # Handle categorical data
        if "ocean_proximity" in X.columns:
            X = pd.get_dummies(X, columns=["ocean_proximity"], drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train
        model = RandomForestRegressor(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        
        # --- FIX: Calculate RMSE manually ---
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5  # This is the same as square root
        # ------------------------------------
        
        mlflow.log_metric("rmse", rmse)
        print(f"Model trained. RMSE: {rmse}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()
    
    train(args.data_path, args.n_estimators)