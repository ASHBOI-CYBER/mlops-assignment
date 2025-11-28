import mlflow
import argparse
import os

def workflow(raw_data):
    # Define paths
    processed_data = "data/processed_data.csv"
    
    print(">>> Step 1: Running ETL")
    mlflow.run(
        ".", 
        "etl", 
        parameters={
            "input_file": raw_data, 
            "output_file": processed_data
        },
        env_manager="local"  # <--- THIS WAS MISSING
    )

    print(">>> Step 2: Running Training")
    mlflow.run(
        ".", 
        "train", 
        parameters={
            "data_file": processed_data, 
            "n_estimators": 50
        },
        env_manager="local"  # <--- THIS WAS MISSING
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data")
    args = parser.parse_args()
    
    # Optional: Group under a parent run for cleaner UI
    with mlflow.start_run(run_name="Main Pipeline"):
        workflow(args.raw_data)