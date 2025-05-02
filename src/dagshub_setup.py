
import os
import mlflow
from getpass import getpass

def setup_dagshub_tracking():
    print(" Setting up MLflow tracking with DagsHub...")
    username = input("👤 Enter your DagsHub username: ")
    repo_name = input("📘 Enter your DagsHub repo name: ")
    token = getpass("🔐 Enter your DagsHub access token: ")

    tracking_uri = f"https://dagshub.com/{username}/{repo_name}.mlflow"

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_tracking_uri(tracking_uri)
    
    # Create the experiment if it doesn't exist
    experiment_name = "ML Colab Experiment"  
    try:
        mlflow.create_experiment(experiment_name)
        print(f"✅ Created experiment: {experiment_name}") 
    except mlflow.exceptions.MlflowException as e:
        if "Experiment with name" in str(e) and "already exists" in str(e):
            print(f"✅ Experiment '{experiment_name}' already exists. Skipping creation.")
        else:
            raise e 

    mlflow.set_experiment(experiment_name)

    print(f"✅ Tracking URI set to: {tracking_uri}")
