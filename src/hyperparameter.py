import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn
import mlflow.keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ✅ Set up MLflow to track to DagsHub
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/nwanduben/easylaborprediction.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "nwanduben"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "b608dde0fc91ff555273d7f349c426b2fa610a5d"

SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

def run_grid_search(model_name, model, param_grid, X_train, y_train, X_test, y_test):
    search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    
    with mlflow.start_run(run_name=model_name):
        search.fit(X_train, y_train)
        
        y_pred = search.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(search.best_estimator_, "model")

        with open(os.path.join(SAVE_DIR, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(search.best_estimator_, f)

        print(f"✅ {model_name} done | Accuracy: {acc:.4f}")
        return model_name, acc, search.best_params_

def train_lstm_with_tuning(X_train, y_train, X_test, y_test, save_dir="models"):
    param_grid = [
        {"units": 32, "epochs": 10, "batch_size": 32},
        {"units": 64, "epochs": 15, "batch_size": 64},
        {"units": 128, "epochs": 20, "batch_size": 32},
    ]

    best_acc = 0
    best_config = None

    for config in param_grid:
        with mlflow.start_run(run_name=f"LSTM_u{config['units']}_bs{config['batch_size']}"):
            model = Sequential([
                Dense(config["units"], activation='relu', input_shape=(X_train.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            model.fit(X_train, y_train,
                      epochs=config["epochs"],
                      batch_size=config["batch_size"],
                      validation_split=0.2,
                      verbose=0)

            loss, acc = model.evaluate(X_test, y_test, verbose=0)

            mlflow.log_params(config)
            mlflow.log_metric("accuracy", acc)
            mlflow.keras.log_model(model, "model")

            if acc > best_acc:
                best_acc = acc
                best_config = config
                model.save(os.path.join(save_dir, "lstm_best_model.h5"))

            print(f"✅ LSTM with {config} — Accuracy: {acc:.4f}")

    print(f"\n🏆 Best LSTM Config: {best_config} | Accuracy: {best_acc:.4f}")
    return "lstm_model", best_acc, best_config

def main():
    print("🚀 Starting hyperparameter tuning...")

    X = pd.read_csv("../data/processed/X_processed.csv")
    y = pd.read_csv("../data/processed/y_processed.csv").squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000),
            {
                "C": [0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"]
            }
        ),
        "decision_tree": (
            DecisionTreeClassifier(),
            {
                "max_depth": [3, 5, 10],
                "min_samples_split": [2, 5, 10]
            }
        ),
        "xgboost": (
            XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            {
                "n_estimators": [50, 100],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1]
            }
        )
    }

    results = []
    for name, (model, param_grid) in models.items():
        result = run_grid_search(name, model, param_grid, X_train, y_train, X_test, y_test)
        results.append(result)

    # ✅ Add LSTM tuning
    lstm_result = train_lstm_with_tuning(X_train, y_train, X_test, y_test)
    results.append(lstm_result)

    print("\n📊 Summary of Best Accuracies:")
    for name, acc, params in results:
        print(f"{name}: {acc:.4f} | Best Params: {params}")

    best_model = max(results, key=lambda x: x[1])
    print(f"\n🏆 Best Model Overall: {best_model[0]} | Accuracy: {best_model[1]:.4f}")

if __name__ == "__main__":
    main()
