import pandas as pd
import numpy as np
import pickle
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import mlflow
import mlflow.sklearn
import mlflow.keras

def train_classical_models(X_train, y_train, X_test, y_test, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42)
    }

    results = {}

    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name}_default"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            report = classification_report(y_test, y_pred, output_dict=True)

            results[name] = {
                "accuracy": acc,
                "f1_score": f1,
                "report": report
            }

            # Save locally
            model_path = os.path.join(save_dir, f"{name}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Log to MLflow
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, name)

            # Log classification report as artifact
            report_path = os.path.join(save_dir, f"{name}_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f)
            mlflow.log_artifact(report_path)

            print(f"✅ {name} trained and logged. Accuracy: {acc:.4f}")

    return results


def train_lstm_model(X_train, y_train, X_test, y_test, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)

    input_dim = X_train.shape[1]

    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    with mlflow.start_run(run_name="lstm_dense_default"):
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        model_path = os.path.join(save_dir, "lstm_model.h5")
        model.save(model_path)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.keras.log_model(model, "lstm_model")

        print(f"✅ LSTM model trained and logged. Accuracy: {accuracy:.4f}")

    return accuracy


if __name__ == "__main__":
    print("🔧 Running model training...")

    # Load preprocessed data
    X = pd.read_csv("../data/processed/X_processed.csv")
    y = pd.read_csv("../data/processed/y_processed.csv").squeeze()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train classical models
    classical_results = train_classical_models(X_train, y_train, X_test, y_test)

    # Train dense neural net
    lstm_accuracy = train_lstm_model(X_train, y_train, X_test, y_test)

    # Summarise
    all_accuracies = {model: result['accuracy'] for model, result in classical_results.items()}
    all_accuracies["lstm_model"] = lstm_accuracy

    print("\n🔵 All Model Accuracies:")
    for model, acc in all_accuracies.items():
        print(f"{model}: {acc:.4f}")

    best_model_name = max(all_accuracies, key=all_accuracies.get)
    best_model_accuracy = all_accuracies[best_model_name]
    print(f"\n🏆 Best Model: {best_model_name} with Accuracy: {best_model_accuracy:.4f}")
