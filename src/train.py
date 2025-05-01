# src/train.py

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def train_classical_models(X_train, y_train, X_test, y_test, save_dir="models"):
    """
    Trains Logistic Regression, XGBoost, and Decision Tree models.
    Saves the models into the specified directory.
    """
    os.makedirs(save_dir, exist_ok=True)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results[name] = {
            "accuracy": acc,
            "report": classification_report(y_test, y_pred, output_dict=True)
        }

        # Save each model
        with open(os.path.join(save_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(model, f)

        print(f"✅ {name} trained and saved. Accuracy: {acc:.4f}")

    return results


def train_lstm_model(X_train, y_train, X_test, y_test, save_dir="models"):
    """
    Trains a simple dense-based model (inspired by LSTM architecture).
    Saves the model into the specified directory.
    """
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

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    model.save(os.path.join(save_dir, "lstm_model.h5"))

    print(f"✅ LSTM model trained and saved. Accuracy: {accuracy:.4f}")

    return accuracy

if __name__ == "__main__":
    print("Running model training script...")

    # ✅ Load preprocessed data
    X = pd.read_csv("../data/processed/X_processed.csv")
    y = pd.read_csv("../data/processed/y_processed.csv").squeeze()

    # ✅ Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ✅ Train classical models
    classical_results = train_classical_models(X_train, y_train, X_test, y_test)

    # ✅ Train LSTM model
    lstm_accuracy = train_lstm_model(X_train, y_train, X_test, y_test)

    # ✅ Combine all results
    all_accuracies = {model: result['accuracy'] for model, result in classical_results.items()}
    all_accuracies["lstm_model"] = lstm_accuracy

    print("\n🔵 All Model Accuracies:")
    for model, acc in all_accuracies.items():
        print(f"{model}: {acc:.4f}")

    # ✅ Find the best model
    best_model_name = max(all_accuracies, key=all_accuracies.get)
    best_model_accuracy = all_accuracies[best_model_name]
    print(f"\n🏆 Best Model: {best_model_name} with Accuracy: {best_model_accuracy:.4f}")
    

