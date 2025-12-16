# -*- coding: utf-8 -*-
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_PATH = "gender_classification_preprocessed.csv"
RUN_ID_PATH = os.path.join(os.path.dirname(__file__), "run_id.txt")

def train():
    print("Memuat dataset preprocessed...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop("gender", axis=1)
    y = df["gender"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    with mlflow.start_run() as run:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log parameter
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        # Log metric
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        # Simpan run_id
        with open(RUN_ID_PATH, "w") as f:
            f.write(run.info.run_id)

        print(f"Training selesai. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train()
