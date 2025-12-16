import os
import shutil

import pandas as pd
import mlflow
from mlflow.sklearn import save_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_FILE = "gender_classification_preprocessed.csv"
RUN_ID_OUTPUT = "run_id.txt"
TEMP_MODEL_DIR = "tmp_saved_model"


def load_dataset(path):
    print("Membaca dataset...")
    return pd.read_csv(path)


def prepare_data(df):
    X = df.drop("gender", axis=1)
    y = df["gender"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_and_log():
    df = load_dataset(DATA_FILE)
    X_train, X_test, y_train, y_test = prepare_data(df)

    model = LogisticRegression(max_iter=1000)

    if os.path.exists(TEMP_MODEL_DIR):
        shutil.rmtree(TEMP_MODEL_DIR)

    with mlflow.start_run() as run:
        print("Training model dimulai...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("algorithm", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        print(f"Akurasi model: {accuracy}")
        
        save_model(model, TEMP_MODEL_DIR)

        mlflow.log_artifacts(TEMP_MODEL_DIR, artifact_path="model")

        with open(RUN_ID_OUTPUT, "w") as f:
            f.write(run.info.run_id)

        print(f"Run ID disimpan: {run.info.run_id}")

    shutil.rmtree(TEMP_MODEL_DIR)


if __name__ == "__main__":
    train_and_log()
