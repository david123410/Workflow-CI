import os
import shutil
import pandas as pd
import mlflow
from mlflow.sklearn import save_model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
DATA_FILE = "gender_classification_preprocessed.csv"
RUN_ID_FILE = "run_id.txt"
TEMP_MODEL_DIR = "temp_model_dir"


def load_dataset(path):
    print(" Membaca dataset...")
    return pd.read_csv(path)


def prepare_data(df):
    X = df.drop("gender", axis=1)
    y = df["gender"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_and_log():
    df = load_dataset(DATA_FILE)
    X_train, X_test, y_train, y_test = prepare_data(df)

    model = LogisticRegression(max_iter=1000)

    # Bersihkan folder sementara
    if os.path.exists(TEMP_MODEL_DIR):
        shutil.rmtree(TEMP_MODEL_DIR)

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log metric & param
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        print(f"Akurasi model: {acc}")

        # Simpan model ke folder sementara
        save_model(model, TEMP_MODEL_DIR)

        # Upload sebagai artifact "model"
        mlflow.log_artifacts(TEMP_MODEL_DIR, artifact_path="model")

        # Simpan run_id
        with open(RUN_ID_FILE, "w") as f:
            f.write(run.info.run_id)

        print(f"🆔 Run ID disimpan: {run.info.run_id}")

    # Bersihkan folder lokal
    shutil.rmtree(TEMP_MODEL_DIR)
    print("Folder sementara dibersihkan")


if __name__ == "__main__":
    train_and_log()
