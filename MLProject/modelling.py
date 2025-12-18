import os
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
DATA_FILE = "gender_classification_preprocessed.csv"
RUN_ID_FILE = "run_id.txt"
LOCAL_MODEL_DIR = "model"  

mlflow.autolog()

def load_dataset(path):
    print("Membaca dataset...")
    return pd.read_csv(path)

def prepare_data(df):
    X = df.drop("gender", axis=1)
    y = df["gender"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

def train_and_log():
    df = load_dataset(DATA_FILE)
    X_train, X_test, y_train, y_test = prepare_data(df)

    if os.path.exists(LOCAL_MODEL_DIR):
        shutil.rmtree(LOCAL_MODEL_DIR)

    with mlflow.start_run() as run:
        print("Training dimulai...")

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy_manual", acc)

        print(f"Akurasi: {acc}")

        print("Menyimpan model ke folder 'model/' ...")
        mlflow.sklearn.save_model(model, LOCAL_MODEL_DIR)

        print("Upload artefak model ke MLflow...")
        mlflow.log_artifacts(LOCAL_MODEL_DIR, artifact_path="model")

        with open(RUN_ID_FILE, "w") as f:
            f.write(run.info.run_id)

        print(f"Run ID disimpan: {run.info.run_id}")

    shutil.rmtree(LOCAL_MODEL_DIR)
    print("Folder model lokal dibersihkan")

if __name__ == "__main__":
    train_and_log()
