import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_FILE = "gender_classification_preprocessed.csv"
RUN_ID_FILE = "run_id.txt"

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

    with mlflow.start_run() as run:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Optional (boleh ada)
        mlflow.log_metric("accuracy_manual", acc)

        # Simpan run_id untuk CI / Docker
        with open(RUN_ID_FILE, "w") as f:
            f.write(run.info.run_id)

        print(f"Run ID disimpan: {run.info.run_id}")

if __name__ == "__main__":
    train_and_log()
