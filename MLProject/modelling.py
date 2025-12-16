import pandas as pd
import mlflow
from mlflow.sklearn import save_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import shutil

DATA_PATH = "gender_classification_preprocessed.csv"
RUN_ID_FILE = "run_id.txt"
LOCAL_MODEL_DIR = "local_model_output"

df = pd.read_csv(DATA_PATH)

X = df.drop("gender", axis=1)
y = df["gender"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)

try:
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", acc)

        print(f"Akurasi: {acc}")
        run_id = run.info.run_id

        if os.path.exists(LOCAL_MODEL_DIR):
            shutil.rmtree(LOCAL_MODEL_DIR)

        save_model(model, LOCAL_MODEL_DIR)

        mlflow.log_artifacts(LOCAL_MODEL_DIR, artifact_path="model")

        shutil.rmtree(LOCAL_MODEL_DIR)

        with open(RUN_ID_FILE, "w") as f:
            f.write(run_id)

except Exception as e:
    print("ERROR:", e)
    if os.path.exists(LOCAL_MODEL_DIR):
        shutil.rmtree(LOCAL_MODEL_DIR)
    exit(1)
