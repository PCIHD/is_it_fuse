import os

import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score

df = pd.read_parquet("./data_foundation/curated/curated_dataset.parquet")
classes = ["cable", "lighting", "fuses"]
df["text"] = df["text"].str.lower()


def simple_check(df_row):
    for class_type in classes:
        if class_type in df_row["text"]:
            df_row["class_pred"] = class_type
        else:
            df_row["class_pred"] = "others"
    return df_row


df = df.apply(simple_check, axis=1)
df["class_name"] = df["class_name"].astype("category")
df["class_pred"] = df["class_pred"].astype("category")
accuracy = sklearn.metrics.accuracy_score(
    df["class_name"].tolist(), df["class_pred"].tolist()
)
confusion_matrix = sklearn.metrics.confusion_matrix(
    df["class_name"].tolist(), df["class_pred"].tolist()
)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment = mlflow.get_experiment_by_name("simple_content_check").experiment_id

with mlflow.start_run(experiment_id=experiment) as trial:
    mlflow.log_metric("accuracy", accuracy)
os.makedirs("./results/simple_content_check/")
df.to_parquet("./results/simple_content_check/results.parquet")
