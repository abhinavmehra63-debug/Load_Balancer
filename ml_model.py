import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Shared feature list to guarantee consistency
FEATURES = ["cpu_need", "memory_need", "cpu_usage", "memory_usage", "runtime"]

def train_model():
    # Check file existence and non-empty
    if not os.path.isfile("progress.csv") or os.path.getsize("progress.csv") == 0:
        print("No progress data found")
        return None, None
    
    df = pd.read_csv("progress.csv")

    if len(df) < 50:
        print("Not enough data to train!")
        return None, None

    # Use consistent feature order
    X = df[FEATURES]
    y = df["success"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))*100

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/model_{len(df)}rows.pkl")

    return model, acc


def predict_best_server(task, model, servers):
    scores = {}
    for s in servers:
        if s.status == "shutdown":
            continue  # skip this server
        task_features = pd.DataFrame([{
            "cpu_need": task["cpu_need"],
            "memory_need": task["memory_need"],
            "cpu_usage": s.cpu_usage,
            "memory_usage": s.memory_usage,
            "runtime": task["runtime"]
        }])[FEATURES]
        prob_success = model.predict_proba(task_features)[0][1]
        scores[s.id] = prob_success
    return max(scores, key=scores.get) if scores else None