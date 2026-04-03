import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

def train():
    df = pd.read_csv("data/servers.csv")
    X = df.drop('chosen_server', axis=1)
    y = df['chosen_server']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print("✅ Model trained with accuracy:", acc)

    joblib.dump(model, "model.pkl")
    print("✅ Model saved as model.pkl")

if __name__ == "__main__":
    train()
