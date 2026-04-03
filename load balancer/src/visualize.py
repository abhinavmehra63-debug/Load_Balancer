import pandas as pd
import matplotlib.pyplot as plt
import joblib

def plot_server_loads():
    df = pd.read_csv("data/servers.csv")

    # Plot CPU usage distribution
    plt.figure(figsize=(6,4))
    plt.hist(df['cpu_usage'], bins=10, color='skyblue', edgecolor='black')
    plt.title("CPU Usage Distribution")
    plt.xlabel("CPU Usage (%)")
    plt.ylabel("Frequency")
    plt.show()

    # Plot memory usage distribution
    plt.figure(figsize=(6,4))
    plt.hist(df['memory_usage'], bins=10, color='lightgreen', edgecolor='black')
    plt.title("Memory Usage Distribution")
    plt.xlabel("Memory Usage (%)")
    plt.ylabel("Frequency")
    plt.show()

def plot_model_accuracy():
    df = pd.read_csv("data/servers.csv")
    X = df.drop('chosen_server', axis=1)
    y = df['chosen_server']

    model = joblib.load("model.pkl")
    acc = model.score(X, y)

    # Simple bar chart for accuracy
    plt.figure(figsize=(4,4))
    plt.bar(["DecisionTree"], [acc], color='orange')
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.show()

if __name__ == "__main__":
    plot_server_loads()
    plot_model_accuracy()
