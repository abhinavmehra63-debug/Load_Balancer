import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import joblib
import os

# Import backend functions
from main import generate_task, update_model
from ml_model import FEATURES   # reuse the shared feature list

MODEL_FILE = "trained_model.pkl"

st.title("⚡ LOAD BALANCER DASHBOARD")

# --- Backend buttons ---
st.header("Backend Controls")

if st.button("➕ Generate New Tasks"):
    generate_task()   # calls main.py logic
    st.success("New tasks generated and logged successfully!")

# --- Load progress data ---
if os.path.isfile("progress.csv"):
    df = pd.read_csv("progress.csv")
    st.header("📂 Task Progress Log")
    st.dataframe(df)

    model, acc = None, None
    if os.path.isfile(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        st.info("✅ Loaded previously trained model.")

    # --- Train model button ---
    if st.button("📈 Train Model"):
        model, acc = update_model()
        if model:
            st.success(f"Model trained successfully! Accuracy: {acc:.2f}")
            joblib.dump(model, MODEL_FILE)
            st.info("💾 Model saved for future runs.")

            server_labels = [f"server-{i+1}" for i in range(len(model.classes_))]

            if acc >= 0.70 and hasattr(model, "estimators_"):
                st.header("🌳 Random Forest - Top 5 Trees (Graphical View)")
                for i in range(min(5, len(model.estimators_))):
                    fig, ax = plt.subplots(figsize=(24, 12))
                    tree.plot_tree(
                        model.estimators_[i],
                        filled=True,
                        ax=ax,
                        fontsize=15,
                        max_depth=10,
                        feature_names=FEATURES,
                        class_names=server_labels,
                        label="all"
                    )
                    st.pyplot(fig)
            else:
                st.warning("Accuracy below 70% → showing Weighted Round Robin fallback.")

            # --- Server performance insights ---
            st.header("📊 Server Performance Insights")
            server_stats = df.groupby("server_id").agg({
                "success": ["sum", "count"],
                "cpu_need": "mean",
                "memory_need": "mean"
            })

            best_server = server_stats["success"]["sum"].idxmax()
            worst_server = server_stats["success"]["sum"].idxmin()
            most_tasks = server_stats["success"]["count"].idxmax()
            least_tasks = server_stats["success"]["count"].idxmin()

            st.write(f"✅ Best performing server: {best_server}")
            st.write(f"❌ Most failures: {worst_server}")
            st.write(f"⚡ Most tasks handled: {most_tasks}")
            st.write(f"💤 Least load: {least_tasks}")

            # --- Task distribution chart ---
            st.header("📊 Task Distribution Across Servers")
            rr_counts = df["server_id"].value_counts()
            st.bar_chart(rr_counts)
        else:
            st.warning("Not enough data to train yet.")

    # --- Update model button ---
    if st.button("🔄 Update Model"):
        model, acc = update_model()
        if model:
            st.success(f"Model updated incrementally! Accuracy: {acc:.2f} %")
            joblib.dump(model, MODEL_FILE)
            st.info("💾 Updated model saved.")

            # Show updated server stats
            st.header("📊 Updated Server Performance Insights")
            server_stats = df.groupby("server_id").agg({
                "success": ["sum", "count"],
                "cpu_need": "mean",
                "memory_need": "mean"
            })

            best_server = server_stats["success"]["sum"].idxmax()
            worst_server = server_stats["success"]["sum"].idxmin()
            most_tasks = server_stats["success"]["count"].idxmax()
            least_tasks = server_stats["success"]["count"].idxmin()

            st.write(f"✅ Best performing server: {best_server}")
            st.write(f"❌ Most failures: {worst_server}")
            st.write(f"⚡ Most tasks handled: {most_tasks}")
            st.write(f"💤 Least load: {least_tasks}")

            st.header("📊 Task Distribution Across Servers")
            rr_counts = df["server_id"].value_counts()
            st.bar_chart(rr_counts)
        else:
            st.warning("No existing model found. Train a model first.")
else:
    st.error("No progress.csv found. Use 'Generate New Tasks' first.")
