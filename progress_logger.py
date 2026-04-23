import pandas as pd
import os
import threading

lock = threading.Lock()

def log_progress(task, server):
    new_row = pd.DataFrame([{
        "task_id": task["id"],
        "server_id": server.id,
        "cpu_need": task["cpu_need"],
        "memory_need": task["memory_need"],
        "runtime": task["runtime"],
        "success": task["success"],          # ✅ now always set by assign_task
        "cpu_usage": server.cpu_usage,
        "memory_usage": server.memory_usage,
        "avg_runtime": server.avg_runtime,
        "failure_rate": server.failure_rate,
        "requests_fulfilled": server.requests_fulfilled,
        "status": server.status
    }])

    with lock:  # prevent race conditions
        header = not os.path.isfile("progress.csv") or os.path.getsize("progress.csv") == 0
        new_row.to_csv("progress.csv", mode="a", header=header, index=False)
