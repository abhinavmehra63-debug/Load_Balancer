import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from server import Server
from task_generator import generate_task_batch
from progress_logger import log_progress
from ml_model import train_model, predict_best_server
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from round_robin import round_robin


def process_task(task, servers, model, server_index):
    rr_server = servers[server_index]

    if model:
        ml_choice = predict_best_server(task, model, servers)
        ml_server = next((s for s in servers if s.id == ml_choice), rr_server)
        ml_choice_str = f"{ml_server.id} (C:{ml_server.cpu_usage:.2f}, M:{ml_server.memory_usage:.2f})"
    else:
        ml_server = rr_server
        ml_choice_str = "(no model yet)"

    chosen_server = ml_server
    chosen_server.assign_task(task)
    log_progress(task, chosen_server)
    chosen_server.request_usage()

    return {
        "Task": task["id"],
        "CPU Need": f"{task['cpu_need']:.2f}",
        "Memory Need": f"{task['memory_need']:.2f}",
        "Runtime": f"{task['runtime']:.2f}",
        "Round Robin": rr_server.id,
        "ML Prediction": ml_choice_str,
        "Final Assignment": chosen_server.id,
        "Success": task["success"]
    }


def generate_task(batch_size=100, num_servers=5):
    servers = [Server(i + 1) for i in range(num_servers)]
    tasks = generate_task_batch(batch_size)
    model, acc = train_model()

    # Split tasks into two phases
    rr_tasks = tasks[:30]       # first 30 tasks → round robin
    ml_tasks = tasks[30:]       # remaining tasks → ML (if model is good enough)

    results = []

    # Round Robin phase
    with ThreadPoolExecutor(max_workers=len(servers)) as executor:
        futures = []
        for task in rr_tasks:
            chosen_server = round_robin(servers)
            futures.append(executor.submit(process_task, task, servers, None, servers.index(chosen_server)))
        for future in as_completed(futures):
            results.append(future.result())

    # ML phase (only if model is trained and accurate enough)
    if model and acc is not None and acc >= 0.70:
        with ThreadPoolExecutor(max_workers=len(servers)) as executor:
            futures = []
            for i, task in enumerate(ml_tasks):
                futures.append(executor.submit(process_task, task, servers, model, i % len(servers)))
            for future in as_completed(futures):
                results.append(future.result())
    else:
        # Fallback: Round Robin again for ML tasks
        with ThreadPoolExecutor(max_workers=len(servers)) as executor:
            futures = []
            for task in ml_tasks:
                chosen_server = round_robin(servers)
                futures.append(executor.submit(process_task, task, servers, None, servers.index(chosen_server)))
            for future in as_completed(futures):
                results.append(future.result())

    if results:
        df = pd.DataFrame(results)
        return df
    else:
        return pd.DataFrame()


def update_model():
    """Helper function so app.py can call update_model."""
    model, acc = train_model()
    return model, acc


def main():
    df = generate_task()
    if not df.empty:
        print("\n=== Parallel Task Assignment Report ===\n")
        print(df.to_string(index=False))
    else:
        print("No tasks processed.")
    print("\n=== End of Parallel Report ===\n")


if __name__ == "__main__":
    main()
