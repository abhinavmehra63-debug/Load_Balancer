import pandas as pd
import numpy as np

def generate_data(n=200):
    # Simulated request dataset
    data = {
        'cpu_usage': np.random.randint(10, 90, n),
        'memory_usage': np.random.randint(10, 90, n),
        'latency': np.random.randint(5, 50, n),
        'active_connections': np.random.randint(10, 200, n),
        'response_time': np.random.randint(100, 1000, n),
        'chosen_server': np.random.choice([1, 2, 3], n)
    }
    df = pd.DataFrame(data)
    df.to_csv("data/servers.csv", index=False)
    print("✅ Request dataset generated and saved to data/servers.csv")

if __name__ == "__main__":
    generate_data()
