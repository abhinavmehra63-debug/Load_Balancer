import joblib
import pandas as pd

def simulate_request(features):
    model = joblib.load("model.pkl")
    prediction = model.predict([features])[0]

    # Load server info
    servers_info = pd.read_csv("data/servers_info.csv")
    server_row = servers_info[servers_info['server_id'] == prediction].iloc[0]

    print("➡️ Best server for this request:", prediction)
    print(f"   Specs: CPU={server_row['max_cpu']}%, Memory={server_row['max_memory']}GB, Location={server_row['location']}")

if __name__ == "__main__":
    # Example request: [cpu, memory, latency, connections, response_time]
    new_request = [45, 60, 20, 80, 300]
    simulate_request(new_request)
