class Server:
    def __init__(self, server_id):
        self.id = server_id
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.avg_runtime = 0.0
        self.requests_fulfilled = 0
        self.failure_rate = 0.0
        self.status = "Active"
        self.total_runtime = 0.0
        self.failed_requests = 0

    def assign_task(self, task):
        # update usage
        self.cpu_usage = min(100.0, self.cpu_usage + task["cpu_need"])
        self.memory_usage = min(100.0, self.memory_usage + task["memory_need"])
        self.total_runtime += task["runtime"]
        self.requests_fulfilled += 1
        self.avg_runtime = self.total_runtime / self.requests_fulfilled

        # ✅ compute success based on capacity
        success = (task["cpu_need"] + self.cpu_usage <= 100) and \
                  (task["memory_need"] + self.memory_usage <= 100)

        # assign success back into the task dictionary
        task["success"] = success

        # update failure stats if unsuccessful
        if not success:
            self.failed_requests += 1
            self.failure_rate = min(100.0, (self.failed_requests / self.requests_fulfilled) * 100)

        # shutdown if overloaded
        if self.cpu_usage > 90 or self.memory_usage > 90:
            self.status = "shutdown"

    def request_usage(self):
        self.cpu_usage = max(0.0, self.cpu_usage - 10.0)
        self.memory_usage = max(0.0, self.memory_usage - 10.0)

    def load_score(self):
        remaining_cpu = max(0.0, 100.0 - self.cpu_usage)
        remaining_mem = max(0.0, 100.0 - self.memory_usage)
        return remaining_cpu + remaining_mem
