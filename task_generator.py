import random

def generate_task(task_id):
    task={
        "id":task_id,
        "cpu_need": random.uniform(5.0,40.0),
        "memory_need" :random.uniform(5.0,55.0),
        "runtime":random.uniform(0.5,5.0),
    }
    return task

def generate_task_batch(num_tasks):
    tasks=[]
    for i in range(num_tasks):
        tasks.append(generate_task(i+1))
    return tasks