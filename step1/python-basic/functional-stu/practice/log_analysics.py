from collections import defaultdict

logs = [
    {"ip": "10.0.0.1", "path": "/api/users", "status": 200, "duration": 120},
    {"ip": "10.0.0.2", "path": "/api/orders", "status": 500, "duration": 350},
    {"ip": "10.0.0.1", "path": "/api/users", "status": 200, "duration": 90},
    {"ip": "10.0.0.3", "path": "/api/products", "status": 404, "duration": 80},
    {"ip": "10.0.0.2", "path": "/api/orders", "status": 200, "duration": 260},
    {"ip": "10.0.0.1", "path": "/api/orders", "status": 200, "duration": 180},
    {"ip": "10.0.0.4", "path": "/api/users", "status": 500, "duration": 400},
]


def success_logs(logs):
    return [log for log in logs if log["status"] // 100 == 2]


def error_logs(logs):
    return [log for log in logs if log["status"] >= 400]


def paths(logs):
    return {log["path"] for log in logs}


def count_by_status(logs):
    count_by_status = defaultdict(int)
    for log in logs:
        count_by_status[log["status"]] += 1
    return count_by_status


def count_by_path(logs):
    count_by_path = defaultdict(int)
    for log in logs:
        count_by_path[log["path"]] += 1
    return count_by_path


def average_duration(logs):
    return sum(log["duration"] for log in logs) / len(logs)


def logs_by_path(logs):
    logs_by_path = defaultdict(list)
    for log in logs:
        logs_by_path[log["path"]].append(log)
    return logs_by_path


def average_duration_by_path(logs):
    average_duration = defaultdict(float)
    total_duration = defaultdict(int)
    logs_by_path_dict = logs_by_path(logs)
    for log in logs:
        total_duration[log["path"]] += log["duration"]
    for key, value in total_duration.items():
        average_duration[key] = value / len(logs_by_path_dict[key])
    return average_duration


def slow_logs(logs, threshold):
    return [log for log in logs if log["duration"] >= threshold]


def top_n_slowest_logs(logs, n):
    return sorted(logs, key=lambda log: log["duration"], reverse=True)[:n]


def most_visited_path(logs):
    count_by_path_dict = count_by_path(logs)
    return max(count_by_path_dict.keys(), key=lambda key: count_by_path_dict[key])
