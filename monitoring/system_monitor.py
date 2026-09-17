import psutil


def get_system_metrics():
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "process_count": len(psutil.pids())
    }


def get_top_processes(n=5):
    procs = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent"]):
        try:
            procs.append(p.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return sorted(procs, key=lambda x: x["cpu_percent"] or 0, reverse=True)[:n]


if __name__ == "__main__":
    print("System metrics:", get_system_metrics())
    print()
    print("Top processes:")
    for p in get_top_processes():
        print(p)