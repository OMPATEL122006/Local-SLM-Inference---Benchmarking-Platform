import subprocess
import time

import psutil


def get_gpu_stats():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,"
                "memory.used,"
                "memory.total,"
                "power.draw,"
                "temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        values = result.stdout.strip().split(",")

        if len(values) < 5:
            return None

        return {
            "gpu_utilization_percent": float(values[0].strip()),
            "vram_used_mb": float(values[1].strip()),
            "vram_total_mb": float(values[2].strip()),
            "gpu_power_w": float(values[3].strip()),
            "gpu_temperature_c": float(values[4].strip()),
        }

    except (subprocess.SubprocessError, ValueError):
        return None


def get_system_stats():
    memory = psutil.virtual_memory()

    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_gb": round(
            memory.used / (1024 ** 3),
            2,
        ),
        "ram_available_gb": round(
            memory.available / (1024 ** 3),
            2,
        ),
        "ram_percent": memory.percent,
    }


def collect_snapshot():
    snapshot = {
        "timestamp": time.time(),
    }

    snapshot.update(get_system_stats())

    gpu_stats = get_gpu_stats()

    if gpu_stats:
        snapshot.update(gpu_stats)

    return snapshot


def summarize_snapshots(snapshots):
    if not snapshots:
        return {}

    def values(key):
        return [
            snapshot[key]
            for snapshot in snapshots
            if key in snapshot
        ]

    def average(key):
        data = values(key)
        return round(sum(data) / len(data), 2) if data else None

    def maximum(key):
        data = values(key)
        return round(max(data), 2) if data else None

    return {
        "samples": len(snapshots),

        "cpu_avg_percent": average("cpu_percent"),
        "cpu_peak_percent": maximum("cpu_percent"),

        "ram_avg_percent": average("ram_percent"),
        "ram_peak_percent": maximum("ram_percent"),

        "gpu_avg_percent": average(
            "gpu_utilization_percent"
        ),
        "gpu_peak_percent": maximum(
            "gpu_utilization_percent"
        ),

        "vram_avg_mb": average("vram_used_mb"),
        "vram_peak_mb": maximum("vram_used_mb"),

        "gpu_power_avg_w": average("gpu_power_w"),
        "gpu_power_peak_w": maximum("gpu_power_w"),

        "gpu_temperature_avg_c": average(
            "gpu_temperature_c"
        ),
        "gpu_temperature_peak_c": maximum(
            "gpu_temperature_c"
        ),
    }
