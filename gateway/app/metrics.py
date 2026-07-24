"""System and inference metrics for the Performance page.

No psutil, no jtop, no tegrastats: everything comes from files the container
can already see (/proc, /sys) or from the predictions we already store. On a
non-Jetson dev box the Jetson-only readings (GPU load, thermal zones) just
come back empty and the page still renders.
"""

import glob
import time

from app import config, db
from app.runner import runner


def _read_int(path: str) -> int | None:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def cpu_percent() -> float | None:
    """Instantaneous CPU busy %, from two /proc/stat samples 100ms apart.
    No cross-request state, so it's correct even on the first page load."""
    def sample():
        try:
            with open("/proc/stat") as f:
                n = [int(x) for x in f.readline().split()[1:]]
        except OSError:
            return None
        return sum(n), n[3] + (n[4] if len(n) > 4 else 0)  # total, idle+iowait

    a = sample()
    if not a:
        return None
    time.sleep(0.1)
    b = sample()
    dt, di = b[0] - a[0], b[1] - a[1]
    return round(100 * (1 - di / dt), 1) if dt else None


def memory() -> dict:
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                info[key] = int(rest.strip().split()[0])  # kB
    except OSError:
        return {}
    total = info.get("MemTotal", 0)
    avail = info.get("MemAvailable", 0)
    swap_total = info.get("SwapTotal", 0)
    swap_free = info.get("SwapFree", 0)
    return {
        "used_gb": round((total - avail) / 1e6, 2),
        "total_gb": round(total / 1e6, 2),
        "percent": round(100 * (total - avail) / total, 1) if total else None,
        "swap_used_gb": round((swap_total - swap_free) / 1e6, 2),
        "swap_total_gb": round(swap_total / 1e6, 2),
    }


def loadavg() -> list[float] | None:
    try:
        with open("/proc/loadavg") as f:
            return [float(x) for x in f.read().split()[:3]]
    except (OSError, ValueError):
        return None


def gpu_load() -> float | None:
    """Jetson GPU busy %, from sysfs (0–1000 per mille). Path varies by board,
    so try the known ones then fall back to any *gpu*/load node."""
    candidates = [
        "/sys/devices/platform/gpu.0/load",
        "/sys/devices/gpu.0/load",
        "/sys/devices/platform/17000000.gpu/load",
        "/sys/devices/17000000.gpu/load",
    ]
    candidates += [p for p in glob.glob("/sys/devices/**/load", recursive=True)
                   if "gpu" in p]
    for path in candidates:
        val = _read_int(path)
        if val is not None:
            return round(val / 10, 1)
    return None


def temperatures() -> list[dict]:
    """Every thermal zone that reports a sane value, e.g. CPU-therm, GPU-therm."""
    out = []
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        milli = _read_int(f"{zone}/temp")
        try:
            with open(f"{zone}/type") as f:
                name = f.read().strip()
        except OSError:
            name = zone.rsplit("/", 1)[-1]
        # Jetson reports millidegrees; ignore the -256000 "disabled" sentinels.
        if milli is not None and milli > 0:
            out.append({"name": name, "c": round(milli / 1000, 1)})
    return out


def disk() -> dict:
    import shutil
    try:
        total, used, free = shutil.disk_usage(config.MODEL_DIR)
    except OSError:
        return {}
    return {
        "used_gb": round(used / 1e9, 1),
        "total_gb": round(total / 1e9, 1),
        "free_gb": round(free / 1e9, 1),
        "percent": round(100 * used / total, 1) if total else None,
    }


def system() -> dict:
    resident = None
    if runner.model_uuid:
        rows = db.query("SELECT model_id, version FROM models WHERE id = %s",
                        (runner.model_uuid,))
        if rows:
            resident = f"{rows[0]['model_id']}:{rows[0]['version']}"
    return {
        "cpu_percent": cpu_percent(),
        "memory": memory(),
        "loadavg": loadavg(),
        "gpu_load": gpu_load(),
        "temperatures": temperatures(),
        "disk": disk(),
        "device": runner.device,
        "resident_model": resident,
    }


def model_performance() -> list[dict]:
    """Latency and throughput per model, from stored predictions. This is the
    number that actually matters for a serving config — real per-board latency
    on this hardware, split by whether TTA was on."""
    rows = db.query(
        "SELECT m.model_id, m.version, m.task,"
        " count(*) AS n,"
        " round(avg(p.latency_ms)::numeric, 1) AS avg_ms,"
        " round(min(p.latency_ms)::numeric, 1) AS min_ms,"
        " round((percentile_cont(0.95) WITHIN GROUP"
        "   (ORDER BY p.latency_ms))::numeric, 1) AS p95_ms,"
        " count(*) FILTER (WHERE p.tta) AS tta_n,"
        " max(p.created_at) AS last_at"
        " FROM predictions p JOIN models m ON m.id = p.model"
        " WHERE p.source = 'server' AND p.latency_ms IS NOT NULL"
        " GROUP BY m.id, m.model_id, m.version, m.task"
        " ORDER BY last_at DESC NULLS LAST"
    )
    return rows


def throughput() -> dict:
    pred = db.query(
        "SELECT"
        " count(*) FILTER (WHERE created_at > now() - interval '1 hour') AS h1,"
        " count(*) FILTER (WHERE created_at > now() - interval '24 hours') AS d1,"
        " count(*) FILTER (WHERE created_at > now() - interval '7 days') AS d7"
        " FROM predictions WHERE source = 'server'"
    )[0]
    ingest = db.query(
        "SELECT"
        " count(*) FILTER (WHERE received_at > now() - interval '1 hour') AS h1,"
        " count(*) FILTER (WHERE received_at > now() - interval '24 hours') AS d1,"
        " count(*) FILTER (WHERE received_at > now() - interval '7 days') AS d7"
        " FROM boards"
    )[0]
    return {"predictions": pred, "boards": ingest}
