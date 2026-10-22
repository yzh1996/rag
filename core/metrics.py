"""Prometheus counters and latency histograms, without content labels."""
import threading
import time
from collections import defaultdict

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily
import json

from config.settings import settings

_lock = threading.Lock()
_counters = defaultdict(float)
_latency_sum = defaultdict(float)
_latency_count = defaultdict(float)
_started_at = time.time()
registry = CollectorRegistry()
_prom_counters = {}
_prom_histograms = {}


def increment(name: str, value: float = 1.0) -> None:
    with _lock:
        _counters[name] += value
        if name not in _prom_counters:
            _prom_counters[name] = Counter("rag_" + name, name.replace("_", " "), registry=registry)
        _prom_counters[name].inc(value)


def observe(name: str, value_ms: float) -> None:
    with _lock:
        _latency_sum[name] += value_ms
        _latency_count[name] += 1
        if name not in _prom_histograms:
            _prom_histograms[name] = Histogram("rag_" + name + "_seconds", name.replace("_", " "),
                buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120), registry=registry)
        _prom_histograms[name].observe(value_ms / 1000)


def snapshot() -> dict:
    with _lock:
        return {"uptime_seconds": round(time.time() - _started_at, 2), "counters": dict(_counters),
            "latency_ms": {key: {"count": _latency_count[key], "sum": round(_latency_sum[key], 2),
                "average": round(_latency_sum[key] / _latency_count[key], 2)} for key in _latency_count if _latency_count[key]}}


def prometheus_text() -> str:
    return generate_latest(registry).decode("utf-8")


class BackupCollector:
    def collect(self):
        path = settings.DATA_DIR / "backup_status.json"
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {}
        yield GaugeMetricFamily("rag_backup_last_success_timestamp", "Last verified backup timestamp", value=status.get("last_success_timestamp", 0))
        yield GaugeMetricFamily("rag_backup_last_attempt_failed", "Last backup attempt failed", value=int(status.get("last_attempt_failed", False)))
        yield GaugeMetricFamily("rag_cost_tracking_configured", "LLM token prices have been configured", value=int(settings.LLM_INPUT_COST_PER_MILLION > 0 or settings.LLM_OUTPUT_COST_PER_MILLION > 0))


registry.register(BackupCollector())


for _name in ("http_requests_total", "http_server_errors_total", "qa_requests_total", "qa_errors_total",
    "qa_refusals_total", "retrieval_errors_total", "retrieval_empty_total", "feedback_total",
    "feedback_negative_total", "feedback_positive_total", "handoff_requests_total", "ingestion_failed_total",
    "llm_input_tokens_total", "llm_output_tokens_total", "llm_cost_usd_total", "quality_rejections_total",
    "backup_success_total", "backup_errors_total"):
    increment(_name, 0)
