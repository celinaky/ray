"""Router-local queue calibration and optional reproducible audit records."""

import atexit
import json
import logging
import math
import os
import queue
import threading
import time
import uuid
from collections import deque
from pathlib import Path

PREFIX = "prefill_queue_model_"


class CalibrationAudit:
    """Bounded asynchronous JSONL writer; audit failures never stop serving."""

    def __init__(self, directory, identity):
        self.queue = queue.Queue(maxsize=10000)
        self.dropped = 0
        self.failed = False
        self.closed = False
        self.path = Path(directory) / f"{identity}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def emit(self, record):
        if self.closed:
            return
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        try:
            with self.path.open("a", buffering=1) as stream:
                while True:
                    record = self.queue.get()
                    if record is None:
                        stream.write(
                            json.dumps(
                                {"event": "audit_closed", "dropped": self.dropped}
                            )
                            + "\n"
                        )
                        break
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
        except Exception:
            self.failed = True

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.thread.is_alive():
            try:
                self.queue.put(None, timeout=1)
                self.thread.join(timeout=2)
            except queue.Full:
                self.dropped += 1


class PrefillQueueCalibrator:
    def __init__(self, min_samples=100, window_samples=1000, refresh_interval_s=120):
        if (
            isinstance(min_samples, bool)
            or not isinstance(min_samples, int)
            or isinstance(window_samples, bool)
            or not isinstance(window_samples, int)
            or min_samples < 2
            or window_samples < min_samples
        ):
            raise ValueError("Require 2 <= min_samples <= window_samples")
        if not math.isfinite(refresh_interval_s) or refresh_interval_s <= 0:
            raise ValueError("refresh_interval_s must be positive and finite")
        self.min_samples = min_samples
        self.samples = deque(maxlen=window_samples)
        self.refresh_interval_s = refresh_interval_s
        self.identity = uuid.uuid4().hex
        self.sequence = 0
        self.attempt = 0
        self.version = 0
        self.model = {}
        self.last_refresh = time.monotonic()
        directory = os.environ.get("RAY_SERVE_LLM_PREFILL_CALIBRATION_TRACE_DIR")
        self.audit = None
        if directory:
            try:
                self.audit = CalibrationAudit(directory, self.identity)
            except Exception:
                logging.getLogger(__name__).exception("Calibration audit unavailable")

    def _emit(self, event, **fields):
        if self.audit:
            self.audit.emit(
                dict(
                    schema_version=1,
                    event=event,
                    router_id=self.identity,
                    wall_time_s=time.time(),
                    **fields,
                )
            )

    def record_routing(self, request_id, snapshot):
        # Record at admission as well as at observation, so requests cancelled
        # before their first output can still be joined to scheduler traces.
        self._emit("request_routed", request_id=request_id, **snapshot)

    def observe(self, request_id, snapshot, queue_time_ms):
        if (
            queue_time_ms is None
            or not math.isfinite(queue_time_ms)
            or queue_time_ms < 0
        ):
            return
        self.sequence += 1
        sample = dict(
            sample_id=self.sequence,
            request_id=request_id,
            queue_time_ms=queue_time_ms,
            observed_at_s=time.time(),
            **snapshot,
        )
        self.samples.append(sample)
        self._emit("sample_accepted", sample=sample)
        # Bootstrap at the first eligible observation instead of waiting for a
        # refresh boundary. Subsequent attempts retain the regular cadence.
        if self.sequence == self.min_samples:
            self.refresh(bootstrap=True)

    def refresh(self, now=None, *, bootstrap=False):
        now = time.monotonic() if now is None else now
        if not bootstrap and now - self.last_refresh < self.refresh_interval_s:
            return False
        self.last_refresh = now
        self.attempt += 1
        rows = list(self.samples)
        reason = "insufficient_samples"
        candidate = None
        if len(rows) >= self.min_samples:
            mean_x = math.fsum(r["workload_tokens"] for r in rows) / len(rows)
            mean_y = math.fsum(r["queue_time_ms"] for r in rows) / len(rows)
            variance = math.fsum((r["workload_tokens"] - mean_x) ** 2 for r in rows)
            if variance > 0:
                slope = (
                    math.fsum(
                        (r["workload_tokens"] - mean_x) * (r["queue_time_ms"] - mean_y)
                        for r in rows
                    )
                    / variance
                )
                intercept = mean_y - slope * mean_x
                if math.isfinite(slope) and math.isfinite(intercept) and slope > 0:
                    candidate = dict(
                        intercept_ms=intercept,
                        slope_ms_per_token=slope,
                        sample_count=len(rows),
                        updated_at_s=time.time(),
                    )
                    self.version += 1
                    self.model = {PREFIX + k: v for k, v in candidate.items()}
                    self.model[PREFIX + "version"] = self.version
                    reason = "published"
                else:
                    reason = "invalid_fit"
            else:
                reason = "no_workload_variation"
        self._emit(
            "fit_snapshot",
            attempt_id=self.attempt,
            samples=rows,
            reason=reason,
            candidate=candidate,
            published_model=self.model.copy(),
        )
        return candidate is not None
