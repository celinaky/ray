import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ray.llm._internal.serve.routing_policies.kv_aware.kv_token_tracker import (
    KVTokenTracker,
)
from ray.llm._internal.serve.routing_policies.kv_aware.prefill_calibration import (
    PREFIX,
    PrefillQueueCalibrator,
)
from ray.llm._internal.serve.routing_policies.kv_aware.vllm.token_tracking import (
    RequestTokenTracker,
)
from ray.serve.experimental.llm_token_autoscaling_policy import (
    LLMTokenAutoscalingPolicy,
)


def observe(model, i):
    model.observe(
        str(i),
        dict(workload_tokens=i * 10, worker_id=1, routed_at_s=1, prompt_tokens=42),
        5 + 2 * i * 10,
    )


def test_readiness_refresh_and_window(tmp_path, monkeypatch):
    monkeypatch.setenv("RAY_SERVE_LLM_PREFILL_CALIBRATION_TRACE_DIR", str(tmp_path))
    clock = [0.0]
    monkeypatch.setattr(
        "ray.llm._internal.serve.routing_policies.kv_aware.prefill_calibration.time.monotonic",
        lambda: clock[0],
    )
    m = PrefillQueueCalibrator(window_samples=100)
    start = m.last_refresh
    for i in range(99):
        observe(m, i)
    assert not m.refresh(start + 120)
    clock[0] = start + 121
    observe(m, 99)
    assert m.version == 1
    assert not m.refresh(start + 121)
    assert not m.refresh(start + 240)
    assert m.model[PREFIX + "slope_ms_per_token"] == pytest.approx(2)
    assert m.model[PREFIX + "intercept_ms"] == pytest.approx(5)
    old = m.model.copy()
    for i in range(100):
        observe(m, 500)
    assert not m.refresh(start + 241)
    assert m.model == old
    m.audit.close()
    records = [
        json.loads(line)
        for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()
    ]
    fits = [r for r in records if r["event"] == "fit_snapshot"]
    assert [len(r["samples"]) for r in fits] == [99, 100, 100]
    assert fits[-1]["reason"] == "no_workload_variation"


def test_bootstrap_before_first_interval_then_waits(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "ray.llm._internal.serve.routing_policies.kv_aware.prefill_calibration.time.monotonic",
        lambda: clock[0],
    )
    m = PrefillQueueCalibrator()
    for i in range(99):
        observe(m, i)
    assert not m.model
    clock[0] = 10
    observe(m, 99)
    assert m.model[PREFIX + "sample_count"] == 100
    assert m.version == 1
    observe(m, 100)
    assert not m.refresh(120)
    assert m.refresh(130)
    assert m.version == 2
    assert m.model[PREFIX + "sample_count"] == 101


@pytest.mark.asyncio
async def test_tracker_publishes_bootstrap_in_next_snapshot():
    tracker = LocalTracker(enable_prefill_queue_learning=True)
    sink = Mock()
    tracker.set_autoscaling_metrics_sink(sink)
    for i in range(99):
        observe(tracker._calibrator, i)
    tracker._register_block_size(16, "worker")
    tracker._track_request_state("last", 1, 100, None, 100)
    tracker._calibration_pending["last"] = dict(workload_tokens=990)
    await tracker.on_prefill_complete("last", 1985)
    assert sink.call_args.args[0][PREFIX + "sample_count"] == 100
    assert sink.call_args.args[0][PREFIX + "version"] == 1
    tracker.close_calibration()


def test_native_queue_time_once():
    forwarder = Mock()
    tracker = RequestTokenTracker(forwarder, "r")
    output = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[1])],
        metrics=SimpleNamespace(queued_ts=1, scheduled_ts=1.25),
    )
    tracker.on_output(output)
    tracker.on_output(output)
    forwarder.report.assert_called_once_with("on_prefill_complete", "r", 250)


def context(models):
    raw = {"outstanding_effective_prefill_tokens": {}}
    for source, fields in models.items():
        raw["outstanding_effective_prefill_tokens"][source] = [
            SimpleNamespace(timestamp=99, value=1000)
        ]
        for key, value in fields.items():
            raw.setdefault(PREFIX + key, {})[source] = [
                SimpleNamespace(timestamp=99, value=value)
            ]
    return SimpleNamespace(
        raw_router_metrics=raw,
        current_time=100,
        config=SimpleNamespace(look_back_period_s=60),
        target_num_replicas=7,
        total_queued_requests=0,
    )


def test_policy_holds_then_inverts_conservatively():
    policy = LLMTokenAutoscalingPolicy(target_queue_time_ms=100)
    assert policy(context({"a": {}}))[0] == 7
    fields = dict(
        intercept_ms=0,
        slope_ms_per_token=0.1,
        sample_count=100,
        updated_at_s=99,
        version=1,
    )
    other = dict(fields, slope_ms_per_token=0.2)
    replicas, state = policy(context({"a": fields, "b": other}))
    assert replicas == 2
    assert state["__policy_metrics"]["learned_prefill_tokens_per_replica"] == 500
    assert policy(context({"a": dict(fields, intercept_ms=200)}))[0] == 7


class LocalTracker(KVTokenTracker):
    def _create_selection_service(self):
        self._svc = SimpleNamespace(
            prefill_complete=AsyncMock(), free_reservation=AsyncMock()
        )

    def _start_replica_tracking(self):
        pass


@pytest.mark.asyncio
async def test_tracker_worker_accounting_and_pair_once():
    tracker = LocalTracker(enable_prefill_queue_learning=True)
    tracker._register_block_size(16, "worker")
    tracker._track_request_state("a", 1, 100, None, 100)
    tracker._track_request_state("b", 2, 200, None, 200)
    tracker._calibration_pending["a"] = dict(
        workload_tokens=0, worker_id=1, routed_at_s=1, prompt_tokens=100
    )
    await tracker.on_prefill_complete("a", 3)
    await tracker.on_prefill_complete("a", 3)
    assert len(tracker._calibrator.samples) == 1
    assert tracker._worker_prefill_tokens == {1: 0, 2: 200}
    await tracker.on_request_completed("b")
    assert tracker._worker_prefill_tokens[2] == 0


@pytest.mark.asyncio
async def test_selection_freezes_only_selected_worker_and_skips_mirrors(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RAY_SERVE_LLM_PREFILL_CALIBRATION_TRACE_DIR", str(tmp_path))
    tracker = LocalTracker(enable_prefill_queue_learning=True)
    tracker._register_block_size(16, "worker")
    tracker._track_request_state("existing", 1, 100, None, 100)
    tracker._track_request_state("other", 2, 500, None, 500)
    tracker._svc.select_and_reserve = AsyncMock(
        return_value={
            "worker_id": 1,
            "dp_rank": 0,
            "overlap": {"longest_matched": 0},
            "effective_prefill_tokens": 40,
        }
    )
    await tracker.select_worker("incoming", list(range(40)), [1, 2])
    assert tracker._calibration_pending["incoming"]["workload_tokens"] == 100
    assert (
        tracker._calibration_pending["incoming"]["router_effective_prefill_tokens"]
        == 40
    )
    tracker._track_request_state("later", 1, 200, None, 200)
    await tracker.on_prefill_complete("incoming", 25)
    assert tracker._calibrator.samples[0]["workload_tokens"] == 100
    await tracker.on_prefill_complete("other", 50)
    assert len(tracker._calibrator.samples) == 1
    assert (
        tracker._get_autoscaling_snapshot()["outstanding_effective_prefill_tokens"]
        == 300
    )
    tracker.close_calibration()
    events = [
        json.loads(line)
        for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()
    ]
    routed = next(e for e in events if e["event"] == "request_routed")
    assert routed["request_id"] == "incoming"
    assert routed["workload_tokens"] == 100
    assert routed["router_effective_prefill_tokens"] == 40
