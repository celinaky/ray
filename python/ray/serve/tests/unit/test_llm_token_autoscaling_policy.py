from types import SimpleNamespace

import pytest

from ray.serve._private.common import TimeStampedValue
from ray.serve._private.constants import SERVE_AUTOSCALING_POLICY_METRICS_KEY
from ray.serve.experimental.llm_token_autoscaling_policy import (
    ACTIVE_DECODE_REQUESTS,
    DECODE_REQUIRED_REPLICAS,
    OUTSTANDING_EFFECTIVE_PREFILL_TOKENS,
    PREFILL_ARRIVAL_AHEAD_TOKENS,
    PREFILL_ARRIVAL_P95_TOKENS,
    PREFILL_REQUIRED_REPLICAS,
    REQUIRED_REPLICAS,
    LLMTokenAutoscalingPolicy,
)


def _context(
    *,
    prefill_by_replica=None,
    decode_by_replica=None,
    prefill_arrivals_by_router=None,
    now=100.0,
    current_num_replicas=None,
    target_num_replicas=2,
    look_back_period_s=30,
):
    replica_ids = {*(prefill_by_replica or {}), *(decode_by_replica or {})}

    def to_series(values):
        series = {}
        for replica_id, samples in values.items():
            if isinstance(samples, tuple):
                samples = [samples]
            series[replica_id] = [
                TimeStampedValue(timestamp=timestamp, value=value)
                for timestamp, value in samples
            ]
        return series

    return SimpleNamespace(
        raw_router_metrics={
            OUTSTANDING_EFFECTIVE_PREFILL_TOKENS: to_series(prefill_by_replica or {}),
            ACTIVE_DECODE_REQUESTS: to_series(decode_by_replica or {}),
            PREFILL_ARRIVAL_AHEAD_TOKENS: to_series(prefill_arrivals_by_router or {}),
        },
        current_time=now,
        config=SimpleNamespace(look_back_period_s=look_back_period_s),
        running_replicas=list(replica_ids),
        current_num_replicas=(
            target_num_replicas
            if current_num_replicas is None
            else current_num_replicas
        ),
        target_num_replicas=target_num_replicas,
        total_queued_requests=0,
    )


def test_scales_from_max_global_snapshot_across_reporters():
    policy = LLMTokenAutoscalingPolicy(
        target_prefill_tokens_per_replica=100,
        target_decode_requests_per_replica=4,
    )
    ctx = _context(
        prefill_by_replica={"a": (99, 250), "b": (99, 200)},
        decode_by_replica={"a": (99, 3), "b": (99, 9)},
    )

    replicas, state = policy(ctx)

    assert replicas == 3
    assert state[OUTSTANDING_EFFECTIVE_PREFILL_TOKENS] == 250
    assert state[ACTIVE_DECODE_REQUESTS] == 9
    assert state["required_replicas"] == 3


def test_scales_to_zero_when_idle():
    policy = LLMTokenAutoscalingPolicy(100, 4)

    replicas, _ = policy(
        _context(
            prefill_by_replica={"a": (99, 0)},
            decode_by_replica={"a": (99, 0)},
            target_num_replicas=1,
        )
    )

    assert replicas == 0


def test_keeps_one_replica_when_requests_are_queued():
    policy = LLMTokenAutoscalingPolicy(100, 4)
    ctx = _context(
        prefill_by_replica={"a": (99, 0)},
        decode_by_replica={"a": (99, 0)},
    )
    ctx.total_queued_requests = 1

    replicas, _ = policy(ctx)

    assert replicas == 1


@pytest.mark.parametrize(
    "prefill_by_replica, decode_by_replica",
    [
        ({}, {"a": (99, 2)}),
        ({"a": (99, 100)}, {}),
        ({"a": (60, 100)}, {"a": (60, 2)}),
    ],
)
def test_holds_target_for_partial_or_stale_snapshot(
    prefill_by_replica, decode_by_replica
):
    policy = LLMTokenAutoscalingPolicy(100, 4)

    replicas, state = policy(
        _context(
            prefill_by_replica=prefill_by_replica,
            decode_by_replica=decode_by_replica,
            target_num_replicas=7,
        )
    )

    assert replicas == 7
    assert state == {}


def test_holds_target_for_mismatched_snapshot_timestamps():
    policy = LLMTokenAutoscalingPolicy(100, 4)

    replicas, state = policy(
        _context(
            prefill_by_replica={"a": (98, 100)},
            decode_by_replica={"a": (99, 2)},
            target_num_replicas=7,
        )
    )

    assert replicas == 7
    assert state == {}


@pytest.mark.skip(reason="Asymmetric rule is disabled.")
def test_max_aggregation_retains_then_expires_peak():
    policy = LLMTokenAutoscalingPolicy(
        100,
        4,
        upscale_lookback_s=30,
        upscale_aggregation="max",
    )
    metrics = {
        "a": [(75, 700), (99, 100)],
    }
    decode = {
        "a": [(75, 0), (99, 0)],
    }

    replicas, state = policy(
        _context(
            prefill_by_replica=metrics,
            decode_by_replica=decode,
            target_num_replicas=1,
        )
    )
    assert replicas == 7
    assert state["required_replicas"] == 7

    replicas, state = policy(
        _context(
            prefill_by_replica=metrics,
            decode_by_replica=decode,
            now=106,
            target_num_replicas=1,
            look_back_period_s=60,
        )
    )
    assert replicas == 1
    assert state["required_replicas"] == 1


def test_prefill_p95_concatenates_arrivals_across_routers():
    policy = LLMTokenAutoscalingPolicy(100, 4)
    arrivals = {
        "router-a": [(80 + i, 10) for i in range(10)],
        "router-b": [
            *[(90 + i, 10) for i in range(8)],
            (98, 150),
            (99, 1000),
        ],
    }

    replicas, state = policy(
        _context(
            prefill_by_replica={"a": (99, 1)},
            decode_by_replica={"a": (99, 0)},
            prefill_arrivals_by_router=arrivals,
            current_num_replicas=2,
            target_num_replicas=2,
        )
    )

    # Nearest-rank p95 of 20 samples is the 19th value (150), not the max.
    assert state[PREFILL_ARRIVAL_P95_TOKENS] == 150
    assert state[SERVE_AUTOSCALING_POLICY_METRICS_KEY][
        PREFILL_REQUIRED_REPLICAS
    ] == pytest.approx(3)
    assert replicas == 3


def test_prefill_p95_uses_current_replica_count_as_scaling_basis():
    policy = LLMTokenAutoscalingPolicy(100, 4)

    replicas, _ = policy(
        _context(
            prefill_by_replica={"a": (99, 1)},
            decode_by_replica={"a": (99, 0)},
            prefill_arrivals_by_router={"router": (99, 80)},
            current_num_replicas=3,
            target_num_replicas=3,
        )
    )

    assert replicas == 3  # ceil(3 * 80 / 100)


def test_prefill_p95_ignores_arrivals_before_window():
    policy = LLMTokenAutoscalingPolicy(100, 4, upscale_lookback_s=30)

    replicas, state = policy(
        _context(
            prefill_by_replica={"a": (99, 1)},
            decode_by_replica={"a": (99, 0)},
            prefill_arrivals_by_router={"router": [(69, 1000), (99, 20)]},
            current_num_replicas=3,
            target_num_replicas=3,
            look_back_period_s=60,
        )
    )

    assert state[PREFILL_ARRIVAL_P95_TOKENS] == 20
    assert replicas == 1


def test_decode_mean_remains_time_weighted():
    policy = LLMTokenAutoscalingPolicy(100, 4, upscale_lookback_s=30)

    replicas, state = policy(
        _context(
            prefill_by_replica={"a": [(70, 0), (90, 0), (95, 0)]},
            decode_by_replica={"a": [(70, 12), (90, 28), (95, 12)]},
            current_num_replicas=1,
            target_num_replicas=1,
        )
    )

    assert state[SERVE_AUTOSCALING_POLICY_METRICS_KEY][
        DECODE_REQUIRED_REPLICAS
    ] == pytest.approx(11 / 3)
    assert replicas == 4


def test_holds_target_when_no_complete_snapshot_is_in_upscale_window():
    policy = LLMTokenAutoscalingPolicy(100, 4, upscale_lookback_s=30)

    replicas, state = policy(
        _context(
            prefill_by_replica={"a": (65, 100)},
            decode_by_replica={"a": (65, 0)},
            target_num_replicas=7,
            look_back_period_s=60,
        )
    )

    assert replicas == 7
    assert state == {}


@pytest.mark.skip(reason="Asymmetric rule is disabled.")
def test_scale_down_removes_at_most_one_replica_when_survivors_are_below_target():
    policy = LLMTokenAutoscalingPolicy(100, 4, scale_down_sensitivity=0.8)

    replicas, state = policy(
        _context(
            prefill_by_replica={"a": (99, 479)},
            decode_by_replica={"a": (99, 19)},
            target_num_replicas=7,
        )
    )

    assert replicas == 6
    assert state["projected_prefill_utilization"] == pytest.approx(479 / 600)
    assert state["projected_decode_utilization"] == pytest.approx(19 / 24)
    assert state["scale_down_blocked"] is False


@pytest.mark.skip(reason="Asymmetric rule is disabled.")
@pytest.mark.parametrize(
    "prefill, decode",
    [
        (480, 0),
        (0, 19.2),
    ],
)
def test_scale_down_requires_strictly_less_than_sensitivity(prefill, decode):
    policy = LLMTokenAutoscalingPolicy(100, 4, scale_down_sensitivity=0.8)

    replicas, state = policy(
        _context(
            prefill_by_replica={"a": (99, prefill)},
            decode_by_replica={"a": (99, decode)},
            target_num_replicas=7,
        )
    )

    assert replicas == 7
    assert state["scale_down_blocked"] is True


@pytest.mark.skip(reason="Asymmetric rule is disabled.")
def test_every_reporter_must_be_safe_to_scale_down():
    policy = LLMTokenAutoscalingPolicy(100, 4, scale_down_sensitivity=0.8)

    replicas, state = policy(
        _context(
            prefill_by_replica={"a": (99, 100), "b": (99, 480)},
            decode_by_replica={"a": (99, 0), "b": (99, 0)},
            target_num_replicas=7,
        )
    )

    assert replicas == 7
    assert state["scale_down_blocked"] is True


@pytest.mark.skip(reason="Asymmetric rule is disabled.")
def test_scales_down_only_one_replica_when_idle():
    policy = LLMTokenAutoscalingPolicy(100, 4)

    replicas, _ = policy(
        _context(
            prefill_by_replica={"a": (99, 0)},
            decode_by_replica={"a": (99, 0)},
            target_num_replicas=7,
        )
    )

    assert replicas == 6


def test_scales_down_to_arrival_p95():
    policy = LLMTokenAutoscalingPolicy(100, 4, upscale_lookback_s=30)

    replicas, state = policy(
        _context(
            prefill_by_replica={"a": (99, 1)},
            decode_by_replica={"a": (99, 0)},
            prefill_arrivals_by_router={"router": (99, 25)},
            current_num_replicas=7,
            target_num_replicas=7,
        )
    )

    # ceil(7 current replicas * 25 arrival p95 / 100 target) = 2.
    assert replicas == 2
    assert state["required_replicas"] == 2


def test_publishes_prefill_p95_and_decode_mean():
    policy = LLMTokenAutoscalingPolicy(100, 4, upscale_lookback_s=30)

    replicas, state = policy(
        _context(
            prefill_by_replica={"a": [(70, 0), (85, 0)]},
            decode_by_replica={"a": [(70, 0), (85, 16)]},
            prefill_arrivals_by_router={"router": (99, 150)},
            current_num_replicas=1,
            target_num_replicas=1,
        )
    )

    # Prefill asks for 1.5 replicas and decode's lookback mean asks for 2.
    assert replicas == 2
    assert state[SERVE_AUTOSCALING_POLICY_METRICS_KEY] == {
        PREFILL_REQUIRED_REPLICAS: pytest.approx(1.5),
        PREFILL_ARRIVAL_P95_TOKENS: 150,
        DECODE_REQUIRED_REPLICAS: pytest.approx(2.0),
        REQUIRED_REPLICAS: 2,
    }


def test_proposes_zero_when_lookback_window_is_idle():
    policy = LLMTokenAutoscalingPolicy(100, 4)

    replicas, _ = policy(
        _context(
            prefill_by_replica={"a": (99, 0)},
            decode_by_replica={"a": (99, 0)},
            target_num_replicas=7,
        )
    )

    # Serve's delay logic floors this at 1 unless the current target is 1.
    assert replicas == 0


@pytest.mark.parametrize("prefill_target, decode_target", [(0, 1), (1, 0)])
def test_rejects_nonpositive_targets(prefill_target, decode_target):
    with pytest.raises(ValueError, match="must be positive"):
        LLMTokenAutoscalingPolicy(prefill_target, decode_target)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"upscale_lookback_s": 0}, "upscale_lookback_s"),
        ({"upscale_lookback_s": float("inf")}, "upscale_lookback_s"),
        ({"upscale_aggregation": "median"}, "upscale_aggregation"),
        ({"scale_down_sensitivity": 0}, "scale_down_sensitivity"),
        ({"scale_down_sensitivity": 1.1}, "scale_down_sensitivity"),
    ],
)
def test_rejects_invalid_damping_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LLMTokenAutoscalingPolicy(100, 4, **kwargs)


def test_rejects_upscale_window_larger_than_retained_history():
    policy = LLMTokenAutoscalingPolicy(100, 4, upscale_lookback_s=31)

    with pytest.raises(ValueError, match="look_back_period_s"):
        policy(
            _context(
                prefill_by_replica={"a": (99, 100)},
                decode_by_replica={"a": (99, 0)},
                look_back_period_s=30,
            )
        )
