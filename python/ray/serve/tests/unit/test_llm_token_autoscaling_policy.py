import math
import sys
from types import SimpleNamespace

import pytest

from ray.serve._private.common import TimeStampedValue
from ray.serve._private.constants import (
    SERVE_AUTOSCALING_DECISION_COUNTERS_KEY,
    SERVE_AUTOSCALING_DECISION_TIMESTAMP_KEY,
)
from ray.serve.autoscaling_policy import _apply_autoscaling_config
from ray.serve.config import AutoscalingConfig
from ray.serve.experimental.llm_token_autoscaling_policy import (
    OUTSTANDING_EFFECTIVE_PREFILL_TOKENS,
    REQUIRED_REPLICAS,
    LLMTokenAutoscalingPolicy,
)


def _context(
    *,
    prefill_by_handle=None,
    target_num_replicas=2,
    total_queued_requests=0,
    look_back_period_s=30,
    config=None,
    current_num_replicas=2,
    current_time=100,
    last_scale_down_time=None,
    policy_state=None,
):
    prefill_by_handle = prefill_by_handle or {}
    series_by_handle = {}
    for handle_id, samples in prefill_by_handle.items():
        if isinstance(samples, tuple):
            samples = [samples]
        series_by_handle[handle_id] = [
            TimeStampedValue(timestamp=timestamp, value=value)
            for timestamp, value in samples
        ]

    return SimpleNamespace(
        raw_router_metrics={
            OUTSTANDING_EFFECTIVE_PREFILL_TOKENS: series_by_handle,
        },
        config=config
        or AutoscalingConfig(
            min_replicas=0, max_replicas=100, look_back_period_s=look_back_period_s
        ),
        target_num_replicas=target_num_replicas,
        current_num_replicas=current_num_replicas,
        total_queued_requests=total_queued_requests,
        total_num_requests=total_queued_requests,
        current_time=current_time,
        last_scale_down_time=last_scale_down_time,
        policy_state=policy_state or {},
    )


def test_scales_from_time_weighted_mean():
    policy = LLMTokenAutoscalingPolicy(
        target_prefill_tokens_per_replica=100,
        upscale_lookback_s=30,
    )

    replicas, state = policy(
        _context(
            prefill_by_handle={"a": [(0, 300), (20, 700), (25, 300), (30, 300)]},
            target_num_replicas=1,
        )
    )

    assert replicas == 4
    assert state == {
        OUTSTANDING_EFFECTIVE_PREFILL_TOKENS: 300,
        REQUIRED_REPLICAS: 4,
    }


def test_takes_max_mean_across_replicated_histories():
    policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

    replicas, _ = policy(
        _context(
            prefill_by_handle={
                "a": [(0, 200), (30, 200)],
                "b": [(0, 300), (30, 300)],
            },
            target_num_replicas=1,
        )
    )

    assert replicas == 3


def test_router_clock_offsets_do_not_change_mean():
    policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

    replicas, _ = policy(
        _context(
            prefill_by_handle={
                "behind": [(-10_030, 100), (-10_015, 300), (-10_000, 300)],
                "ahead": [(9_970, 100), (9_985, 300), (10_000, 300)],
            },
            target_num_replicas=1,
        )
    )

    assert replicas == 2


def test_carries_gauge_across_window_boundary():
    policy = LLMTokenAutoscalingPolicy(
        target_prefill_tokens_per_replica=100,
        upscale_lookback_s=20,
    )

    replicas, _ = policy(
        _context(
            prefill_by_handle={"a": [(0, 300), (20, 0), (30, 0)]},
            target_num_replicas=1,
        )
    )

    assert replicas == 2


@pytest.mark.parametrize("outstanding_tokens", [0, 100])
def test_scales_down(outstanding_tokens):
    policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

    replicas, state = policy(
        _context(
            prefill_by_handle={
                "a": [(0, outstanding_tokens), (30, outstanding_tokens)]
            },
            target_num_replicas=7,
        )
    )

    assert replicas == math.ceil(outstanding_tokens / 100)
    assert state[REQUIRED_REPLICAS] == math.ceil(outstanding_tokens / 100)


@pytest.mark.parametrize("prefill_by_handle", [{}, {"a": []}])
def test_holds_target_for_missing_snapshot(prefill_by_handle):
    policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

    replicas, state = policy(
        _context(prefill_by_handle=prefill_by_handle, target_num_replicas=7)
    )

    assert replicas == 7
    assert state == {}


def test_keeps_one_replica_when_requests_are_queued():
    policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

    replicas, state = policy(
        _context(
            prefill_by_handle={"a": [(0, 0), (30, 0)]},
            target_num_replicas=0,
            total_queued_requests=1,
        )
    )

    assert replicas == 1
    assert state[REQUIRED_REPLICAS] == 1


def test_clamps_negative_snapshots_to_zero():
    policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

    replicas, state = policy(
        _context(
            prefill_by_handle={"a": [(0, -100), (30, -1)]},
            target_num_replicas=0,
        )
    )

    assert replicas == 0
    assert state[OUTSTANDING_EFFECTIVE_PREFILL_TOKENS] == 0


def test_rejects_lookback_larger_than_collected_history():
    policy = LLMTokenAutoscalingPolicy(
        target_prefill_tokens_per_replica=100,
        upscale_lookback_s=31,
    )

    with pytest.raises(ValueError, match="must not exceed"):
        policy(_context(prefill_by_handle={"a": (0, 100)}))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"target_prefill_tokens_per_replica": 0}, "must be positive"),
        (
            {
                "target_prefill_tokens_per_replica": 100,
                "upscale_lookback_s": 0,
            },
            "positive and finite",
        ),
    ],
)
def test_rejects_invalid_config(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LLMTokenAutoscalingPolicy(**kwargs)


@pytest.mark.parametrize("tokens", [0, 1000])
@pytest.mark.parametrize("running_replicas", [2, 3, 4])
def test_cooldown_holds_accepted_target_and_reports_workload(tokens, running_replicas):
    policy = LLMTokenAutoscalingPolicy(
        target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=60
    )
    replicas, state = policy(
        _context(
            prefill_by_handle={"a": (1_000_000, tokens)},
            target_num_replicas=3,
            current_num_replicas=running_replicas,
            last_scale_down_time=100,
            current_time=159,
        )
    )
    assert replicas == 3
    assert state[OUTSTANDING_EFFECTIVE_PREFILL_TOKENS] == tokens
    assert state[REQUIRED_REPLICAS] == math.ceil(tokens / 100)


@pytest.mark.parametrize("tokens", [0, 1000])
@pytest.mark.parametrize("now", [160, 200])
def test_cooldown_expires_without_extending_on_policy_calls(tokens, now):
    policy = LLMTokenAutoscalingPolicy(
        target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=60
    )
    for current_time in [100, 130, 159, now]:
        replicas, _ = policy(
            _context(
                prefill_by_handle={"a": (0, tokens)},
                target_num_replicas=3,
                last_scale_down_time=100,
                current_time=current_time,
            )
        )
        assert replicas == (3 if current_time < 160 else math.ceil(tokens / 100))


@pytest.mark.parametrize("last_scale_down_time", [None, 0])
def test_cooldown_does_not_start_on_running_replica_loss(last_scale_down_time):
    policy = LLMTokenAutoscalingPolicy(
        target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=60
    )
    replicas, _ = policy(
        _context(
            prefill_by_handle={"a": (0, 1000)},
            target_num_replicas=3,
            current_num_replicas=2,
            last_scale_down_time=last_scale_down_time,
            current_time=100,
        )
    )
    assert replicas == 10


def test_cooldown_uses_latest_downscale_timestamp():
    policy = LLMTokenAutoscalingPolicy(
        target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=60
    )
    for last_downscale, target, expected in [(100, 3, 10), (160, 2, 2)]:
        replicas, _ = policy(
            _context(
                prefill_by_handle={"a": (0, 1000)},
                target_num_replicas=target,
                last_scale_down_time=last_downscale,
                current_time=170,
            )
        )
        assert replicas == expected


def test_cooldown_disabled_by_default():
    policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)
    replicas, _ = policy(
        _context(prefill_by_handle={"a": (0, 1000)}, last_scale_down_time=100)
    )
    assert replicas == 10


def test_cooldown_with_missing_workload():
    policy = LLMTokenAutoscalingPolicy(
        target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=60
    )
    assert policy(_context(target_num_replicas=3, last_scale_down_time=100)) == (3, {})


@pytest.mark.parametrize("wrapped", [False, True])
def test_cooldown_preserves_scale_from_zero(wrapped):
    policy = LLMTokenAutoscalingPolicy(
        target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=60
    )
    if wrapped:
        policy = _apply_autoscaling_config(policy)
    replicas, _ = policy(
        _context(
            prefill_by_handle={"a": (0, 0)},
            target_num_replicas=0,
            current_num_replicas=0,
            total_queued_requests=1,
            last_scale_down_time=100,
        )
    )
    assert replicas == 1


@pytest.mark.parametrize("tokens, expected", [(100, 1), (1000, 10)])
def test_wrapped_policy_resets_delays_during_cooldown(monkeypatch, tokens, expected):
    policy = _apply_autoscaling_config(
        LLMTokenAutoscalingPolicy(
            target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=60
        )
    )
    config = AutoscalingConfig(
        min_replicas=0,
        max_replicas=100,
        upscale_delay_s=10,
        downscale_delay_s=10,
    )
    state = {
        SERVE_AUTOSCALING_DECISION_COUNTERS_KEY: 1,
        SERVE_AUTOSCALING_DECISION_TIMESTAMP_KEY: 100,
    }
    for now in [150, 159, 160, 169, 170]:
        monkeypatch.setattr("ray.serve.autoscaling_policy.time.time", lambda: now)
        replicas, state = policy(
            _context(
                prefill_by_handle={"a": (0, tokens)},
                config=config,
                current_num_replicas=4,
                target_num_replicas=3,
                current_time=now,
                last_scale_down_time=100,
                policy_state=state,
            )
        )
        assert replicas == (3 if now < 170 else expected)
        if now < 160:
            assert state[SERVE_AUTOSCALING_DECISION_COUNTERS_KEY] == 0
            assert state[SERVE_AUTOSCALING_DECISION_TIMESTAMP_KEY] is None


@pytest.mark.parametrize("duration", [-1, math.inf, -math.inf, math.nan])
def test_rejects_invalid_cooldown(duration):
    with pytest.raises(ValueError, match="nonnegative and finite"):
        LLMTokenAutoscalingPolicy(
            target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=duration
        )


@pytest.mark.parametrize(
    "factor",
    [
        {"upscaling_factor": 2},
        {"downscaling_factor": 0.5},
        {"smoothing_factor": 0.5},
    ],
)
def test_cooldown_rejects_factors_that_can_change_held_target(factor):
    policy = LLMTokenAutoscalingPolicy(
        target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=60
    )
    with pytest.raises(
        ValueError, match="requires upscaling and downscaling factors of 1"
    ):
        policy(_context(config=AutoscalingConfig(**factor)))


def test_disabled_cooldown_allows_custom_scaling_factors():
    policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)
    replicas, _ = policy(
        _context(
            config=AutoscalingConfig(upscaling_factor=2, downscaling_factor=0.5),
            prefill_by_handle={"a": (0, 1000)},
            last_scale_down_time=100,
        )
    )
    assert replicas == 10


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
