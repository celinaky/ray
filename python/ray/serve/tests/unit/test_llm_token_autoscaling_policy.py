import math
import sys

import pytest

from ray.serve._private.common import DeploymentID
from ray.serve._private.constants import (
    SERVE_AUTOSCALING_DECISION_COUNTERS_KEY,
    SERVE_AUTOSCALING_DECISION_TIMESTAMP_KEY,
)
from ray.serve.autoscaling_policy import _apply_autoscaling_config
from ray.serve.config import AutoscalingConfig, AutoscalingContext
from ray.serve.experimental.llm_token_autoscaling_policy import (
    OUTSTANDING_EFFECTIVE_PREFILL_TOKENS,
    REQUIRED_REPLICAS,
    LLMTokenAutoscalingPolicy,
)

_UNSET = object()


class TestLLMTokenAutoscalingPolicy:
    """Tests for LLMTokenAutoscalingPolicy, which scales on the uncached prefill
    tokens reported by KV routers."""

    def _make_ctx(
        self,
        *,
        prefill_by_handle=None,
        aggregated_router_metrics=_UNSET,
        target_num_replicas=2,
        total_num_requests=0,
        current_num_replicas=None,
        config=None,
        current_time=100,
        last_scale_down_time=None,
        policy_state=None,
    ):
        if aggregated_router_metrics is _UNSET:
            aggregated_router_metrics = {
                OUTSTANDING_EFFECTIVE_PREFILL_TOKENS: prefill_by_handle or {},
            }
        return AutoscalingContext(
            deployment_id=DeploymentID(name="test", app_name="app"),
            deployment_name="test",
            app_name="app",
            current_num_replicas=(
                target_num_replicas
                if current_num_replicas is None
                else current_num_replicas
            ),
            target_num_replicas=target_num_replicas,
            running_replicas=[],
            total_num_requests=total_num_requests,
            total_queued_requests=0,
            aggregated_metrics=None,
            raw_metrics=None,
            capacity_adjusted_min_replicas=0,
            capacity_adjusted_max_replicas=100,
            policy_state=policy_state or {},
            last_scale_up_time=None,
            last_scale_down_time=last_scale_down_time,
            current_time=current_time,
            config=config or AutoscalingConfig(min_replicas=0, max_replicas=100),
            aggregated_router_metrics=aggregated_router_metrics,
        )

    def test_scales_from_aggregated_tokens(self):
        """Requests ceil(tokens / target) replicas and returns the inputs as state."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, state = policy(
            self._make_ctx(prefill_by_handle={"a": 350.0}, target_num_replicas=1)
        )

        assert replicas == 4
        assert state == {
            OUTSTANDING_EFFECTIVE_PREFILL_TOKENS: 350.0,
            REQUIRED_REPLICAS: 4,
        }

    def test_takes_max_across_handles(self):
        """Uses the largest handle's tokens, since each router sees the global load."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, state = policy(
            self._make_ctx(
                prefill_by_handle={"a": 200.0, "b": 300.0}, target_num_replicas=1
            )
        )

        assert replicas == 3
        assert state[OUTSTANDING_EFFECTIVE_PREFILL_TOKENS] == 300.0

    @pytest.mark.parametrize("outstanding_tokens", [0, 100])
    def test_scales_down(self, outstanding_tokens):
        """Scales below the current target when tokens drop, including to zero."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, state = policy(
            self._make_ctx(
                prefill_by_handle={"a": float(outstanding_tokens)},
                target_num_replicas=7,
            )
        )

        assert replicas == math.ceil(outstanding_tokens / 100)
        assert state[REQUIRED_REPLICAS] == math.ceil(outstanding_tokens / 100)

    @pytest.mark.parametrize("aggregated_router_metrics", [None, {}, {"other": {}}])
    def test_holds_target_for_missing_metric(self, aggregated_router_metrics):
        """Keeps the current target when no router has reported the metric."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, state = policy(
            self._make_ctx(
                aggregated_router_metrics=aggregated_router_metrics,
                target_num_replicas=7,
            )
        )

        assert replicas == 7
        assert state == {}

    @pytest.mark.parametrize("target_num_replicas", [0, 3])
    def test_scales_from_zero_for_requests_without_metric(self, target_num_replicas):
        """Starts at least one replica for requests before any router reports."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, state = policy(
            self._make_ctx(
                aggregated_router_metrics={},
                target_num_replicas=target_num_replicas,
                total_num_requests=2,
            )
        )

        assert replicas == max(1, target_num_replicas)
        assert state == {}

    def test_keeps_one_replica_when_requests_are_in_flight(self):
        """Never scales to zero while requests are in flight."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, state = policy(
            self._make_ctx(
                prefill_by_handle={"a": 0.0},
                target_num_replicas=0,
                total_num_requests=1,
            )
        )

        assert replicas == 1
        assert state[REQUIRED_REPLICAS] == 1

    def test_clamps_negative_tokens_to_zero(self):
        """Treats a negative token count as zero."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, state = policy(
            self._make_ctx(prefill_by_handle={"a": -50.0}, target_num_replicas=0)
        )

        assert replicas == 0
        assert state[OUTSTANDING_EFFECTIVE_PREFILL_TOKENS] == 0

    @pytest.mark.parametrize("target", [0, -1, float("nan"), float("inf")])
    def test_rejects_invalid_target(self, target):
        """Rejects a target that is not positive and finite."""
        with pytest.raises(ValueError, match="positive and finite"):
            LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=target)

    @pytest.mark.parametrize("tokens", [0, 1000])
    @pytest.mark.parametrize("running_replicas", [2, 3, 4])
    def test_cooldown_holds_accepted_target_and_reports_workload(
        self, tokens, running_replicas
    ):
        """Holds the accepted target after a downscale but still reports workload."""
        policy = LLMTokenAutoscalingPolicy(
            target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=60
        )

        replicas, state = policy(
            self._make_ctx(
                prefill_by_handle={"a": float(tokens)},
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
    def test_cooldown_expires_without_extending_on_policy_calls(self, tokens, now):
        """Ends the hold at the deadline even though the policy runs every tick."""
        policy = LLMTokenAutoscalingPolicy(
            target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=60
        )

        for current_time in [100, 130, 159, now]:
            replicas, _ = policy(
                self._make_ctx(
                    prefill_by_handle={"a": float(tokens)},
                    target_num_replicas=3,
                    last_scale_down_time=100,
                    current_time=current_time,
                )
            )
            assert replicas == (3 if current_time < 160 else math.ceil(tokens / 100))

    @pytest.mark.parametrize("last_scale_down_time", [None, 0])
    def test_cooldown_does_not_start_on_running_replica_loss(
        self, last_scale_down_time
    ):
        """Losing a running replica without an accepted downscale starts no hold."""
        policy = LLMTokenAutoscalingPolicy(
            target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=60
        )

        replicas, _ = policy(
            self._make_ctx(
                prefill_by_handle={"a": 1000.0},
                target_num_replicas=3,
                current_num_replicas=2,
                last_scale_down_time=last_scale_down_time,
                current_time=100,
            )
        )

        assert replicas == 10

    def test_cooldown_uses_latest_downscale_timestamp(self):
        """Times the hold from the most recent accepted downscale."""
        policy = LLMTokenAutoscalingPolicy(
            target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=60
        )

        for last_downscale, target, expected in [(100, 3, 10), (160, 2, 2)]:
            replicas, _ = policy(
                self._make_ctx(
                    prefill_by_handle={"a": 1000.0},
                    target_num_replicas=target,
                    last_scale_down_time=last_downscale,
                    current_time=170,
                )
            )
            assert replicas == expected

    def test_cooldown_disabled_by_default(self):
        """Scales normally right after a downscale when no cooldown is set."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, _ = policy(
            self._make_ctx(prefill_by_handle={"a": 1000.0}, last_scale_down_time=100)
        )

        assert replicas == 10

    def test_cooldown_with_missing_metric(self):
        """Holds the target when no router has reported the metric."""
        policy = LLMTokenAutoscalingPolicy(
            target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=60
        )

        assert policy(
            self._make_ctx(
                aggregated_router_metrics={},
                target_num_replicas=3,
                last_scale_down_time=100,
            )
        ) == (3, {})

    @pytest.mark.parametrize("wrapped", [False, True])
    def test_cooldown_preserves_scale_from_zero(self, wrapped):
        """Starts a replica for queued requests even during the hold."""
        policy = LLMTokenAutoscalingPolicy(
            target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=60
        )
        if wrapped:
            policy = _apply_autoscaling_config(policy)

        replicas, _ = policy(
            self._make_ctx(
                prefill_by_handle={"a": 0.0},
                target_num_replicas=0,
                current_num_replicas=0,
                total_num_requests=1,
                last_scale_down_time=100,
            )
        )

        assert replicas == 1

    @pytest.mark.parametrize("tokens, expected", [(100, 1), (1000, 10)])
    def test_wrapped_policy_resets_delays_during_cooldown(
        self, monkeypatch, tokens, expected
    ):
        """Serve's delay timers restart after the hold instead of firing at its end."""
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
                self._make_ctx(
                    prefill_by_handle={"a": float(tokens)},
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
    def test_rejects_invalid_cooldown(self, duration):
        """Rejects a cooldown that is negative or not finite."""
        with pytest.raises(ValueError, match="nonnegative and finite"):
            LLMTokenAutoscalingPolicy(
                target_prefill_tokens_per_replica=100,
                post_downscale_cooldown_s=duration,
            )

    @pytest.mark.parametrize(
        "factor",
        [
            {"upscaling_factor": 2},
            {"downscaling_factor": 0.5},
            {"smoothing_factor": 0.5},
        ],
    )
    def test_cooldown_rejects_factors_that_can_change_held_target(self, factor):
        """Rejects scaling factors that would let Serve change the held target."""
        policy = LLMTokenAutoscalingPolicy(
            target_prefill_tokens_per_replica=100, post_downscale_cooldown_s=60
        )

        with pytest.raises(
            ValueError, match="requires upscaling and downscaling factors of 1"
        ):
            policy(self._make_ctx(config=AutoscalingConfig(**factor)))

    def test_disabled_cooldown_allows_custom_scaling_factors(self):
        """Allows any scaling factors when the cooldown is off."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, _ = policy(
            self._make_ctx(
                config=AutoscalingConfig(upscaling_factor=2, downscaling_factor=0.5),
                prefill_by_handle={"a": 1000.0},
                last_scale_down_time=100,
            )
        )

        assert replicas == 10

    @pytest.mark.parametrize(
        "tokens, target, step, expected",
        [
            (100, 4, 1, 3),  # asks for 1, steps down one
            (100, 4, 2, 2),  # asks for 1, steps down two
            (300, 4, 2, 3),  # asks for 3, already within the step
            (1000, 4, 1, 10),  # scale-up is not limited
            (400, 4, 1, 4),  # no change
        ],
    )
    def test_max_downscale_step_limits_each_downscale(
        self, tokens, target, step, expected
    ):
        """Asks for at most `step` fewer replicas than the current target."""
        policy = LLMTokenAutoscalingPolicy(
            target_prefill_tokens_per_replica=100, max_downscale_step=step
        )

        replicas, state = policy(
            self._make_ctx(
                prefill_by_handle={"a": float(tokens)}, target_num_replicas=target
            )
        )

        assert replicas == expected
        # The unlimited ask is still reported.
        assert state[REQUIRED_REPLICAS] == math.ceil(tokens / 100)

    def test_max_downscale_step_keeps_cooldown_hold(self):
        """The cooldown still holds the target while a step limit is set."""
        policy = LLMTokenAutoscalingPolicy(
            target_prefill_tokens_per_replica=100,
            post_downscale_cooldown_s=60,
            max_downscale_step=1,
        )

        replicas, _ = policy(
            self._make_ctx(
                prefill_by_handle={"a": 100.0},
                target_num_replicas=3,
                current_time=120,
                last_scale_down_time=100,
            )
        )

        assert replicas == 3

    def test_wrapped_max_downscale_step_removes_one_replica_per_delay(
        self, monkeypatch
    ):
        """Through Serve's delay logic, each accepted downscale removes one replica."""
        policy = _apply_autoscaling_config(
            LLMTokenAutoscalingPolicy(
                target_prefill_tokens_per_replica=100, max_downscale_step=1
            )
        )
        config = AutoscalingConfig(
            min_replicas=1, max_replicas=100, downscale_delay_s=10
        )
        state, target, history = {}, 4, []
        for now in range(100, 140, 2):
            monkeypatch.setattr("ray.serve.autoscaling_policy.time.time", lambda: now)
            replicas, state = policy(
                self._make_ctx(
                    prefill_by_handle={"a": 50.0},
                    config=config,
                    target_num_replicas=target,
                    current_time=now,
                    policy_state=state,
                )
            )
            if replicas != target:
                history.append((now, target, replicas))
                target = replicas

        assert [(a, b) for _, a, b in history] == [(4, 3), (3, 2), (2, 1)]
        times = [t for t, _, _ in history]
        assert all(b - a >= 10 for a, b in zip(times, times[1:]))

    @pytest.mark.parametrize("step", [0, -1, 1.5, True])
    def test_rejects_invalid_max_downscale_step(self, step):
        """Rejects a step that is not a positive integer."""
        with pytest.raises(ValueError, match="max_downscale_step"):
            LLMTokenAutoscalingPolicy(
                target_prefill_tokens_per_replica=100, max_downscale_step=step
            )


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
