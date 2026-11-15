import math
from typing import Any, Dict, Tuple

from ray.serve.config import AutoscalingContext

OUTSTANDING_EFFECTIVE_PREFILL_TOKENS = "outstanding_effective_prefill_tokens"
REQUIRED_REPLICAS = "required_replicas"


class LLMTokenAutoscalingPolicy:
    """Scale an LLM deployment based on outstanding uncached prefill tokens.

    Serve aggregates each KV router's reported tokens over ``look_back_period_s``.
    Each router reports the same global workload, so the policy takes the maximum
    across routers to conservatively reconcile their replicated global views.

    Args:
        target_prefill_tokens_per_replica: Target mean outstanding uncached
            prefill tokens per replica.
        post_downscale_cooldown_s: Hold the current replica target for this many
            seconds after Serve accepts a downscale. Defaults to zero (disabled).
            When enabled, both autoscaling factors must be one so Serve preserves
            the held target while replicas are draining.
    """

    def __init__(
        self,
        target_prefill_tokens_per_replica: int,
        post_downscale_cooldown_s: float = 0,
    ):
        if (
            not math.isfinite(target_prefill_tokens_per_replica)
            or target_prefill_tokens_per_replica <= 0
        ):
            raise ValueError(
                "target_prefill_tokens_per_replica must be positive and finite"
            )
        if (
            not math.isfinite(post_downscale_cooldown_s)
            or post_downscale_cooldown_s < 0
        ):
            raise ValueError("post_downscale_cooldown_s must be nonnegative and finite")

        self._target_prefill_tokens_per_replica = target_prefill_tokens_per_replica
        self._post_downscale_cooldown_s = post_downscale_cooldown_s

    def _in_cooldown(self, ctx: AutoscalingContext) -> bool:
        return (
            self._post_downscale_cooldown_s > 0
            and ctx.last_scale_down_time is not None
            and ctx.current_time - ctx.last_scale_down_time
            < self._post_downscale_cooldown_s
        )

    def __call__(self, ctx: AutoscalingContext) -> Tuple[int, Dict[str, Any]]:
        if self._post_downscale_cooldown_s > 0 and (
            ctx.config.get_upscaling_factor() != 1
            or ctx.config.get_downscaling_factor() != 1
        ):
            raise ValueError(
                "post_downscale_cooldown_s requires upscaling and downscaling "
                "factors of 1"
            )

        prefill_tokens_by_handle = (ctx.aggregated_router_metrics or {}).get(
            OUTSTANDING_EFFECTIVE_PREFILL_TOKENS, {}
        )
        if not prefill_tokens_by_handle:
            if ctx.total_num_requests > 0:
                return max(1, ctx.target_num_replicas), {}
            return ctx.target_num_replicas, {}

        # The KV Router dashboard panel mirrors this max; update it if this changes.
        prefill_tokens = max(0, max(prefill_tokens_by_handle.values()))
        required_replicas = math.ceil(
            prefill_tokens / self._target_prefill_tokens_per_replica
        )
        # Keep the accepted target, not the running count: draining replicas can
        # still be running after Serve has committed a downscale.
        decision_replicas = (
            ctx.target_num_replicas if self._in_cooldown(ctx) else required_replicas
        )
        if ctx.total_num_requests > 0:
            required_replicas = max(1, required_replicas)
            decision_replicas = max(1, decision_replicas)

        return decision_replicas, {
            OUTSTANDING_EFFECTIVE_PREFILL_TOKENS: prefill_tokens,
            REQUIRED_REPLICAS: required_replicas,
        }
