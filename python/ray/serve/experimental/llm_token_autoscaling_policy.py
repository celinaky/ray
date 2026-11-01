import math
import time
from typing import Any, Dict, List, Literal, Tuple

from ray.serve._private.constants import SERVE_AUTOSCALING_POLICY_METRICS_KEY
from ray.serve.config import AutoscalingContext

OUTSTANDING_EFFECTIVE_PREFILL_TOKENS = "outstanding_effective_prefill_tokens"
PREFILL_ARRIVAL_AHEAD_TOKENS = "prefill_arrival_ahead_tokens"

# Policy metrics exported by the controller as serve_autoscaling_policy_metric.
PREFILL_REQUIRED_REPLICAS = "prefill_required_replicas"
PREFILL_ARRIVAL_MEAN_TOKENS = "prefill_arrival_mean_tokens"
PREFILL_ARRIVAL_SAMPLES = "prefill_arrival_samples"
REQUIRED_REPLICAS = "required_replicas"

_UpscaleAggregation = Literal["max", "mean"]
_Snapshot = Tuple[float, float]


def _now(ctx: AutoscalingContext) -> float:
    return ctx.current_time if ctx.current_time is not None else time.time()


def _fresh_snapshots_by_source(
    ctx: AutoscalingContext,
) -> Dict[str, List[_Snapshot]]:
    """Return fresh ``(timestamp, prefill)`` series per reporting source."""
    raw_metrics = ctx.raw_router_metrics or {}
    prefill_by_source = raw_metrics.get(OUTSTANDING_EFFECTIVE_PREFILL_TOKENS, {})
    cutoff = _now(ctx) - ctx.config.look_back_period_s
    snapshots: Dict[str, List[_Snapshot]] = {}

    for source_id, prefill_series in prefill_by_source.items():
        if not prefill_series:
            continue
        if prefill_series[-1].timestamp < cutoff:
            continue

        source_snapshots = [
            (sample.timestamp, sample.value)
            for sample in prefill_series
            if sample.timestamp >= cutoff
        ]
        if source_snapshots:
            snapshots[source_id] = source_snapshots

    return snapshots


class LLMTokenAutoscalingPolicy:
    """Scale an aggregated LLM deployment on outstanding uncached prefill tokens.

    Prefill capacity follows the request-weighted mean of uncached prefill tokens
    already ahead on the selected worker when each request arrived. Each router
    owns a disjoint subset of those arrival observations, so their samples are
    combined before taking the mean.

    The prefill arrival mean is a per-replica load observation. Its ratio to
    ``target_prefill_tokens_per_replica`` is therefore a scaling factor applied
    to the current replica count, not an absolute replica count.

    ``upscale_aggregation`` and ``scale_down_sensitivity`` are validated but
    unused by the arrival-mean policy.
    """

    def __init__(
        self,
        target_prefill_tokens_per_replica: int,
        upscale_lookback_s: float = 30,
        min_arrival_samples: int = 50,
        upscale_aggregation: _UpscaleAggregation = "max",
        scale_down_sensitivity: float = 0.8,
    ):
        if target_prefill_tokens_per_replica <= 0:
            raise ValueError("target_prefill_tokens_per_replica must be positive")
        if min_arrival_samples < 1:
            raise ValueError("min_arrival_samples must be at least 1")
        if not math.isfinite(upscale_lookback_s) or upscale_lookback_s <= 0:
            raise ValueError("upscale_lookback_s must be positive and finite")
        if upscale_aggregation not in ("max", "mean"):
            raise ValueError("upscale_aggregation must be 'max' or 'mean'")
        if (
            not math.isfinite(scale_down_sensitivity)
            or scale_down_sensitivity <= 0
            or scale_down_sensitivity > 1
        ):
            raise ValueError("scale_down_sensitivity must be in the interval (0, 1]")

        self._target_prefill_tokens_per_replica = target_prefill_tokens_per_replica
        self._upscale_lookback_s = upscale_lookback_s
        self._min_arrival_samples = min_arrival_samples
        self._upscale_aggregation = upscale_aggregation
        self._scale_down_sensitivity = scale_down_sensitivity

    def _prefill_arrival_mean(
        self, ctx: AutoscalingContext, now: float
    ) -> Tuple[float, int]:
        """Return the mean and count of fresh router-local arrival samples."""
        window_start = now - self._upscale_lookback_s
        raw_metrics = ctx.raw_router_metrics or {}
        samples = [
            max(0, point.value)
            for series in raw_metrics.get(PREFILL_ARRIVAL_AHEAD_TOKENS, {}).values()
            for point in series
            if point.timestamp >= window_start
        ]
        if not samples:
            return 0.0, 0

        return sum(samples) / len(samples), len(samples)

    def __call__(self, ctx: AutoscalingContext) -> Tuple[int, Dict[str, Any]]:
        if self._upscale_lookback_s > ctx.config.look_back_period_s:
            raise ValueError(
                "upscale_lookback_s must not exceed the autoscaling config's "
                "look_back_period_s"
            )

        snapshots_by_source = _fresh_snapshots_by_source(ctx)
        # A missing or stale observation is not evidence of zero work.
        if not snapshots_by_source:
            return ctx.target_num_replicas, {}

        now = _now(ctx)
        window_start = now - self._upscale_lookback_s
        if not any(
            timestamp >= window_start
            for snapshots in snapshots_by_source.values()
            for timestamp, _ in snapshots
        ):
            # No observation in the lookback window is also fail-closed.
            return ctx.target_num_replicas, {}

        prefill_arrival_mean, sample_count = self._prefill_arrival_mean(ctx, now)
        prefill_required_replicas = (
            ctx.current_num_replicas
            * prefill_arrival_mean
            / self._target_prefill_tokens_per_replica
        )
        if sample_count < self._min_arrival_samples:
            # Too few arrivals to trust the aggregate: hold the current count.
            required_replicas = ctx.target_num_replicas
        else:
            required_replicas = math.ceil(prefill_required_replicas)
        if ctx.total_queued_requests > 0:
            required_replicas = max(1, required_replicas)

        latest_snapshots = [snapshots[-1] for snapshots in snapshots_by_source.values()]
        outstanding_prefill_tokens = max(
            max(0, prefill) for _, prefill in latest_snapshots
        )
        if outstanding_prefill_tokens > 0:
            required_replicas = max(1, required_replicas)
        state: Dict[str, Any] = {
            OUTSTANDING_EFFECTIVE_PREFILL_TOKENS: outstanding_prefill_tokens,
            PREFILL_ARRIVAL_MEAN_TOKENS: prefill_arrival_mean,
            PREFILL_ARRIVAL_SAMPLES: sample_count,
            REQUIRED_REPLICAS: required_replicas,
            "lookback_s": self._upscale_lookback_s,
            "min_arrival_samples": self._min_arrival_samples,
            SERVE_AUTOSCALING_POLICY_METRICS_KEY: {
                OUTSTANDING_EFFECTIVE_PREFILL_TOKENS: outstanding_prefill_tokens,
                PREFILL_REQUIRED_REPLICAS: prefill_required_replicas,
                PREFILL_ARRIVAL_MEAN_TOKENS: prefill_arrival_mean,
                PREFILL_ARRIVAL_SAMPLES: sample_count,
                REQUIRED_REPLICAS: required_replicas,
            },
        }
        return required_replicas, state
