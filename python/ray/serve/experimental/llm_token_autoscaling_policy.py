import math
import time
from typing import Any, Dict, List, Literal, Tuple

from ray.serve._private.constants import SERVE_AUTOSCALING_POLICY_METRICS_KEY
from ray.serve.config import AutoscalingContext

OUTSTANDING_EFFECTIVE_PREFILL_TOKENS = "outstanding_effective_prefill_tokens"

# Policy metrics exported by the controller as serve_autoscaling_policy_metric.
PREFILL_REQUIRED_REPLICAS = "prefill_required_replicas"
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


def _time_weighted_mean(
    samples: List[Tuple[float, float]], window_start: float, window_end: float
) -> float:
    """Average a right-continuous gauge over its observed part of a window."""
    observed_start = max(window_start, samples[0][0])
    if window_end <= observed_start:
        return samples[-1][1]

    weighted_sum = 0.0
    for index, (timestamp, value) in enumerate(samples):
        interval_start = max(timestamp, observed_start)
        interval_end = samples[index + 1][0] if index + 1 < len(samples) else window_end
        interval_end = min(interval_end, window_end)
        if interval_end > interval_start:
            weighted_sum += value * (interval_end - interval_start)

    return weighted_sum / (window_end - observed_start)


class LLMTokenAutoscalingPolicy:
    """Scale an aggregated LLM deployment on outstanding uncached prefill tokens.

    Both directions follow the time-weighted mean over ``upscale_lookback_s``
    of the router's outstanding effective (uncached) prefill tokens divided by
    ``target_prefill_tokens_per_replica``. Router replicas report the same
    global snapshot, so per-reporter means are maximized rather than summed.
    The latest raw snapshot and the mean are published under
    ``SERVE_AUTOSCALING_POLICY_METRICS_KEY`` for the controller to export. Asymmetry comes from Serve's ``upscale_delay_s``
    and ``downscale_delay_s``.

    ``upscale_aggregation`` and ``scale_down_sensitivity`` are validated but
    unused: the policy always uses the lookback mean.
    """

    def __init__(
        self,
        target_prefill_tokens_per_replica: int,
        upscale_lookback_s: float = 30,
        upscale_aggregation: _UpscaleAggregation = "max",
        scale_down_sensitivity: float = 0.8,
    ):
        if target_prefill_tokens_per_replica <= 0:
            raise ValueError("target_prefill_tokens_per_replica must be positive")
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
        self._upscale_aggregation = upscale_aggregation
        self._scale_down_sensitivity = scale_down_sensitivity

    def _required_capacity(self, prefill: float) -> float:
        return max(0, prefill) / self._target_prefill_tokens_per_replica

    def _mean_required_capacity(
        self, snapshots_by_source: Dict[str, List[_Snapshot]], now: float
    ) -> float:
        """Time-weighted mean required replicas over the lookback window."""
        window_start = now - self._upscale_lookback_s
        capacity_by_source = []
        for snapshots in snapshots_by_source.values():
            capacity_samples = [
                (timestamp, self._required_capacity(prefill))
                for timestamp, prefill in snapshots
            ]
            samples = [
                sample for sample in capacity_samples if sample[0] >= window_start
            ]
            if not samples:
                continue
            # Metrics are gauges. Carry the last observation before the window
            # boundary forward so the mean covers the full window, provided a
            # new observation confirms that the source is live.
            samples_before_window = [
                sample for sample in capacity_samples if sample[0] < window_start
            ]
            if samples_before_window:
                samples.insert(0, samples_before_window[-1])
            capacity_by_source.append(_time_weighted_mean(samples, window_start, now))

        return max(capacity_by_source, default=0.0)

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

        prefill_required_replicas = self._mean_required_capacity(
            snapshots_by_source, now
        )
        required_replicas = math.ceil(prefill_required_replicas)
        if ctx.total_queued_requests > 0:
            required_replicas = max(1, required_replicas)

        latest_snapshots = [snapshots[-1] for snapshots in snapshots_by_source.values()]
        outstanding_prefill_tokens = max(
            max(0, prefill) for _, prefill in latest_snapshots
        )
        state: Dict[str, Any] = {
            OUTSTANDING_EFFECTIVE_PREFILL_TOKENS: outstanding_prefill_tokens,
            REQUIRED_REPLICAS: required_replicas,
            "lookback_s": self._upscale_lookback_s,
            SERVE_AUTOSCALING_POLICY_METRICS_KEY: {
                OUTSTANDING_EFFECTIVE_PREFILL_TOKENS: outstanding_prefill_tokens,
                PREFILL_REQUIRED_REPLICAS: prefill_required_replicas,
                REQUIRED_REPLICAS: required_replicas,
            },
        }
        return required_replicas, state
