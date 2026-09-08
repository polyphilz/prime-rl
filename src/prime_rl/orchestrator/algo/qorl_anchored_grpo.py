from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field

from prime_rl.configs.algorithm import QorlAnchoredGRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient

MIN_TRAINING_SPEEDUP = 0.1
MAX_TRAINING_SPEEDUP = 10.0


DecisionKind = Literal[
    "candidate",
    "keep_default",
    "default_duplicate",
    "timeout",
    "invalid",
]


@dataclass(frozen=True)
class QorlDecision:
    kind: DecisionKind
    speedup: float | None = None
    timing_reuse_key: str | None = None
    observed_speedup: float | None = None
    reuse_group_size: int = 1


@dataclass(frozen=True)
class QorlAdvantage:
    kind: DecisionKind
    quality: float | None
    reference: float | None
    protocol_cost: float
    advantage: float


def soft_threshold(value: float, tau: float) -> float:
    if value > 0:
        return max(value - tau, 0.0)
    return min(value + tau, 0.0)


def decision_quality(decision: QorlDecision, tau: float, t: float) -> float | None:
    if decision.kind == "invalid":
        return None
    if decision.kind in {"keep_default", "default_duplicate"}:
        return 0.0
    if decision.speedup is None or not math.isfinite(decision.speedup) or decision.speedup <= 0:
        raise ValueError(f"{decision.kind} requires a finite positive speedup")
    clipped = min(MAX_TRAINING_SPEEDUP, max(MIN_TRAINING_SPEEDUP, decision.speedup))
    quality = soft_threshold(math.log(clipped), tau)
    return quality - t if decision.kind == "timeout" else quality


def anchored_advantages(
    decisions: list[QorlDecision],
    *,
    tau: float,
    c: float,
    d: float,
    t: float,
    min_peers: int,
) -> list[QorlAdvantage]:
    qualities = [decision_quality(decision, tau, t) for decision in decisions]
    results: list[QorlAdvantage] = []
    for index, (decision, quality) in enumerate(zip(decisions, qualities, strict=True)):
        if decision.kind == "invalid":
            results.append(QorlAdvantage(decision.kind, None, None, 0.0, -c))
            continue

        assert quality is not None
        peers = [value for peer, value in enumerate(qualities) if peer != index and value is not None]
        reference = max(0.0, statistics.fmean(peers)) if len(peers) >= min_peers else 0.0
        protocol_cost = d if decision.kind == "default_duplicate" else 0.0
        results.append(
            QorlAdvantage(
                decision.kind,
                quality,
                reference,
                protocol_cost,
                quality - reference - protocol_cost,
            )
        )
    return results


class FinalEvidence(BaseModel):
    """Only the finalized measurement fields consumed by credit assignment."""

    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)
    kind: Literal[
        "kept_default", "default_duplicate", "measured", "timed_out", "no_valid_candidate", "selection_failed"
    ]
    speedup: float | None
    selected_candidate_id: str | None = None
    selected_plan_sha256: str | None = None
    timing_reuse_key: str | None = None
    initial_default_median_execution_time_ms: float | None = Field(default=None, gt=0)
    timeout_ms: int | None = Field(default=None, gt=0)


class PostgresScope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    id: str
    pg_conf_sha256: str
    expected_sha256: str


class PoolScope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    config_sha256: str
    postgres_config: PostgresScope


class RolloutScope(BaseModel):
    """Sharing is confined to one query and a fixed database/resource configuration."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    schema_version: Literal[2]
    task_id: str
    database_pool: PoolScope


def decision_from_final(final: dict[str, Any]) -> QorlDecision:
    evidence = FinalEvidence.model_validate(final)
    if evidence.kind in {"kept_default", "default_duplicate"}:
        if evidence.speedup != 1.0:
            raise ValueError("default reuse requires speedup 1.0")
        return QorlDecision("keep_default" if evidence.kind == "kept_default" else "default_duplicate")
    if evidence.kind in {"no_valid_candidate", "selection_failed"}:
        if evidence.speedup is not None:
            raise ValueError(f"{evidence.kind} cannot have a measured speedup")
        return QorlDecision("invalid")
    if not evidence.selected_candidate_id:
        raise ValueError("selected outcome requires a candidate ID")
    if evidence.kind == "measured":
        if not evidence.selected_plan_sha256 or not evidence.timing_reuse_key:
            raise ValueError("measured outcome requires plan identity and timing_reuse_key")
        if evidence.speedup is None or evidence.speedup <= 0:
            raise ValueError("measured outcome requires a finite positive speedup")
        speedup = evidence.speedup
        kind: DecisionKind = "candidate"
    else:
        if evidence.speedup is not None:
            raise ValueError("timed_out cannot claim a measured speedup")
        if evidence.initial_default_median_execution_time_ms is None or evidence.timeout_ms is None:
            raise ValueError("timed_out requires its initial baseline and cutoff")
        if (evidence.selected_plan_sha256 is None) != (evidence.timing_reuse_key is None):
            raise ValueError("timeout plan and timing-reuse identity must both be present or absent")
        speedup = evidence.initial_default_median_execution_time_ms / evidence.timeout_ms
        kind = "timeout"
    # Clip each rollout before sharing, then apply the log threshold to the shared value.
    return QorlDecision(
        kind,
        min(MAX_TRAINING_SPEEDUP, max(MIN_TRAINING_SPEEDUP, speedup)),
        evidence.timing_reuse_key,
        observed_speedup=evidence.speedup,
    )


def share_reusable_speedups(decisions: list[QorlDecision]) -> list[QorlDecision]:
    by_reuse_key: dict[str, list[float]] = {}
    for decision in decisions:
        if decision.kind not in {"candidate", "timeout"}:
            continue
        if decision.speedup is None:
            raise ValueError(f"{decision.kind} requires speedup evidence")
        if decision.timing_reuse_key is None:
            if decision.kind == "timeout":
                continue
            raise ValueError("measured candidate requires a timing-reuse key")
        by_reuse_key.setdefault(decision.timing_reuse_key, []).append(decision.speedup)

    shared = {timing_reuse_key: statistics.median(speedups) for timing_reuse_key, speedups in by_reuse_key.items()}
    return [
        replace(
            decision,
            speedup=shared[decision.timing_reuse_key],
            reuse_group_size=len(by_reuse_key[decision.timing_reuse_key]),
        )
        if decision.kind in {"candidate", "timeout"} and decision.timing_reuse_key is not None
        else decision
        for decision in decisions
    ]


class QorlAnchoredGRPO(Algorithm):
    """QORL credit assignment relative to valid siblings and the default plan."""

    def __init__(self, config: QorlAnchoredGRPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.config = config

    @staticmethod
    def _all_traces(episodes: list[vf.Episode]) -> list[vf.Trace]:
        return [trace for episode in episodes for trace in episode.traces]

    def _discard_reason(
        self,
        episodes: list[vf.Episode],
        trainable_count: int,
    ) -> str | None:
        if len(episodes) != self.config.expected_group_size:
            return "incomplete_group"
        if any(not episode.ok for episode in episodes) or any(trace.has_error for trace in self._all_traces(episodes)):
            return "errored_group"
        if trainable_count != self.config.expected_group_size:
            return "unexpected_trainable_trace_count"
        return None

    @staticmethod
    def _record(trace: vf.Trace, decision: QorlDecision, result: QorlAdvantage) -> None:
        trace.info["qorl_advantage"] = {
            "rule": "qorl_anchored_grpo",
            "discarded": False,
            "timing_reuse_key": decision.timing_reuse_key,
            "observed_speedup": decision.observed_speedup,
            "shared_speedup": decision.speedup,
            "reuse_group_size": decision.reuse_group_size,
            **asdict(result),
        }

    @staticmethod
    def _final_identity(trace: vf.Trace) -> dict[str, Any]:
        qorl = trace.info.get("qorl")
        final = qorl.get("final") if isinstance(qorl, dict) else None
        if not isinstance(final, dict):
            return {"kind": None}
        return {
            "kind": final.get("kind"),
        }

    def _discard(
        self,
        episodes: list[vf.Episode],
        trainable: list[vf.Trace],
        reason: str,
    ) -> None:
        for trace in self._all_traces(episodes):
            trace.info["qorl_advantage"] = {
                "rule": "qorl_anchored_grpo",
                "discarded": True,
                "discard_reason": reason,
                "kind": None,
                "quality": None,
                "reference": None,
                "protocol_cost": 0.0,
                "advantage": 0.0,
            }
        for trace in trainable:
            assign_advantages(trace, 0.0)

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        trainable = [trace for _, trace in iter_trainable_traces(episodes)]
        if reason := self._discard_reason(episodes, len(trainable)):
            self._discard(episodes, trainable, reason)
            return

        try:
            scopes = [RolloutScope.model_validate(trace.info["qorl"]) for trace in trainable]
            if any(scope != scopes[0] for scope in scopes[1:]):
                self._discard(episodes, trainable, "mismatched_measurement_scope")
                return
            if any(trace.info["qorl"].get("failure") is not None for trace in trainable):
                self._discard(episodes, trainable, "unscored_failure")
                return
            decisions = share_reusable_speedups(
                [decision_from_final(trace.info["qorl"]["final"]) for trace in trainable]
            )
            results = anchored_advantages(
                decisions,
                tau=self.config.tau,
                c=self.config.c,
                d=self.config.d,
                t=self.config.t,
                min_peers=self.config.min_peers,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            identities = [self._final_identity(trace) for trace in trainable]
            get_logger().warning(
                f"Discarding QORL group with unsupported final record: finals={identities}, error={error!r}"
            )
            self._discard(episodes, trainable, "unsupported_final")
            return

        for trace, decision, result in zip(trainable, decisions, results, strict=True):
            assign_advantages(trace, result.advantage)
            self._record(trace, decision, result)
