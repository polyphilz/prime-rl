from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import verifiers.v1 as vf

from prime_rl.configs.algorithm import QorlAnchoredGRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


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
    fingerprint: str | None = None
    observed_speedup: float | None = None
    fingerprint_group_size: int = 1


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
    clipped = min(10.0, max(0.1, decision.speedup))
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


def measured_decision(kind: DecisionKind, final: dict[str, Any]) -> QorlDecision:
    fingerprint = final.get("winning_plan_sha256")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"{kind} requires a winning_plan_sha256")
    speedup = float(final.get("score", 0.1))
    return QorlDecision(
        kind,
        speedup,
        fingerprint,
        observed_speedup=speedup,
    )


def decision_from_final(final: dict[str, Any]) -> QorlDecision:
    status = final.get("status")
    source = final.get("score_source")
    if status == "completed" and source == "explicit_keep_default":
        return QorlDecision("keep_default")
    if status == "completed" and source == "default_fingerprint":
        return QorlDecision("default_duplicate")
    if status == "completed" and source == "interleaved_measurement":
        return measured_decision("candidate", final)
    if status == "candidate_timeout":
        return measured_decision("timeout", final)
    if status == "no_valid_candidate":
        return QorlDecision("invalid")
    raise ValueError(f"unsupported QORL final result: status={status!r} score_source={source!r}")


def share_fingerprint_speedups(decisions: list[QorlDecision]) -> list[QorlDecision]:
    by_fingerprint: dict[str, list[float]] = {}
    for decision in decisions:
        if decision.kind not in {"candidate", "timeout"}:
            continue
        if decision.fingerprint is None or decision.speedup is None:
            raise ValueError(f"{decision.kind} requires fingerprinted speedup evidence")
        by_fingerprint.setdefault(decision.fingerprint, []).append(decision.speedup)

    shared = {fingerprint: statistics.median(speedups) for fingerprint, speedups in by_fingerprint.items()}
    return [
        replace(
            decision,
            speedup=shared[decision.fingerprint],
            fingerprint_group_size=len(by_fingerprint[decision.fingerprint]),
        )
        if decision.kind in {"candidate", "timeout"}
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
            "fingerprint": decision.fingerprint,
            "observed_speedup": decision.observed_speedup,
            "shared_speedup": decision.speedup,
            "fingerprint_group_size": decision.fingerprint_group_size,
            **asdict(result),
        }

    @staticmethod
    def _final_identity(trace: vf.Trace) -> dict[str, Any]:
        qorl = trace.info.get("qorl")
        final = qorl.get("final") if isinstance(qorl, dict) else None
        if not isinstance(final, dict):
            return {"status": None, "score_source": None}
        return {
            "status": final.get("status"),
            "score_source": final.get("score_source"),
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
            decisions = share_fingerprint_speedups(
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
