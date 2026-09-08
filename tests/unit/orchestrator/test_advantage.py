import asyncio
import json
from unittest.mock import Mock

import pytest
import verifiers.v1 as vf

from prime_rl import monitors
from prime_rl.configs.algorithm import (
    GRPOAlgoConfig,
    LinearLengthPenaltyConfig,
    MaxRLAlgoConfig,
    QorlAnchoredGRPOAlgoConfig,
)
from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.configs.orchestrator import ModelConfig, OrchestratorConfig
from prime_rl.monitors.file import FileMonitor
from prime_rl.monitors.file.traces import get_annotations_dir, get_trace_stream
from prime_rl.monitors.file.traces.update import fold_trace_updates
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.max_rl import MaxRLAlgorithm
from prime_rl.orchestrator.algo.qorl_anchored_grpo import (
    QorlAnchoredGRPO,
    QorlDecision,
    anchored_advantages,
    decision_from_final,
    share_reusable_speedups,
)
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.orchestrator.annotations import stamp_arrival, stamp_batch
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Progress


def _build_episode(
    reward: float,
    *,
    sampled_lengths: list[int],
    obs_lengths: list[int] | None = None,
    env_name: str = "test",
    metrics: dict | None = None,
) -> vf.Episode:
    """Build a training trace as an alternating message graph.

    ``sampled_lengths`` gives the token count of each model turn (a sampled
    ``AssistantMessage`` node); ``obs_lengths`` (one shorter, if given) gives the
    token count of the non-sampled observation node injected *after* each turn
    (tool output / user feedback).
    """
    obs_lengths = obs_lengths or []
    nodes: list[vf.MessageNode] = []
    parent: int | None = None
    next_token = 0

    def _take(n: int) -> list[int]:
        nonlocal next_token
        ids = list(range(next_token, next_token + n))
        next_token += n
        return ids

    # Leading user prompt (never trainable).
    prompt_ids = _take(1)
    nodes.append(
        vf.MessageNode(
            message=vf.UserMessage(content="q"),
            token_ids=prompt_ids,
            mask=[False] * len(prompt_ids),
            logprobs=[0.0] * len(prompt_ids),
            sampled=False,
            parent=parent,
        )
    )
    parent = len(nodes) - 1

    # Trace token counts are usage-based, so carry provider usage on the final turn's call:
    # every model-generated token as completion, the leading prompt + tool observations as the
    # fed-in context (num_input_tokens = num_total_tokens - num_output_tokens).
    output_tokens = sum(sampled_lengths)
    input_tokens = 1 + sum(obs_lengths)
    calls: list[vf.ModelCall] = []

    for i, n_sampled in enumerate(sampled_lengths):
        ids = _take(n_sampled)
        is_last = i == len(sampled_lengths) - 1
        nodes.append(
            vf.MessageNode(
                message=vf.AssistantMessage(content="a"),
                token_ids=ids,
                mask=[True] * n_sampled,
                logprobs=[-0.1] * n_sampled,
                sampled=True,
                parent=parent,
            )
        )
        parent = len(nodes) - 1
        if is_last:
            calls.append(
                vf.ModelCall(
                    node=parent,
                    usage=vf.Usage(prompt_tokens=input_tokens, completion_tokens=output_tokens),
                )
            )
        if i < len(obs_lengths):
            obs_ids = _take(obs_lengths[i])
            nodes.append(
                vf.MessageNode(
                    message=vf.ToolMessage(content="t", tool_call_id="x"),
                    token_ids=obs_ids,
                    mask=[False] * obs_lengths[i],
                    logprobs=[0.0] * obs_lengths[i],
                    sampled=False,
                    parent=parent,
                )
            )
            parent = len(nodes) - 1

    trace = vf.Trace[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        calls=calls,
        rewards={"reward": vf.Reward(score=reward)},
        metrics=metrics or {},
        ok=True,
    )
    episode = vf.Episode(
        env=vf.EnvInfo(id=env_name, name=env_name),
        task=trace.task,
        group=vf.GroupInfo(id="group"),
        traces=[trace],
        ok=True,
    )
    return episode


def _make_episode(
    reward: float,
    completion_len: int = 1,
    num_turns: int = 1,
    env_name: str = "test",
    metrics: dict | None = None,
) -> vf.Episode:
    """Build a training trace carrying ``completion_len`` model-sampled tokens split
    across ``num_turns`` sampled turns. Always carries at least one trainable
    token so credit broadcasts somewhere."""
    num_turns = max(num_turns, 1)
    per_turn, rem = divmod(max(completion_len, 1), num_turns)
    sampled_lengths = [per_turn + (rem if i == 0 else 0) for i in range(num_turns)]
    sampled_lengths = [max(n, 1) for n in sampled_lengths]
    return _build_episode(reward, sampled_lengths=sampled_lengths, env_name=env_name, metrics=metrics)


def _make_group(rewards, completion_lengths=None, num_turns=None) -> list[vf.Episode]:
    """Build one group of training traces from 1D arrays of rewards/lengths/turns —
    exactly what ``score_group`` sees."""
    episodes = []
    for i, reward in enumerate(rewards):
        cl = int(completion_lengths[i]) if completion_lengths is not None else 1
        nt = int(num_turns[i]) if num_turns is not None else 1
        episodes.append(_make_episode(float(reward), cl, nt))
    return episodes


def _scalar(episode: vf.Episode) -> float:
    """The per-rollout advantage scalar an algorithm assigned — broadcast over
    the rollout's trainable (mask-True) tokens, so any trainable position holds it."""
    for node in episode.traces[0].nodes:
        if node.advantages:
            return node.advantages[0]
    raise AssertionError("episode has no trainable token")


def _grpo(group: list[vf.Episode], length_penalty=None) -> list[float]:
    """Drive ``GRPOAlgorithm.score_group`` and read back each per-rollout scalar."""
    algo = GRPOAlgorithm(GRPOAlgoConfig(length_penalty=length_penalty), clients=None)
    asyncio.run(algo.score_group(group))
    return [_scalar(episode) for episode in group]


def _max_rl(group: list[vf.Episode]) -> list[float]:
    """Drive ``MaxRLAlgorithm.score_group`` and read back each per-rollout scalar."""
    algo = MaxRLAlgorithm(MaxRLAlgoConfig(), clients=None)
    asyncio.run(algo.score_group(group))
    return [_scalar(episode) for episode in group]


def _qorl(decisions: list[QorlDecision]) -> list[float]:
    return [
        result.advantage
        for result in anchored_advantages(
            decisions,
            tau=0.05,
            c=0.10,
            d=0.02,
            t=0.10,
            min_peers=2,
        )
    ]


def _qorl_group(finals: list[dict]) -> list[vf.Episode]:
    group = _make_group([0.0] * len(finals))
    for episode, final in zip(group, finals, strict=True):
        episode.traces[0].info["qorl"] = {
            "schema_version": 2,
            "task_id": "task",
            "database_pool": {
                "config_sha256": "pool",
                "postgres_config": {"id": "pg", "pg_conf_sha256": "pg-conf", "expected_sha256": "pg-expected"},
            },
            "final": final,
        }
    return group


def _score_qorl_group(group: list[vf.Episode], expected_group_size: int = 4) -> list[float]:
    config = QorlAnchoredGRPOAlgoConfig(expected_group_size=expected_group_size)
    asyncio.run(QorlAnchoredGRPO(config, clients=None).score_group(group))
    return [_scalar(episode) for episode in group if not episode.traces[0].has_error]


# --------------------------------------------------------------------------
# GRPO / MaxRL: group-relative credit, assigned in score_group.
# --------------------------------------------------------------------------


def test_grpo_plain_mean():
    advs = _grpo(_make_group(rewards=[1.0, 0.5, 0.8], completion_lengths=[10, 12, 8]))
    assert len(advs) == 3
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_grpo_singleton_group_is_zero():
    # A group of size 1 has reward == mean, so its advantage is 0.
    assert _grpo([_build_episode(0.7, sampled_lengths=[2])]) == pytest.approx([0.0], abs=1e-6)


def test_max_rl_mean_normalized():
    # mean 0.25: the success gets (1 - 0.25)/0.25 = 3, failures (0 - 0.25)/0.25 = -1
    assert _max_rl(_make_group(rewards=[1.0, 0.0, 0.0, 0.0])) == pytest.approx([3.0, -1.0, -1.0, -1.0])
    # no-success groups carry no signal (the paper's K=0 convention) ...
    assert _max_rl(_make_group(rewards=[0.0, 0.0])) == pytest.approx([0.0, 0.0])
    # ... and all-success groups center to zero like GRPO
    assert _max_rl(_make_group(rewards=[1.0, 1.0])) == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize(
    ("decisions", "expected"),
    [
        (
            [QorlDecision("candidate", score) for score in (1.10, 1.05, 1.17, 1.40)],
            [-0.086, -0.146, -0.004, 0.236],
        ),
        (
            [QorlDecision("candidate", score) for score in (0.95, 0.90, 0.80, 0.70)],
            [-0.001, -0.055, -0.173, -0.307],
        ),
        (
            [QorlDecision("candidate", 1.40), *[QorlDecision("invalid")] * 3],
            [0.286, -0.100, -0.100, -0.100],
        ),
        (
            [QorlDecision("timeout", 0.1), *[QorlDecision("candidate", score) for score in (1.05, 1.03, 1.02)]],
            [-2.353, 0.0, 0.0, 0.0],
        ),
        (
            [QorlDecision("keep_default"), *[QorlDecision("candidate", score) for score in (0.90, 0.80, 0.70)]],
            [0.0, -0.055, -0.173, -0.307],
        ),
        (
            [
                QorlDecision("keep_default"),
                QorlDecision("candidate", 1.40),
                QorlDecision("keep_default"),
                QorlDecision("keep_default"),
            ],
            [-0.095, 0.286, -0.095, -0.095],
        ),
        (
            [
                QorlDecision("candidate", 1.40),
                QorlDecision("candidate", 1.10),
                QorlDecision("invalid"),
                QorlDecision("invalid"),
            ],
            [0.286, 0.045, -0.100, -0.100],
        ),
        (
            [
                QorlDecision("candidate", 1.40),
                QorlDecision("candidate", 1.10),
                QorlDecision("candidate", 1.05),
                QorlDecision("timeout", 0.1),
            ],
            [0.286, 0.045, 0.0, -2.463],
        ),
        (
            [
                QorlDecision("keep_default"),
                QorlDecision("default_duplicate"),
                QorlDecision("keep_default"),
                QorlDecision("default_duplicate"),
            ],
            [0.0, -0.02, 0.0, -0.02],
        ),
        (
            [QorlDecision("invalid")] * 4,
            [-0.10] * 4,
        ),
    ],
)
def test_qorl_anchored_grpo_worked_examples(decisions, expected):
    assert _qorl(decisions) == pytest.approx(expected, abs=1e-3)


def test_qorl_anchored_grpo_penalizes_timeout_at_the_same_measured_score():
    candidate, timeout = anchored_advantages(
        [
            QorlDecision("candidate", 1 / 3),
            QorlDecision("timeout", 1 / 3),
        ],
        tau=0.05,
        c=0.10,
        d=0.02,
        t=0.10,
        min_peers=2,
    )

    assert timeout.quality == pytest.approx(candidate.quality - 0.10)


@pytest.mark.parametrize("invalid_kind", ["no_valid_candidate", "selection_failed"])
def test_qorl_anchored_grpo_reads_qorl_final_results(invalid_kind):
    group = _qorl_group(
        [
            {"kind": "kept_default", "speedup": 1.0},
            {"kind": "default_duplicate", "speedup": 1.0},
            {
                "kind": "measured",
                "selected_candidate_id": "candidate-01",
                "selected_plan_sha256": "candidate-plan",
                "timing_reuse_key": "candidate-plan",
                "speedup": 1.4,
            },
            {"kind": invalid_kind, "speedup": None},
        ]
    )

    advantages = _score_qorl_group(group)

    assert advantages == pytest.approx([-0.143, -0.163, 0.286, -0.1], abs=1e-3)
    assert all(episode.traces[0].info["qorl_advantage"]["discarded"] is False for episode in group)
    logged = group[2].traces[0].info["qorl_advantage"]
    assert logged["rule"] == "qorl_anchored_grpo"
    assert logged["discarded"] is False
    assert logged["kind"] == "candidate"
    numeric = {key: logged[key] for key in ("quality", "reference", "protocol_cost", "advantage")}
    assert numeric == pytest.approx(
        {"quality": 0.286, "reference": 0.0, "protocol_cost": 0.0, "advantage": 0.286},
        abs=1e-3,
    )


def test_selection_failure_cannot_claim_measured_speedup():
    with pytest.raises(ValueError, match="selection_failed cannot have a measured speedup"):
        decision_from_final({"kind": "selection_failed", "speedup": 1.0})


@pytest.mark.parametrize("case", ["zero", "discarded", "rejected", "shipped"])
def test_qorl_group_credit_is_saved_before_shipping(case, tmp_path, monkeypatch):
    group = _qorl_group([{"kind": "kept_default", "speedup": 1.0}] * 4)
    if case in ("rejected", "shipped"):
        group[0].traces[0].info["qorl"]["final"] = {"kind": "no_valid_candidate", "speedup": None}
    if case == "discarded":
        group[-1].ok = False
        group[-1].traces[0].errors = [vf.Error(type="DatabaseError", message="unavailable")]
    algorithm = QorlAnchoredGRPO(QorlAnchoredGRPOAlgoConfig(expected_group_size=4), clients=None)
    env = Mock(algorithm=algorithm, sampling_args={"temperature": 1.0}, requires_sampling_masks=False)
    sink = TrainSink(
        OrchestratorConfig(model=ModelConfig(name="Qwen/Qwen3-0.6B"), constant_trainer_batch_size=True),
        tokenizer=None,
        train_envs=Mock(get=Mock(return_value=env)),
        progress=Progress(),
        batch_size=4,
        token_batch_size=None,
        on_result=lambda _: case != "rejected",
    )
    monitor = FileMonitor(FileMonitorConfig(compress=False, float_decimals=None))
    monkeypatch.setattr(monitors, "MONITORS", [monitor])

    async def run():
        await monitor.init(tmp_path, producer="orchestrator")
        try:
            stamp_arrival(group, "train", 1)
            await monitor.log(group, step=1, kind="train", subset="all")
            sink.pending_groups["group"] = group
            await sink.process_group("group")
            if case == "shipped":
                assert sink.pending_batch
                await monitor.log_annotations(stamp_batch(group, 1))
            else:
                assert not sink.pending_batch
        finally:
            await monitor.finalize()

    asyncio.run(run())
    arrivals = [json.loads(line) for line in (get_trace_stream(tmp_path) / "00000.jsonl").read_text().splitlines()]
    updates = [
        json.loads(line)
        for line in (get_annotations_dir(tmp_path) / "orchestrator/00000.jsonl").read_text().splitlines()
    ]
    for episode, arrival in zip(group, arrivals, strict=True):
        trace = arrival["traces"][0]
        assert "qorl_advantage" not in trace["info"]
        fold_trace_updates(trace, [update for update in updates if update["trace_id"] == trace["id"]])
        assert trace["info"]["qorl_advantage"] == episode.traces[0].info["qorl_advantage"]
        assert trace["info"]["qorl_advantage"]["discarded"] == (case == "discarded")
        assert ("ship" in trace["info"]) == (case == "shipped")


def test_qorl_anchored_grpo_shares_speedup_by_non_default_fingerprint():
    decisions = share_reusable_speedups(
        [
            QorlDecision("candidate", 1.4, "shared", observed_speedup=1.4),
            QorlDecision("candidate", 1.2, "shared", observed_speedup=1.2),
            QorlDecision("candidate", 0.9, "other", observed_speedup=0.9),
            QorlDecision("invalid"),
        ]
    )

    assert [decision.speedup for decision in decisions] == pytest.approx([1.3, 1.3, 0.9, None])
    assert [decision.reuse_group_size for decision in decisions] == [2, 2, 1, 1]
    assert [decision.observed_speedup for decision in decisions[:2]] == [1.4, 1.2]


def test_qorl_anchored_grpo_assigns_equal_advantage_to_equal_fingerprints():
    group = _qorl_group(
        [
            {
                "kind": "measured",
                "selected_candidate_id": "candidate-01",
                "selected_plan_sha256": "shared",
                "timing_reuse_key": "shared",
                "speedup": 1.4,
            },
            {
                "kind": "measured",
                "selected_candidate_id": "candidate-01",
                "selected_plan_sha256": "shared",
                "timing_reuse_key": "shared",
                "speedup": 1.2,
            },
            {
                "kind": "measured",
                "selected_candidate_id": "candidate-01",
                "selected_plan_sha256": "other",
                "timing_reuse_key": "other",
                "speedup": 0.9,
            },
            {"kind": "no_valid_candidate", "speedup": None},
        ]
    )

    advantages = _score_qorl_group(group)

    assert advantages[0] == pytest.approx(advantages[1])
    first = group[0].traces[0].info["qorl_advantage"]
    second = group[1].traces[0].info["qorl_advantage"]
    assert first["observed_speedup"] == 1.4
    assert second["observed_speedup"] == 1.2
    assert first["shared_speedup"] == pytest.approx(1.3)
    assert second["shared_speedup"] == pytest.approx(1.3)
    assert first["reuse_group_size"] == 2
    assert second["reuse_group_size"] == 2


def test_qorl_anchored_grpo_clips_before_sharing_but_records_raw_speedup():
    decisions = [
        decision_from_final(
            {
                "kind": "measured",
                "selected_candidate_id": "candidate-01",
                "selected_plan_sha256": "same-plan",
                "timing_reuse_key": "same-key",
                "speedup": speedup,
            }
        )
        for speedup in (20.0, 0.2)
    ]
    shared = share_reusable_speedups(decisions)
    assert [item.speedup for item in shared] == pytest.approx([5.1, 5.1])
    assert [item.observed_speedup for item in shared] == [20.0, 0.2]


def test_qorl_anchored_grpo_does_not_share_unknown_planning_timeouts():
    finals = [
        {
            "kind": "timed_out",
            "selected_candidate_id": "candidate-01",
            "selected_plan_sha256": None,
            "timing_reuse_key": None,
            "speedup": None,
            "initial_default_median_execution_time_ms": baseline,
            "timeout_ms": 5000,
        }
        for baseline in (1000.0, 2000.0)
    ]
    decisions = share_reusable_speedups([decision_from_final(final) for final in finals])
    assert [item.speedup for item in decisions] == [0.2, 0.4]
    assert [item.observed_speedup for item in decisions] == [None, None]
    assert [item.reuse_group_size for item in decisions] == [1, 1]


def test_qorl_anchored_grpo_does_not_share_different_overrides():
    decisions = [
        decision_from_final(
            {
                "kind": "measured",
                "selected_candidate_id": "candidate-01",
                "selected_plan_sha256": "same-plan",
                "timing_reuse_key": key,
                "speedup": speedup,
            }
        )
        for key, speedup in (("settings-a", 1.4), ("settings-b", 1.2))
    ]
    assert [item.speedup for item in share_reusable_speedups(decisions)] == [1.4, 1.2]


@pytest.mark.parametrize("changed", ["task", "pool", "postgres", "failure"])
def test_qorl_anchored_grpo_discards_unscored_or_different_measurement_scopes(changed):
    group = _qorl_group([{"kind": "kept_default", "speedup": 1.0}] * 4)
    record = group[0].traces[0].info["qorl"]
    if changed == "task":
        record["task_id"] = "other"
    elif changed == "pool":
        record["database_pool"]["config_sha256"] = "other"
    elif changed == "postgres":
        record["database_pool"]["postgres_config"]["pg_conf_sha256"] = "other"
    else:
        record["final"] = None
        record["failure"] = {"error": "default query timed out"}
    assert _score_qorl_group(group) == [0.0] * 4
    expected = "unscored_failure" if changed == "failure" else "mismatched_measurement_scope"
    assert all(episode.traces[0].info["qorl_advantage"]["discard_reason"] == expected for episode in group)


def test_qorl_anchored_grpo_discards_incomplete_group():
    group = _qorl_group([{"kind": "measured", "selected_candidate_id": "candidate-01", "speedup": 1.4}] * 3)

    advantages = _score_qorl_group(group)

    assert advantages == [0.0, 0.0, 0.0]
    assert all(episode.traces[0].info["qorl_advantage"]["discard_reason"] == "incomplete_group" for episode in group)


def test_qorl_anchored_grpo_discards_group_with_error():
    group = _qorl_group([{"kind": "measured", "selected_candidate_id": "candidate-01", "speedup": 1.4}] * 4)
    group[0].traces[0].ok = False

    advantages = _score_qorl_group(group)

    assert advantages == [0.0, 0.0, 0.0]
    assert all(episode.traces[0].info["qorl_advantage"]["discard_reason"] == "errored_group" for episode in group)


@pytest.mark.parametrize(
    ("case", "bad_final"),
    [
        ("missing", None),
        ("unknown_kind", {"kind": "unknown"}),
        (
            "invalid_score",
            {
                "kind": "measured",
                "selected_candidate_id": "candidate-01",
                "selected_plan_sha256": "candidate-plan",
                "timing_reuse_key": "candidate-plan",
                "speedup": "not-a-number",
            },
        ),
        (
            "missing_fingerprint",
            {
                "kind": "measured",
                "selected_candidate_id": "candidate-01",
                "speedup": 1.4,
            },
        ),
    ],
)
def test_qorl_anchored_grpo_discards_unsupported_final(case, bad_final):
    group = _qorl_group([{"kind": "measured", "selected_candidate_id": "candidate-01", "speedup": 1.4}] * 4)
    if case == "missing":
        group[0].traces[0].info["qorl"].pop("final")
    else:
        group[0].traces[0].info["qorl"]["final"] = bad_final

    advantages = _score_qorl_group(group)

    assert advantages == [0.0] * 4
    assert all(episode.traces[0].info["qorl_advantage"]["discard_reason"] == "unsupported_final" for episode in group)


# --------------------------------------------------------------------------
# GRPO linear length penalty: pass_rate-scaled penalty before the baseline.
# --------------------------------------------------------------------------


def test_linear_equal_lengths_reduce_to_plain_grpo():
    """Equal completion length and turns → every rollout takes the same penalty
    fraction, so subtracting it leaves the centered advantages unchanged."""
    penalized = _grpo(
        _make_group(rewards=[1.0, 0.0, 1.0], completion_lengths=[10, 10, 10], num_turns=[2, 2, 2]),
        length_penalty=LinearLengthPenaltyConfig(),
    )
    plain = _grpo(_make_group(rewards=[1.0, 0.0, 1.0], completion_lengths=[10, 10, 10], num_turns=[2, 2, 2]))
    assert penalized == pytest.approx(plain, abs=1e-6)


def test_linear_completion_term_penalizes_longer():
    """With only the completion term, longer completions get a larger penalty and a
    lower advantage; advantages stay zero-mean."""
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.25, num_input_tokens_weight=0.0, num_turns_weight=0.0)
    advs = _grpo(_make_group(rewards=[1.0, 1.0, 1.0], completion_lengths=[10, 20, 30]), length_penalty=cfg)
    assert advs[0] > advs[1] > advs[2]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_linear_context_term_penalizes_more_context():
    """The context term penalizes non-completion (prompt / tool-response) tokens: at
    equal completion length, more context tokens yields a lower advantage."""
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.0, num_input_tokens_weight=0.25, num_turns_weight=0.0)
    group = [
        _build_episode(1.0, sampled_lengths=[10], obs_lengths=[]),
        _build_episode(1.0, sampled_lengths=[10], obs_lengths=[100]),
    ]
    asyncio.run(GRPOAlgorithm(GRPOAlgoConfig(length_penalty=cfg), clients=None).score_group(group))
    advs = [_scalar(episode) for episode in group]
    assert advs[0] > advs[1]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_linear_turns_term_penalizes_more_turns():
    """The turns term penalizes higher turn counts at equal token lengths."""
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.0, num_input_tokens_weight=0.0, num_turns_weight=0.25)
    advs = _grpo(
        _make_group(rewards=[1.0, 1.0], completion_lengths=[100, 100], num_turns=[1, 4]),
        length_penalty=cfg,
    )
    assert advs[0] > advs[1]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# assign_advantages: scalar broadcast over the rollout's trainable tokens.
# --------------------------------------------------------------------------


def test_assign_advantages_broadcasts_scalar():
    """A scalar broadcasts uniformly over the rollout's trainable (mask-True) tokens."""
    episode = _build_episode(0.0, sampled_lengths=[2])
    trace = episode.traces[0]
    # one user prompt token (masked) + 2 sampled tokens (trainable)
    assign_advantages(trace, 0.7)
    assert trace_to_samples(trace)[0].advantages == [0.0, 0.7, 0.7]


def test_assign_advantages_zeros_non_trainable():
    """Non-trainable (mask=False) positions stay 0.0 under scalar broadcast."""
    # prompt(1, masked) + sampled(1) + obs(1, masked): mask is [F, T, F]
    episode = _build_episode(0.0, sampled_lengths=[1], obs_lengths=[1])
    trace = episode.traces[0]
    assign_advantages(trace, 0.7)
    assert trace_to_samples(trace)[0].advantages == [0.0, 0.7, 0.0]


def test_assign_advantages_rejects_misaligned():
    episode = _build_episode(0.0, sampled_lengths=[2])
    # full length is 3 (prompt + 2 sampled); a 1-element list must be rejected
    with pytest.raises(ValueError, match="align"):
        assign_advantages(episode.traces[0], [0.5])
