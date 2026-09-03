import asyncio

import pytest
import verifiers.v1 as vf

from prime_rl.configs.algorithm import (
    GRPOAlgoConfig,
    LinearLengthPenaltyConfig,
    MaxRLAlgoConfig,
    QorlAnchoredGRPOAlgoConfig,
)
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.max_rl import MaxRLAlgorithm
from prime_rl.orchestrator.algo.qorl_anchored_grpo import (
    QorlAnchoredGRPO,
    QorlDecision,
    anchored_advantages,
    share_fingerprint_speedups,
)
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.orchestrator.trajectories import trace_to_samples


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
        episode.traces[0].info["qorl"] = {"final": final}
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


def test_qorl_anchored_grpo_reads_qorl_final_results():
    group = _qorl_group(
        [
            {"status": "completed", "score_source": "explicit_keep_default", "score": 1.0},
            {"status": "completed", "score_source": "default_fingerprint", "score": 1.0},
            {
                "status": "completed",
                "score_source": "interleaved_measurement",
                "winning_plan_sha256": "candidate-plan",
                "score": 1.4,
            },
            {"status": "no_valid_candidate", "score": 0.0},
        ]
    )

    advantages = _score_qorl_group(group)

    assert advantages == pytest.approx([-0.143, -0.163, 0.286, -0.1], abs=1e-3)
    logged = group[2].traces[0].info["qorl_advantage"]
    assert logged["rule"] == "qorl_anchored_grpo"
    assert logged["discarded"] is False
    assert logged["kind"] == "candidate"
    numeric = {key: logged[key] for key in ("quality", "reference", "protocol_cost", "advantage")}
    assert numeric == pytest.approx(
        {"quality": 0.286, "reference": 0.0, "protocol_cost": 0.0, "advantage": 0.286},
        abs=1e-3,
    )


def test_qorl_anchored_grpo_shares_speedup_by_non_default_fingerprint():
    decisions = share_fingerprint_speedups(
        [
            QorlDecision("candidate", 1.4, "shared", observed_speedup=1.4),
            QorlDecision("candidate", 1.2, "shared", observed_speedup=1.2),
            QorlDecision("candidate", 0.9, "other", observed_speedup=0.9),
            QorlDecision("invalid"),
        ]
    )

    assert [decision.speedup for decision in decisions] == pytest.approx([1.3, 1.3, 0.9, None])
    assert [decision.fingerprint_group_size for decision in decisions] == [2, 2, 1, 1]
    assert [decision.observed_speedup for decision in decisions[:2]] == [1.4, 1.2]


def test_qorl_anchored_grpo_assigns_equal_advantage_to_equal_fingerprints():
    group = _qorl_group(
        [
            {
                "status": "completed",
                "score_source": "interleaved_measurement",
                "winning_plan_sha256": "shared",
                "score": 1.4,
            },
            {
                "status": "completed",
                "score_source": "interleaved_measurement",
                "winning_plan_sha256": "shared",
                "score": 1.2,
            },
            {
                "status": "completed",
                "score_source": "interleaved_measurement",
                "winning_plan_sha256": "other",
                "score": 0.9,
            },
            {"status": "no_valid_candidate", "score": 0.0},
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
    assert first["fingerprint_group_size"] == 2
    assert second["fingerprint_group_size"] == 2


def test_qorl_anchored_grpo_discards_incomplete_group():
    group = _qorl_group([{"status": "completed", "score_source": "interleaved_measurement", "score": 1.4}] * 3)

    advantages = _score_qorl_group(group)

    assert advantages == [0.0, 0.0, 0.0]
    assert all(episode.traces[0].info["qorl_advantage"]["discard_reason"] == "incomplete_group" for episode in group)


def test_qorl_anchored_grpo_discards_group_with_error():
    group = _qorl_group([{"status": "completed", "score_source": "interleaved_measurement", "score": 1.4}] * 4)
    group[0].traces[0].ok = False

    advantages = _score_qorl_group(group)

    assert advantages == [0.0, 0.0, 0.0]
    assert all(episode.traces[0].info["qorl_advantage"]["discard_reason"] == "errored_group" for episode in group)


@pytest.mark.parametrize(
    ("case", "bad_final"),
    [
        ("missing", None),
        ("unknown_status", {"status": "unknown"}),
        (
            "invalid_score",
            {
                "status": "completed",
                "score_source": "interleaved_measurement",
                "winning_plan_sha256": "candidate-plan",
                "score": "not-a-number",
            },
        ),
        (
            "missing_fingerprint",
            {
                "status": "completed",
                "score_source": "interleaved_measurement",
                "score": 1.4,
            },
        ),
    ],
)
def test_qorl_anchored_grpo_discards_unsupported_final(case, bad_final):
    group = _qorl_group([{"status": "completed", "score_source": "interleaved_measurement", "score": 1.4}] * 4)
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
