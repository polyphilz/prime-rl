"""Prepared rows, smaller final updates, token normalization and resumable cursors."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from prime_rl.configs.sft import PreparedDataConfig, SFTConfig, SFTValConfig
from prime_rl.trainer.sft.prepared import (
    PreparedDataset,
    PreparedRow,
    gradient_scale,
    load_rng,
    prepared_dataloader,
    save_rng,
    validation_mode,
)


@pytest.fixture
def config(tmp_path: Path) -> PreparedDataConfig:
    path = tmp_path / "rows.jsonl"
    rows = [
        PreparedRow(
            input_ids=[index, 1, 2, 3],
            target_ids=[3, 2, 1, index],
            loss_mask=[True, index % 2 == 0, False, True],
            position_ids=[0, 1, 0, 1],
            seq_lens=[2, 2],
        )
        for index in range(10)
    ]
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
    return PreparedDataConfig(path=path, seq_len=4, batch_size=8, micro_batch_size=2, epochs=2, shuffle=False)


def test_rows_and_boundaries_are_unchanged(config):
    dataset = PreparedDataset(config)
    assert len(dataset) == 4
    assert [dataset[i].row_indices for i in range(len(dataset))] == [list(range(8)), [8, 9]] * 2
    batch = dataset[1].micro_batches[0]
    assert batch["input_ids"].tolist() == [[8, 1, 2, 3, 9, 1, 2, 3]]
    assert batch["target_ids"].tolist() == [[3, 2, 1, 8, 3, 2, 1, 9]]
    assert batch["position_ids"].tolist() == [[0, 1, 0, 1, 0, 1, 0, 1]]
    assert batch["seq_lens"].tolist() == [2, 2, 2, 2]
    assert batch["loss_mask"].tolist() == [[True, True, False, True, True, False, False, True]]


def test_all_ranks_cover_partial_batch_without_repeating_rows(config):
    steps = [PreparedDataset(config, rank, 4)[1] for rank in range(4)]
    assert [step.local_row_indices for step in steps] == [[8], [9], [], []]
    assert [len(step.micro_batches) for step in steps] == [1] * 4
    assert sum(int(batch["loss_mask"].sum()) for step in steps for batch in step.micro_batches) == 5


def test_shuffle_and_resume_preserve_every_row(config):
    dataset = PreparedDataset(config.model_copy(update={"shuffle": True, "seed": 7}))
    loader = prepared_dataloader(dataset)
    iterator = iter(loader)
    first = next(iterator)
    state = loader.state_dict()
    expected = [step.row_indices for step in iterator]
    resumed = prepared_dataloader(dataset)
    resumed.load_state_dict(state)
    actual = [step.row_indices for step in resumed]
    assert actual == expected
    assert sorted(first.row_indices + actual[0]) == list(range(10))
    assert sorted(actual[1] + actual[2]) == list(range(10))


def update(model, optimizer, step):
    tokens = 0
    for batch in step.micro_batches:
        prediction = model(batch["input_ids"].float().unsqueeze(-1)).squeeze(-1)
        loss = ((prediction - batch["target_ids"].float()) ** 2)[batch["loss_mask"]].sum()
        tokens += int(batch["loss_mask"].sum())
        (loss / len(step.micro_batches)).backward()
    for parameter in model.parameters():
        parameter.grad.mul_(gradient_scale(tokens, len(step.micro_batches), 1))
    optimizer.step()
    optimizer.zero_grad()


def test_gradient_uses_actual_tokens_in_both_batches(config):
    for step in PreparedDataset(config):
        model = torch.nn.Linear(1, 1)
        reference = deepcopy(model)
        update(model, torch.optim.SGD(model.parameters(), lr=0.01), step)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        batches = step.micro_batches
        inputs = torch.cat([batch["input_ids"] for batch in batches], dim=-1).float().unsqueeze(-1)
        targets = torch.cat([batch["target_ids"] for batch in batches], dim=-1).float()
        masks = torch.cat([batch["loss_mask"] for batch in batches], dim=-1)
        ((reference(inputs).squeeze(-1) - targets) ** 2)[masks].mean().backward()
        reference_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)


def test_resume_restores_optimizer_scheduler_rng_and_cursor(config, tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=4)
    loader = prepared_dataloader(PreparedDataset(config))
    iterator = iter(loader)
    update(model, optimizer, next(iterator))
    scheduler.step()
    saved = deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), loader.state_dict()))
    rng_path = tmp_path / "rng.pt"
    save_rng(rng_path)
    expected_random = torch.rand(3)
    expected_rates = []
    for step in iterator:
        update(model, optimizer, step)
        scheduler.step()
        expected_rates.append(scheduler.get_last_lr())
    resumed = torch.nn.Linear(1, 1)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.LinearLR(resumed_optimizer, total_iters=4)
    resumed_loader = prepared_dataloader(PreparedDataset(config))
    resumed.load_state_dict(saved[0])
    resumed_optimizer.load_state_dict(saved[1])
    resumed_scheduler.load_state_dict(saved[2])
    resumed_loader.load_state_dict(saved[3])
    resumed_iterator = iter(resumed_loader)
    load_rng(rng_path)
    torch.testing.assert_close(torch.rand(3), expected_random, rtol=0, atol=0)
    actual_rates = []
    for step in resumed_iterator:
        update(resumed, resumed_optimizer, step)
        resumed_scheduler.step()
        actual_rates.append(resumed_scheduler.get_last_lr())
    assert actual_rates == expected_rates
    for actual, expected in zip(resumed.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_validation_has_no_gradients_and_preserves_training_state():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(0.5))
    model(torch.ones(3, 1)).sum().backward()
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    rng = torch.get_rng_state().clone()
    with validation_mode(model):
        assert not model.training
        assert not model(torch.ones(3, 1)).requires_grad
        torch.rand(1)
    assert model.training
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for parameter, gradient in zip(model.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient)


def test_prepared_config_does_not_require_renderer(config):
    native = SFTConfig(data=config)
    assert native.data.type == "prepared"


def test_conversation_validation_keeps_its_existing_default_type():
    validation = SFTValConfig.model_validate({"data": {"name": "conversation-dataset"}})
    assert validation.data.type == "sft"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="optimizer offload requires CUDA")
@pytest.mark.parametrize("warmup", [False, True])
def test_offloaded_optimizer_checkpoint_restores_moments_and_next_update(tmp_path, warmup):
    import torch.distributed as dist
    from torch.distributed.checkpoint import load, save
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    from prime_rl.trainer.ckpt import AppState, Progress
    from prime_rl.trainer.optim.state_offload import CPUOffloadOptimizer

    def setup():
        model = torch.nn.Linear(2, 1, device="cuda", dtype=torch.bfloat16)
        fully_shard(model)
        optimizer = CPUOffloadOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01))
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer.base_optimizer, start_factor=1 / 3 if warmup else 1.0, total_iters=4
        )
        progress = Progress(step=1, total_samples=8)
        return model, optimizer, scheduler, progress

    def step(model, optimizer, scheduler, gradient):
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, gradient)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        model, optimizer, scheduler, progress = setup()
        step(model, optimizer, scheduler, 0.25)
        checkpoint = tmp_path / "checkpoint"
        save({"app": AppState(model, [optimizer], scheduler, progress)}, checkpoint_id=checkpoint)
        resumed, resumed_optimizer, resumed_scheduler, resumed_progress = setup()
        load(
            {"app": AppState(resumed, [resumed_optimizer], resumed_scheduler, resumed_progress)},
            checkpoint_id=checkpoint,
        )
        assert resumed_progress == progress
        assert resumed_scheduler.state_dict() == scheduler.state_dict()
        assert resumed_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
        for actual, expected in zip(resumed_optimizer.state.values(), optimizer.state.values()):
            for name in ("step", "exp_avg", "exp_avg_sq"):
                actual_value = actual[name].to_local() if isinstance(actual[name], DTensor) else actual[name]
                expected_value = expected[name].to_local() if isinstance(expected[name], DTensor) else expected[name]
                assert actual_value.device.type == "cpu"
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        step(model, optimizer, scheduler, 0.5)
        step(resumed, resumed_optimizer, resumed_scheduler, 0.5)
        for actual, expected in zip(resumed.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
