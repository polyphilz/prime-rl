"""Prepared rows, smaller final updates, token normalization and validation."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from prime_rl.configs.sft import PreparedDataConfig, SFTConfig, SFTValConfig
from prime_rl.configs.shared import ResumeConfig, RunConfig
from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.trainer.sft.prepared import (
    PreparedDataset,
    PreparedRow,
    gradient_scale,
    prepared_dataloader,
    validate_continuation,
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


def test_shuffle_preserves_every_row(config):
    dataset = PreparedDataset(config.model_copy(update={"shuffle": True, "seed": 7}))
    loader = prepared_dataloader(dataset)
    batches = [step.row_indices for step in loader]
    assert batches == [step.row_indices for step in prepared_dataloader(dataset)]
    assert sorted(batches[0] + batches[1]) == list(range(10))
    assert sorted(batches[2] + batches[3]) == list(range(10))


@pytest.mark.parametrize("workers", [0, 1])
def test_continuation_starts_at_next_absolute_epoch(config, workers):
    dataset = PreparedDataset(config.model_copy(update={"shuffle": True, "seed": 42, "num_workers": workers}))
    complete = list(prepared_dataloader(dataset))
    resumed = list(prepared_dataloader(dataset, completed_steps=2))
    assert [step.row_indices for step in resumed] == [step.row_indices for step in complete[2:]]
    assert [step.epoch for step in resumed] == [1, 1]
    assert [len(step.row_indices) for step in resumed] == [8, 2]
    for actual, expected in zip(resumed, complete[2:]):
        for actual_batch, expected_batch in zip(actual.micro_batches, expected.micro_batches):
            for key in ("input_ids", "target_ids", "loss_mask", "position_ids", "seq_lens"):
                torch.testing.assert_close(actual_batch[key], expected_batch[key])


@pytest.mark.parametrize("completed", [-1, 1, 3, 4])
def test_continuation_rejects_non_epoch_or_exhausted_position(config, completed):
    with pytest.raises(ValueError, match="completed epoch"):
        prepared_dataloader(PreparedDataset(config), completed_steps=completed)


def test_external_native_sibling_runs_and_symlink_safety(config, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    native = SFTConfig(
        data=config,
        output_dir=tmp_path,
        run=RunConfig(dir="destination"),
        resume=ResumeConfig(dir=source / "checkpoints/step_2"),
    )
    validate_continuation(native)
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    native.run.dir = "alias/nested"
    with pytest.raises(ValueError, match="overlap"):
        validate_continuation(native)
    native.run.dir = "destination"
    native.ckpt = CheckpointConfig(output_dir=alias)
    with pytest.raises(ValueError, match="overlap"):
        validate_continuation(native)


@pytest.mark.parametrize("offload", [False, True])
def test_checkpoint_continuation_matches_uninterrupted_adamw(config, tmp_path, offload):
    pytest.importorskip("dion", reason="native optimizer dependencies require the GPU installation")
    if offload and not torch.cuda.is_available():
        pytest.skip("state-only CPU offload requires CUDA")
    from torch.distributed.checkpoint import load, save

    from prime_rl.trainer.ckpt import AppState, Progress
    from prime_rl.trainer.optim.state_offload import CPUOffloadOptimizer
    from prime_rl.trainer.scheduler import setup_constant_scheduler

    device = "cuda" if offload else "cpu"
    torch.manual_seed(42)
    model = torch.nn.Linear(1, 1).to(device)
    reference = deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, betas=(0.8, 0.95))
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.003, betas=(0.8, 0.95))
    if offload:
        optimizer = CPUOffloadOptimizer(optimizer)
        reference_optimizer = CPUOffloadOptimizer(reference_optimizer)
    scheduler = setup_constant_scheduler(optimizer.base_optimizer if offload else optimizer)
    reference_scheduler = setup_constant_scheduler(
        reference_optimizer.base_optimizer if offload else reference_optimizer
    )
    dataset = PreparedDataset(config.model_copy(update={"shuffle": True, "seed": 42}))

    def run_step(target, optim, scheduled):
        for batch in scheduled.micro_batches:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
        update(target, optim, scheduled)

    for index in range(4):
        run_step(reference, reference_optimizer, dataset[index])
        reference_scheduler.step()
    for index in range(2):
        run_step(model, optimizer, dataset[index])
        scheduler.step()
    expected_states = deepcopy(optimizer.state_dict())
    expected_scheduler = deepcopy(scheduler.state_dict())
    progress = Progress(step=2, total_samples=10, total_tokens=40)
    save({"app": AppState(model, [optimizer], scheduler, progress)}, checkpoint_id=tmp_path / "checkpoint")
    restored = torch.nn.Linear(1, 1).to(device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.9)
    if offload:
        restored_optimizer = CPUOffloadOptimizer(restored_optimizer)
    restored_scheduler = setup_constant_scheduler(restored_optimizer.base_optimizer if offload else restored_optimizer)
    restored_progress = Progress()
    load(
        {"app": AppState(restored, [restored_optimizer], restored_scheduler, restored_progress)},
        checkpoint_id=tmp_path / "checkpoint",
    )
    assert restored_progress == progress
    assert restored_scheduler.state_dict() == expected_scheduler
    actual_states = restored_optimizer.state_dict()
    assert actual_states["param_groups"] == expected_states["param_groups"]
    for key, state in expected_states["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(actual_states["state"][key][name], value)
    for scheduled in prepared_dataloader(dataset, completed_steps=restored_progress.step):
        run_step(restored, restored_optimizer, scheduled)
        restored_scheduler.step()
    for actual, expected in zip(restored.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


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
