"""Prepared rows, smaller final updates, token normalization and validation."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from prime_rl.configs.sft import PreparedDataConfig, SFTConfig, SFTValConfig
from prime_rl.trainer.sft.prepared import (
    PreparedDataset,
    PreparedRow,
    gradient_scale,
    prepared_dataloader,
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
