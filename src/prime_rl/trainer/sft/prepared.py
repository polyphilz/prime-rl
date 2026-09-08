"""Read prepared optimizer batches without rendering, shifting or repacking rows."""

import math
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from prime_rl.configs.sft import PreparedDataConfig
from prime_rl.trainer.sft.data import Batch


def gradient_scale(supervised_tokens: int, accumulation_steps: int, fsdp_divisor: float) -> float:
    """Undo accumulation/FSDP scaling and average over actual supervised tokens."""
    return fsdp_divisor * accumulation_steps / supervised_tokens if supervised_tokens > 0 else 1.0


@contextmanager
def validation_mode(model: torch.nn.Module) -> Iterator[None]:
    """Evaluate without training gradients, dropout or changes to training randomness."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.random.fork_rng():
            yield
    finally:
        model.train(was_training)


class PreparedRow(BaseModel):
    """The text-only on-disk training sample; targets have already been shifted."""

    model_config = ConfigDict(extra="forbid", strict=True)
    input_ids: list[int] = Field(min_length=1)
    target_ids: list[int]
    loss_mask: list[bool]
    position_ids: list[int]
    seq_lens: list[int]
    mm_kwargs: None = None
    mm_token_type_ids: None = None

    @model_validator(mode="after")
    def aligned(self) -> Self:
        length = len(self.input_ids)
        if any(len(values) != length for values in (self.target_ids, self.loss_mask, self.position_ids)):
            raise ValueError("prepared token, target, mask and position lengths differ")
        if not self.seq_lens or min(self.seq_lens) < 1 or sum(self.seq_lens) != length:
            raise ValueError("prepared seq_lens must cover the row exactly")
        if not any(self.loss_mask):
            raise ValueError("prepared row has no supervised tokens")
        return self


def collate_rows(rows: list[PreparedRow], seq_len: int) -> Batch:
    """Concatenate rows for a forward pass, preserving every sequence boundary.

    An empty rank uses synthetic, fully masked tokens solely to participate in
    FSDP collectives. These are not dataset rows and never enter token accounting.
    """
    return {
        "input_ids": torch.tensor([token for row in rows for token in row.input_ids] or [0] * seq_len).unsqueeze(0),
        "target_ids": torch.tensor([token for row in rows for token in row.target_ids] or [0] * seq_len).unsqueeze(0),
        "position_ids": torch.tensor(
            [position for row in rows for position in row.position_ids] or list(range(seq_len))
        ).unsqueeze(0),
        "loss_mask": torch.tensor(
            [mask for row in rows for mask in row.loss_mask] or [False] * seq_len, dtype=torch.bool
        ).unsqueeze(0),
        "seq_lens": torch.tensor([length for row in rows for length in row.seq_lens] or [seq_len]),
        "mm_kwargs": None,
        "mm_token_type_ids": None,
    }


@dataclass(frozen=True)
class PreparedStep:
    """One optimizer update, with global row IDs for exact consumption accounting."""

    epoch: int
    row_indices: list[int]
    local_row_indices: list[int]
    micro_batches: list[Batch]


class PreparedDataset(Dataset[PreparedStep]):
    """Index optimizer steps rather than rows so a partial batch cannot cross epochs."""

    def __init__(self, config: PreparedDataConfig, data_rank: int = 0, data_world_size: int = 1):
        self.config = config
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        if not 0 <= data_rank < data_world_size:
            raise ValueError("prepared data rank must belong to the data world")
        self.offsets: list[int] = []
        with config.path.open("rb") as stream:
            while line := stream.readline():
                self.offsets.append(stream.tell() - len(line))
                row = PreparedRow.model_validate_json(line)
                if len(row.input_ids) != config.seq_len:
                    raise ValueError("prepared row length must equal data.seq_len")
        if not self.offsets:
            raise ValueError("prepared dataset is empty")
        self.steps_per_epoch = math.ceil(len(self.offsets) / config.batch_size)
        self._epoch: int | None = None
        self._order: list[int] = []

    def __len__(self) -> int:
        return self.steps_per_epoch * self.config.epochs

    def __getitem__(self, index: int) -> PreparedStep:
        if not 0 <= index < len(self):
            raise IndexError(index)
        epoch, batch_index = divmod(index, self.steps_per_epoch)
        if self._epoch != epoch:
            self._order = list(range(len(self.offsets)))
            if self.config.shuffle:
                random.Random(self.config.seed + epoch).shuffle(self._order)
            self._epoch = epoch
        start = batch_index * self.config.batch_size
        indices = self._order[start : start + self.config.batch_size]
        local_indices = indices[self.data_rank :: self.data_world_size]
        rows: list[PreparedRow] = []
        with self.config.path.open("rb") as stream:
            for row_index in local_indices:
                stream.seek(self.offsets[row_index])
                rows.append(PreparedRow.model_validate_json(stream.readline()))
        micro_size = self.config.micro_batch_size
        micro_count = math.ceil(len(indices) / (self.data_world_size * micro_size))
        return PreparedStep(
            epoch=epoch,
            row_indices=indices,
            local_row_indices=local_indices,
            micro_batches=[
                collate_rows(rows[start : start + micro_size], self.config.seq_len)
                for start in range(0, micro_count * micro_size, micro_size)
            ],
        )


def prepared_dataloader(dataset: PreparedDataset) -> StatefulDataLoader:
    """Checkpoint the ordered step cursor, including worker prefetch state."""
    return StatefulDataLoader(dataset, batch_size=None, num_workers=dataset.config.num_workers)


def save_rng(path: Path) -> None:
    """Save per-rank model randomness alongside an SFT checkpoint."""
    torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}, path)


def load_rng(path: Path) -> None:
    """Restore randomness after constructing the model and dataloader iterator."""
    state = torch.load(path, weights_only=True)
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])
