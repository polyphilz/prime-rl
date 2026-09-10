"""Weights-only initialization of plain exported LoRA adapters."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.tensor import DTensor, distribute_tensor

from prime_rl.configs.trainer import LoRAConfig


def load_initial_adapter(model: nn.Module, directory: Path, lora: LoRAConfig) -> None:
    """Validate every adapter tensor before loading; never load optimizer or base weights.

    The caller must verify the export belongs to the supplied base model. Tensor
    names use the plain A/B format exported from this trainer's linear layers.
    """
    tensors = load_file(directory / "adapter_model.safetensors", device="cpu")
    if not all(name.endswith((".lora_A.weight", ".lora_B.weight")) for name in tensors):
        raise ValueError("initial adapter contains unsupported tensor names")
    recorded = json.loads((directory / "adapter_config.json").read_text())
    expected = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": recorded.get("base_model_name_or_path"),
        "r": lora.rank,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "bias": "none",
        "target_modules": sorted({name.split(".")[-3] for name in tensors}),
        "modules_to_save": None,
    }
    if recorded != expected or lora.modules_to_save:
        raise ValueError("initial adapter configuration differs from plain training LoRA settings")

    parameters = {
        name: value for name, value in get_model_state_dict(model).items() if name.endswith((".lora_A.0", ".lora_B.0"))
    }
    if len(parameters) != sum(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("initial adapter requires only trainable linear LoRA A/B parameters")
    translated: dict[str, torch.Tensor] = {}
    for name, parameter in parameters.items():
        exported = name.removesuffix(".0") + ".weight"
        if exported not in tensors or tensors[exported].shape != parameter.shape:
            raise ValueError(f"initial adapter tensor missing or wrong shape: {exported}")
        if not tensors[exported].is_floating_point() or not torch.isfinite(tensors[exported]).all():
            raise ValueError(f"initial adapter tensor is not finite: {exported}")
        translated[name] = tensors[exported]
    expected_names = {name.removesuffix(".0") + ".weight" for name in parameters}
    if not parameters or set(tensors) != expected_names:
        raise ValueError("initial adapter tensor names differ from the trainable LoRA parameters")

    # State-dict tensors share parameter storage and have canonical names even
    # under activation checkpointing. Copy only A/B weights, preserving FSDP shards.
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = translated[name].to(device=parameter.device, dtype=parameter.dtype)
            if isinstance(parameter, DTensor):
                value = distribute_tensor(value, parameter.device_mesh, parameter.placements)
            parameter.copy_(value)
