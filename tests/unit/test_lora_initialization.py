"""Compare fresh adapter initialization, before sampling or optimizer updates."""

import pytest
import torch
from torch import nn

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.lora import LoRAState
from prime_rl.trainer.models.layers.lora import MultiLoRALinear


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_seeded_adapter_initialization(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config = LoRAConfig(rank=16, alpha=32)
    state = LoRAState(config, torch.device(device))
    for name in ("q_proj", "v_proj"):
        module = MultiLoRALinear(nn.Linear(32, 32, device=device), rank=16, n_adapters=1)
        state.register_module(name, module)
    state.reset_adapter_parameters(seed=42)
    first = {name: tensor.clone() for name, tensor in state.adapter_state_dict().items()}
    torch.rand(137, device=device)
    state.reset_adapter_parameters(seed=42)
    assert all(torch.equal(first[name], tensor) for name, tensor in state.adapter_state_dict().items())
    state.reset_adapter_parameters(seed=43)
    different = state.adapter_state_dict()
    assert all(not torch.equal(first[name], tensor) for name, tensor in different.items() if "lora_A" in name)
    assert all(torch.count_nonzero(tensor) == 0 for name, tensor in different.items() if "lora_B" in name)
