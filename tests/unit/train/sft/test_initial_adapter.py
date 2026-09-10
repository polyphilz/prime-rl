"""Weights-only loading must preserve the base and reject partial adapters."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.trainer import LoRAConfig, ModelConfig
from prime_rl.trainer.sft.adapter import load_initial_adapter


class PlainLoRALayer(nn.Module):
    @property
    def weight(self):
        return self.base_layer.weight


@pytest.fixture
def adapter(tmp_path):
    model = nn.Module()
    model.q_proj = PlainLoRALayer()
    model.q_proj.base_layer = nn.Linear(3, 3, bias=False)
    model.q_proj.base_layer.weight = nn.Parameter(torch.eye(3), requires_grad=False)
    model.q_proj.lora_A = nn.ParameterList([nn.Parameter(torch.zeros(2, 3))])
    model.q_proj.lora_B = nn.ParameterList([nn.Parameter(torch.zeros(3, 2))])

    def flatten_base(module, state, prefix, metadata):
        state[prefix + "weight"] = state.pop(prefix + "base_layer.weight")

    def restore_base(module, state, prefix, *args):
        if prefix + "weight" in state:
            state[prefix + "base_layer.weight"] = state.pop(prefix + "weight")

    model.q_proj.register_state_dict_post_hook(flatten_base)
    model.q_proj.register_load_state_dict_pre_hook(restore_base)
    settings = LoRAConfig(rank=2, alpha=4, dropout=0, target_modules=["q_proj"])
    save_file(
        {"q_proj.lora_A.weight": torch.full((2, 3), 0.25), "q_proj.lora_B.weight": torch.full((3, 2), 0.5)},
        tmp_path / "adapter_model.safetensors",
    )
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": "base",
                "r": 2,
                "lora_alpha": 4,
                "lora_dropout": 0,
                "bias": "none",
                "target_modules": ["q_proj"],
                "modules_to_save": None,
            }
        )
    )
    return model, settings, tmp_path


@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_initial_adapter_loads_all_weights_before_fresh_optimizer(adapter, activation_checkpointing):
    model, settings, directory = adapter
    if activation_checkpointing:
        model.q_proj = checkpoint_wrapper(model.q_proj)
    original_files = {p.name: p.read_bytes() for p in directory.iterdir()}
    base = model.q_proj.base_layer.weight.detach().clone()
    load_initial_adapter(model, directory, settings)
    assert torch.equal(model.q_proj.lora_A[0], torch.full((2, 3), 0.25))
    assert torch.equal(model.q_proj.lora_B[0], torch.full((3, 2), 0.5))
    assert torch.equal(model.q_proj.base_layer.weight, base)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    assert not optimizer.state
    loss = (model.q_proj.lora_B[0] @ model.q_proj.lora_A[0]).square().sum()
    loss.backward()
    optimizer.step()
    assert not torch.equal(model.q_proj.lora_B[0], torch.full((3, 2), 0.5))
    assert torch.equal(model.q_proj.base_layer.weight, base)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == original_files


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "nan", "alpha", "rank", "scaling"])
def test_initial_adapter_rejects_invalid_export_before_changing_weights(adapter, change):
    model, settings, directory = adapter
    weights = directory / "adapter_model.safetensors"
    tensors = load_file(weights)
    config_path = directory / "adapter_config.json"
    config = json.loads(config_path.read_text())
    if change == "missing":
        del tensors["q_proj.lora_B.weight"]
    elif change == "extra":
        tensors["other.lora_A.weight"] = torch.ones(2, 3)
    elif change == "shape":
        tensors["q_proj.lora_B.weight"] = torch.ones(4, 2)
    elif change == "nan":
        tensors["q_proj.lora_B.weight"].fill_(float("nan"))
    elif change == "alpha":
        config["lora_alpha"] = 8
    elif change == "rank":
        config["r"] = 3
    else:
        config["use_rslora"] = True
    save_file(tensors, weights)
    config_path.write_text(json.dumps(config))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="initial adapter"):
        load_initial_adapter(model, directory, settings)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_initial_adapter_config_rejects_resume_and_non_lora(tmp_path):
    with pytest.raises(ValueError, match="requires LoRA"):
        SFTConfig(initial_adapter=tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        SFTConfig(model=ModelConfig(lora=LoRAConfig()), initial_adapter=tmp_path, resume={"step": 2})
    with pytest.raises(ValueError, match="clean must not overlap"):
        SFTConfig(
            model=ModelConfig(lora=LoRAConfig()),
            initial_adapter=tmp_path / "adapter",
            clean=True,
            ckpt={"output_dir": tmp_path},
        )
    assert "initial_adapter" not in SFTConfig().model_dump()
