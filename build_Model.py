import os
import shutil
import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.my_mixtral_ckpt.configuration_my_mixtral import MyMixtralConfig
from .my_mixtral_ckpt.modeling_my_mixtral import MyMixtralForCausalLM

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
SRC_DIR     = os.path.join(ROOT, "mixtral/mixtral_base")   # model source code
CKPT_DIR    = os.path.join(ROOT, "mixtral/mixtral_base")   # output checkpoint
CONFIG_FILE = os.path.join(ROOT, "mixtral/config/base_config.json")

if_build = True
if_load  = True

os.makedirs(CKPT_DIR, exist_ok=True)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Build ─────────────────────────────────────────────────────────────────────
if if_build:
    # 1. Config
    config = MyMixtralConfig.from_pretrained(CONFIG_FILE)
    config.auto_map = {
        "AutoConfig":           "configuration_my_mixtral.MyMixtralConfig",
        "AutoModelForCausalLM": "modeling_my_mixtral.MyMixtralForCausalLM",
    }
    config.architectures = ["MyMixtralForCausalLM"]
    config.save_pretrained(CKPT_DIR)

    # 2. Model
    MyMixtralConfig.register_for_auto_class()
    MyMixtralForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    model = MyMixtralForCausalLM(config)

    # ── 正交初始化 hidden_proj + 小随机 expert_bias ───────────────────────────
    from mixtral.my_mixtral_ckpt.modeling_my_mixtral import MyMixtralRouter
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, MyMixtralRouter):
                nn.init.orthogonal_(module.hidden_proj.weight, gain=1.0)
                nn.init.normal_(module.expert_bias, mean=0.0, std=0.1)
    print("Orthogonal init applied to all MyMixtralRouter.hidden_proj.")

    state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    save_file(state_dict, os.path.join(CKPT_DIR, "model.safetensors"))

    # 3. Tokenizer
    tokenizer.save_pretrained(CKPT_DIR, push_to_hub=False)

    print(f"MyMixtral saved to {CKPT_DIR}")
    n_params  = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    print(f"  Parameters: {n_params:,}  |  Buffers: {n_buffers:,}")

# ── Verify ────────────────────────────────────────────────────────────────────
if if_load:
    model = AutoModelForCausalLM.from_pretrained(CKPT_DIR, trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        ["asgriohsfoihoihoiweho", "afvsafgf"],
        return_tensors="pt", padding=True,
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}

    outputs = model(**inputs, labels=inputs["input_ids"])
    outputs.loss.backward()
    print(f"Verification loss: {outputs.loss.item():.4f}")
