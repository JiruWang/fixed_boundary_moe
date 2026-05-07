# test_moe

## 1. Build

`build.py` initializes model weights from scratch and runs a single forward pass on a small batch from OpenWebText, then saves an expert usage heatmap.

```bash
python build.py --mode <base|sim_moe|rand>
```

Three modes are available:

- **base** — Standard Mixtral MoE with the default router.
- **sim_moe** — Mixtral with a similarity-based router.
- **rand** — Mixtral with a random router.

Outputs (saved to `mixtral_<mode>/`):

- `model.safetensors` — initialized weights
- `<mode>_moe.png` — expert usage heatmap from the first forward pass

## 2. Eval

The evaluation script uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Install it first:

```bash
pip install lm-eval
```

Then set `CKPT_DIR` in `eval/eval_my_mixtral.sh` to your checkpoint path and run:

```bash
bash eval/eval_my_mixtral.sh
```
