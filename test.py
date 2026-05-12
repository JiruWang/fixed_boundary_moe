import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import time

def tic(msg):
    print(f"\n[START] {msg}", flush=True)
    return time.time(), msg

def toc(t0, msg):
    print(f"[DONE]  {msg}  ({time.time()-t0:.1f}s)", flush=True)



# MODEL_PATH = "mistralai/Mixtral-8x7B-v0.1"
# MODEL_PATH = "mistralai/Mixtral-8x22B-v0.1"
# MODEL_PATH = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_PATH = "mistralai/Mixtral-8x22B-Instruct-v0.1"
LAYER_ID = 20
print(LAYER_ID)
print(MODEL_PATH)
SEQ_LEN = 4000

EPS = 1e-5
RANK = 1248


# ---------- tokenizer & model ----------
t0, msg = tic("load tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
toc(t0, msg)

t0, msg = tic("load model")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=False,
    attn_implementation="flash_attention_2",
)

model.eval()
toc(t0, msg)
print(f"Model loaded on: {next(model.parameters()).device}")


t0, msg = tic("load dataset + tokenize")
ds = load_dataset(
    "wikitext",
    "wikitext-103-raw-v1",
    split="train",
    streaming=True,
)

token_ids = []

for sample in ds:
    text = sample["text"].strip()

    if not text:
        continue

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    token_ids.extend(ids)

    if len(token_ids) >= SEQ_LEN:
        break

input_ids = torch.tensor(
    [token_ids[:SEQ_LEN]],
    dtype=torch.long,
)

device = next(model.parameters()).device
input_ids = input_ids.to(device)

toc(t0, msg)
print(f"Input shape: {input_ids.shape}")


# ---------- collect router input hidden ----------
router_inputs = {}
router = model.model.layers[LAYER_ID].mlp.gate


def save_router_hook(module, inputs, output):
    router_inputs["hidden"] = inputs[0].detach().float()


handle = router.register_forward_hook(save_router_hook)

t0, msg = tic("forward pass")
with torch.no_grad():
    _ = model(input_ids=input_ids)
toc(t0, msg)

handle.remove()

H = router_inputs["hidden"].reshape(-1, router_inputs["hidden"].shape[-1])

print("Router input H:", H.shape)


# ============================================================
# Whitening hidden states  (truncated SVD on GPU, then move to CPU)
# ============================================================

mu = H.mean(dim=0, keepdim=True)
X = H - mu

t0, msg = tic("SVD (svd_lowrank)")
U, S, Vh = torch.svd_lowrank(X, q=RANK, niter=4)
toc(t0, msg)
# svd_lowrank returns V as [hidden_dim, RANK] (columns = right singular vectors)
V = Vh.T                              # [RANK, hidden_dim]
eig = (S ** 2) / (X.shape[0] - 1)    # [RANK]

Z = (X @ V.T) / torch.sqrt(eig + EPS)   # [T, RANK]  — still on GPU

print("\nWhitened H:", Z.shape)
print("Mean abs:", Z.mean(dim=0).abs().mean().item())
print("Std mean:", Z.std(dim=0).mean().item())

Cov_Z = (Z.T @ Z) / (Z.shape[0] - 1)   # [RANK, RANK] on GPU

print("Cov diag mean:", Cov_Z.diag().mean().item())
print("Cov diag std:", Cov_Z.diag().std().item())

off_mask = ~torch.eye(Cov_Z.shape[0], dtype=torch.bool, device=Cov_Z.device)
print("Cov offdiag mean abs:", Cov_Z[off_mask].abs().mean().item())


# ============================================================
# Router weight in original and whitened space
# ============================================================

# Load gate weight directly from HF cache (avoids meta tensor from offloaded model)
def load_gate_weight(model_path, layer_id, device):
    import glob, os
    from safetensors import safe_open

    model_id = model_path.replace("/", "--")
    snapshots = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model_id}/snapshots/*/"))
    snapshot = sorted(snapshots)[-1]

    candidates = [
        f"model.layers.{layer_id}.block_sparse_moe.gate.weight",
        f"model.layers.{layer_id}.mlp.gate.weight",
    ]
    for shard in sorted(glob.glob(snapshot + "*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in candidates:
                if key in f.keys():
                    print(f"  gate weight loaded from {os.path.basename(shard)}, key={key}")
                    return f.get_tensor(key).float().to(device)
    raise RuntimeError(f"Gate weight for layer {layer_id} not found in {snapshot}")

W = load_gate_weight(MODEL_PATH, LAYER_ID, V.device)

# Original router directions
W_orig = F.normalize(W, dim=-1, eps=1e-6)

# Project router to PCA coordinates
W_pca = W @ V.T  # [E, r]

# Correct whitening transform for directions:
# hidden whitening: z = (h - mu) V^T / sqrt(eig)
# router direction should be mapped by the same linear transform
W_white = W_pca / torch.sqrt(eig + EPS).unsqueeze(0)

W_white = F.normalize(W_white, dim=-1, eps=1e-6)

print("W_orig:", W_orig.shape)
print("W_white:", W_white.shape)


# ============================================================
# ETF metrics
# ============================================================

def etf_metrics(W_dir):
    E = W_dir.shape[0]

    C = W_dir @ W_dir.T
    dev = W_dir.device
    mask = ~torch.eye(E, dtype=torch.bool, device=dev)

    off = C[mask]

    target = -1.0 / (E - 1)

    cos_mean = off.mean().item()
    cos_std = off.std().item()
    etf_dev = abs(cos_mean - target)
    max_pair_dev = (off - target).abs().max().item()

    G = torch.full((E, E), target, device=dev)
    G.fill_diagonal_(1.0)

    gram_dev = torch.norm(C - G, p="fro") / torch.norm(G, p="fro")

    return {
        "E": E,
        "target": target,
        "cos_mean": cos_mean,
        "cos_std": cos_std,
        "etf_dev": etf_dev,
        "max_pair_dev": max_pair_dev,
        "gram_dev": gram_dev.item(),
    }


print("\nOriginal space ETF:")
print(etf_metrics(W_orig))

print("\nWhitened space ETF:")
print(etf_metrics(W_white))


# ============================================================
# vMF-like clustering
# ============================================================

def fit_vmf_mixture(W_dir, G=6, seed=42):
    X_np = W_dir.cpu().numpy()

    km = KMeans(
        n_clusters=G,
        n_init=20,
        random_state=seed,
    ).fit(X_np)

    labels = km.labels_

    mus, kappas, pis = [], [], []
    D = W_dir.shape[1]

    for g in range(G):
        idx = labels == g
        Wg = W_dir[idx]
        ng = Wg.shape[0]

        mu = F.normalize(Wg.mean(dim=0), dim=0)

        R = torch.norm(Wg.sum(dim=0)) / max(ng, 1)

        kappa = (R * (D - R**2)) / (1 - R**2 + 1e-6)

        mus.append(mu)
        kappas.append(kappa.item())
        pis.append(ng / len(W_dir))

    return {
        "mus": mus,
        "kappas": kappas,
        "pis": pis,
        "labels": labels,
    }


def cluster_cosine_stats(W_dir, labels, title=""):
    C = W_dir @ W_dir.T
    L = torch.tensor(labels)

    same = []
    diff = []

    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            if L[i] == L[j]:
                same.append(C[i, j].item())
            else:
                diff.append(C[i, j].item())

    same = np.asarray(same)
    diff = np.asarray(diff)

    print(f"\n===== Cosine stats: {title} =====")

    print("Intra-cluster:")
    print("  mean:", same.mean(), "std:", same.std())
    print("  min:", same.min(), "max:", same.max())

    print("Inter-cluster:")
    print("  mean:", diff.mean(), "std:", diff.std())
    print("  min:", diff.min(), "max:", diff.max())

    print("Gap intra - inter:", same.mean() - diff.mean())


def print_cluster_summary(res, title):
    labels = res["labels"]
    unique, counts = np.unique(labels, return_counts=True)

    print(f"\n================ {title} ================")

    print("\nCluster weights (pi):")
    print(res["pis"])

    print("\nConcentration (kappa):")
    print(res["kappas"])

    print("\nCluster sizes:")
    for u, c in zip(unique, counts):
        print(f"cluster {u}: {c}")


# move to CPU only here — KMeans / numpy require CPU tensors
W_orig  = W_orig.cpu()
W_white = W_white.cpu()
Z       = Z.cpu()

t0, msg = tic("KMeans whitened")
res_white = fit_vmf_mixture(W_white, G=6, seed=42)
toc(t0, msg)
labels_white = res_white["labels"]

print_cluster_summary(res_white, "WHITENED SPACE")
cluster_cosine_stats(W_white, labels_white, title="W_white with labels_white")


t0, msg = tic("KMeans original")
res_orig = fit_vmf_mixture(W_orig, G=6, seed=42)
toc(t0, msg)
labels_orig = res_orig["labels"]

print_cluster_summary(res_orig, "ORIGINAL SPACE")
cluster_cosine_stats(W_orig, labels_orig, title="W_orig with labels_orig")


