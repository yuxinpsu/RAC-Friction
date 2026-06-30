import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import retrieval_utils as sem

BASE = PROJECT_ROOT / "output" / "features" / "embedding_rac_features.npz"
ENCODER_DIR = PROJECT_ROOT / "ckpt" / "encoder"
RETRIEVAL_DIR = PROJECT_ROOT / "ckpt" / "retrieval"
METRIC_DIR = PROJECT_ROOT / "output" / "metrics"
ENCODER_DIR.mkdir(parents=True, exist_ok=True)
RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)
OUT = RETRIEVAL_DIR / "raw_plus_contrastive64_c65536_analog.npz"
EMBED_OUT = ENCODER_DIR / "contrastive64_embeddings.npz"
HISTORY = METRIC_DIR / "contrastive64_history.json"

SEED = 42
EMBED_DIM = 64
BATCH_SIZE = 2048
EPOCHS = 8
TAU = 0.12
SIGMA_Y = 0.075
SIGMA_Z = 1.5
CONTRASTIVE_WEIGHT = 0.18
CLS_WEIGHT = 0.45

np.random.seed(SEED)
torch.manual_seed(SEED)


class ContrastiveBackbone(nn.Module):
    def __init__(self, input_dim, embed_dim=EMBED_DIM, n_classes=10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(256, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, embed_dim),
        )
        self.reg = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.cls = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, n_classes))

    def forward(self, x):
        h = self.body(x)
        hn = torch.nn.functional.normalize(h, p=2, dim=1)
        return hn, self.reg(h).squeeze(1), self.cls(h)


def soft_contrastive_loss(h, y, z):
    # A continuous target-aware contrastive loss. Nearby friction values and nearby
    # ordinal classes receive high target probability in the batch.
    sim = (h @ h.T) / TAU
    n = sim.shape[0]
    eye = torch.eye(n, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)

    dy = torch.abs(y[:, None] - y[None, :])
    dz = torch.abs(z.float()[:, None] - z.float()[None, :])
    target = torch.exp(-dy / SIGMA_Y - dz / SIGMA_Z).masked_fill(eye, 0.0)
    target = target / (target.sum(dim=1, keepdim=True) + 1e-12)
    logp = torch.log_softmax(sim, dim=1)
    return -(target * logp).sum(dim=1).mean()


def train_encoder(x_train, y_train, z_train):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training contrastive encoder on {device}...", flush=True)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-6)
    y_scaled = ((y_train - y_mean) / y_std).astype(np.float32)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x_train))
    val_n = 50_000
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]

    counts = np.bincount(z_train, minlength=10).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / weights.mean()

    model = ContrastiveBackbone(x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=2.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    reg_loss = nn.SmoothL1Loss(beta=0.5)
    cls_loss = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    ds = TensorDataset(
        torch.from_numpy(x_train[tr_idx]),
        torch.from_numpy(y_scaled[tr_idx]),
        torch.from_numpy(y_train[tr_idx]),
        torch.from_numpy(z_train[tr_idx].astype(np.int64)),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    x_val = torch.from_numpy(x_train[val_idx]).to(device)
    y_val_scaled = torch.from_numpy(y_scaled[val_idx]).to(device)
    z_val = torch.from_numpy(z_train[val_idx].astype(np.int64)).to(device)

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        reg_losses = []
        cls_losses = []
        con_losses = []
        for xb, yb_scaled, yb_raw, zb in loader:
            xb = xb.to(device, non_blocking=True)
            yb_scaled = yb_scaled.to(device, non_blocking=True)
            yb_raw = yb_raw.to(device, non_blocking=True)
            zb = zb.to(device, non_blocking=True)
            h, yr, zlogit = model(xb)
            lr = reg_loss(yr, yb_scaled)
            lc = cls_loss(zlogit, zb)
            lcon = soft_contrastive_loss(h, yb_raw, zb)
            loss = lr + CLS_WEIGHT * lc + CONTRASTIVE_WEIGHT * lcon
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            reg_losses.append(float(lr.detach().cpu()))
            cls_losses.append(float(lc.detach().cpu()))
            con_losses.append(float(lcon.detach().cpu()))
        scheduler.step()

        model.eval()
        with torch.no_grad():
            h, yr, zlogit = model(x_val)
            val_mse = float(torch.mean((yr - y_val_scaled) ** 2).detach().cpu())
            val_acc = float((zlogit.argmax(1) == z_val).float().mean().detach().cpu())
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "reg_loss": float(np.mean(reg_losses)),
            "cls_loss": float(np.mean(cls_losses)),
            "contrastive_loss": float(np.mean(con_losses)),
            "val_mse_scaled": val_mse,
            "val_acc": val_acc,
        }
        history.append(rec)
        print(
            f"epoch={epoch} loss={rec['loss']:.4f} reg={rec['reg_loss']:.4f} "
            f"cls={rec['cls_loss']:.4f} con={rec['contrastive_loss']:.4f} "
            f"val_mse={val_mse:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )
    return model, history


def encode(model, x):
    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            xb = torch.from_numpy(x[start:start + 65536]).to(device)
            h, _, _ = model(xb)
            out.append(h.cpu().numpy().astype(np.float32))
    return np.vstack(out)


def main():
    start = time.time()
    base = np.load(BASE)
    x_train = base["x_train"].astype(np.float32)
    x_test = base["x_test"].astype(np.float32)
    y_train = base["y_train"].astype(np.float32)
    z_train = base["z_train"].astype(int)

    model, history = train_encoder(x_train, y_train, z_train)
    print("encoding train/test contrastive embeddings...", flush=True)
    train_emb = encode(model, x_train)
    test_emb = encode(model, x_test)
    np.savez_compressed(EMBED_OUT, train_emb=train_emb, test_emb=test_emb)

    sem.CANDIDATE_MIN = 65536
    sem.N_CLUSTERS = 1024
    train_space, test_space = sem.build_space([
        (0.65, x_train, x_test),
        (0.35, train_emb, test_emb),
    ])

    print("building train analog features with self-neighbor removed...", flush=True)
    train_analog = sem.make_analog_features(train_space, train_space, y_train, z_train, query_is_train=True)
    print("building test analog features...", flush=True)
    test_analog = sem.make_analog_features(train_space, test_space, y_train, z_train, query_is_train=False)
    np.savez_compressed(OUT, train_analog=train_analog, test_analog=test_analog)

    result = {"saved": str(OUT), "runtime_seconds": time.time() - start, "history": history}
    HISTORY.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
