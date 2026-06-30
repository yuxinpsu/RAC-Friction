import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "output" / "features" / "embedding_rac_features.npz"
SIGNALS = PROJECT_ROOT / "output" / "signals" / "predictive_oof_signals.npz"
OUT_DIR = PROJECT_ROOT / "ckpt" / "retrieval"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
K = 32
N_CLUSTERS = 1024
CANDIDATE_MIN = 1024
BATCH_SIZE = 8192
EPOCHS = 12
EMBED_DIM = 64

np.random.seed(SEED)
torch.manual_seed(SEED)


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(z_true, z_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        z_true, z_pred, average="macro", zero_division=0
    )
    return {
        "Accuracy": float(accuracy_score(z_true, z_pred)),
        "Macro_Precision": float(p),
        "Macro_Recall": float(r),
        "Macro_F1": float(f1),
        "Weighted_F1": float(f1_score(z_true, z_pred, average="weighted", zero_division=0)),
    }


class StrongBackbone(nn.Module):
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
        return h, self.reg(h).squeeze(1), self.cls(h)


def train_backbone(x_train, y_train, z_train):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training stronger neural backbone on {device}...", flush=True)
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

    model = StrongBackbone(x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    reg_loss = nn.SmoothL1Loss(beta=0.5)
    cls_loss = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    ds = TensorDataset(
        torch.from_numpy(x_train[tr_idx]),
        torch.from_numpy(y_scaled[tr_idx]),
        torch.from_numpy(z_train[tr_idx].astype(np.int64)),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    x_val = torch.from_numpy(x_train[val_idx]).to(device)
    y_val = torch.from_numpy(y_scaled[val_idx]).to(device)
    z_val = torch.from_numpy(z_train[val_idx].astype(np.int64)).to(device)

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb, zb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            zb = zb.to(device, non_blocking=True)
            _, yr, zlogit = model(xb)
            prob = torch.softmax(zlogit, dim=1)
            expected_cls = (prob * torch.arange(10, device=device, dtype=torch.float32)[None, :]).sum(1)
            target_cls = zb.float()
            ordinal_loss = torch.mean(torch.abs(expected_cls - target_cls) / 9.0)
            loss = reg_loss(yr, yb) + 0.55 * cls_loss(zlogit, zb) + 0.15 * ordinal_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        model.eval()
        with torch.no_grad():
            _, yr, zlogit = model(x_val)
            val_mse = float(torch.mean((yr - y_val) ** 2).detach().cpu())
            val_acc = float((zlogit.argmax(1) == z_val).float().mean().detach().cpu())
            val_macro = float(
                f1_score(
                    z_val.detach().cpu().numpy(),
                    zlogit.argmax(1).detach().cpu().numpy(),
                    average="macro",
                    zero_division=0,
                )
            )
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_mse_scaled": val_mse,
            "val_acc": val_acc,
            "val_macro_f1": val_macro,
        }
        history.append(rec)
        print(
            f"epoch={epoch} loss={rec['loss']:.4f} val_mse={val_mse:.4f} "
            f"val_acc={val_acc:.4f} val_macro_f1={val_macro:.4f}",
            flush=True,
        )
    return model, history


def encode_backbone(model, x):
    device = next(model.parameters()).device
    model.eval()
    h_out, yr_out, zp_out = [], [], []
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            xb = torch.from_numpy(x[start:start + 65536]).to(device)
            h, yr, zlogit = model(xb)
            h = torch.nn.functional.normalize(h, p=2, dim=1)
            h_out.append(h.cpu().numpy().astype(np.float32))
            yr_out.append(yr.cpu().numpy().astype(np.float32))
            zp_out.append(torch.softmax(zlogit, dim=1).cpu().numpy().astype(np.float32))
    return np.vstack(h_out), np.concatenate(yr_out), np.vstack(zp_out)


def standardize(train_block, test_block):
    train_block = np.nan_to_num(train_block.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    test_block = np.nan_to_num(test_block.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = train_block.mean(axis=0, keepdims=True)
    std = train_block.std(axis=0, keepdims=True) + 1e-6
    return ((train_block - mean) / std).astype(np.float32), ((test_block - mean) / std).astype(np.float32)


def row_normalize(a):
    return (a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def build_space(blocks):
    train_parts = []
    test_parts = []
    for weight, tr, te in blocks:
        tr_s, te_s = standardize(tr, te)
        train_parts.append(float(weight) * row_normalize(tr_s))
        test_parts.append(float(weight) * row_normalize(te_s))
    train_space = row_normalize(np.hstack(train_parts).astype(np.float32))
    test_space = row_normalize(np.hstack(test_parts).astype(np.float32))
    return train_space, test_space


def topk(q_block, cand_block, k):
    q = torch.from_numpy(q_block)
    c = torch.from_numpy(cand_block.T.copy())
    sims = q @ c
    vals, inds = torch.topk(sims, k=min(k, cand_block.shape[0]), dim=1)
    return vals.numpy().astype(np.float32), inds.numpy().astype(np.int64)


def analog_from_neighbors(inds, sims, y_train, z_train):
    dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sims))
    weights = 1.0 / (dist + 1e-4)
    weights = weights / weights.sum(axis=1, keepdims=True)
    yy = y_train[inds]
    zz = z_train[inds]
    mean = (weights * yy).sum(axis=1)
    var = (weights * (yy - mean[:, None]) ** 2).sum(axis=1)
    std = np.sqrt(np.maximum(var, 0.0))
    q25 = np.quantile(yy, 0.25, axis=1)
    med = np.quantile(yy, 0.50, axis=1)
    q75 = np.quantile(yy, 0.75, axis=1)
    cont = [
        mean,
        std,
        q25,
        med,
        q75,
        yy.min(axis=1),
        yy.max(axis=1),
        yy[:, 0],
        yy[:, :3].mean(axis=1),
        yy[:, :5].mean(axis=1),
    ]
    probs = np.zeros((len(yy), 10), dtype=np.float32)
    for cls in range(10):
        probs[:, cls] = (weights * (zz == cls)).sum(axis=1)
    entropy = -(probs * np.log(probs + 1e-6)).sum(axis=1)
    maxp = probs.max(axis=1)
    exp_cls = (probs * np.arange(10, dtype=np.float32)[None, :]).sum(axis=1)
    dmean = (weights * dist).sum(axis=1)
    dstd = np.sqrt(np.maximum((weights * (dist - dmean[:, None]) ** 2).sum(axis=1), 0.0))
    return np.column_stack(cont + [probs, entropy, maxp, exp_cls, dmean, dstd]).astype(np.float32)


def make_analog_features(train_space, query_space, y_train, z_train, query_is_train):
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=8192,
        random_state=SEED,
        n_init=3,
        max_iter=120,
        reassignment_ratio=0.01,
    )
    train_cluster = kmeans.fit_predict(train_space)
    query_cluster = train_cluster if query_is_train else kmeans.predict(query_space)
    centers = kmeans.cluster_centers_.astype(np.float32)
    centers = row_normalize(centers)
    center_sims = centers @ centers.T
    nearest_clusters = np.argsort(-center_sims, axis=1)
    cluster_members = [np.where(train_cluster == c)[0] for c in range(N_CLUSTERS)]

    feats = np.zeros((len(query_space), 25), dtype=np.float32)
    processed = 0
    local_k = K + 12 if query_is_train else K

    for c_id in range(N_CLUSTERS):
        q_idx = np.where(query_cluster == c_id)[0]
        if len(q_idx) == 0:
            continue
        cand_parts = []
        cand_count = 0
        for nc in nearest_clusters[c_id]:
            members = cluster_members[int(nc)]
            cand_parts.append(members)
            cand_count += len(members)
            if cand_count >= max(CANDIDATE_MIN, K + 32):
                break
        cand = np.concatenate(cand_parts)
        cand_space = train_space[cand]
        out = []
        for start in range(0, len(q_idx), 2048):
            rows = q_idx[start:start + 2048]
            vals, inds_local = topk(query_space[rows], cand_space, local_k)
            inds_global = cand[inds_local]
            if query_is_train:
                cleaned = []
                cleaned_sims = []
                for row_no, qi in enumerate(rows):
                    keep = inds_global[row_no][inds_global[row_no] != qi][:K]
                    if len(keep) < K:
                        fallback = inds_global[row_no][inds_global[row_no] != qi]
                        keep = np.resize(fallback, K)
                    cleaned.append(keep)
                    cleaned_sims.append(np.sum(train_space[keep] * query_space[qi][None, :], axis=1))
                inds_global = np.vstack(cleaned)
                sims = np.vstack(cleaned_sims).astype(np.float32)
            else:
                inds_global = inds_global[:, :K]
                sims = vals[:, :K]
            out.append(analog_from_neighbors(inds_global, sims, y_train, z_train))
        feats[q_idx] = np.vstack(out)
        processed += len(q_idx)
        if processed % 50_000 < len(q_idx):
            print(f"retrieval processed {processed}/{len(query_space)}", flush=True)
    return feats


def main():
    start_time = time.time()
    base = np.load(BASE)
    signals = np.load(SIGNALS)
    x_train = base["x_train"].astype(np.float32)
    x_test = base["x_test"].astype(np.float32)
    y_train = base["y_train"].astype(np.float32)
    y_test = base["y_test"].astype(np.float32)
    z_train = base["z_train"].astype(int)
    z_test = base["z_test"].astype(int)

    model, history = train_backbone(x_train, y_train, z_train)
    train_emb, train_nn_reg, train_nn_proba = encode_backbone(model, x_train)
    test_emb, test_nn_reg, test_nn_proba = encode_backbone(model, x_test)

    oof_reg = signals["oof_reg"].astype(np.float32)[:, None]
    test_reg = signals["test_reg"].astype(np.float32)[:, None]
    oof_proba = signals["oof_proba"].astype(np.float32)
    test_proba = signals["test_proba"].astype(np.float32)
    nn_reg_train = train_nn_reg[:, None]
    nn_reg_test = test_nn_reg[:, None]

    variants = {
        "raw_improved": [(1.0, x_train, x_test)],
        "strong_emb64": [(1.0, train_emb, test_emb)],
        "raw_plus_emb64": [(0.65, x_train, x_test), (0.35, train_emb, test_emb)],
        "semantic_pred": [(0.45, oof_reg, test_reg), (0.55, oof_proba, test_proba)],
        "emb_plus_semantic": [(0.45, train_emb, test_emb), (0.25, oof_reg, test_reg), (0.30, oof_proba, test_proba)],
        "raw_emb_semantic": [(0.40, x_train, x_test), (0.30, train_emb, test_emb), (0.15, oof_reg, test_reg), (0.15, oof_proba, test_proba)],
        "raw_semantic": [(0.50, x_train, x_test), (0.25, oof_reg, test_reg), (0.25, oof_proba, test_proba)],
        "nn_semantic": [(0.45, train_emb, test_emb), (0.25, nn_reg_train, nn_reg_test), (0.30, train_nn_proba, test_nn_proba)],
    }

    summary = {"backbone_history": history, "variants": {}}
    for name, blocks in variants.items():
        print(f"\n=== variant: {name} ===", flush=True)
        train_space, test_space = build_space(blocks)
        print(f"space dims train={train_space.shape} test={test_space.shape}", flush=True)
        print("train retrieval with self-neighbor removed...", flush=True)
        train_analog = make_analog_features(train_space, train_space, y_train, z_train, query_is_train=True)
        print("test retrieval...", flush=True)
        test_analog = make_analog_features(train_space, test_space, y_train, z_train, query_is_train=False)
        analog_reg = test_analog[:, 0]
        analog_cls = np.argmax(test_analog[:, 10:20], axis=1)
        metrics = {
            "regression": regression_metrics(y_test, analog_reg),
            "classification": classification_metrics(z_test, analog_cls),
        }
        summary["variants"][name] = metrics
        np.savez_compressed(
            OUT_DIR / f"{name}_analog.npz",
            train_analog=train_analog,
            test_analog=test_analog,
        )
        print(json.dumps(metrics, indent=2), flush=True)

    summary["runtime_seconds"] = time.time() - start_time
    with open(OUT_DIR / "variant_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== VARIANT_SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
