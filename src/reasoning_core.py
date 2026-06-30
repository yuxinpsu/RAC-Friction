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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "output" / "features" / "embedding_rac_features.npz"
EMB = PROJECT_ROOT / "ckpt" / "encoder" / "contrastive64_embeddings.npz"
RETRIEVAL_DIR = PROJECT_ROOT / "ckpt" / "retrieval"
REASONER_DIR = PROJECT_ROOT / "ckpt" / "reasoner"
METRIC_DIR = PROJECT_ROOT / "output" / "metrics"
RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
REASONER_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

NEIGHBOR_FILE = RETRIEVAL_DIR / "contrastive_hybrid_c65536_neighbors.npz"
REASON_FILE = REASONER_DIR / "reasoned_memory_features.npz"
SUMMARY_FILE = METRIC_DIR / "reasoner_summary.json"

SEED = 42
K = 32
N_CLUSTERS = 1024
CANDIDATE_MIN = 65536
BATCH_QUERY = 1024
EPOCHS = 5
EMBED_WEIGHT = 0.35
RAW_WEIGHT = 0.65
ATTN_SIM_PRIOR = 2.0

np.random.seed(SEED)
torch.manual_seed(SEED)


def regression_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
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


def standardize(train_block, test_block):
    train_block = np.nan_to_num(train_block.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    test_block = np.nan_to_num(test_block.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = train_block.mean(axis=0, keepdims=True)
    std = train_block.std(axis=0, keepdims=True) + 1e-6
    return ((train_block - mean) / std).astype(np.float32), ((test_block - mean) / std).astype(np.float32)


def row_normalize(a):
    return (a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def build_hybrid_space(x_train, x_test, h_train, h_test):
    x_tr, x_te = standardize(x_train, x_test)
    h_tr, h_te = standardize(h_train, h_test)
    train_space = row_normalize(
        np.hstack([
            RAW_WEIGHT * row_normalize(x_tr),
            EMBED_WEIGHT * row_normalize(h_tr),
        ]).astype(np.float32)
    )
    test_space = row_normalize(
        np.hstack([
            RAW_WEIGHT * row_normalize(x_te),
            EMBED_WEIGHT * row_normalize(h_te),
        ]).astype(np.float32)
    )
    return train_space, test_space


def topk_gpu(q_block, cand_block, k, device):
    q = torch.from_numpy(q_block).to(device=device, dtype=torch.float32)
    c = torch.from_numpy(cand_block.T.copy()).to(device=device, dtype=torch.float32)
    sims = q @ c
    vals, inds = torch.topk(sims, k=min(k, cand_block.shape[0]), dim=1)
    return vals.detach().cpu().numpy().astype(np.float32), inds.detach().cpu().numpy().astype(np.int64)


def build_neighbor_indices(train_space, test_space):
    if NEIGHBOR_FILE.exists():
        print(f"loading existing neighbors: {NEIGHBOR_FILE}", flush=True)
        return np.load(NEIGHBOR_FILE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"building contrastive-hybrid retrieval indices on {device}...", flush=True)
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=8192,
        random_state=SEED,
        n_init=3,
        max_iter=120,
        reassignment_ratio=0.01,
    )
    train_cluster = kmeans.fit_predict(train_space)
    test_cluster = kmeans.predict(test_space)
    centers = row_normalize(kmeans.cluster_centers_.astype(np.float32))
    center_sims = centers @ centers.T
    nearest_clusters = np.argsort(-center_sims, axis=1)
    cluster_members = [np.where(train_cluster == c)[0].astype(np.int64) for c in range(N_CLUSTERS)]

    def one_side(query_space, query_cluster, query_is_train):
        n_query = len(query_space)
        out_inds = np.zeros((n_query, K), dtype=np.int32)
        out_sims = np.zeros((n_query, K), dtype=np.float32)
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
            for start in range(0, len(q_idx), 1024):
                rows = q_idx[start:start + 1024]
                vals, inds_local = topk_gpu(query_space[rows], cand_space, local_k, device)
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
                    vals = np.vstack(cleaned_sims).astype(np.float32)
                else:
                    inds_global = inds_global[:, :K]
                    vals = vals[:, :K]
                out_inds[rows] = inds_global.astype(np.int32)
                out_sims[rows] = vals.astype(np.float32)
            processed += len(q_idx)
            if processed % 50_000 < len(q_idx):
                side = "train" if query_is_train else "test"
                print(f"{side} retrieval processed {processed}/{n_query}", flush=True)
        return out_inds, out_sims

    train_inds, train_sims = one_side(train_space, train_cluster, True)
    test_inds, test_sims = one_side(test_space, test_cluster, False)
    np.savez_compressed(
        NEIGHBOR_FILE,
        train_inds=train_inds,
        train_sims=train_sims,
        test_inds=test_inds,
        test_sims=test_sims,
    )
    print(f"saved neighbors: {NEIGHBOR_FILE}", flush=True)
    return np.load(NEIGHBOR_FILE)


class TransitionReasoner(nn.Module):
    def __init__(self, pair_dim, n_classes=10):
        super().__init__()
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(256, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(160, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.delta_head = nn.Linear(128, 1)
        self.score_head = nn.Linear(128, 1)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(128 + 4),
            nn.Linear(128 + 4, 96),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(96, n_classes),
        )

    def forward(self, pair_feat, y_neighbor_scaled, sims):
        bsz, k, dim = pair_feat.shape
        token = self.pair_encoder(pair_feat.reshape(bsz * k, dim)).reshape(bsz, k, 128)
        delta = self.delta_head(token).squeeze(-1)
        score = self.score_head(token).squeeze(-1) + ATTN_SIM_PRIOR * sims
        alpha = torch.softmax(score, dim=1)
        adapted = y_neighbor_scaled + delta
        pred = (alpha * adapted).sum(dim=1)
        mean = pred[:, None]
        spread = torch.sqrt(torch.clamp((alpha * (adapted - mean) ** 2).sum(dim=1), min=0.0))
        entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=1) / math.log(k)
        max_alpha = alpha.max(dim=1).values
        pooled = (alpha[:, :, None] * token).sum(dim=1)
        cls_logits = self.cls_head(torch.cat([pooled, pred[:, None], spread[:, None], entropy[:, None], max_alpha[:, None]], dim=1))
        return pred, adapted, delta, alpha, cls_logits


class BatchMaker:
    def __init__(self, x_query, h_query, x_train, h_train, y_scaled, z_train, inds, sims, device):
        self.x_query = x_query
        self.h_query = h_query
        self.x_train = x_train
        self.h_train = h_train
        self.y_scaled = y_scaled
        self.z_train = z_train
        self.inds = inds
        self.sims = sims
        self.device = device

    def make(self, query_idx):
        n_idx = self.inds[query_idx].astype(np.int64)
        qx = self.x_query[query_idx]
        qh = self.h_query[query_idx]
        nx = self.x_train[n_idx]
        nh = self.h_train[n_idx]
        bsz, k, _ = nx.shape
        qx_rep = np.repeat(qx[:, None, :], k, axis=1)
        qh_rep = np.repeat(qh[:, None, :], k, axis=1)
        yi = self.y_scaled[n_idx]
        zi = self.z_train[n_idx].astype(np.float32) / 9.0
        sim = self.sims[query_idx]
        dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sim))
        pair = np.concatenate(
            [
                qh_rep,
                nh,
                qh_rep - nh,
                qh_rep * nh,
                qx_rep,
                nx,
                qx_rep - nx,
                qx_rep * nx,
                yi[:, :, None],
                zi[:, :, None],
                sim[:, :, None],
                dist[:, :, None],
            ],
            axis=2,
        ).astype(np.float32)
        return (
            torch.from_numpy(pair).to(self.device),
            torch.from_numpy(yi.astype(np.float32)).to(self.device),
            torch.from_numpy(sim.astype(np.float32)).to(self.device),
        )


def train_reasoner(data, neighbors):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train = data["x_train"].astype(np.float32)
    h_train = data["train_emb"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    z_train = data["z_train"].astype(np.int64)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-6)
    y_scaled = ((y_train - y_mean) / y_std).astype(np.float32)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x_train))
    val_n = 50_000
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]

    maker = BatchMaker(
        x_train,
        h_train,
        x_train,
        h_train,
        y_scaled,
        z_train,
        neighbors["train_inds"],
        neighbors["train_sims"],
        device,
    )
    pair_dim = 4 * h_train.shape[1] + 4 * x_train.shape[1] + 4
    model = TransitionReasoner(pair_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=2.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    reg_loss = nn.SmoothL1Loss(beta=0.35)
    delta_loss = nn.SmoothL1Loss(beta=0.50, reduction="none")
    counts = np.bincount(z_train, minlength=10).astype(np.float32)
    class_weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    class_weights = class_weights / class_weights.mean()
    cls_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    history = []
    print(f"training analogical transition reasoner on {device}...", flush=True)
    for epoch in range(1, EPOCHS + 1):
        rng.shuffle(train_idx)
        model.train()
        losses = []
        main_losses = []
        aux_losses = []
        cls_losses = []
        attn_losses = []
        for start in range(0, len(train_idx), BATCH_QUERY):
            rows = train_idx[start:start + BATCH_QUERY]
            pair, yi, sims = maker.make(rows)
            yj = torch.from_numpy(y_scaled[rows]).to(device)
            zj = torch.from_numpy(z_train[rows]).to(device)
            pred, adapted, delta, alpha, cls_logits = model(pair, yi, sims)

            target_delta = yj[:, None] - yi
            l_main = reg_loss(pred, yj)
            l_delta = delta_loss(delta, target_delta).mean()
            with torch.no_grad():
                raw_diff = torch.abs(torch.from_numpy(y_train[neighbors["train_inds"][rows]]).to(device) - torch.from_numpy(y_train[rows]).to(device)[:, None])
                target_alpha = torch.softmax(-raw_diff / 0.045 + 1.5 * sims, dim=1)
            l_attn = -(target_alpha * torch.log(alpha + 1e-8)).sum(dim=1).mean()
            l_cls = cls_loss(cls_logits, zj)
            loss = l_main + 0.20 * l_delta + 0.05 * l_attn + 0.35 * l_cls

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            main_losses.append(float(l_main.detach().cpu()))
            aux_losses.append(float(l_delta.detach().cpu()))
            cls_losses.append(float(l_cls.detach().cpu()))
            attn_losses.append(float(l_attn.detach().cpu()))
        scheduler.step()

        val_pred, val_proba = predict_direct(model, maker, val_idx, y_mean, y_std)
        val_reg = regression_metrics(y_train[val_idx], val_pred)
        val_cls = classification_metrics(z_train[val_idx], val_proba.argmax(axis=1))
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "main": float(np.mean(main_losses)),
            "delta": float(np.mean(aux_losses)),
            "attention": float(np.mean(attn_losses)),
            "classification": float(np.mean(cls_losses)),
            "val_regression": val_reg,
            "val_classification": val_cls,
        }
        history.append(rec)
        print(
            f"epoch={epoch} loss={rec['loss']:.4f} main={rec['main']:.4f} "
            f"delta={rec['delta']:.4f} cls={rec['classification']:.4f} "
            f"val_MAE={val_reg['MAE']:.5f} val_R2={val_reg['R2']:.4f} "
            f"val_macroF1={val_cls['Macro_F1']:.4f}",
            flush=True,
        )
    return model, history, y_mean, y_std


def predict_direct(model, maker, query_idx, y_mean, y_std):
    model.eval()
    preds = []
    probas = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(query_idx), BATCH_QUERY):
            rows = query_idx[start:start + BATCH_QUERY]
            pair, yi, sims = maker.make(rows)
            pred, _, _, _, cls_logits = model(pair, yi, sims)
            preds.append((pred.detach().cpu().numpy() * y_std + y_mean).astype(np.float32))
            probas.append(torch.softmax(cls_logits, dim=1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds), np.vstack(probas)


def generate_reason_features(model, maker, query_idx, y_mean, y_std):
    model.eval()
    features = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(query_idx), BATCH_QUERY):
            rows = query_idx[start:start + BATCH_QUERY]
            pair, yi, sims = maker.make(rows)
            pred, adapted, delta, alpha, cls_logits = model(pair, yi, sims)
            adapted_raw = adapted.detach().cpu().numpy() * y_std + y_mean
            delta_raw = delta.detach().cpu().numpy() * y_std
            alpha_np = alpha.detach().cpu().numpy()
            pred_raw = pred.detach().cpu().numpy() * y_std + y_mean
            spread = np.sqrt(np.maximum((alpha_np * (adapted_raw - pred_raw[:, None]) ** 2).sum(axis=1), 0.0))
            qs = np.quantile(adapted_raw, [0.25, 0.50, 0.75], axis=1).T
            ymin = adapted_raw.min(axis=1)
            ymax = adapted_raw.max(axis=1)
            dmean = (alpha_np * delta_raw).sum(axis=1)
            dstd = np.sqrt(np.maximum((alpha_np * (delta_raw - dmean[:, None]) ** 2).sum(axis=1), 0.0))
            entropy = -(alpha_np * np.log(alpha_np + 1e-8)).sum(axis=1) / math.log(alpha_np.shape[1])
            max_alpha = alpha_np.max(axis=1)
            proba = torch.softmax(cls_logits, dim=1).detach().cpu().numpy().astype(np.float32)
            cls_entropy = -(proba * np.log(proba + 1e-8)).sum(axis=1)
            cls_expected = (proba * np.arange(10, dtype=np.float32)[None, :]).sum(axis=1)
            block = np.column_stack(
                [
                    pred_raw,
                    spread,
                    qs,
                    ymin,
                    ymax,
                    dmean,
                    dstd,
                    entropy,
                    max_alpha,
                    proba,
                    cls_entropy,
                    cls_expected,
                ]
            ).astype(np.float32)
            features.append(block)
            if (start + len(rows)) % 100_000 < len(rows):
                print(f"reason features processed {start + len(rows)}/{len(query_idx)}", flush=True)
    return np.vstack(features).astype(np.float32)

