import json
import math
import time
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import contextual_reasoner as ctx_mod
from src import reasoning_core as base_mod

BASE = PROJECT_ROOT / "output" / "features" / "embedding_rag_features.npz"
EMB = PROJECT_ROOT / "ckpt" / "encoder" / "contrastive64_embeddings.npz"
RETRIEVAL_DIR = PROJECT_ROOT / "ckpt" / "retrieval"
REASONER_DIR = PROJECT_ROOT / "ckpt" / "reasoner"
METRIC_DIR = PROJECT_ROOT / "output" / "metrics"
RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
REASONER_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_FILE = RETRIEVAL_DIR / "contrastive_hybrid_c65536_top128_candidates.npz"
RERANKED_NEIGHBOR_FILE = RETRIEVAL_DIR / "cross_encoder_reranked_top32_neighbors.npz"
RERANKED_ANALOG_FILE = RETRIEVAL_DIR / "cross_encoder_reranked_analog.npz"
REASON_FILE = REASONER_DIR / "reasoned_memory_features_rerank_contextual.npz"
SUMMARY_FILE = METRIC_DIR / "rerank_contextual_summary.json"

SEED = 42
K_CANDIDATES = 128
K_FINAL = 32
N_CLUSTERS = 1024
CANDIDATE_MIN = 65536
BATCH_QUERY = 768
RERANK_EPOCHS = 4
RERANK_LR = 8.0e-4
TAU_Y = 0.045
TAU_Z = 1.5

np.random.seed(SEED)
torch.manual_seed(SEED)


def row_normalize(a):
    return (a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def topk_gpu(q_block, cand_block, k, device):
    q = torch.from_numpy(q_block).to(device=device, dtype=torch.float32)
    c = torch.from_numpy(cand_block.T.copy()).to(device=device, dtype=torch.float32)
    sims = q @ c
    vals, inds = torch.topk(sims, k=min(k, cand_block.shape[0]), dim=1)
    return vals.detach().cpu().numpy().astype(np.float32), inds.detach().cpu().numpy().astype(np.int64)


def build_candidate_indices(train_space, test_space):
    if CANDIDATE_FILE.exists():
        print(f"loading existing top-{K_CANDIDATES} candidates: {CANDIDATE_FILE}", flush=True)
        return np.load(CANDIDATE_FILE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"building top-{K_CANDIDATES} hybrid retrieval candidates on {device}...", flush=True)
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
        out_inds = np.zeros((n_query, K_CANDIDATES), dtype=np.int32)
        out_sims = np.zeros((n_query, K_CANDIDATES), dtype=np.float32)
        processed = 0
        local_k = K_CANDIDATES + 16 if query_is_train else K_CANDIDATES
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
                if cand_count >= max(CANDIDATE_MIN, K_CANDIDATES + 64):
                    break
            cand = np.concatenate(cand_parts)
            cand_space = train_space[cand]
            for start in range(0, len(q_idx), 768):
                rows = q_idx[start:start + 768]
                vals, inds_local = topk_gpu(query_space[rows], cand_space, local_k, device)
                inds_global = cand[inds_local]
                if query_is_train:
                    cleaned = []
                    cleaned_sims = []
                    for row_no, qi in enumerate(rows):
                        keep = inds_global[row_no][inds_global[row_no] != qi][:K_CANDIDATES]
                        if len(keep) < K_CANDIDATES:
                            fallback = inds_global[row_no][inds_global[row_no] != qi]
                            keep = np.resize(fallback, K_CANDIDATES)
                        cleaned.append(keep)
                        cleaned_sims.append(np.sum(train_space[keep] * query_space[qi][None, :], axis=1))
                    inds_global = np.vstack(cleaned)
                    vals = np.vstack(cleaned_sims).astype(np.float32)
                else:
                    inds_global = inds_global[:, :K_CANDIDATES]
                    vals = vals[:, :K_CANDIDATES]
                out_inds[rows] = inds_global.astype(np.int32)
                out_sims[rows] = vals.astype(np.float32)
            processed += len(q_idx)
            if processed % 50_000 < len(q_idx):
                side = "train" if query_is_train else "test"
                print(f"{side} top-{K_CANDIDATES} retrieval processed {processed}/{n_query}", flush=True)
        return out_inds, out_sims

    train_inds, train_sims = one_side(train_space, train_cluster, True)
    test_inds, test_sims = one_side(test_space, test_cluster, False)
    np.savez_compressed(
        CANDIDATE_FILE,
        train_inds=train_inds,
        train_sims=train_sims,
        test_inds=test_inds,
        test_sims=test_sims,
    )
    print(f"saved candidates: {CANDIDATE_FILE}", flush=True)
    return np.load(CANDIDATE_FILE)


class CrossEncoderReranker(nn.Module):
    def __init__(self, pair_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pair_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(256, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(160, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Linear(96, 1),
        )

    def forward(self, pair_feat):
        bsz, k, dim = pair_feat.shape
        return self.net(pair_feat.reshape(bsz * k, dim)).reshape(bsz, k)


class CandidateBatchMaker:
    def __init__(self, x_query, h_query, x_train, h_train, y_train, z_train, inds, sims, device):
        self.x_query = x_query
        self.h_query = h_query
        self.x_train = x_train
        self.h_train = h_train
        self.y_train = y_train
        self.z_train = z_train
        self.inds = inds
        self.sims = sims
        self.device = device

    def make(self, rows):
        n_idx = self.inds[rows].astype(np.int64)
        qx = self.x_query[rows]
        qh = self.h_query[rows]
        nx = self.x_train[n_idx]
        nh = self.h_train[n_idx]
        bsz, k, _ = nx.shape
        qx_rep = np.repeat(qx[:, None, :], k, axis=1)
        qh_rep = np.repeat(qh[:, None, :], k, axis=1)
        sim = self.sims[rows]
        dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sim))
        yi = self.y_train[n_idx]
        zi = self.z_train[n_idx].astype(np.float32) / 9.0
        pair = np.concatenate(
            [
                qh_rep,
                nh,
                np.abs(qh_rep - nh),
                qh_rep * nh,
                qx_rep,
                nx,
                np.abs(qx_rep - nx),
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
            torch.from_numpy(self.z_train[n_idx].astype(np.float32)).to(self.device),
            torch.from_numpy(sim.astype(np.float32)).to(self.device),
        )


def train_reranker(x_train, h_train, y_train, z_train, candidates):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    maker = CandidateBatchMaker(
        x_train,
        h_train,
        x_train,
        h_train,
        y_train,
        z_train,
        candidates["train_inds"],
        candidates["train_sims"],
        device,
    )
    pair_dim = 4 * h_train.shape[1] + 4 * x_train.shape[1] + 4
    model = CrossEncoderReranker(pair_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=RERANK_LR, weight_decay=2.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=RERANK_EPOCHS)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x_train))
    val_n = 50_000
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    history = []
    print(f"training cross-encoder analog reranker on {device}...", flush=True)
    for epoch in range(1, RERANK_EPOCHS + 1):
        rng.shuffle(train_idx)
        model.train()
        losses = []
        for start in range(0, len(train_idx), BATCH_QUERY):
            rows = train_idx[start:start + BATCH_QUERY]
            pair, yi, zi, sims = maker.make(rows)
            score = model(pair)
            yq = torch.from_numpy(y_train[rows]).to(device)[:, None]
            zq = torch.from_numpy(z_train[rows].astype(np.float32)).to(device)[:, None]
            rel = torch.exp(-torch.abs(yq - yi) / TAU_Y - torch.abs(zq - zi) / TAU_Z + 0.50 * sims)
            target = rel / (rel.sum(dim=1, keepdim=True) + 1e-8)
            logp = torch.log_softmax(score, dim=1)
            loss = -(target * logp).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_metrics = evaluate_reranker_direct(model, maker, val_idx, y_train, z_train)
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_analog": val_metrics,
        }
        history.append(rec)
        print(
            f"rerank epoch={epoch} loss={rec['loss']:.4f} "
            f"val_MAE={val_metrics['MAE']:.5f} val_RMSE={val_metrics['RMSE']:.5f}",
            flush=True,
        )
    return model, history


def evaluate_reranker_direct(model, maker, rows, y_train, z_train):
    model.eval()
    preds = []
    true = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_QUERY):
            batch = rows[start:start + BATCH_QUERY]
            pair, _, _, _ = maker.make(batch)
            score = model(pair)
            top = torch.topk(score, k=K_FINAL, dim=1).indices.detach().cpu().numpy()
            cand = maker.inds[batch].astype(np.int64)
            selected = np.take_along_axis(cand, top, axis=1)
            selected_scores = np.take_along_axis(score.detach().cpu().numpy(), top, axis=1)
            w = np.exp(selected_scores - selected_scores.max(axis=1, keepdims=True))
            w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
            preds.append((w * y_train[selected]).sum(axis=1))
            true.append(y_train[batch])
    pred = np.concatenate(preds)
    y = np.concatenate(true)
    return base_mod.regression_metrics(y, pred)


def score_and_select(model, maker, query_count):
    model.eval()
    out_inds = np.zeros((query_count, K_FINAL), dtype=np.int32)
    out_sims = np.zeros((query_count, K_FINAL), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, query_count, BATCH_QUERY):
            rows = np.arange(start, min(start + BATCH_QUERY, query_count))
            pair, _, _, _ = maker.make(rows)
            score = model(pair)
            vals, top = torch.topk(score, k=K_FINAL, dim=1)
            top_np = top.detach().cpu().numpy()
            vals_np = vals.detach().cpu().numpy().astype(np.float32)
            cand = maker.inds[rows].astype(np.int64)
            out_inds[rows] = np.take_along_axis(cand, top_np, axis=1).astype(np.int32)
            out_sims[rows] = (1.0 / (1.0 + np.exp(-vals_np))).astype(np.float32)
            if (rows[-1] + 1) % 100_000 < len(rows):
                print(f"reranking processed {rows[-1] + 1}/{query_count}", flush=True)
    return out_inds, out_sims


def analog_from_neighbors(inds, sims, y_train, z_train):
    score = np.log(np.clip(sims, 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - sims, 1e-6, 1))
    weights = np.exp(score - score.max(axis=1, keepdims=True))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
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
    dist = 1.0 - sims
    dmean = (weights * dist).sum(axis=1)
    dstd = np.sqrt(np.maximum((weights * (dist - dmean[:, None]) ** 2).sum(axis=1), 0.0))
    return np.column_stack(cont + [probs, entropy, maxp, exp_cls, dmean, dstd]).astype(np.float32)


def main():
    start = time.time()
    base = np.load(BASE)
    emb = np.load(EMB)
    x_train = base["x_train"].astype(np.float32)
    x_test = base["x_test"].astype(np.float32)
    h_train = emb["train_emb"].astype(np.float32)
    h_test = emb["test_emb"].astype(np.float32)
    y_train = base["y_train"].astype(np.float32)
    y_test = base["y_test"].astype(np.float32)
    z_train = base["z_train"].astype(np.int64)
    z_test = base["z_test"].astype(np.int64)

    train_space, test_space = base_mod.build_hybrid_space(x_train, x_test, h_train, h_test)
    candidates = build_candidate_indices(train_space, test_space)
    reranker, rerank_history = train_reranker(x_train, h_train, y_train, z_train, candidates)
    torch.save(
        {"state_dict": reranker.state_dict(), "candidate_count": K_CANDIDATES, "final_neighbors": K_FINAL},
        REASONER_DIR / "cross_encoder_reranker.pt",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_maker = CandidateBatchMaker(
        x_train, h_train, x_train, h_train, y_train, z_train,
        candidates["train_inds"], candidates["train_sims"], device
    )
    test_maker = CandidateBatchMaker(
        x_test, h_test, x_train, h_train, y_train, z_train,
        candidates["test_inds"], candidates["test_sims"], device
    )
    print("reranking train candidates to top-32...", flush=True)
    train_inds, train_sims = score_and_select(reranker, train_maker, len(x_train))
    print("reranking test candidates to top-32...", flush=True)
    test_inds, test_sims = score_and_select(reranker, test_maker, len(x_test))
    np.savez_compressed(
        RERANKED_NEIGHBOR_FILE,
        train_inds=train_inds,
        train_sims=train_sims,
        test_inds=test_inds,
        test_sims=test_sims,
    )
    print("building reranked analog memory...", flush=True)
    train_analog = analog_from_neighbors(train_inds.astype(np.int64), train_sims, y_train, z_train)
    test_analog = analog_from_neighbors(test_inds.astype(np.int64), test_sims, y_train, z_train)
    np.savez_compressed(RERANKED_ANALOG_FILE, train_analog=train_analog, test_analog=test_analog)
    analog_reg = base_mod.regression_metrics(y_test, test_analog[:, 0])
    analog_cls = base_mod.classification_metrics(z_test, test_analog[:, 10:20].argmax(axis=1))

    neighbors = {
        "train_inds": train_inds,
        "train_sims": train_sims,
        "test_inds": test_inds,
        "test_sims": test_sims,
    }
    data = {
        "x_train": x_train,
        "x_test": x_test,
        "train_emb": h_train,
        "test_emb": h_test,
        "y_train": y_train,
        "y_test": y_test,
        "z_train": z_train,
        "z_test": z_test,
    }
    print("training contextual reasoner on reranked analog evidence...", flush=True)
    reasoner, reason_history, y_mean, y_std = ctx_mod.train_reasoner(data, neighbors)
    torch.save(
        {
            "state_dict": reasoner.state_dict(),
            "y_mean": y_mean,
            "y_std": y_std,
            "module": "contextual_reasoner_with_heteroscedastic_transition_loss",
        },
        REASONER_DIR / "contextual_reasoner_heteroscedastic.pt",
    )
    y_scaled = ((y_train - y_mean) / y_std).astype(np.float32)
    reason_train_maker = base_mod.BatchMaker(
        x_train, h_train, x_train, h_train, y_scaled, z_train, train_inds, train_sims, device
    )
    reason_test_maker = base_mod.BatchMaker(
        x_test, h_test, x_train, h_train, y_scaled, z_train, test_inds, test_sims, device
    )
    print("generating train rerank-contextual reasoned memory features...", flush=True)
    train_reason = ctx_mod.generate_reason_features(reasoner, reason_train_maker, np.arange(len(x_train)), y_mean, y_std)
    print("generating test rerank-contextual reasoned memory features...", flush=True)
    test_reason = ctx_mod.generate_reason_features(reasoner, reason_test_maker, np.arange(len(x_test)), y_mean, y_std)
    np.savez_compressed(REASON_FILE, train_reason=train_reason, test_reason=test_reason)
    direct_reg = base_mod.regression_metrics(y_test, test_reason[:, 0])
    direct_cls = base_mod.classification_metrics(z_test, test_reason[:, 11:21].argmax(axis=1))

    summary = {
        "saved_candidates": str(CANDIDATE_FILE),
        "saved_reranked_neighbors": str(RERANKED_NEIGHBOR_FILE),
        "saved_reranked_analog": str(RERANKED_ANALOG_FILE),
        "saved_reason_features": str(REASON_FILE),
        "runtime_seconds": time.time() - start,
        "reranker_history": rerank_history,
        "reranked_analog_only_test": {
            "regression": analog_reg,
            "classification": analog_cls,
        },
        "reasoner_history": reason_history,
        "direct_reasoner_test": {
            "regression": direct_reg,
            "classification": direct_cls,
        },
        "reason_feature_columns": [
            "reasoned_friction",
            "reasoned_dispersion",
            "reasoned_q25",
            "reasoned_median",
            "reasoned_q75",
            "reasoned_min",
            "reasoned_max",
            "transition_delta_mean",
            "transition_delta_std",
            "reasoning_attention_entropy",
            "reasoning_max_attention",
            "reason_class_prob_0",
            "reason_class_prob_1",
            "reason_class_prob_2",
            "reason_class_prob_3",
            "reason_class_prob_4",
            "reason_class_prob_5",
            "reason_class_prob_6",
            "reason_class_prob_7",
            "reason_class_prob_8",
            "reason_class_prob_9",
            "reason_class_entropy",
            "reason_expected_class",
            "heteroscedastic_transition_std",
            "heteroscedastic_aleatoric_std",
            "heteroscedastic_disagreement_std",
        ],
        "notes": {
            "module": "top-128 hybrid retrieval followed by cross-encoder analog reranking to top-32",
            "candidate_count": K_CANDIDATES,
            "final_neighbors": K_FINAL,
            "uncertainty_policy": "heteroscedastic uncertainty is trained in the reasoner and evaluated as a reliability indicator; it is not used by the final predictor",
        },
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== RERANK_CONTEXTUAL_SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
