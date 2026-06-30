import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src import reasoning_core as base_mod


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "output" / "features" / "embedding_rac_features.npz"
EMB = PROJECT_ROOT / "ckpt" / "encoder" / "contrastive64_embeddings.npz"
REASONER_DIR = PROJECT_ROOT / "ckpt" / "reasoner"
METRIC_DIR = PROJECT_ROOT / "output" / "metrics"
REASONER_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)
REASON_FILE = REASONER_DIR / "reasoned_memory_features_contextual.npz"
SUMMARY_FILE = METRIC_DIR / "contextual_reasoner_summary.json"

SEED = 42
EPOCHS = 5
BATCH_QUERY = 768
ATTN_SIM_PRIOR = 1.85
TOKEN_DIM = 128
N_HEADS = 4
N_CONTEXT_LAYERS = 2
HETERO_LOSS_WEIGHT = 0.08
LOGVAR_MIN = -6.0
LOGVAR_MAX = 3.0

np.random.seed(SEED)
torch.manual_seed(SEED)


class ContextualAnalogReasoner(nn.Module):
    def __init__(self, pair_dim, k_neighbors=32, n_classes=10):
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
            nn.Linear(160, TOKEN_DIM),
            nn.LayerNorm(TOKEN_DIM),
            nn.SiLU(),
        )
        self.rank_embedding = nn.Parameter(torch.zeros(1, k_neighbors, TOKEN_DIM))
        nn.init.normal_(self.rank_embedding, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=TOKEN_DIM,
            nhead=N_HEADS,
            dim_feedforward=256,
            dropout=0.06,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(layer, num_layers=N_CONTEXT_LAYERS)
        self.context_norm = nn.LayerNorm(TOKEN_DIM)
        self.delta_head = nn.Linear(TOKEN_DIM, 1)
        self.logvar_head = nn.Linear(TOKEN_DIM, 1)
        self.score_head = nn.Linear(TOKEN_DIM, 1)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(TOKEN_DIM + 5),
            nn.Linear(TOKEN_DIM + 5, 96),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(96, n_classes),
        )

    def forward(self, pair_feat, y_neighbor_scaled, sims):
        bsz, k, dim = pair_feat.shape
        token = self.pair_encoder(pair_feat.reshape(bsz * k, dim)).reshape(bsz, k, TOKEN_DIM)
        token = token + self.rank_embedding[:, :k, :]
        contextual = self.context_norm(self.context_encoder(token))

        delta = self.delta_head(contextual).squeeze(-1)
        logvar = torch.clamp(self.logvar_head(contextual).squeeze(-1), LOGVAR_MIN, LOGVAR_MAX)
        score = self.score_head(contextual).squeeze(-1) + ATTN_SIM_PRIOR * sims
        alpha = torch.softmax(score, dim=1)
        adapted = y_neighbor_scaled + delta
        pred = (alpha * adapted).sum(dim=1)

        spread = torch.sqrt(torch.clamp((alpha * (adapted - pred[:, None]) ** 2).sum(dim=1), min=0.0))
        transition_std = torch.sqrt(
            torch.clamp(
                (alpha * (delta - (alpha * delta).sum(dim=1)[:, None]) ** 2).sum(dim=1),
                min=0.0,
            )
        )
        entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=1) / math.log(k)
        max_alpha = alpha.max(dim=1).values
        pooled = (alpha[:, :, None] * contextual).sum(dim=1)
        cls_logits = self.cls_head(
            torch.cat(
                [pooled, pred[:, None], spread[:, None], transition_std[:, None], entropy[:, None], max_alpha[:, None]],
                dim=1,
            )
        )
        return pred, adapted, delta, logvar, alpha, cls_logits


def predict_direct(model, maker, query_idx, y_mean, y_std):
    model.eval()
    preds = []
    probas = []
    with torch.no_grad():
        for start in range(0, len(query_idx), BATCH_QUERY):
            rows = query_idx[start:start + BATCH_QUERY]
            pair, yi, sims = maker.make(rows)
            pred, _, _, _, _, cls_logits = model(pair, yi, sims)
            preds.append((pred.detach().cpu().numpy() * y_std + y_mean).astype(np.float32))
            probas.append(torch.softmax(cls_logits, dim=1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds), np.vstack(probas)


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

    maker = base_mod.BatchMaker(
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
    model = ContextualAnalogReasoner(pair_dim, k_neighbors=base_mod.K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=6.5e-4, weight_decay=2.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    reg_loss = nn.SmoothL1Loss(beta=0.35)
    delta_loss = nn.SmoothL1Loss(beta=0.50, reduction="none")
    counts = np.bincount(z_train, minlength=10).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    cls_loss = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    history = []
    print(f"training contextual analog reasoner on {device}...", flush=True)
    for epoch in range(1, EPOCHS + 1):
        rng.shuffle(train_idx)
        model.train()
        losses = []
        main_losses = []
        delta_losses = []
        hetero_losses = []
        cls_losses = []
        attn_losses = []
        for start in range(0, len(train_idx), BATCH_QUERY):
            rows = train_idx[start:start + BATCH_QUERY]
            pair, yi, sims = maker.make(rows)
            yj = torch.from_numpy(y_scaled[rows]).to(device)
            zj = torch.from_numpy(z_train[rows]).to(device)
            pred, _, delta, logvar, alpha, cls_logits = model(pair, yi, sims)

            target_delta = yj[:, None] - yi
            l_main = reg_loss(pred, yj)
            l_delta = delta_loss(delta, target_delta).mean()
            transition_residual = target_delta - delta
            l_hetero = 0.5 * (torch.exp(-logvar) * transition_residual.pow(2) + logvar)
            l_hetero = l_hetero.mean()
            with torch.no_grad():
                yy_neighbors = torch.from_numpy(y_train[neighbors["train_inds"][rows]]).to(device)
                yy_query = torch.from_numpy(y_train[rows]).to(device)[:, None]
                raw_diff = torch.abs(yy_neighbors - yy_query)
                target_alpha = torch.softmax(-raw_diff / 0.045 + 1.5 * sims, dim=1)
            l_attn = -(target_alpha * torch.log(alpha + 1e-8)).sum(dim=1).mean()
            l_cls = cls_loss(cls_logits, zj)
            loss = l_main + 0.18 * l_delta + HETERO_LOSS_WEIGHT * l_hetero + 0.05 * l_attn + 0.35 * l_cls

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            main_losses.append(float(l_main.detach().cpu()))
            delta_losses.append(float(l_delta.detach().cpu()))
            hetero_losses.append(float(l_hetero.detach().cpu()))
            cls_losses.append(float(l_cls.detach().cpu()))
            attn_losses.append(float(l_attn.detach().cpu()))
        scheduler.step()

        val_pred, val_proba = predict_direct(model, maker, val_idx, y_mean, y_std)
        val_reg = base_mod.regression_metrics(y_train[val_idx], val_pred)
        val_cls = base_mod.classification_metrics(z_train[val_idx], val_proba.argmax(axis=1))
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "main": float(np.mean(main_losses)),
            "delta": float(np.mean(delta_losses)),
            "heteroscedastic_transition": float(np.mean(hetero_losses)),
            "attention": float(np.mean(attn_losses)),
            "classification": float(np.mean(cls_losses)),
            "val_regression": val_reg,
            "val_classification": val_cls,
        }
        history.append(rec)
        print(
            f"epoch={epoch} loss={rec['loss']:.4f} main={rec['main']:.4f} "
            f"delta={rec['delta']:.4f} het={rec['heteroscedastic_transition']:.4f} "
            f"cls={rec['classification']:.4f} "
            f"val_MAE={val_reg['MAE']:.5f} val_R2={val_reg['R2']:.4f} "
            f"val_macroF1={val_cls['Macro_F1']:.4f}",
            flush=True,
        )
    return model, history, y_mean, y_std


def generate_reason_features(model, maker, query_idx, y_mean, y_std):
    model.eval()
    features = []
    with torch.no_grad():
        for start in range(0, len(query_idx), BATCH_QUERY):
            rows = query_idx[start:start + BATCH_QUERY]
            pair, yi, sims = maker.make(rows)
            pred, adapted, delta, logvar, alpha, cls_logits = model(pair, yi, sims)
            adapted_raw = adapted.detach().cpu().numpy() * y_std + y_mean
            delta_raw = delta.detach().cpu().numpy() * y_std
            pair_var_raw = np.exp(logvar.detach().cpu().numpy()) * (y_std ** 2)
            alpha_np = alpha.detach().cpu().numpy()
            pred_raw = pred.detach().cpu().numpy() * y_std + y_mean
            spread = np.sqrt(np.maximum((alpha_np * (adapted_raw - pred_raw[:, None]) ** 2).sum(axis=1), 0.0))
            qs = np.quantile(adapted_raw, [0.25, 0.50, 0.75], axis=1).T
            ymin = adapted_raw.min(axis=1)
            ymax = adapted_raw.max(axis=1)
            dmean = (alpha_np * delta_raw).sum(axis=1)
            dstd = np.sqrt(np.maximum((alpha_np * (delta_raw - dmean[:, None]) ** 2).sum(axis=1), 0.0))
            hetero_aleatoric_var = (alpha_np * pair_var_raw).sum(axis=1)
            hetero_transition_std = np.sqrt(np.maximum(hetero_aleatoric_var + dstd ** 2, 0.0))
            hetero_aleatoric_std = np.sqrt(np.maximum(hetero_aleatoric_var, 0.0))
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
                    hetero_transition_std,
                    hetero_aleatoric_std,
                    dstd,
                ]
            ).astype(np.float32)
            features.append(block)
            if (start + len(rows)) % 100_000 < len(rows):
                print(f"contextual reason features processed {start + len(rows)}/{len(query_idx)}", flush=True)
    return np.vstack(features).astype(np.float32)

