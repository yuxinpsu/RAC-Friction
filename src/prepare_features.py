import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.parent / "Data"
OUT = PROJECT_ROOT / "output" / "features"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

SEED = 42
K = 32
EMBED_DIM = 24
N_CLUSTERS = 512
EPOCHS = 6
BATCH_SIZE = 8192

np.random.seed(SEED)
torch.manual_seed(SEED)


def parse_linestring(wkt: str):
    if not isinstance(wkt, str):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", wkt)
    if len(nums) < 4:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pts = [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums) - 1, 2)]
    if len(pts) < 2:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    lon = np.array([p[0] for p in pts], dtype=np.float64)
    lat = np.array([p[1] for p in pts], dtype=np.float64)
    lat0 = np.deg2rad(float(lat.mean()))
    x = lon * 111_320.0 * np.cos(lat0)
    y = lat * 110_540.0
    dx = np.diff(x)
    dy = np.diff(y)
    seg = np.sqrt(dx * dx + dy * dy)
    length = float(seg.sum())
    chord = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
    straightness = chord / (length + 1e-6)
    bearing = math.atan2(y[-1] - y[0], x[-1] - x[0])
    span_x = float(x.max() - x.min())
    span_y = float(y.max() - y.min())
    return (length, straightness, math.sin(bearing), math.cos(bearing), span_x, span_y)


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    ts = pd.to_datetime(all_df["timestamp"], errors="coerce", utc=True)
    hour = ts.dt.hour.fillna(0).astype(float).to_numpy()
    month = ts.dt.month.fillna(1).astype(float).to_numpy()
    dow = ts.dt.dayofweek.fillna(0).astype(float).to_numpy()

    feats = {}
    for c in [
        "rain",
        "snowfall",
        "snow_depth",
        "temperatureAverage",
        "wiperSpeedAverage",
        "inrix_speed",
        "inrix_hist_avg_speed",
        "inrix_ref_speed",
    ]:
        feats[c] = pd.to_numeric(all_df[c], errors="coerce").to_numpy(dtype=np.float32)

    temp_f = feats["temperatureAverage"]
    ref = feats["inrix_ref_speed"]
    speed = feats["inrix_speed"]
    hist = feats["inrix_hist_avg_speed"]

    numeric = [
        feats["rain"],
        feats["snowfall"],
        feats["snow_depth"],
        feats["temperatureAverage"],
        feats["wiperSpeedAverage"],
        feats["inrix_speed"],
        feats["inrix_hist_avg_speed"],
        feats["inrix_ref_speed"],
        np.log1p(np.maximum(feats["rain"], 0)),
        np.log1p(np.maximum(feats["snowfall"], 0)),
        np.log1p(np.maximum(feats["snow_depth"], 0)),
        np.log1p(np.maximum(feats["wiperSpeedAverage"], 0)),
        (temp_f <= 32.0).astype(np.float32),
        (np.abs(temp_f - 32.0) <= 3.0).astype(np.float32),
        speed / (ref + 1e-3),
        (ref - speed) / (ref + 1e-3),
        speed / (hist + 1e-3),
        (hist - speed) / (hist + 1e-3),
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * month / 12.0),
        np.cos(2 * np.pi * month / 12.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        (dow >= 5).astype(np.float32),
    ]

    geom = np.array([parse_linestring(v) for v in all_df["geometryWkt"].tolist()], dtype=np.float32)
    for i in range(geom.shape[1]):
        numeric.append(geom[:, i])
    numeric = np.vstack(numeric).T.astype(np.float32)

    materials = sorted(train_df["Road_Material"].fillna("missing").astype(str).unique().tolist())
    mat = all_df["Road_Material"].fillna("missing").astype(str)
    one_hot = np.zeros((len(all_df), len(materials)), dtype=np.float32)
    index = {m: i for i, m in enumerate(materials)}
    for r, v in enumerate(mat):
        if v in index:
            one_hot[r, index[v]] = 1.0

    n_train = len(train_df)
    scaler = StandardScaler()
    numeric_train = scaler.fit_transform(numeric[:n_train]).astype(np.float32)
    numeric_test = scaler.transform(numeric[n_train:]).astype(np.float32)
    x_train = np.hstack([numeric_train, one_hot[:n_train]]).astype(np.float32)
    x_test = np.hstack([numeric_test, one_hot[n_train:]]).astype(np.float32)
    return x_train, x_test


class Backbone(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, embed_dim),
        )
        self.reg = nn.Linear(embed_dim, 1)
        self.cls = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        h = self.net(x)
        return h, self.reg(h).squeeze(1), self.cls(h)


def train_backbone(x_train, y_train, z_train):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-6)
    y_scaled = ((y_train - y_mean) / y_std).astype(np.float32)

    n = len(x_train)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    val_n = min(50_000, max(10_000, int(0.1 * n)))
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]

    model = Backbone(x_train.shape[1], EMBED_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

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
            _, pr, pc = model(xb)
            loss = mse(pr, yb) + 0.35 * ce(pc, zb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            _, vr, vc = model(x_val)
            val_mse = float(mse(vr, y_val).detach().cpu())
            val_acc = float((vc.argmax(1) == z_val).float().mean().detach().cpu())
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_mse_scaled": val_mse, "val_acc": val_acc})
        print(f"epoch={epoch} loss={history[-1]['loss']:.4f} val_mse={val_mse:.4f} val_acc={val_acc:.4f}", flush=True)
    return model, history


def encode(model, x):
    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            xb = torch.from_numpy(x[start:start + 65536]).to(device)
            h, _, _ = model(xb)
            h = torch.nn.functional.normalize(h, p=2, dim=1)
            out.append(h.cpu().numpy().astype(np.float32))
    return np.vstack(out)


def topk_in_cluster(query_emb, cand_emb, k):
    q = torch.from_numpy(query_emb)
    c = torch.from_numpy(cand_emb.T.copy())
    sims = q @ c
    vals, inds = torch.topk(sims, k=min(k, cand_emb.shape[0]), dim=1)
    return vals.numpy().astype(np.float32), inds.numpy().astype(np.int64)


def make_analog_features(query_emb, train_emb, y_train, z_train, query_is_train=False, k=K):
    print("clustering embeddings...", flush=True)
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=8192,
        random_state=SEED,
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
    )
    train_cluster = kmeans.fit_predict(train_emb)
    query_cluster = train_cluster if query_is_train else kmeans.predict(query_emb)
    centers = kmeans.cluster_centers_.astype(np.float32)
    centers /= (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    center_sims = centers @ centers.T
    nearest_clusters = np.argsort(-center_sims, axis=1)
    cluster_members = [np.where(train_cluster == c)[0] for c in range(N_CLUSTERS)]

    n = len(query_emb)
    n_features = 10 + 10 + 5
    feats = np.zeros((n, n_features), dtype=np.float32)
    neighbor_idx = np.zeros((n, k), dtype=np.int64)
    eps = 1e-6
    processed = 0

    for c_id in range(N_CLUSTERS):
        q_idx = np.where(query_cluster == c_id)[0]
        if len(q_idx) == 0:
            continue
        cand = []
        for nc in nearest_clusters[c_id]:
            cand.append(cluster_members[int(nc)])
            if sum(len(a) for a in cand) >= max(k + 2, 256):
                break
        cand = np.concatenate(cand)
        cand_emb = train_emb[cand]
        q_emb = query_emb[q_idx]
        local_k = k + 1 if query_is_train else k

        out_rows = []
        out_nei = []
        for start in range(0, len(q_idx), 4096):
            qe = q_emb[start:start + 4096]
            vals, inds_local = topk_in_cluster(qe, cand_emb, local_k + 8 if query_is_train else local_k)
            inds_global = cand[inds_local]
            if query_is_train:
                cleaned = []
                for row, qi in enumerate(q_idx[start:start + 4096]):
                    keep = inds_global[row][inds_global[row] != qi][:k]
                    if len(keep) < k:
                        extra = inds_global[row][:k]
                        keep = np.unique(np.concatenate([keep, extra]))[:k]
                    cleaned.append(keep)
                inds_global = np.vstack(cleaned)
                sims = np.sum(train_emb[inds_global] * query_emb[q_idx[start:start + 4096], None, :], axis=2)
            else:
                inds_global = inds_global[:, :k]
                sims = vals[:, :k]
            out_nei.append(inds_global)

            dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sims))
            weights = 1.0 / (dist + 1e-4)
            weights = weights / weights.sum(axis=1, keepdims=True)
            yy = y_train[inds_global]
            zz = z_train[inds_global]
            mean = (weights * yy).sum(axis=1)
            var = (weights * (yy - mean[:, None]) ** 2).sum(axis=1)
            std = np.sqrt(np.maximum(var, 0))
            q25 = np.quantile(yy, 0.25, axis=1)
            med = np.quantile(yy, 0.50, axis=1)
            q75 = np.quantile(yy, 0.75, axis=1)
            mn = yy.min(axis=1)
            mx = yy.max(axis=1)
            nearest_y = yy[:, 0]
            top3 = yy[:, :3].mean(axis=1)
            cont = [mean, std, q25, med, q75, mn, mx, nearest_y, top3, yy[:, :5].mean(axis=1)]
            probs = np.zeros((len(yy), 10), dtype=np.float32)
            for qclass in range(10):
                probs[:, qclass] = (weights * (zz == qclass)).sum(axis=1)
            entropy = -(probs * np.log(probs + eps)).sum(axis=1)
            maxp = probs.max(axis=1)
            exp_cls = (probs * np.arange(10, dtype=np.float32)[None, :]).sum(axis=1)
            dmean = (weights * dist).sum(axis=1)
            dstd = np.sqrt(np.maximum((weights * (dist - dmean[:, None]) ** 2).sum(axis=1), 0))
            support = [entropy, maxp, exp_cls, dmean, dstd]
            rows = np.column_stack(cont + [probs] + support).astype(np.float32)
            out_rows.append(rows)
        feats[q_idx] = np.vstack(out_rows)
        neighbor_idx[q_idx] = np.vstack(out_nei)
        processed += len(q_idx)
        if processed % 50_000 < len(q_idx):
            print(f"retrieval processed {processed}/{n}", flush=True)
    return feats, neighbor_idx


def main():
    print("reading data...", flush=True)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    y_train = pd.to_numeric(train_df["avgMidPointFriction"], errors="coerce").to_numpy(dtype=np.float32)
    y_test = pd.to_numeric(test_df["avgMidPointFriction"], errors="coerce").to_numpy(dtype=np.float32)
    z_train = pd.to_numeric(train_df["Label_10_class"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    z_test = pd.to_numeric(test_df["Label_10_class"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)

    print("building engineered features...", flush=True)
    x_train, x_test = build_features(train_df, test_df)
    print("x_train", x_train.shape, "x_test", x_test.shape, flush=True)

    print("training backbone...", flush=True)
    model, history = train_backbone(x_train, y_train, z_train)

    print("encoding train/test...", flush=True)
    train_emb = encode(model, x_train)
    test_emb = encode(model, x_test)
    print("embeddings", train_emb.shape, test_emb.shape, flush=True)

    print("building train analog features with self-neighbor removed...", flush=True)
    train_analog, _ = make_analog_features(train_emb, train_emb, y_train, z_train, query_is_train=True)
    print("building test analog features...", flush=True)
    test_analog, _ = make_analog_features(test_emb, train_emb, y_train, z_train, query_is_train=False)

    np.savez_compressed(
        OUT / "embedding_rag_features.npz",
        x_train=x_train,
        x_test=x_test,
        train_analog=train_analog,
        test_analog=test_analog,
        y_train=y_train,
        y_test=y_test,
        z_train=z_train,
        z_test=z_test,
    )
    with open(OUT / "backbone_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("saved", OUT / "embedding_rag_features.npz", flush=True)


if __name__ == "__main__":
    main()
