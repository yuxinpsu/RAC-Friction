import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_FILE = PROJECT_ROOT / "output" / "predictions" / "rerank_contextual_predictions.npz"
METRIC_DIR = PROJECT_ROOT / "output" / "metrics"
FIG_DIR = PROJECT_ROOT / "output" / "figures"
OUT = METRIC_DIR / "reasoning_uncertainty_validation.json"

METRIC_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def regression_metrics(y, p):
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
    }


def corr_pair(signal, error):
    return {
        "pearson": float(pearsonr(signal, error).statistic),
        "spearman": float(spearmanr(signal, error).statistic),
    }


def quantile_groups(y, pred, uncertainty, n_groups=4):
    order = np.argsort(uncertainty)
    groups = np.array_split(order, n_groups)
    rows = []
    for i, idx in enumerate(groups, start=1):
        rows.append(
            {
                "group": i,
                "n": int(len(idx)),
                "uncertainty_mean": float(np.mean(uncertainty[idx])),
                "abs_error_mean": float(np.mean(np.abs(y[idx] - pred[idx]))),
                **regression_metrics(y[idx], pred[idx]),
            }
        )
    return rows


def selective_curve(y, pred, uncertainty):
    order = np.argsort(uncertainty)
    coverages = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    rows = []
    for cov in coverages:
        n = max(1, int(round(cov * len(order))))
        idx = order[:n]
        rows.append(
            {
                "coverage": float(cov),
                "n": int(n),
                "uncertainty_cutoff": float(np.max(uncertainty[idx])),
                **regression_metrics(y[idx], pred[idx]),
            }
        )
    return rows


def plot_groups(rows, out_path):
    labels = [f"Q{r['group']}" for r in rows]
    errors = [r["abs_error_mean"] for r in rows]
    unc = [r["uncertainty_mean"] for r in rows]
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(6.2, 3.6), dpi=180)
    ax1.bar(x - 0.18, errors, width=0.36, color="#4477AA", label="Mean absolute error")
    ax1.set_ylabel("Mean absolute error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Reasoning uncertainty quartile")

    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, unc, marker="o", color="#CC6677", label="Mean uncertainty")
    ax2.set_ylabel("Mean uncertainty")

    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels1 + labels2, loc="upper left", frameon=False)
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_selective(rows, out_path):
    cov = [r["coverage"] for r in rows]
    mae = [r["MAE"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.8, 3.4), dpi=180)
    ax.plot(cov, mae, marker="o", color="#228833")
    ax.set_xlabel("Prediction coverage kept")
    ax.set_ylabel("MAE on retained records")
    ax.set_xlim(0.48, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    data = np.load(PRED_FILE)
    y = data["y_test"].astype(float)
    pred = data["rmse_pred"].astype(float)
    abs_error = np.abs(y - pred)

    signals = {
        "heteroscedastic_transition_std": data["heteroscedastic_transition_std"].astype(float),
        "heteroscedastic_aleatoric_std": data["heteroscedastic_aleatoric_std"].astype(float),
        "transition_delta_std": data["transition_delta_std"].astype(float),
        "attention_entropy": data["attention_entropy"].astype(float),
        "class_entropy": data["class_entropy"].astype(float),
        "reasoned_dispersion": data["reasoned_dispersion"].astype(float),
    }
    correlation = {name: corr_pair(signal, abs_error) for name, signal in signals.items()}

    primary = signals["heteroscedastic_transition_std"]
    groups = quantile_groups(y, pred, primary)
    selective = selective_curve(y, pred, primary)

    plot_groups(groups, FIG_DIR / "reasoning_uncertainty_groups.png")
    plot_selective(selective, FIG_DIR / "reasoning_uncertainty_coverage.png")

    result = {
        "primary_uncertainty_signal": "heteroscedastic_transition_std",
        "interpretation": (
            "Higher heteroscedastic_transition_std means the contextual reasoner estimates "
            "a less reliable analog-to-target transition after combining learned pair-wise "
            "transition variance and disagreement among analog corrections."
        ),
        "correlation_with_absolute_error": correlation,
        "quartile_groups_by_heteroscedastic_transition_std": groups,
        "selective_prediction_by_heteroscedastic_transition_std": selective,
        "figures": {
            "quartile_groups": str(FIG_DIR / "reasoning_uncertainty_groups.png"),
            "selective_prediction": str(FIG_DIR / "reasoning_uncertainty_coverage.png"),
        },
        "note": "These uncertainty diagnostics are not used as input features by the final predictor.",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
