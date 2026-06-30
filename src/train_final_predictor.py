import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "output" / "features" / "embedding_rag_features.npz"
SIGNALS = PROJECT_ROOT / "output" / "signals" / "predictive_oof_signals.npz"
ANALOG = PROJECT_ROOT / "ckpt" / "retrieval" / "cross_encoder_reranked_analog.npz"
REASON = PROJECT_ROOT / "ckpt" / "reasoner" / "reasoned_memory_features_rerank_contextual.npz"
FINAL_DIR = PROJECT_ROOT / "ckpt" / "final"
METRIC_DIR = PROJECT_ROOT / "output" / "metrics"
PRED_DIR = PROJECT_ROOT / "output" / "predictions"
OUT = METRIC_DIR / "rerank_contextual_results.json"

FINAL_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)


def reg_metrics(y, p):
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
    }


def cls_metrics(y, p):
    pr, re, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {
        "Accuracy": float(accuracy_score(y, p)),
        "Macro_Precision": float(pr),
        "Macro_Recall": float(re),
        "Macro_F1": float(f1),
        "Weighted_F1": float(f1_score(y, p, average="weighted", zero_division=0)),
    }


def reason_predictive_block(reason_features):
    """Keep only reasoner evidence used by the final predictor.

    The diagnostic uncertainty columns are intentionally excluded here. They are
    saved and evaluated separately by script/05_validate_reasoning_uncertainty.py.
    """
    return np.column_stack(
        [
            reason_features[:, 0],
            reason_features[:, 11:21],
            reason_features[:, 22],
        ]
    ).astype(np.float32)


def load_features():
    base = np.load(BASE)
    sig = np.load(SIGNALS)
    analog = np.load(ANALOG)
    reason = np.load(REASON)

    x_train = base["x_train"].astype(np.float32)
    x_test = base["x_test"].astype(np.float32)
    y_train = base["y_train"].astype(np.float32)
    y_test = base["y_test"].astype(np.float32)
    z_train = base["z_train"].astype(int)
    z_test = base["z_test"].astype(int)

    train_analog = analog["train_analog"].astype(np.float32)
    test_analog = analog["test_analog"].astype(np.float32)
    train_reason = reason["train_reason"].astype(np.float32)
    test_reason = reason["test_reason"].astype(np.float32)

    signal_train = np.hstack(
        [
            sig["oof_reg"].astype(np.float32)[:, None],
            sig["oof_proba"].astype(np.float32),
        ]
    ).astype(np.float32)
    signal_test = np.hstack(
        [
            sig["test_reg"].astype(np.float32)[:, None],
            sig["test_proba"].astype(np.float32),
        ]
    ).astype(np.float32)

    train_reason_core = reason_predictive_block(train_reason)
    test_reason_core = reason_predictive_block(test_reason)

    reason_inter_train = np.column_stack(
        [
            train_reason[:, 0] - train_analog[:, 0],
            np.abs(train_reason[:, 0] - train_analog[:, 0]),
            train_reason[:, 0] - signal_train[:, 0],
            np.abs(train_reason[:, 0] - signal_train[:, 0]),
        ]
    ).astype(np.float32)
    reason_inter_test = np.column_stack(
        [
            test_reason[:, 0] - test_analog[:, 0],
            np.abs(test_reason[:, 0] - test_analog[:, 0]),
            test_reason[:, 0] - signal_test[:, 0],
            np.abs(test_reason[:, 0] - signal_test[:, 0]),
        ]
    ).astype(np.float32)

    x_reason = np.hstack(
        [x_train, train_analog, signal_train, train_reason_core, reason_inter_train]
    ).astype(np.float32)
    t_reason = np.hstack(
        [x_test, test_analog, signal_test, test_reason_core, reason_inter_test]
    ).astype(np.float32)

    return (
        x_reason,
        t_reason,
        y_train,
        y_test,
        z_train,
        z_test,
        test_analog,
        test_reason,
    )


def fit_reg(name, x_tr, y_tr, x_te, y_te, loss, save_path):
    print(f"\n--- regression {loss}: {name} ---", flush=True)
    if loss == "RMSE":
        params = dict(iterations=9000, depth=10, learning_rate=0.009, l2_leaf_reg=10.0, verbose=1000)
    else:
        params = dict(iterations=2600, depth=8, learning_rate=0.022, l2_leaf_reg=4.0, verbose=500)
    model = CatBoostRegressor(
        loss_function=loss,
        random_seed=42,
        allow_writing_files=False,
        task_type="GPU",
        devices="0",
        **params,
    )
    model.fit(x_tr, y_tr)
    model.save_model(str(save_path))
    pred = model.predict(x_te).astype(float)
    metrics = reg_metrics(y_te, pred)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics, pred


def fit_cls(x_tr, z_tr, x_te, z_te, save_path):
    print("\n--- classification: RAG-Friction ---", flush=True)
    counts = np.bincount(z_tr, minlength=10).astype(float)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = (weights / weights.mean()).tolist()
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=2000,
        depth=10,
        learning_rate=0.022,
        l2_leaf_reg=5.0,
        class_weights=weights,
        random_seed=84,
        verbose=400,
        allow_writing_files=False,
        task_type="GPU",
        devices="0",
    )
    model.fit(x_tr, z_tr)
    model.save_model(str(save_path))
    pred = model.predict(x_te).reshape(-1).astype(int)
    proba = model.predict_proba(x_te).astype(np.float32)
    metrics = cls_metrics(z_te, pred)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics, pred, proba


def main():
    x_tr, x_te, y_tr, y_te, z_tr, z_te, test_analog, test_reason = load_features()

    results = {
        "reranked_analog_only": {
            "regression": reg_metrics(y_te, test_analog[:, 0]),
            "classification": cls_metrics(z_te, test_analog[:, 10:20].argmax(axis=1)),
        },
        "direct_rerank_contextual_reasoner": {
            "regression": reg_metrics(y_te, test_reason[:, 0]),
            "classification": cls_metrics(z_te, test_reason[:, 11:21].argmax(axis=1)),
        },
        "feature_dim": int(x_tr.shape[1]),
        "feature_policy": {
            "final_predictor_uses_reasoned_friction": True,
            "final_predictor_uses_reason_class_probabilities": True,
            "final_predictor_uses_reasoner_uncertainty_columns": False,
            "excluded_reasoner_diagnostics": [
                "reasoned_dispersion",
                "reasoned_quantiles",
                "reasoned_range",
                "transition_delta_mean",
                "transition_delta_std",
                "heteroscedastic_transition_std",
                "heteroscedastic_aleatoric_std",
                "attention_entropy",
                "max_attention",
                "class_entropy",
            ],
        },
    }
    print("\n--- direct references ---", flush=True)
    print(json.dumps(results, indent=2), flush=True)

    results["rmse_final"], pred_rmse = fit_reg(
        "RAG-Friction",
        x_tr,
        y_tr,
        x_te,
        y_te,
        "RMSE",
        FINAL_DIR / "rag_friction_rerank_contextual_rmse_regressor.cbm",
    )
    results["mae_final"], pred_mae = fit_reg(
        "RAG-Friction",
        x_tr,
        y_tr,
        x_te,
        y_te,
        "MAE",
        FINAL_DIR / "rag_friction_rerank_contextual_mae_regressor.cbm",
    )
    results["classification_final"], pred_cls, proba_cls = fit_cls(
        x_tr,
        z_tr,
        x_te,
        z_te,
        FINAL_DIR / "rag_friction_rerank_contextual_classifier.cbm",
    )

    np.savez_compressed(
        PRED_DIR / "rerank_contextual_predictions.npz",
        y_test=y_te.astype(np.float32),
        z_test=z_te.astype(np.int32),
        rmse_pred=pred_rmse.astype(np.float32),
        mae_pred=pred_mae.astype(np.float32),
        cls_pred=pred_cls.astype(np.int32),
        cls_proba=proba_cls.astype(np.float32),
        reason_pred=test_reason[:, 0].astype(np.float32),
        transition_delta_std=test_reason[:, 8].astype(np.float32),
        attention_entropy=test_reason[:, 9].astype(np.float32),
        class_entropy=test_reason[:, 21].astype(np.float32),
        reasoned_dispersion=test_reason[:, 1].astype(np.float32),
        heteroscedastic_transition_std=test_reason[:, 23].astype(np.float32),
        heteroscedastic_aleatoric_std=test_reason[:, 24].astype(np.float32),
    )

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== RAG_FRICTION_RESULTS ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
