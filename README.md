# RAC-Friction

Retrieval-Augmented Correction for network-level road surface friction prediction from sparse mobile sensing records.

RAC-Friction treats a target road condition as a query and uses a historical friction archive as analog evidence. The model first retrieves candidate historical analogs, then reranks and corrects the retrieved evidence before making continuous friction and friction-risk predictions.

## Repository Layout

```text
RAC-Friction/
  config/                  Experiment configuration.
  data/                    Place train.csv and test.csv here. Data are not tracked.
  src/                     Core reusable modules.
  script/                  Single command-line entry point.
  ckpt/                    Model checkpoints and retrieval artifacts. Not tracked.
  output/                  Metrics, predictions, and figures. Not tracked.
  docs/                    Reproducibility notes and artifact manifest.
```

## Main Pipeline

Run the full model pipeline from the repository root:

```bash
python script/run_rac_friction.py --stage all
```

Individual stages can also be run with `--stage prepare`, `--stage retrieval`, `--stage reasoner`, `--stage predictor`, or `--stage uncertainty`.

The main RAC-Friction pipeline produces:

- `output/features/embedding_rag_features.npz`
- `ckpt/encoder/contrastive64_embeddings.npz`
- `ckpt/retrieval/raw_plus_contrastive64_c65536_analog.npz`
- `ckpt/retrieval/cross_encoder_reranked_analog.npz`
- `ckpt/reasoner/contextual_reasoner_heteroscedastic.pt`
- `ckpt/reasoner/reasoned_memory_features_rerank_contextual.npz`
- `ckpt/final/rag_friction_rerank_contextual_rmse_regressor.cbm`
- `ckpt/final/rag_friction_rerank_contextual_classifier.cbm`

The historical `rag_friction_*` filenames are retained for compatibility with the current scripts, but they correspond to the RAC-Friction model described in the paper.

## Validation and Paper Metrics

After the full pipeline finishes, the final paper-table metrics are written by the `predictor` stage:

```bash
python script/run_rac_friction.py --stage predictor
```

This stage evaluates the held-out test set and saves the main results to:

```text
output/metrics/rerank_contextual_results.json
```

If `output/signals/predictive_oof_signals.npz` is not present, the `predictor` stage first builds leakage-controlled out-of-fold direct prediction signals from the training set and held-out prediction signals for the test set.

The expected held-out metrics for RAC-Friction reported in the paper are:

| Method | MAE | RMSE | R2 | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| RAC-Friction | 0.0991 | 0.1471 | 0.6245 | 0.5775 | 0.5892 |

These values are computed from the final RAC-Friction regression and classification heads using the held-out test set.

Reliability validation is produced by the `uncertainty` stage:

```bash
python script/run_rac_friction.py --stage uncertainty
```

It writes:

```text
output/metrics/reasoning_uncertainty_validation.json
output/figures/reasoning_uncertainty_validation.pdf
```

## Data Policy

The original friction archive is not included in this repository. Place the cleaned files as:

```text
data/train.csv
data/test.csv
```

By default, the pipeline reads data from the repository-local `data/` directory. To use another data location, set `RAC_FRICTION_DATA_DIR=/path/to/Data` before running the pipeline.

Segment and sub-segment identifiers are excluded from predictors by the feature preparation script.

## Environment

The code was developed with Python 3.10. Core dependencies include PyTorch, NumPy, pandas, scikit-learn, CatBoost, and Matplotlib.

Install the base dependencies with:

```bash
pip install -r requirements.txt
```

See `docs/reproducibility.md` for the expected execution order and artifact policy.

## Reproducibility

All reported tables and figures should be generated from files under `output/metrics`, `output/predictions`, and `output/figures`. Large artifacts are intentionally ignored by Git; use `docs/artifact_manifest.md` to track which generated files correspond to each paper result.
