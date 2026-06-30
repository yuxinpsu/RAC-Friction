# Artifact Manifest

This file records the major generated artifacts used by the paper. Paths are relative to the repository root.

## Main RAC-Friction Artifacts

| Artifact | Purpose |
|---|---|
| `output/features/embedding_rac_features.npz` | Processed train/test features and labels. |
| `ckpt/encoder/contrastive64_embeddings.npz` | Learned road-weather-material embeddings. |
| `ckpt/retrieval/raw_plus_contrastive64_c65536_analog.npz` | Hybrid retrieval analog features. |
| `ckpt/retrieval/cross_encoder_reranked_analog.npz` | Reranked analog features. |
| `ckpt/reasoner/cross_encoder_reranker.pt` | Cross-encoder reranking checkpoint. |
| `ckpt/reasoner/contextual_reasoner_heteroscedastic.pt` | Contextual analog correction checkpoint. |
| `ckpt/reasoner/reasoned_memory_features_rerank_contextual.npz` | Reasoned analog context features. |
| `ckpt/final/rac_friction_rerank_contextual_rmse_regressor.cbm` | Final continuous-friction prediction head. |
| `ckpt/final/rac_friction_rerank_contextual_classifier.cbm` | Final friction-risk classification head. |

## Main Result Files

| Artifact | Purpose |
|---|---|
| `output/metrics/rerank_contextual_results.json` | Main RAC-Friction regression and classification results. |
| `output/metrics/rerank_contextual_summary.json` | Reranking and contextual correction training summary. |
| `output/metrics/reasoning_uncertainty_validation.json` | Reliability and uncertainty validation results, if generated. |

Large artifacts should not be committed. Store them in a separate release archive or institutional storage if needed.
