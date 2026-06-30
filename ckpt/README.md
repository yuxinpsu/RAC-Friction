# Checkpoint Directory

This directory stores generated model checkpoints and retrieval artifacts.

Large files are intentionally ignored by Git. After running the training pipeline, expected subdirectories include:

```text
ckpt/encoder/
ckpt/retrieval/
ckpt/reasoner/
ckpt/final/
ckpt/baselines/
```

Use `docs/artifact_manifest.md` to track which checkpoint supports each paper result.
