# Reproducibility Notes

## Main Command

Run the full RAC-Friction pipeline with:

```bash
python script/run_rac_friction.py --stage all
```

The command executes the following stages internally.

1. Prepare processed features:
   ```bash
   python script/run_rac_friction.py --stage prepare
   ```
2. Train the contrastive retrieval backbone and build hybrid retrieval memory:
   ```bash
   python script/run_rac_friction.py --stage retrieval
   ```
3. Train cross-encoder reranking and contextual analog correction:
   ```bash
   python script/run_rac_friction.py --stage reasoner
   ```
4. Train the final RAC-Friction predictors:
   ```bash
   python script/run_rac_friction.py --stage predictor
   ```
5. Validate reliability diagnostics:
   ```bash
   python script/run_rac_friction.py --stage uncertainty
   ```

## Leakage Control

The pipeline excludes segment and sub-segment identifiers from predictors. During training-time retrieval for training records, self-neighbors are removed. During test-time evaluation, analogs are retrieved only from the training archive.
