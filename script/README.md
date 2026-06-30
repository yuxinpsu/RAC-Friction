# Script Directory

This folder intentionally contains only one public entry point:

```bash
python script/run_rac_friction.py --stage all
```

Available stages are:

- `prepare`
- `retrieval`
- `reasoner`
- `predictor`
- `uncertainty`

The underlying implementation lives in `src/` so that the GitHub repository stays clean and model-focused.
