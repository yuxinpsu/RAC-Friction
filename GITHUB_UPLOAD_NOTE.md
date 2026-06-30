# GitHub Upload Note

This folder is organized as a code-first repository. Before pushing:

1. Keep `data/`, `ckpt/`, and `output/` artifacts out of Git unless a small demo artifact is intentionally added.
2. Confirm `git status` does not include proprietary friction records, trained checkpoints, or generated figures/results.
3. Commit source code, configuration, README files, and documentation only.

Suggested first commit:

```bash
git init
git add README.md requirements.txt .gitignore config src script docs data/README.md ckpt/README.md output/README.md GITHUB_UPLOAD_NOTE.md
git commit -m "Initial RAC-Friction code release"
```
