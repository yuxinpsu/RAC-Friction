#!/usr/bin/env python3
"""Command-line entry point for the RAC-Friction pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

STAGES = {
    "prepare": "src.prepare_features",
    "retrieval": "src.train_contrastive_retrieval",
    "reasoner": "src.train_rerank_contextual_reasoner",
    "predictor": "src.train_final_predictor",
    "uncertainty": "src.validate_reasoning_uncertainty",
}

PIPELINE = ["prepare", "retrieval", "reasoner", "predictor", "uncertainty"]


def run_module(module_name: str) -> None:
    cmd = [sys.executable, "-m", module_name]
    print(f"\n[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAC-Friction training and evaluation.")
    parser.add_argument(
        "--stage",
        choices=["all", *STAGES.keys()],
        default="all",
        help="Pipeline stage to run. Use 'all' for the full RAC-Friction pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = PIPELINE if args.stage == "all" else [args.stage]
    for stage in stages:
        run_module(STAGES[stage])


if __name__ == "__main__":
    main()
