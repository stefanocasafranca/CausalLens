"""Full reproduction pipeline for CausalLens (RecSys 2026).

Downloads datasets, trains models, runs the Phase 3 experiment (200 users x
2 models x 2 datasets), and generates all figures and tables.

Usage:
    PYTHONPATH=. python run_all.py                  # Full run
    PYTHONPATH=. python run_all.py --skip-training  # Regenerate figures from existing CSV
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, desc):
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nERROR: {desc} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="CausalLens full reproduction pipeline")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and experiment; regenerate figures from existing CSV")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    csv_path = root / "results" / "phase3_results.csv"

    if not args.skip_training:
        # Step 1: Download datasets (triggered automatically by loaders)
        print("Datasets will be downloaded automatically on first use.\n")

        # Step 2+3: Train models and run Phase 3 experiment
        run(f"PYTHONPATH={root} python {root / 'experiments' / 'run_phase3.py'}",
            "Phase 3: Training models and running 800 user evaluations")
    else:
        if not csv_path.exists():
            print(f"ERROR: --skip-training requires {csv_path} to exist.")
            print("Run without --skip-training first.")
            sys.exit(1)
        print("Skipping training — using existing results CSV.\n")

    # Step 4: Generate figures, tables, and summary
    run(f"PYTHONPATH={root} python {root / 'experiments' / 'generate_figures.py'}",
        "Generating figures, tables, and results summary")

    print(f"\n{'='*70}")
    print("  Full reproduction complete. See results/")
    print(f"{'='*70}")
    print(f"\n  CSV:     results/phase3_results.csv")
    print(f"  Figures: results/figures/")
    print(f"  Tables:  results/tables/")
    print(f"  Summary: results/RESULTS_SUMMARY.md")
    print(f"  Paper:   paper/causallens_paper.tex\n")


if __name__ == "__main__":
    main()
