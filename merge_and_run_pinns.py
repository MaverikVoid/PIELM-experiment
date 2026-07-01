"""
Merge results:
Loads Vanilla-PIELM and KAPI-ELM results from the Kaggle run outputs,
runs the PINN methods (SCIO, Adam, L-BFGS) locally (GPU-enabled),
merges them, and generates the final summary and plots.
"""

import csv
from pathlib import Path
import numpy as np
import time

# Add current directory to path to import convdiff_benchmark
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from convdiff_benchmark import (
    NU_VALUES,
    SEEDS,
    run_pinn_method,
    diagnose_failure_mode,
    write_results_csv,
    plot_headline,
    print_failure_mode_analysis,
)

def load_kaggle_rows(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Parse fields correctly
            method = r["method"]
            if method not in ["Vanilla-PIELM", "KAPI-ELM"]:
                continue
            rows.append({
                "method": method,
                "nu": float(r["nu"]),
                "seed": int(r["seed"]),
                "rel_l2": float(r["rel_l2"]),
                "wall_time_s": float(r["wall_time_s"]),
                "final_loss": float(r["final_loss"]) if r["final_loss"] != "nan" else float("nan"),
                "final_nfe": int(r["final_nfe"]),
                "nan_divergence": r["nan_divergence"] == "True",
                "inf_norm_J": float(r["inf_norm_J"]) if r["inf_norm_J"] != "nan" else float("nan"),
                "best_w": float(r["best_w"]) if r["best_w"] != "nan" else float("nan"),
                "failure_mode": r["failure_mode"],
            })
    return rows

def main():
    workspace_dir = Path(__file__).resolve().parent
    kaggle_csv = workspace_dir / "kaggle_output" / "convdiff_results" / "results.csv"
    output_dir = workspace_dir / "convdiff_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = output_dir / "results.csv"

    if merged_csv.exists():
        with open(merged_csv, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) >= 76:
            print("Found existing complete merged results.csv. Skipping training and running generation directly.")
            merged_rows = []
            with open(merged_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    merged_rows.append({
                        "method": r["method"],
                        "nu": float(r["nu"]),
                        "seed": int(r["seed"]),
                        "rel_l2": float(r["rel_l2"]) if r["rel_l2"] != "nan" else float("nan"),
                        "wall_time_s": float(r["wall_time_s"]) if r["wall_time_s"] != "nan" else float("nan"),
                        "final_loss": float(r["final_loss"]) if r["final_loss"] != "nan" else float("nan"),
                        "final_nfe": int(r["final_nfe"]),
                        "nan_divergence": r["nan_divergence"] == "True",
                        "inf_norm_J": float(r["inf_norm_J"]) if r["inf_norm_J"] != "nan" else float("nan"),
                        "best_w": float(r["best_w"]) if r["best_w"] != "nan" else float("nan"),
                        "failure_mode": r["failure_mode"],
                    })
            run_training = False
        else:
            run_training = True
    else:
        run_training = True

    if run_training:
        print("Loading Vanilla-PIELM and KAPI-ELM results from Kaggle outputs...")
        if not kaggle_csv.exists():
            print(f"Error: Kaggle CSV not found at {kaggle_csv}")
            sys.exit(1)

        merged_rows = load_kaggle_rows(kaggle_csv)
        print(f"Loaded {len(merged_rows)} rows from Kaggle (Vanilla-PIELM and KAPI-ELM).")

        # Now run the local PINN methods on local GPU
        pinn_methods = ["SCIO", "Adam", "L-BFGS"]
        print("\n" + "=" * 70)
        print("RUNNING PINN METHODS LOCALLY (ON GPU)")
        print("=" * 70)

        for method_name in pinn_methods:
            print(f"\nRunning {method_name}...")
            for nu in NU_VALUES:
                for seed in SEEDS:
                    print(f"  nu={nu:.0e} seed={seed}: training...", end=" ", flush=True)
                    try:
                        result = run_pinn_method(method_name, nu, seed)
                        nan_flag = any(not np.isfinite(l) for l in result["history"]["loss"])
                        failure = diagnose_failure_mode(result)

                        row = {
                            "method": method_name,
                            "nu": nu,
                            "seed": seed,
                            "rel_l2": result["rel_l2"],
                            "wall_time_s": result["total_time"],
                            "final_loss": result["final_loss"],
                            "final_nfe": result["final_nfe"],
                            "nan_divergence": nan_flag,
                            "inf_norm_J": float("nan"),
                            "best_w": float("nan"),
                            "failure_mode": failure,
                        }
                        merged_rows.append(row)
                        print(f"rel_l2={result['rel_l2']:.6f}  "
                              f"loss={result['final_loss']:.3e}  "
                              f"nfe={result['final_nfe']}  "
                              f"mode={failure}  "
                              f"time={result['total_time']:.1f}s")
                    except Exception as exc:
                        print(f"FAILED: {exc}")
                        row = {
                            "method": method_name,
                            "nu": nu,
                            "seed": seed,
                            "rel_l2": float("nan"),
                            "wall_time_s": float("nan"),
                            "final_loss": float("nan"),
                            "final_nfe": 0,
                            "nan_divergence": True,
                            "inf_norm_J": float("nan"),
                            "best_w": float("nan"),
                            "failure_mode": "exception",
                        }
                        merged_rows.append(row)

    # Save merged results
    write_results_csv(merged_rows, output_dir / "results.csv")
    plot_headline(merged_rows, output_dir / "convdiff_spp_results.png")

    # Failure mode analysis
    print_failure_mode_analysis(merged_rows)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (mean ± std rel L2 across seeds)")
    print("=" * 70)
    methods = ["Vanilla-PIELM", "KAPI-ELM", "SCIO", "Adam", "L-BFGS"]
    header = f"{'Method':<16}" + "".join(f"{'nu='+str(nu):<16}" for nu in NU_VALUES)
    print(header)
    print("-" * len(header))
    for method in methods:
        line = f"{method:<16}"
        for nu in NU_VALUES:
            errs = [r["rel_l2"] for r in merged_rows
                    if r["method"] == method and r["nu"] == nu
                    and np.isfinite(r["rel_l2"])]
            if errs:
                mean = np.mean(errs)
                std = np.std(errs, ddof=0)
                line += f"{mean:.3e}±{std:.1e} "
            else:
                line += f"{'N/A':<16}"
        print(line)

    print(f"\nAll merged results saved to: {output_dir}")

if __name__ == "__main__":
    main()
