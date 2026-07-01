"""
Residual Adaptive Sampling (RAS) PINN benchmark: RAR-SCIO and RAD-SCIO
on the 1D convection-diffusion singular perturbation benchmark BVP:

    u_x - nu * u_xx = 0,   x in [0, 1],   u(0) = 0,   u(1) = 1

We benchmark point-adaptation algorithms (RAR and RAD) using SCIO
to see if they escape the decoy basin (Failure Mode 2) or stuck-at-initialization (Failure Mode 1).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# Add workspace directory to path to import SCIO
sys.path.insert(0, str(Path(__file__).resolve().parent))
from SCIO import (
    DEVICE,
    SCIO,
    ScaledPINN,
    clear_input_grads,
    set_seed,
    to_float,
)

# ---------------------------------------------------------------------------
# Exact solution (overflow-safe)
# ---------------------------------------------------------------------------

def exact_solution(x: np.ndarray, nu: float) -> np.ndarray:
    """Overflow-safe exact solution for u_x - nu * u_xx = 0, u(0)=0, u(1)=1."""
    x = np.asarray(x, dtype=np.float64)
    overflow_threshold = 1.0 / np.log(np.finfo(np.float64).max)
    if nu > overflow_threshold:
        return np.expm1(x / nu) / np.expm1(1.0 / nu)
    else:
        exponent = (x - 1.0) / nu
        threshold = -np.log(np.finfo(np.float64).eps)
        u = np.exp(exponent)
        u = np.where(exponent < -threshold, 0.0, u)
        u = np.where(np.isclose(x, 1.0), 1.0, u)
        return u


def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    pred, true = np.asarray(pred).ravel(), np.asarray(true).ravel()
    denom = np.linalg.norm(true)
    return float(np.linalg.norm(pred - true) / (denom if denom > 1e-12 else 1.0))


# ---------------------------------------------------------------------------
# Adaptive Point Selection Core Functions
# ---------------------------------------------------------------------------

def evaluate_pde_residuals(model: nn.Module, x_cand: np.ndarray, nu: float) -> np.ndarray:
    """Compute absolute PDE residual |u_x - nu * u_xx| on a candidate grid."""
    model.eval()
    x_cand_t = torch.tensor(
        x_cand, dtype=torch.float32, device=DEVICE
    ).reshape(-1, 1).requires_grad_(True)
    
    with torch.enable_grad():
        u = model(x_cand_t)
        u_x = torch.autograd.grad(
            u, x_cand_t, torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x_cand_t, torch.ones_like(u_x), create_graph=True
        )[0]
        residual = torch.abs(u_x - nu * u_xx).detach().cpu().numpy().ravel()
        
    model.train()
    return residual


def run_adaptive_training(
    method: str,
    nu: float,
    seed: int,
    target_nfe: int = 800,
    bc_weight: float = 50.0,
) -> dict:
    """Train SCIO using RAR or RAD point adaptation."""
    set_seed(seed)
    start_time = time.time()
    
    # Grid for evaluating residual adaptation
    x_cand = np.linspace(0.0, 1.0, 10000)
    
    # 1. Initialize collocation points
    if method == "RAR-SCIO":
        # RAR starts with N_initial = 200 uniform collocation points
        colloc_pts = np.linspace(0.0, 1.0, 200)
        max_colloc = 2000
    elif method == "RAD-SCIO":
        # RAD starts with N = 512 uniform collocation points
        colloc_pts = np.linspace(0.0, 1.0, 512)
        max_colloc = 512
    else:
        raise ValueError(f"Unknown method: {method}")

    # Set up model
    model = ScaledPINN(
        in_dim=1, out_dim=1, bounds=[(0.0, 1.0)],
        width=64, depth=4,
    ).to(DEVICE)
    
    # Fixed BC points on device
    x_bc0 = torch.zeros(1, 1, device=DEVICE)
    x_bc1 = torch.ones(1, 1, device=DEVICE)
    
    # Evaluation grid (2000 points)
    x_eval_np = np.linspace(0, 1, 2000)
    u_exact_np = exact_solution(x_eval_np, nu)
    x_eval_t = torch.tensor(
        x_eval_np, dtype=torch.float32, device=DEVICE
    ).reshape(-1, 1)
    
    # SCIO optimizer setup (same parameters as baseline)
    optimizer = SCIO(
        model.parameters(),
        lr=0.001, memory_size=50, burn_in_steps=50,
        T_high=1.25, max_iter=10, max_step_norm=1.0,
    )
    
    # Make collocation points PyTorch-accessible
    colloc_pts_t = torch.tensor(
        colloc_pts, dtype=torch.float32, device=DEVICE
    ).reshape(-1, 1).requires_grad_(True)
    
    nfe_counter = 0
    last_adapt_nfe = 0
    
    # Log loss terms separately
    pde_losses = []
    bc_losses = []
    total_losses = []
    
    def closure() -> Tensor:
        nonlocal nfe_counter
        optimizer.zero_grad()
        
        # Forward pass on active collocation points
        u = model(colloc_pts_t)
        u_x = torch.autograd.grad(
            u, colloc_pts_t, torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, colloc_pts_t, torch.ones_like(u_x), create_graph=True
        )[0]
        
        # PDE and BC loss terms
        loss_pde = torch.mean((u_x - nu * u_xx) ** 2)
        loss_bc = bc_weight * (
            (model(x_bc0) - 0.0) ** 2 + (model(x_bc1) - 1.0) ** 2
        ).mean()
        
        total = loss_pde + loss_bc
        total.backward()
        
        nfe_counter += 1
        pde_losses.append(loss_pde.item())
        bc_losses.append(loss_bc.item())
        total_losses.append(total.item())
        
        clear_input_grads([colloc_pts_t])
        return total

    # Training and Adaptation loop
    while nfe_counter < target_nfe:
        # Perform one optimizer outer step (consisting of max_iter iterations inside PyTorch)
        optimizer.step(closure)
        
        # Adaptation Trigger: adaptation is performed only between optimizer steps
        # to ensure line search remains consistent during a step.
        if nfe_counter - last_adapt_nfe >= 100:
            last_adapt_nfe = nfe_counter
            
            # Evaluate current PDE residuals on the 10,000 candidate grid
            res = evaluate_pde_residuals(model, x_cand, nu)
            
            if method == "RAR-SCIO":
                # Only add points if we have not hit the N_max capacity
                if len(colloc_pts) < max_colloc:
                    # Find candidates that are not already collocation points (approximate matching)
                    # To keep it simple and robust, we select the top-50 largest residual points
                    # from the candidate set, filtering out points that are extremely close to existing ones
                    # (min distance threshold of 1e-4)
                    sorted_indices = np.argsort(res)[::-1]
                    added = 0
                    x_new = []
                    for idx in sorted_indices:
                        candidate = x_cand[idx]
                        if np.min(np.abs(colloc_pts - candidate)) > 1e-4:
                            x_new.append(candidate)
                            added += 1
                        if added >= 50:
                            break
                    
                    if len(x_new) > 0:
                        colloc_pts = np.concatenate([colloc_pts, x_new])
                        if len(colloc_pts) > max_colloc:
                            colloc_pts = colloc_pts[:max_colloc]
                        
                        # Re-create collocation points tensor
                        colloc_pts_t = torch.tensor(
                            colloc_pts, dtype=torch.float32, device=DEVICE
                        ).reshape(-1, 1).requires_grad_(True)
                        
            elif method == "RAD-SCIO":
                # Distribution Resampling (replace the entire set)
                # P_i proportional to R_i + 1
                weights = res + 1.0
                p = weights / weights.sum()
                
                # Sample 512 points from candidate grid with probability p
                colloc_pts = np.random.choice(x_cand, size=512, p=p, replace=True)
                
                # Re-create collocation points tensor
                colloc_pts_t = torch.tensor(
                    colloc_pts, dtype=torch.float32, device=DEVICE
                ).reshape(-1, 1).requires_grad_(True)

    elapsed_time = time.time() - start_time
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred = model(x_eval_t).cpu().numpy().ravel()
    rel_l2_err = relative_l2(pred, u_exact_np)
    
    # Final loss metrics
    final_pde = pde_losses[-1] if pde_losses else float("nan")
    final_bc = bc_losses[-1] if bc_losses else float("nan")
    final_total = total_losses[-1] if total_losses else float("nan")
    
    # Check for NaN/divergence in loss history
    nan_flag = any(not np.isfinite(l) for l in total_losses)
    
    return {
        "method": method,
        "nu": nu,
        "seed": seed,
        "rel_l2": rel_l2_err,
        "wall_time_s": elapsed_time,
        "loss_pde": final_pde,
        "loss_bc": final_bc,
        "loss_total": final_total,
        "final_nfe": nfe_counter,
        "nan_divergence": nan_flag,
        "n_collocation_points": len(colloc_pts),
        "final_colloc_pts": colloc_pts.tolist(),
        "model_state_dict": model.state_dict(),
        "colloc_pts": colloc_pts,
        "predictions": pred,
        "exact": u_exact_np,
        "x_eval": x_eval_np,
    }


# ---------------------------------------------------------------------------
# Output files and Directories setup
# ---------------------------------------------------------------------------

def main():
    workspace_dir = Path(__file__).resolve().parent
    output_dir = workspace_dir / "convdiff_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ras_plots_dir = output_dir / "ras_plots"
    ras_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load baseline SCIO results from results.csv
    baseline_csv = output_dir / "results.csv"
    if not baseline_csv.exists():
        # Check alternative path kaggle_output/convdiff_results/results.csv
        baseline_csv = workspace_dir / "kaggle_output" / "convdiff_results" / "results.csv"
        
    baseline_scio_rows = []
    if baseline_csv.exists():
        print(f"Loading baseline SCIO results from {baseline_csv}...")
        with open(baseline_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["method"] == "SCIO":
                    baseline_scio_rows.append({
                        "method": "SCIO (Baseline)",
                        "nu": float(r["nu"]),
                        "seed": int(r["seed"]),
                        "rel_l2": float(r["rel_l2"]),
                        "wall_time_s": float(r["wall_time_s"]),
                        "loss_pde": float(r["final_loss"]) if r["final_loss"] != "nan" else float("nan"),
                        "loss_bc": float("nan"),
                        "loss_total": float(r["final_loss"]) if r["final_loss"] != "nan" else float("nan"),
                        "final_nfe": int(r["final_nfe"]),
                        "nan_divergence": r["nan_divergence"] == "True",
                        "n_collocation_points": 512,
                    })
        print(f"Loaded {len(baseline_scio_rows)} baseline SCIO rows.")
    else:
        print("Warning: Baseline results.csv not found. Summary comparison table will not show baseline.")

    # 2. Run RAR and RAD sweeps
    NU_VALUES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    SEEDS = [42, 123, 7]
    methods = ["RAR-SCIO", "RAD-SCIO"]
    
    ras_rows = []
    run_records = []
    
    print("\n" + "=" * 70)
    print("RUNNING ADAPTIVE SAMPLING PINN SWEEP")
    print("=" * 70)
    
    for method in methods:
        print(f"\n--- {method} ---")
        for nu in NU_VALUES:
            for seed in SEEDS:
                print(f"  nu={nu:.0e} seed={seed}: training...", end=" ", flush=True)
                
                try:
                    result = run_adaptive_training(method, nu, seed, target_nfe=800)
                    
                    row = {
                        "method": method,
                        "nu": nu,
                        "seed": seed,
                        "rel_l2": result["rel_l2"],
                        "wall_time_s": result["wall_time_s"],
                        "loss_pde": result["loss_pde"],
                        "loss_bc": result["loss_bc"],
                        "loss_total": result["loss_total"],
                        "final_nfe": result["final_nfe"],
                        "nan_divergence": result["nan_divergence"],
                        "n_collocation_points": result["n_collocation_points"],
                    }
                    ras_rows.append(row)
                    run_records.append(result)
                    
                    print(f"rel_l2={result['rel_l2']:.6f}  "
                          f"loss={result['loss_total']:.3e}  "
                          f"n_colloc={result['n_collocation_points']}  "
                          f"time={result['wall_time_s']:.1f}s")
                except Exception as exc:
                    print(f"FAILED: {exc}")
                    row = {
                        "method": method,
                        "nu": nu,
                        "seed": seed,
                        "rel_l2": float("nan"),
                        "wall_time_s": float("nan"),
                        "loss_pde": float("nan"),
                        "loss_bc": float("nan"),
                        "loss_total": float("nan"),
                        "final_nfe": 0,
                        "nan_divergence": True,
                        "n_collocation_points": 0,
                    }
                    ras_rows.append(row)

    # 3. Save results_ras.csv
    csv_fields = [
        "method", "nu", "seed", "rel_l2", "wall_time_s", 
        "loss_pde", "loss_bc", "loss_total", "final_nfe", 
        "nan_divergence", "n_collocation_points"
    ]
    with open(output_dir / "results_ras.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(ras_rows)
    print(f"\nAll RAS results written to {output_dir / 'results_ras.csv'}")

    # 4. Generate prediction curves for runs where final_loss < 0.1
    print("\nGenerating prediction curve diagnostics...")
    for rec in run_records:
        if rec["loss_total"] < 0.1:
            method = rec["method"]
            nu = rec["nu"]
            seed = rec["seed"]
            loss = rec["loss_total"]
            err = rec["rel_l2"]
            
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.plot(rec["x_eval"], rec["exact"], "-", color="#1E40AF", linewidth=2.5, label="Exact Solution")
            ax.plot(rec["x_eval"], rec["predictions"], "--", color="#EF4444", linewidth=1.8, label="SCIO PINN (Adapted)")
            
            # Overlay final collocation point locations
            colloc_x = rec["colloc_pts"]
            ax.scatter(colloc_x, np.zeros_like(colloc_x), color="#10B981", alpha=0.3, s=6, label=f"Collocation ({len(colloc_x)} pts)", zorder=2)
            
            ax.set_xlabel("x", fontsize=11)
            ax.set_ylabel("u(x)", fontsize=11)
            ax.set_title(f"Method={method}, nu={nu:.0e}, seed={seed}\nfinal_loss={loss:.2e}, rel_L2={err:.2f}", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc="upper left")
            fig.tight_layout()
            
            plot_name = f"pred_curve_{method}_nu{nu:.0e}_seed{seed}.png"
            fig.savefig(ras_plots_dir / plot_name, dpi=200)
            plt.close(fig)
            print(f"  Generated prediction curve: {plot_name}")

    # 5. Generate collocation point histograms for RAR-SCIO at nu=1e-3 and nu=1e-4
    print("\nGenerating RAR collocation point histograms...")
    for rec in run_records:
        if rec["method"] == "RAR-SCIO" and rec["nu"] in [1e-3, 1e-4]:
            nu = rec["nu"]
            seed = rec["seed"]
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(rec["colloc_pts"], bins=50, range=(0.0, 1.0), color="#10B981", edgecolor="black", alpha=0.8)
            ax.set_xlabel("Collocation Point Location x", fontsize=11)
            ax.set_ylabel("Count / Density", fontsize=11)
            ax.set_title(f"RAR-SCIO Collocation Distribution (nu={nu:.0e}, seed={seed})", fontsize=12)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            
            hist_name = f"rar_hist_nu{nu:.0e}_seed{seed}.png"
            fig.savefig(ras_plots_dir / hist_name, dpi=200)
            plt.close(fig)
            print(f"  Generated RAR histogram: {hist_name}")

    # 6. Generate summary table comparing against baseline SCIO
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON TABLE (mean ± std rel L2 across seeds)")
    print("=" * 70)
    
    all_methods = ["SCIO (Baseline)", "RAR-SCIO", "RAD-SCIO"]
    header = f"{'Method':<20}" + "".join(f"{'nu='+str(nu):<16}" for nu in NU_VALUES)
    print(header)
    print("-" * len(header))
    
    # Combine baseline rows and RAS rows
    combined_rows = baseline_scio_rows + ras_rows
    
    for method in all_methods:
        line = f"{method:<20}"
        for nu in NU_VALUES:
            errs = [r["rel_l2"] for r in combined_rows
                    if r["method"] == method and r["nu"] == nu
                    and np.isfinite(r["rel_l2"])]
            if errs:
                mean = np.mean(errs)
                std = np.std(errs, ddof=0)
                line += f"{mean:.3e}±{std:.1e} "
            else:
                line += f"{'N/A':<16}"
        print(line)
        
    print(f"\nAll deliverables written to: {output_dir}")

if __name__ == "__main__":
    main()
