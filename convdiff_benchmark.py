"""
Benchmark: SCIO vs KAPI-ELM vs vanilla PIELM vs Adam vs L-BFGS
on the 1D singularly-perturbed convection-diffusion equation:

    u_x - nu * u_xx = 0,   x in [0, 1],   u(0) = 0,   u(1) = 1

As nu -> 0 a sharp boundary layer forms near x = 1.

References:
  - Dwivedi, V. & Srinivasan, B. (2020). "Physics Informed Extreme Learning
    Machine (PIELM)." Neurocomputing, 391, 96-118.
  - "Soft Partition-based KAPI-ELM for Multi-Scale PDEs," submitted to
    IEEE Transactions on AI (Dwivedi et al.).
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

# ---------------------------------------------------------------------------
# Import SCIO infrastructure from the main file
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from SCIO import (
    DEVICE,
    PROBLEMS,
    ProblemSetup,
    ProblemSpec,
    ScaledPINN,
    build_optimizer,
    clear_input_grads,
    run_problem,
    set_seed,
    to_float,
)

# ---------------------------------------------------------------------------
# Exact solution (overflow-safe) — matches EXACT_SOLUTION.m from the repo
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
# Vanilla PIELM — fixed random basis (the expected failure baseline)
# ---------------------------------------------------------------------------

def vanilla_pielm_solve(nu: float, n_hidden: int = 200, seed: int = 0,
                        w_scale: float = 8.0) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    centers = rng.uniform(0, 1, n_hidden)
    sig_x = np.full(n_hidden, w_scale / n_hidden)
    m = 1.0 / (np.sqrt(2) * sig_x)
    alpha = -m * centers

    X_pde = np.sort(rng.uniform(0, 1, 2000))
    z = X_pde[:, None] * m[None, :] + alpha[None, :]
    z2 = np.minimum(z ** 2, 700)
    phi = np.exp(-z2)
    phi_x = -2 * m[None, :] * z * phi
    phi_xx = 2 * (m[None, :] ** 2) * (2 * z2 - 1) * phi

    LHS_PDE = phi_x - nu * phi_xx
    X_bc = np.array([0.0, 1.0])
    z_bc = X_bc[:, None] * m[None, :] + alpha[None, :]
    phi_bc = np.exp(-z_bc ** 2)

    H = np.vstack([LHS_PDE, phi_bc])
    b = np.concatenate([np.zeros(len(X_pde)), [0.0, 1.0]])
    c, *_ = np.linalg.lstsq(H, b, rcond=None)
    J = np.max(np.abs(H @ c - b))
    return {"J": J, "c": c, "m": m, "alpha": alpha, "elapsed": time.time() - t0}


def elm_eval(fit_result: dict, X_eval: np.ndarray) -> np.ndarray:
    """Evaluate an ELM solution (shared by vanilla PIELM and KAPI-ELM)."""
    m, alpha, c = fit_result["m"], fit_result["alpha"], fit_result["c"]
    X_eval = np.asarray(X_eval, dtype=np.float64)
    z = X_eval[:, None] * m[None, :] + alpha[None, :]
    phi = np.exp(-np.minimum(z ** 2, 700))
    return phi @ c


# ---------------------------------------------------------------------------
# KAPI-ELM — Bayesian-optimized partition width (the adaptive-basis method)
# ---------------------------------------------------------------------------

def samp_points_sigma(N: int, custom_lens: list | np.ndarray,
                      kSigma: float = 5.0):
    """Port of SAMP_POINTS_SIGMA.m — generates partition + global centers."""
    custom_lens = np.asarray(custom_lens, dtype=np.float64)
    custom_lens = custom_lens / custom_lens.sum()
    k = len(custom_lens)
    edges = np.concatenate([[0.0], np.cumsum(custom_lens)])
    edges[-1] = 1.0

    x_part = []
    for j in range(k):
        a, b = edges[j], edges[j + 1]
        if j == 0 and j == k - 1:
            pts = np.linspace(a, b, N)
        elif j == 0:
            pts = np.linspace(a, b, N + 1)[:N]
        elif j == k - 1:
            pts = np.linspace(a, b, N + 1)[1:]
        else:
            pts = np.linspace(a, b, N + 2)[1:-1]
        x_part.append(pts)
    x_part = np.concatenate(x_part)

    sigma_part = np.concatenate(
        [np.full(N, kSigma * (custom_lens[j] / N)) for j in range(k)]
    )

    x_global = np.linspace(0, 1, k * N + 2)[1:-1]
    sigma_global = np.full_like(x_global, kSigma * (1.0 / (k * N)))

    return x_part, sigma_part, x_global, sigma_global, edges


def kapielm_solve(w: float, nu: float, N: int = 1000, kSigma: float = 5.0):
    """Solve the SPP for a given partition width w. Returns (J, c, m, alpha)."""
    lens = [w, 1 - w]
    x_part, sigma_part, x_global, sigma_global, edges = \
        samp_points_sigma(N, lens, kSigma)

    alpha_star = np.concatenate([x_part, x_global])
    sig_x = np.concatenate([sigma_part, sigma_global])
    X_pde = np.sort(alpha_star)

    m = 1.0 / (np.sqrt(2) * sig_x)
    alpha = -m * alpha_star

    z = X_pde[:, None] * m[None, :] + alpha[None, :]
    z2 = np.minimum(z ** 2, 700)
    phi = np.exp(-z2)
    phi_x = -2 * m[None, :] * z * phi
    phi_xx = 2 * (m[None, :] ** 2) * (2 * z2 - 1) * phi

    LHS_PDE = phi_x - nu * phi_xx
    RHS_PDE = np.zeros(len(X_pde))

    X_bc = np.array([0.0, 1.0])
    z_bc = X_bc[:, None] * m[None, :] + alpha[None, :]
    phi_bc = np.exp(-z_bc ** 2)
    RHS_BC = np.array([0.0, 1.0])

    H = np.vstack([LHS_PDE, phi_bc])
    b = np.concatenate([RHS_PDE, RHS_BC])
    c, *_ = np.linalg.lstsq(H, b, rcond=None)

    J = np.max(np.abs(H @ c - b))
    if not np.isfinite(J):
        J = 1e6
    return J, c, m, alpha


def kapielm_fit(nu: float, N: int = 1000, kSigma: float = 5.0,
                n_calls: int = 25, w_bounds: tuple = (0.90, 0.99),
                seed: int = 42) -> dict:
    """Run Bayesian optimization over partition width w for KAPI-ELM."""
    from skopt import gp_minimize
    from skopt.space import Real

    t0 = time.time()

    def objective(params):
        J, *_ = kapielm_solve(params[0], nu, N=N, kSigma=kSigma)
        return J

    res = gp_minimize(
        objective, [Real(*w_bounds, name="w")],
        n_calls=n_calls, random_state=seed, acq_func="EI",
    )

    best_w = res.x[0]
    J, c, m, alpha = kapielm_solve(best_w, nu, N=N, kSigma=kSigma)
    return {
        "best_w": best_w, "J": J, "c": c, "m": m, "alpha": alpha,
        "elapsed": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# ConvDiff-SPP problem setup for SCIO / Adam / L-BFGS (PINN-based methods)
# ---------------------------------------------------------------------------

def make_setup_convdiff_spp(nu: float):
    """Factory that returns a setup function for a specific nu value."""

    def setup_convdiff_spp(seed: int) -> ProblemSetup:
        set_seed(seed)
        n_f = 512  # uniform random PDE collocation points

        # 1D input: x in [0, 1]
        x_f = torch.rand(n_f, 1, device=DEVICE).requires_grad_(True)

        # Architecture: same as other 1D problems in the registry
        model = ScaledPINN(
            in_dim=1, out_dim=1, bounds=[(0.0, 1.0)],
            width=64, depth=4,
        ).to(DEVICE)

        # BC points (fixed)
        x_bc0 = torch.zeros(1, 1, device=DEVICE)
        x_bc1 = torch.ones(1, 1, device=DEVICE)

        def loss_components_fn(network: nn.Module) -> list[Tensor]:
            u = network(x_f)
            u_x = torch.autograd.grad(
                u, x_f, torch.ones_like(u), create_graph=True
            )[0]
            u_xx = torch.autograd.grad(
                u_x, x_f, torch.ones_like(u_x), create_graph=True
            )[0]
            # PDE residual: u_x - nu * u_xx = 0
            loss_pde = torch.mean((u_x - nu * u_xx) ** 2)
            # Boundary conditions: u(0) = 0, u(1) = 1
            loss_bc = 50.0 * (
                (network(x_bc0) - 0.0) ** 2 + (network(x_bc1) - 1.0) ** 2
            ).mean()
            return [loss_pde, loss_bc]

        # Evaluation: rel L2 on dense grid vs exact solution
        x_eval_np = np.linspace(0, 1, 2000)
        u_exact_np = exact_solution(x_eval_np, nu)
        x_eval_t = torch.tensor(
            x_eval_np, dtype=torch.float32, device=DEVICE
        ).reshape(-1, 1)
        u_exact_t = torch.tensor(
            u_exact_np, dtype=torch.float32, device=DEVICE
        ).reshape(-1, 1)

        def rel_l2_fn(network: nn.Module) -> float:
            network.eval()
            with torch.no_grad():
                pred = network(x_eval_t)
                err = torch.linalg.norm(pred - u_exact_t)
                ref = torch.linalg.norm(u_exact_t)
                rel = err / (ref if ref > 1e-12 else torch.tensor(1.0))
            network.train()
            return to_float(rel)

        return ProblemSetup(
            model=model,
            loss_components_fn=loss_components_fn,
            tracked_inputs=[x_f],
            rel_l2_fn=rel_l2_fn,
        )

    return setup_convdiff_spp


# ---------------------------------------------------------------------------
# Validation gates (Section 4 of the task spec)
# ---------------------------------------------------------------------------

def run_validation_gates() -> bool:
    """Run pre-sweep validation. Returns True if all gates pass."""
    print("=" * 70)
    print("VALIDATION GATE 1: Vanilla PIELM monotonic blowup")
    print("=" * 70)

    x_eval = np.linspace(0, 1, 2000)
    gate1_pass = True
    prev_err = -1.0
    pielm_errors = {}

    for nu in [0.1, 0.01, 1e-3, 1e-4]:
        result = vanilla_pielm_solve(nu, seed=0)
        pred = elm_eval(result, x_eval)
        true = exact_solution(x_eval, nu)
        err = relative_l2(pred, true)
        pielm_errors[nu] = err
        monotonic = "[OK]" if err > prev_err else "[FAIL] NOT MONOTONIC"
        if err <= prev_err and prev_err > 0:
            gate1_pass = False
        prev_err = err
        print(f"  nu={nu:.0e}  rel_l2={err:.4f}  J={result['J']:.3e}  {monotonic}")

    if not gate1_pass:
        print("GATE 1 FAILED: Vanilla PIELM errors are not monotonically "
              "increasing with decreasing nu. Debug before proceeding.")
        return False
    print("GATE 1 PASSED: Monotonic blowup confirmed.\n")

    print("=" * 70)
    print("VALIDATION GATE 2: KAPI-ELM recovery at nu=1e-4")
    print("=" * 70)

    nu_test = 1e-4
    print(f"  Running KAPI-ELM fit at nu={nu_test}, N=1000, n_calls=25...")
    kapi_result = kapielm_fit(nu_test, N=1000, n_calls=25, seed=42)
    kapi_pred = elm_eval(kapi_result, x_eval)
    kapi_true = exact_solution(x_eval, nu_test)
    kapi_err = relative_l2(kapi_pred, kapi_true)
    pielm_err = pielm_errors[nu_test]

    print(f"  KAPI-ELM: rel_l2={kapi_err:.6f}, best_w={kapi_result['best_w']:.4f}, "
          f"J={kapi_result['J']:.3e}, time={kapi_result['elapsed']:.1f}s")
    print(f"  Vanilla PIELM: rel_l2={pielm_err:.4f}")
    improvement = pielm_err / max(kapi_err, 1e-15)
    print(f"  Improvement factor: {improvement:.1f}x")

    if kapi_err >= pielm_err:
        print("GATE 2 FAILED: KAPI-ELM did NOT improve over vanilla PIELM. "
              "The Python port may be incorrect.")
        return False
    if improvement < 5.0:
        print(f"GATE 2 WARNING: Improvement is only {improvement:.1f}x — "
              "expected dramatically better. Proceeding with caution.")
    print("GATE 2 PASSED: KAPI-ELM recovery confirmed.\n")
    return True


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------

NU_VALUES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
SEEDS = [42, 123, 7]
PINN_TARGET_NFE = 800


def run_pinn_method(method_name: str, nu: float, seed: int) -> dict:
    """Run a PINN-based method (SCIO, Adam, L-BFGS) for a specific nu."""
    # Register the problem for this nu value
    problem_key = f"ConvDiff-SPP-nu{nu:.0e}"
    PROBLEMS[problem_key] = ProblemSpec(
        name=problem_key,
        build_setup=make_setup_convdiff_spp(nu),
        scion_kwargs=dict(
            lr=0.001, memory_size=50, burn_in_steps=50,
            T_high=1.25, max_iter=10, max_step_norm=1.0,
        ),
        target_nfe=PINN_TARGET_NFE,
        adam_lr=1e-3,
        lbfgs_lr=0.5,
        lbfgs_history_size=50,
        lbfgs_max_iter=20,
        multiadam_kwargs=dict(loss_group_idx=[1], group_weights=(0.5, 0.5)),
    )

    result = run_problem(
        problem_key,
        optimizer_name=method_name,
        seed=seed,
        target_nfe=PINN_TARGET_NFE,
    )
    # Clean up to avoid polluting the registry
    if problem_key in PROBLEMS:
        del PROBLEMS[problem_key]

    return result


def diagnose_failure_mode(result: dict) -> str:
    """Classify the failure mode for a PINN run.

    Returns one of:
      'converged'         — loss is low, rel_l2 is low
      'representation'    — loss plateaued at a non-trivial value
                            (architecture/spectral-bias limit)
      'eval_mismatch'     — loss is low but rel_l2 is high (possible overfitting
                            to collocation points or other pathology)
      'diverged'          — NaN/Inf in the loss history
      'optimizer_stuck'   — loss is still decreasing at budget end (might need
                            more NFEs)
    """
    history = result.get("history", {})
    losses = history.get("loss", [])

    if not losses:
        return "no_data"

    # Check for NaN/divergence
    if any(not np.isfinite(l) for l in losses):
        return "diverged"

    final_loss = losses[-1]
    rel_l2 = result.get("rel_l2", float("inf"))

    # Check if loss is still meaningfully decreasing in the last 20% of training
    cutoff = max(1, int(0.8 * len(losses)))
    late_losses = losses[cutoff:]
    if len(late_losses) >= 2:
        late_start = np.mean(late_losses[:len(late_losses)//4 + 1])
        late_end = np.mean(late_losses[-len(late_losses)//4 - 1:])
        still_decreasing = (late_start - late_end) / (abs(late_start) + 1e-15) > 0.05
    else:
        still_decreasing = False

    # Thresholds
    if rel_l2 < 0.05:
        return "converged"
    if final_loss > 0.01 and not still_decreasing:
        return "representation"  # plateaued at non-trivial loss
    if final_loss < 0.001 and rel_l2 > 0.1:
        return "eval_mismatch"
    if still_decreasing:
        return "optimizer_stuck"
    return "representation"


def run_full_experiment(output_dir: Path):
    """Run the complete experiment sweep and write results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x_eval = np.linspace(0, 1, 2000)
    all_rows = []

    # ---------------------------------------------------------------
    # 1. Vanilla PIELM (pure NumPy, CPU-only, 3 seeds)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING: Vanilla PIELM")
    print("=" * 70)

    for nu in NU_VALUES:
        for seed in SEEDS:
            t0 = time.time()
            result = vanilla_pielm_solve(nu, seed=seed)
            pred = elm_eval(result, x_eval)
            true_vals = exact_solution(x_eval, nu)
            err = relative_l2(pred, true_vals)
            elapsed = time.time() - t0

            row = {
                "method": "Vanilla-PIELM",
                "nu": nu,
                "seed": seed,
                "rel_l2": err,
                "wall_time_s": elapsed,
                "final_loss": float("nan"),
                "final_nfe": 0,
                "nan_divergence": False,
                "inf_norm_J": result["J"],
                "best_w": float("nan"),
                "failure_mode": "n/a",
            }
            all_rows.append(row)
            print(f"  nu={nu:.0e} seed={seed}: rel_l2={err:.6f}  "
                  f"J={result['J']:.3e}  time={elapsed:.2f}s")

    # ---------------------------------------------------------------
    # 2. KAPI-ELM (pure NumPy + skopt BO, CPU-only, 3 seeds)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING: KAPI-ELM (BO over partition width)")
    print("=" * 70)

    for nu in NU_VALUES:
        for seed in SEEDS:
            print(f"  nu={nu:.0e} seed={seed}: fitting...", end=" ", flush=True)
            kapi = kapielm_fit(nu, N=1000, n_calls=25, seed=seed)
            pred = elm_eval(kapi, x_eval)
            true_vals = exact_solution(x_eval, nu)
            err = relative_l2(pred, true_vals)

            row = {
                "method": "KAPI-ELM",
                "nu": nu,
                "seed": seed,
                "rel_l2": err,
                "wall_time_s": kapi["elapsed"],
                "final_loss": float("nan"),
                "final_nfe": 0,
                "nan_divergence": False,
                "inf_norm_J": kapi["J"],
                "best_w": kapi["best_w"],
                "failure_mode": "n/a",
            }
            all_rows.append(row)
            print(f"rel_l2={err:.6f}  best_w={kapi['best_w']:.4f}  "
                  f"J={kapi['J']:.3e}  time={kapi['elapsed']:.1f}s")

    # ---------------------------------------------------------------
    # 3. PINN-based methods: SCIO, Adam, L-BFGS (torch, GPU, 3 seeds)
    # ---------------------------------------------------------------
    for method_name in ["SCIO", "Adam", "L-BFGS"]:
        print(f"\n{'=' * 70}")
        print(f"RUNNING: {method_name} (PINN, {PINN_TARGET_NFE} NFEs)")
        print("=" * 70)

        for nu in NU_VALUES:
            for seed in SEEDS:
                print(f"  nu={nu:.0e} seed={seed}: training...", end=" ",
                      flush=True)
                try:
                    result = run_pinn_method(method_name, nu, seed)
                    nan_flag = any(
                        not np.isfinite(l)
                        for l in result["history"]["loss"]
                    )
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
                    all_rows.append(row)
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
                    all_rows.append(row)

    return all_rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_results_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {path}")


# ---------------------------------------------------------------------------
# Headline plot: rel L2 vs nu (log-log), one line per method
# ---------------------------------------------------------------------------

def plot_headline(rows: list[dict], path: Path):
    """Generate the log-log headline figure: rel L2 error vs nu."""
    path.parent.mkdir(parents=True, exist_ok=True)

    methods = ["Vanilla-PIELM", "KAPI-ELM", "SCIO", "Adam", "L-BFGS"]
    colors = {
        "Vanilla-PIELM": "#9CA3AF",
        "KAPI-ELM": "#10B981",
        "SCIO": "#2563EB",
        "Adam": "#DC2626",
        "L-BFGS": "#F59E0B",
    }
    markers = {
        "Vanilla-PIELM": "s",
        "KAPI-ELM": "D",
        "SCIO": "o",
        "Adam": "^",
        "L-BFGS": "v",
    }

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue

        # Group by nu, compute mean ± std
        nu_vals = sorted(set(r["nu"] for r in method_rows))
        means, stds = [], []
        for nu in nu_vals:
            errs = [r["rel_l2"] for r in method_rows
                    if r["nu"] == nu and np.isfinite(r["rel_l2"])]
            if errs:
                means.append(np.mean(errs))
                stds.append(np.std(errs, ddof=0))
            else:
                means.append(float("nan"))
                stds.append(0.0)

        means, stds = np.array(means), np.array(stds)
        valid = np.isfinite(means)

        ax.errorbar(
            np.array(nu_vals)[valid], means[valid],
            yerr=stds[valid],
            marker=markers[method], color=colors[method],
            linewidth=2.2, markersize=8, capsize=4,
            label=method, zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\nu$ (diffusion coefficient)", fontsize=14)
    ax.set_ylabel("Relative $L^2$ error", fontsize=14)
    ax.set_title(
        r"1D Singularly-Perturbed Convection-Diffusion: $u_x - \nu\, u_{xx} = 0$",
        fontsize=14, pad=12,
    )
    ax.invert_xaxis()  # nu decreases → problem gets harder
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Headline plot saved to {path}")


# ---------------------------------------------------------------------------
# Failure-mode summary for nu=1e-5
# ---------------------------------------------------------------------------

def print_failure_mode_analysis(rows: list[dict]):
    """Print diagnostic summary for the hardest nu values."""
    print("\n" + "=" * 70)
    print("FAILURE-MODE ANALYSIS (nu = 1e-5)")
    print("=" * 70)
    print("Distinguishing: (a) representation/spectral-bias limit vs "
          "(b) optimizer failure vs (c) divergence\n")

    pinn_methods = ["SCIO", "Adam", "L-BFGS"]
    for method in pinn_methods:
        method_rows = [r for r in rows
                       if r["method"] == method and r["nu"] == 1e-5]
        if not method_rows:
            print(f"  {method}: no data at nu=1e-5")
            continue

        losses = [r["final_loss"] for r in method_rows if np.isfinite(r["final_loss"])]
        errors = [r["rel_l2"] for r in method_rows if np.isfinite(r["rel_l2"])]
        modes = [r["failure_mode"] for r in method_rows]
        nan_flags = [r["nan_divergence"] for r in method_rows]

        avg_loss = np.mean(losses) if losses else float("nan")
        avg_err = np.mean(errors) if errors else float("nan")

        print(f"  {method}:")
        print(f"    Avg final loss: {avg_loss:.3e}")
        print(f"    Avg rel L2:     {avg_err:.4f}")
        print(f"    NaN/diverge:    {any(nan_flags)}")
        print(f"    Failure modes:  {modes}")

        # Interpret
        if all(m == "converged" for m in modes):
            print(f"    -> Converged fine at nu=1e-5.")
        elif all(m == "representation" for m in modes):
            print(f"    -> Representation limit (spectral bias). "
                  f"Loss plateaued at {avg_loss:.3e}.")
            print(f"      This is an architecture ceiling, NOT an "
                  f"optimizer-specific finding.")
        elif all(m == "diverged" for m in modes):
            print(f"    -> Genuine optimizer divergence.")
        elif all(m == "optimizer_stuck" for m in modes):
            print(f"    -> Loss still decreasing — may need more NFEs.")
        else:
            print(f"    -> Mixed modes across seeds: {modes}")
        print()

    # Compare: if all three PINN methods have the same failure mode,
    # that's strong evidence it's not optimizer-specific
    pinn_modes = {}
    for method in pinn_methods:
        method_rows = [r for r in rows
                       if r["method"] == method and r["nu"] == 1e-5]
        modes = [r["failure_mode"] for r in method_rows]
        pinn_modes[method] = modes

    all_modes_flat = [m for modes in pinn_modes.values() for m in modes]
    if all_modes_flat and all(m == all_modes_flat[0] for m in all_modes_flat):
        if all_modes_flat[0] == "representation":
            print("  *** ALL three PINN optimizers hit the same representation "
                  "ceiling. ***")
            print("  *** This is spectral bias of the tanh MLP, not an "
                  "optimizer-level difference. ***\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = Path(__file__).resolve().parent / "convdiff_results"

    # Step 1: Validation gates
    print("\n" + "#" * 70)
    print("# VALIDATION GATES")
    print("#" * 70)
    if not run_validation_gates():
        print("\nValidation failed. Aborting.")
        sys.exit(1)

    # Step 2: Full experiment sweep
    print("\n" + "#" * 70)
    print("# FULL EXPERIMENT SWEEP")
    print("#" * 70)
    rows = run_full_experiment(output_dir)

    # Step 3: Write results
    write_results_csv(rows, output_dir / "results.csv")
    plot_headline(rows, output_dir / "convdiff_spp_results.png")

    # Step 4: Failure-mode analysis
    print_failure_mode_analysis(rows)

    # Step 5: Print summary table
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
            errs = [r["rel_l2"] for r in rows
                    if r["method"] == method and r["nu"] == nu
                    and np.isfinite(r["rel_l2"])]
            if errs:
                mean = np.mean(errs)
                std = np.std(errs, ddof=0)
                line += f"{mean:.3e}±{std:.1e} "
            else:
                line += f"{'N/A':<16}"
        print(line)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
