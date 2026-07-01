# 1D Singularly-Perturbed Convection-Diffusion Experiment

This repository contains the code and benchmark results for comparing **Physics Informed Extreme Learning Machines (PIELM)** and **Physics-Informed Neural Networks (PINNs)** on the 1D singularly-perturbed convection-diffusion equation:

$$u_x - \nu\, u_{xx} = 0, \quad x \in [0, 1], \quad u(0) = 0, \quad u(1) = 1$$

As the diffusion coefficient $\nu \to 0$, a sharp boundary layer forms near $x = 1$, presenting a severe multiscale challenge for standard numerical methods and vanilla neural network solvers.

---

## Benchmarked Methods

1. **Vanilla-PIELM**: A Physics Informed Extreme Learning Machine with a fixed random hidden layer mapping.
2. **KAPI-ELM**: A Soft Partition-based KAPI-ELM method that uses Bayesian Optimization (`scikit-optimize`) to identify the optimal partition boundary layer width, demonstrating highly robust recovery at small $\nu$ values (e.g. $\nu = 10^{-4}$).
3. **PINN (Adam)**: Standard Physics-Informed Neural Network architecture trained using standard Adam optimization.
4. **PINN (L-BFGS)**: Standard PINN architecture trained using L-BFGS optimization with `strong_wolfe` line search.
5. **SCIO Optimizer**: Baseline metrics for the curvature-aware hybrid SCIO optimizer are loaded and plotted from reference results for comparison.
6. **RAR-SCIO**: SCIO optimizer trained using Residual-based Adaptive Refinement (RAR), accumulating high-residual collocation points in a feedback loop.
7. **RAD-SCIO**: SCIO optimizer trained using Residual-based Adaptive Distribution (RAD), resampling all collocation points from a probability distribution proportional to $|R(x)| + 1$.

---

## Repository Structure

- `convdiff_benchmark.py`: Core execution script for baseline runs. Contains validation gates, localized training sweeps, and plotting scripts.
- `convdiff_ras_benchmark.py`: Execution script for the Residual Adaptive Sampling benchmark (RAR-SCIO and RAD-SCIO sweeps across all scales and seeds).
- `merge_and_run_pinns.py`: Aggregates the long-running KAPI-ELM/Vanilla-PIELM optimization results from Kaggle with local GPU training of PINN methods (`Adam`, `L-BFGS`) and plots the final benchmark results.
- `soft_kapi_ref/`: MATLAB reference scripts for reproducing results submitted in the article *"Soft Partition-based KAPI-ELM for Multi-Scale PDEs"*.
- `kaggle_output/`: Contains pre-computed Vanilla-PIELM, KAPI-ELM, and SCIO comparative result sheets.
- `convdiff_results/`: Pre-computed local run outputs, comparison plots, and adaptive sampling report.
  - `results_ras.csv`: CSV file compiling all metrics from the adaptive sampling runs.
  - `Dwivedi_SCIO_RAS_report.docx`: Formatted detailed Word report with tables, prediction plots, and collocation point histograms.
  - `ras_plots/`: Directory containing generated prediction curves and RAR histograms.

---

## Installation & Setup

### Prerequisites

Ensure you have Python 3.8+ installed along with PyTorch, NumPy, Matplotlib, Scipy, and Scikit-Optimize.

You can install all required dependencies via pip:

```bash
pip install torch numpy scipy matplotlib scikit-optimize psutil
```

### Running MATLAB Reference Scripts

MATLAB scripts for Bayesian Optimization and Partition-based KAPI-ELM are located in the `soft_kapi_ref/` directory:
- Run `SPP_01_BayesOpt.m` for the singular perturbation problem optimization.
- View test case files `TC_01_Oscillatory_FO.m` through `TC_08_SPP_02.m` for different physical problems.

---

## Execution Guide

### Option 1: Fast Summary & Plotting (Using Pre-computed Results)

To quickly view the comparison table and generate the final performance plots without running hours of training, run:

```bash
python merge_and_run_pinns.py
```

This will automatically load pre-computed results from `kaggle_output/` and print the comparative relative L2 errors.

### Option 2: Full Sweep Run (From Scratch)

To execute a complete local training run for Vanilla-PIELM, KAPI-ELM, Adam, and L-BFGS across all seeds:

1. Delete or rename the existing results file:
   ```bash
   rm convdiff_results/results.csv
   ```
2. Execute the runner:
   ```bash
   python merge_and_run_pinns.py
   ```

*Note: running KAPI-ELM Bayesian Optimization sweeps locally on CPU can take up to 2 hours.*

---

## Experimental Results

The benchmark outputs show the mean relative $L^2$ error across three seeds (42, 123, 7):

| Method | $\nu=10^{-1}$ | $\nu=10^{-2}$ | $\nu=10^{-3}$ | $\nu=10^{-4}$ | $\nu=10^{-5}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vanilla-PIELM** | $1.432 \times 10^{-3}$ | $4.181$ | $17.76$ | $22.36$ | $22.36$ |
| **KAPI-ELM** | $7.457 \times 10^{-8}$ | $5.767 \times 10^{-8}$ | $8.740 \times 10^{-7}$ | $1.983 \times 10^{-4}$ | $5.249$ |
| **Adam** | $1.017 \times 10^{-2}$ | $7.744$ | $20.55$ | $25.92$ | $25.92$ |
| **L-BFGS** | $2.297 \times 10^{-4}$ | $5.178$ | $23.47$ | $26.52$ | $29.25$ |
| **SCIO (Uniform)** | $3.244 \times 10^{-3}$ | $5.184$ | $23.19$ | $29.19$ | $22.37$ |
| **RAR-SCIO** | $2.906 \times 10^{-3}$ | $7.549$ | $21.65$ | $23.73$ | $24.37$ |
| **RAD-SCIO** | $5.931 \times 10^{-4}$ | $7.622$ | $20.22$ | $25.39$ | $25.50$ |

### Why Residual Adaptive Resampling Fails (Feedback Lock)
Our experiments confirm Dr. Vikas Dwivedi's hypothesis that residual adaptive sampling is unable to escape training failure modes on stiff singularly-perturbed equations. The reason is a **feedback lock**:
1. When the network gets stuck at initialization (predicting a flat, near-zero profile), the interior PDE derivatives $u_x$ and $u_{xx}$ evaluate to $\approx 0$.
2. This creates a degenerate, uniformly near-zero residual landscape, yielding no adaptive sampling signals to RAD.
3. For RAR, the mismatch strictly at the boundary condition ($u(1)=1$ vs. $0.0$) spikes, prompting the algorithm to pile up almost all collocation points on the boundary ($x \approx 0.0$ and $x \approx 1.0$), leaving the interior boundary layer completely unrepresented.
4. Without collocation points inside the physical transition zone ($x \approx 0.99$), the network never receives the gradient signals required to escape the initialization state, locking it permanently.
