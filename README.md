# MOACO-Guided Estimation of Covariance of Scalarized Optimal Solutions
### Project Overview
The aim is to **intelligently select a small subset of weight vectors $\lambda_1, ..., \lambda_m$** (with $m \ll N$) that allows for an accurate estimation of the **empirical covariance matrix** $C_e$, such that it approximates the true (and unknown) **global covariance matrix** $C$. This is critical for reducing the computational cost of evaluating scalarized optimization problems while preserving estimation quality.

###  Surrogate Model: Gaussian Process Regression
Implemented in `surrogate_model.py`, the **surrogate model** is trained to predict the sensitivity of the solution to perturbations in $\lambda$, expressed via the Frobenius norm $\|C(\lambda)\|$.

**Key features:**

* **Input:** Weight vectors $\lambda = [\lambda_1, \lambda_2]$ ($\lambda_3 = 1 - \lambda_1 - \lambda_2$).
* **Output:** Predicted sensitivity norm.
* **Kernel:** A hybrid of RBF, Matérn, and Rational Quadratic kernels with a learnable noise model (WhiteKernel).
* **Training:** Based on a normalized dataset of precomputed covariance norms from perturbed $\lambda$ values.
* 
The surrogate is retrained periodically reloaded to maintain alignment with newly sampled $\lambda$.
---

###  MOACO: Multi-Objective Ant Colony Optimization + Active Learning
Implemented in `aco_active_learning.py`, this module integrates Ant Colony Optimization (ACO) with active learning principles to iteratively build an informative set of $\lambda$ vectors.

#### Core Concepts:
* **Ant Representation:** Each ant samples a candidate $\lambda$ vector.
* **Heuristic $\eta$:** Computed using surrogate predictions, normalized and diversity-penalized to encourage exploration of underrepresented regions.
* **Pheromone $\tau$:** Encodes historical utility of each $\lambda$. Updated when new selections reduce the error $\|C - C_e\|$.
* **Delta-k Evaluation:** For each candidate $\lambda$, the change in error is computed as $\Delta_k = \|C - C_{e+\lambda}\| - \|C - C_e\|$, which guides inclusion.
* **Diversity-Preserving Selection:** Ensures selected lambdas are not too close in Euclidean distance ($\text{min\_dist} = 0.08$).
* **Pruning:** If too many $\lambda$ are selected (> max\_lambdas), a diverse subset is reselected based on pheromone intensities

#### Execution Flow (`run_aco_active_learning`):
1. **Initial Selection:** Choose a diverse set of informative $\lambda$ (based on surrogate $\eta$).
2. **Iteration Loop:**

   * Sample new candidates using probabilistic selection from $\tau$ and $\eta$.
   * Evaluate $\Delta_k$ and update $C_e$.
   * Update pheromones and prune if needed.
   * Optionally retrain the surrogate.
3. **Convergence:** Stop if $\|C - C_e\| < \epsilon$ or too few new $\lambda$ remain.
  

