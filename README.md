# BellNet: Unrolling Dynamic Programming via Graph Filters

Research code accompanying the study of Unrolled Policy Iteration (UPI), a neural network approach that parameterizes the Bellman operator as a learnable graph filter and solves MDPs through differentiable dynamic programming.

Based on the paper:
> Rozada, S., Rey, S., Mateos, G., & Marques, A. G. (2025). *Unrolling Dynamic Programming via Graph Filters*. arXiv:2507.21705.

Extended by Miguel Alcocer Pérez (TFG, URJC 2026).

---

## Project structure

```
rl-unrolling/
├── src/                        # Core library
│   ├── __init__.py
│   ├── environments.py         # CliffWalkingEnv, MirroredCliffWalkingEnv,
│   │                           #   GeneralizedCliffWalkingEnv, WindyGridWorldEnv,
│   │                           #   ChainMDP, RandomGraphMDP
│   ├── models.py               # PolicyEvaluationLayer (arch 1 & 2),
│   │                           #   PolicyImprovementLayer, UnrolledPolicyIterationModel
│   ├── models_experimental.py  # Archive: arch 3/5 (matrix filters, joint input)
│   ├── plots.py                # Policy / filter-coef visualisations
│   ├── utils.py                # get_optimal_q, test_pol_err, plot_errors,
│   │                           #   save_error_matrix_to_csv
│   └── algorithms/
│       ├── generalized_policy_iteration.py  # PolicyIterationTrain (pl.LightningModule)
│       └── unrolling_policy_iteration.py    # UnrollingPolicyIterationTrain
├── experiments/
│   ├── runners.py              # Shared sweep loops: run_unroll_sweep,
│   │                           #   run_k_sweep, run_transfer_sweep
│   ├── influence_unroll.py     # Sweep N_unrolls at K ∈ {1,5,10}
│   ├── influence_K.py          # Sweep K at num_unrolls ∈ {5,10}
│   ├── influence_transferability.py  # Transfer CliffWalking → MirroredCliffWalking
│   ├── cliff_variations.py     # Generalised cliff environments comparison
│   ├── influence_unroll.ipynb  # Interactive version of influence_unroll.py
│   ├── influence_K.ipynb
│   └── influence_transferability.ipynb
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/miguelalcocker/rl-unrolling.git
cd rl-unrolling
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optionally configure Weights & Biases (`wandb login`) — set `use_logger=True` in any experiment to enable logging.

---

## Reproducing experiments

All experiment scripts are self-contained and run from the repo root.

```bash
# Sweep number of unrolls (K=1, K=5, K=10)
python -m experiments.influence_unroll

# Sweep filter order K (num_unrolls=5, 10)
python -m experiments.influence_K

# Transfer learning (CliffWalking → MirroredCliffWalking)
python -m experiments.influence_transferability

# Cliff topology variations
python -m experiments.cliff_variations
```

Results are written to `results/{n_unrolls,filter_order,transfer}/`.

---

## Architecture variants

### Architecture 1 (default)
Single monomial feedback term:

```
q̂ = Σ_{k=0}^{K} h_k P_π^k r  +  β h_{K+1} P_π^{K+1} q_0
```

### Architecture 2
Polynomial feedback of degree K₂:

```
q̂ = Σ_{k=0}^{K} h_k P_π^k r  +  β Σ_{k=K-K₂+1}^{K+K₂+1} w_k P_π^k q_0
```

Both architectures use weight sharing across unrolling layers by default.  
Experimental architectures (matrix filters, joint concatenated input) are
preserved in the `research/experimental-architectures` git branch.

---

## Citation

```bibtex
@misc{rozada2025unrollingdynamicprogramminggraph,
  title={Unrolling Dynamic Programming via Graph Filters},
  author={Sergio Rozada and Samuel Rey and Gonzalo Mateos and Antonio G. Marques},
  year={2025},
  eprint={2507.21705},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2507.21705},
}
```
