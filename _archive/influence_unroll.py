"""Influence of number of unrolls on policy quality.

Sweeps N_unrolls ∈ [2,10] step 2 for Architecture 1 and 2 at K ∈ {1, 5, 10}.
Baselines: value iteration (1-step PE) and K-step policy iteration.

Output: results/n_unrolls/<group>_data.npz per K value.
"""

import os
import numpy as np
from time import perf_counter

import torch

from src.utils import get_optimal_q, plot_errors, save_error_matrix_to_csv
from experiments.runners import run_unroll_sweep

torch.set_float32_matmul_precision("medium")

SAVE = True
PATH = "results/n_unrolls/"
os.makedirs(PATH, exist_ok=True)

N_RUNS = 15
N_UNROLLS = np.arange(2, 11, 2)  # [2, 4, 6, 8, 10]
USE_LOGGER = False
LOG_EVERY = 1


def _make_exps(K: int) -> list[dict]:
    common = dict(tau=5, lr=5e-3, weight_sharing=True,
                  loss_type="original_with_detach", init_q="random",
                  use_legacy_init=True)
    return [
        {"model": "pol-it", "args": {"max_eval_iters": 1},
         "fmt": "^-", "name": "val-it"},
        {"model": "pol-it", "args": {"max_eval_iters": K},
         "fmt": "x-", "name": f"pol-it-K{K}"},
        {"model": "unroll", "args": {**common, "K": K, "architecture_type": 1},
         "fmt": "o-", "name": f"arch1-K{K}-WS"},
        {"model": "unroll", "args": {**common, "K": K, "architecture_type": 2},
         "fmt": "s-", "name": f"arch2-K{K}-WS"},
    ]


def run_block(K: int) -> None:
    group_name = f"n_unrolls-K{K}"
    q_opt = get_optimal_q(use_logger=USE_LOGGER, group_name=group_name)
    exps = _make_exps(K)

    n_exp = len(exps)
    errs1 = np.zeros((N_RUNS, n_exp, N_UNROLLS.size))
    errs2 = np.zeros((N_RUNS, n_exp, N_UNROLLS.size))
    bell_errs = np.zeros((N_RUNS, n_exp, N_UNROLLS.size))

    t0 = perf_counter()
    for g in range(N_RUNS):
        errs1[g], errs2[g], bell_errs[g] = run_unroll_sweep(
            g, N_UNROLLS, exps, q_opt,
            use_logger=USE_LOGGER, group_name=group_name, verbose=True,
        )
    print(f"----- K={K} solved in {(perf_counter() - t0) / 60:.1f} min -----")

    if SAVE:
        fp = PATH + f"{group_name}_data.npz"
        np.savez(fp, N_unrolls=N_UNROLLS, exps=exps,
                 errs1=errs1, errs2=errs2, bell_errs=bell_errs)
        print("Saved:", fp)
        save_error_matrix_to_csv(np.median(errs1, axis=0), N_UNROLLS, exps,
                                 PATH + f"{group_name}_med_err.csv")
        save_error_matrix_to_csv(np.percentile(errs1, 25, axis=0), N_UNROLLS, exps,
                                 PATH + f"{group_name}_prctile25.csv")
        save_error_matrix_to_csv(np.percentile(errs1, 75, axis=0), N_UNROLLS, exps,
                                 PATH + f"{group_name}_prctile75.csv")

    xlabel = "Number of unrolls"
    plot_errors(errs1, N_UNROLLS, exps, xlabel, "Q err (rel)", agg="median", deviation="prctile")
    plot_errors(bell_errs, N_UNROLLS, exps, xlabel, "Bellman err", agg="median", deviation="prctile")


if __name__ == "__main__":
    for K in [1, 5, 10]:
        run_block(K)
