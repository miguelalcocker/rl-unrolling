"""Influence of filter order K on policy quality.

Sweeps K ∈ {1, 2, 3, 5, 10, 15} for Architecture 1 and 2 (weight-shared)
at num_unrolls ∈ {5, 10}.  Baseline: policy iteration with same number of
improvement steps.

Output: results/filter_order/<group>_data.npz per num_unrolls value.
"""

import os
import numpy as np
from time import perf_counter

import torch

from src.utils import get_optimal_q, plot_errors, save_error_matrix_to_csv
from experiments.runners import run_k_sweep

torch.set_float32_matmul_precision("medium")

SAVE = True
PATH = "results/filter_order/"
os.makedirs(PATH, exist_ok=True)

N_RUNS = 15
KS = np.array([1, 2, 3, 5, 10, 15])
USE_LOGGER = False
LOG_EVERY = 1


def _make_exps(num_unrolls: int) -> list[dict]:
    common = dict(num_unrolls=num_unrolls, tau=5, lr=5e-3, weight_sharing=True,
                  loss_type="original_with_detach", init_q="random",
                  use_legacy_init=True)
    return [
        {"model": "pol-it", "args": {"max_epochs": num_unrolls},
         "fmt": "x-", "name": f"pol-it-U{num_unrolls}"},
        {"model": "unroll", "args": {**common, "architecture_type": 1},
         "fmt": "o-", "name": f"arch1-U{num_unrolls}-WS"},
        {"model": "unroll", "args": {**common, "architecture_type": 2},
         "fmt": "s-", "name": f"arch2-U{num_unrolls}-WS"},
    ]


def run_block(num_unrolls: int) -> None:
    group_name = f"filter_order-U{num_unrolls}"
    q_opt = get_optimal_q(use_logger=USE_LOGGER, group_name=group_name)
    exps = _make_exps(num_unrolls)

    n_exp = len(exps)
    errs1 = np.zeros((N_RUNS, n_exp, KS.size))
    errs2 = np.zeros((N_RUNS, n_exp, KS.size))
    bell_errs = np.zeros((N_RUNS, n_exp, KS.size))

    t0 = perf_counter()
    for g in range(N_RUNS):
        errs1[g], errs2[g], bell_errs[g] = run_k_sweep(
            g, KS, exps, q_opt,
            use_logger=USE_LOGGER, group_name=group_name, verbose=True,
        )
    print(f"----- U={num_unrolls} solved in {(perf_counter() - t0) / 60:.1f} min -----")

    if SAVE:
        fp = PATH + f"{group_name}_data.npz"
        np.savez(fp, Ks=KS, exps=exps,
                 errs1=errs1, errs2=errs2, bell_errs=bell_errs)
        print("Saved:", fp)
        save_error_matrix_to_csv(np.median(errs1, axis=0), KS, exps,
                                 PATH + f"{group_name}_med_err.csv")
        save_error_matrix_to_csv(np.percentile(errs1, 25, axis=0), KS, exps,
                                 PATH + f"{group_name}_prctile25.csv")
        save_error_matrix_to_csv(np.percentile(errs1, 75, axis=0), KS, exps,
                                 PATH + f"{group_name}_prctile75.csv")

    xlabel = "Filter order K"
    plot_errors(errs1, KS, exps, xlabel, "Q err (rel)", agg="median", deviation="prctile")
    plot_errors(bell_errs, KS, exps, xlabel, "Bellman err", agg="median", deviation="prctile")


if __name__ == "__main__":
    for U in [5, 10]:
        run_block(U)
