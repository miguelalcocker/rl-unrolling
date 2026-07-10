"""Influence of unrolls on transferability (CliffWalking → MirroredCliffWalking).

Trains on CliffWalkingEnv, evaluates on MirroredCliffWalkingEnv.
Sweeps N_unrolls ∈ [2,10] step 2 for Architecture 1 (K ∈ {5, 10})
vs pol-it baseline.

Output: results/transfer/<group>_data.npz per K value.
"""

import os
import numpy as np
from time import perf_counter

import torch

from src.utils import get_optimal_q, plot_errors, save_error_matrix_to_csv
from experiments.runners import run_transfer_sweep

torch.set_float32_matmul_precision("medium")

SAVE = True
PATH = "results/transfer/"
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
    group_name = f"transfer-K{K}"
    q_opt = get_optimal_q(mirror_env=False, use_logger=USE_LOGGER, group_name=group_name)
    q_opt_test = get_optimal_q(mirror_env=True, use_logger=USE_LOGGER, group_name=group_name)
    exps = _make_exps(K)

    n_exp = len(exps)
    errs = np.zeros((N_RUNS, n_exp, N_UNROLLS.size))
    errs_trans = np.zeros((N_RUNS, n_exp, N_UNROLLS.size))
    bell_errs = np.zeros((N_RUNS, n_exp, N_UNROLLS.size))

    t0 = perf_counter()
    for g in range(N_RUNS):
        errs[g], errs_trans[g], bell_errs[g] = run_transfer_sweep(
            g, N_UNROLLS, exps, q_opt, q_opt_test,
            use_logger=USE_LOGGER, group_name=group_name, verbose=True,
        )
    print(f"----- K={K} solved in {(perf_counter() - t0) / 60:.1f} min -----")

    if SAVE:
        fp = PATH + f"{group_name}_data.npz"
        np.savez(fp, N_unrolls=N_UNROLLS, exps=exps,
                 errs=errs, errs_trans=errs_trans, bell_errs=bell_errs)
        print("Saved:", fp)
        save_error_matrix_to_csv(np.median(errs_trans, axis=0), N_UNROLLS, exps,
                                 PATH + f"{group_name}_transfer_med_err.csv")
        save_error_matrix_to_csv(np.percentile(errs_trans, 25, axis=0), N_UNROLLS, exps,
                                 PATH + f"{group_name}_transfer_prctile25.csv")
        save_error_matrix_to_csv(np.percentile(errs_trans, 75, axis=0), N_UNROLLS, exps,
                                 PATH + f"{group_name}_transfer_prctile75.csv")

    xlabel = "Number of unrolls"
    plot_errors(errs, N_UNROLLS, exps, xlabel, "Q err (training)", agg="median", deviation="prctile")
    plot_errors(errs_trans, N_UNROLLS, exps, xlabel, "Q err (transfer)", agg="median", deviation="prctile")
    plot_errors(bell_errs, N_UNROLLS, exps, xlabel, "Bellman err (transfer)", agg="median", deviation="prctile")


if __name__ == "__main__":
    for K in [5, 10]:
        run_block(K)
