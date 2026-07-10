"""Shared experiment runners for UPI benchmarks.

Each function performs one statistical replicate (one 'g') over a grid of
hyper-parameter values and returns numpy arrays of errors.  Call these inside
a loop over n_runs to accumulate statistics.

Exported functions
------------------
run_unroll_sweep   : sweep num_unrolls, single-environment evaluation.
run_transfer_sweep : sweep num_unrolls, evaluate transfer to env_test.
run_k_sweep        : sweep filter order K, fixed num_unrolls.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import wandb

from src.environments import CliffWalkingEnv, MirroredCliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain
from src.algorithms.unrolling_policy_iteration import UnrollingPolicyIterationTrain
from src.utils import test_pol_err


# ── Internal helper ───────────────────────────────────────────────────────────

def _make_trainer(max_epochs: int, log_every_n_steps: int,
                  logger: Any) -> Trainer:
    return Trainer(max_epochs=max_epochs, log_every_n_steps=log_every_n_steps,
                   accelerator="cpu", logger=logger)


def _get_logger(use_logger: bool, name: str, group: str):
    if use_logger:
        return WandbLogger(project="rl-unrolling", name=name, group=group)
    return False


def _finish_wandb(use_logger: bool) -> None:
    if use_logger and wandb.run is not None:
        wandb.finish()


# ── Public runners ────────────────────────────────────────────────────────────

def run_unroll_sweep(
    g: int,
    N_unrolls: np.ndarray,
    exps: list[dict],
    q_opt: torch.Tensor,
    env_factory=CliffWalkingEnv,
    max_epochs: int = 3000,
    log_every_n_steps: int = 1,
    use_logger: bool = False,
    group_name: str = "",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep num_unrolls; evaluate on the training environment."""
    n_exp = len(exps)
    err1 = np.zeros((n_exp, N_unrolls.size))
    err2 = np.zeros((n_exp, N_unrolls.size))
    bell_err = np.zeros((n_exp, N_unrolls.size))
    log = use_logger and g == 0

    for i, n_unrolls in enumerate(N_unrolls):
        n_unrolls = int(n_unrolls)
        t0 = time.perf_counter()
        for j, exp in enumerate(exps):
            env = env_factory()
            if exp["model"] == "unroll":
                model = UnrollingPolicyIterationTrain(
                    env=env, env_test=env, num_unrolls=n_unrolls, **exp["args"])
                logger = _get_logger(log, f"{exp['name']}-U{n_unrolls}", group_name)
                trainer = _make_trainer(max_epochs, log_every_n_steps, logger)
                trainer.fit(model)
                _finish_wandb(log)
                err1[j, i], err2[j, i] = test_pol_err(model.Pi, q_opt)
                bell_err[j, i] = model.bellman_error.cpu().numpy()
            elif exp["model"] == "pol-it":
                model = PolicyIterationTrain(env=env, **exp["args"])
                logger = _get_logger(log, f"{exp['name']}-U{n_unrolls}", group_name)
                trainer = _make_trainer(n_unrolls, log_every_n_steps, logger)
                trainer.fit(model)
                _finish_wandb(log)
                err1[j, i], err2[j, i] = test_pol_err(model.Pi, q_opt)
                bell_err[j, i] = model.bellman_error.cpu().numpy()
            if verbose:
                print(f"  g={g} U={n_unrolls} {exp['name']}: err={err1[j,i]:.4f}")
        if verbose:
            print(f"  U={n_unrolls} done in {time.perf_counter()-t0:.1f}s")

    return err1, err2, bell_err


def run_transfer_sweep(
    g: int,
    N_unrolls: np.ndarray,
    exps: list[dict],
    q_opt: torch.Tensor,
    q_opt_test: torch.Tensor,
    env_factory=CliffWalkingEnv,
    env_test_factory=MirroredCliffWalkingEnv,
    max_epochs: int = 3000,
    log_every_n_steps: int = 1,
    use_logger: bool = False,
    group_name: str = "",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep num_unrolls; evaluate transfer from env to env_test."""
    n_exp = len(exps)
    err = np.zeros((n_exp, N_unrolls.size))
    err_transfer = np.zeros((n_exp, N_unrolls.size))
    bell_err = np.zeros((n_exp, N_unrolls.size))
    log = use_logger and g == 0

    for i, n_unrolls in enumerate(N_unrolls):
        n_unrolls = int(n_unrolls)
        t0 = time.perf_counter()
        for j, exp in enumerate(exps):
            env = env_factory()
            env_test = env_test_factory()
            if exp["model"] == "unroll":
                model = UnrollingPolicyIterationTrain(
                    env=env, env_test=env_test, num_unrolls=n_unrolls, **exp["args"])
                logger = _get_logger(log, f"{exp['name']}-U{n_unrolls}", group_name)
                trainer = _make_trainer(max_epochs, log_every_n_steps, logger)
                trainer.fit(model)
                _finish_wandb(log)
                _, err[j, i] = test_pol_err(model.Pi, q_opt, mirror_env=False,
                                            device=model.device)
                _, err_transfer[j, i] = test_pol_err(model.Pi_test, q_opt_test,
                                                     mirror_env=True, device=model.device)
                bell_err[j, i] = model.bellman_error_test.cpu().numpy()
            elif exp["model"] == "pol-it":
                model = PolicyIterationTrain(env=env_test, goal_row=0, **exp["args"])
                logger = _get_logger(log, f"{exp['name']}-U{n_unrolls}", group_name)
                trainer = _make_trainer(n_unrolls, log_every_n_steps, logger)
                trainer.fit(model)
                _finish_wandb(log)
                _, err[j, i] = test_pol_err(model.Pi, q_opt_test, mirror_env=True,
                                            device=model.device)
                err_transfer[j, i] = err[j, i]
                bell_err[j, i] = model.bellman_error.cpu().numpy()
            if verbose:
                print(f"  g={g} U={n_unrolls} {exp['name']}: err={err[j,i]:.4f} "
                      f"transfer={err_transfer[j,i]:.4f}")
        if verbose:
            print(f"  U={n_unrolls} done in {time.perf_counter()-t0:.1f}s")

    return err, err_transfer, bell_err


def run_k_sweep(
    g: int,
    Ks: np.ndarray,
    exps: list[dict],
    q_opt: torch.Tensor,
    env_factory=CliffWalkingEnv,
    max_epochs: int = 3000,
    log_every_n_steps: int = 1,
    use_logger: bool = False,
    group_name: str = "",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep filter order K at fixed num_unrolls (specified inside each exp)."""
    n_exp = len(exps)
    err1 = np.zeros((n_exp, Ks.size))
    err2 = np.zeros((n_exp, Ks.size))
    bell_err = np.zeros((n_exp, Ks.size))
    log = use_logger and g == 0

    for i, K in enumerate(Ks):
        K = int(K)
        t0 = time.perf_counter()
        for j, exp in enumerate(exps):
            env = env_factory()
            if exp["model"] == "unroll":
                model = UnrollingPolicyIterationTrain(env=env, env_test=env, K=K,
                                                     **exp["args"])
                logger = _get_logger(log, f"{exp['name']}-K{K}", group_name)
                trainer = _make_trainer(max_epochs, log_every_n_steps, logger)
            elif exp["model"] == "pol-it":
                model = PolicyIterationTrain(env=env, max_eval_iters=K)
                logger = _get_logger(log, f"{exp['name']}-K{K}", group_name)
                trainer = _make_trainer(exp["args"].get("max_epochs", max_epochs),
                                        log_every_n_steps, logger)
            else:
                raise ValueError(f"Unknown model type: {exp['model']}")

            trainer.fit(model)
            _finish_wandb(log)
            err1[j, i], err2[j, i] = test_pol_err(model.Pi, q_opt)
            bell_err[j, i] = model.bellman_error.cpu().numpy()
            if verbose:
                print(f"  g={g} K={K} {exp['name']}: err={err1[j,i]:.4f}")
        if verbose:
            print(f"  K={K} done in {time.perf_counter()-t0:.1f}s")

    return err1, err2, bell_err
