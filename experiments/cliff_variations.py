"""Cliff Variations Experiment — rapid iteration on GeneralizedCliffWalkingEnv.

Compares transfer performance across 5 cliff configurations.
Identical metrics to unrolls_experiments_analysis.py.
Results are saved per-variant in cliff_variations_results/{variant}/
with the same CSV layout so visualize_unrolls_results_tfg.py can be run on them.

Variants (all 4×12, nS=48):
  std_mirrored  — bottom cliff → top cliff            [baseline, same as main project]
  mirrored_std  — top cliff → bottom cliff            [reverse: tests asymmetry]
  std_vertical  — bottom cliff → vertical barrier     [topology change: col-6 strip]
  std_narrow    — bottom cliff → narrow top cliff     [cols 3-9, easier hazard]
  std_tall      — bottom cliff → 2-row top cliff      [more hazard, start row 2]

Quick defaults vs main experiments:
  n_runs=3   (vs 15)   K=[5,10]    (vs [3,5,10,12])
  epochs=2000 (vs 3000)  unrolls=[5,10] (vs 7 values)

Usage:
  python cliff_variations_experiment.py                          # all variants
  python cliff_variations_experiment.py --variants std_mirrored  # single variant
  python cliff_variations_experiment.py --variants std_mirrored mirrored_std --runs 5
  python cliff_variations_experiment.py --force                  # re-run everything
"""

import os
import sys
import time
import shutil
import argparse

# Non-interactive backend — must come before any other matplotlib import
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from pytorch_lightning import Trainer

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.algorithms.unrolling_policy_iteration import UnrollingPolicyIterationTrain
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain
from src.environments import GeneralizedCliffWalkingEnv, WindyGridWorldEnv
from src.utils import (compute_optimality_gap, compute_optimality_gap_V,
                       get_optimal_q_for_env)


# ─────────────────────────────────────────────────────────────────────────────
# Quick-run configuration
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_BASE = Path("cliff_variations_results")
RESULTS_BASE.mkdir(exist_ok=True)

QUICK_CONFIG = {
    "n_runs"       : 15,
    "max_epochs"   : 3000,
    "K_values"     : [3, 5, 10, 12],
    "num_unrolls"  : [2, 4, 5, 6, 8, 10, 15],
    "architectures": [1, 2],
    "tau"          : 5.0,
    "lr"           : 5e-3,
    "gamma"        : 0.99,
    "loss_type"    : "original_no_detach",
    "init_q"       : "random",
}


# ─────────────────────────────────────────────────────────────────────────────
# Variant definitions
# ─────────────────────────────────────────────────────────────────────────────

def _narrow_mirrored():
    """Top row cliff but narrower: only cols 3-9 (leaves outer 3 cols safe)."""
    return GeneralizedCliffWalkingEnv(
        nrows=4, ncols=12,
        cliff_cells=[(0, c) for c in range(3, 10)],
        start=(0, 0), goal=(0, 11),
    )


def _tall_mirrored():
    """2-row thick cliff on rows 0-1, cols 1-10. Start/goal pushed to row 2."""
    return GeneralizedCliffWalkingEnv(
        nrows=4, ncols=12,
        cliff_cells=[(0, c) for c in range(1, 11)] + [(1, c) for c in range(1, 11)],
        start=(2, 0), goal=(2, 11),
    )


def _large_mirrored():
    """Scale-2 mirrored grid: 8×24, cliff at top row, nS=192."""
    return GeneralizedCliffWalkingEnv.scaled(scale=2, mirrored=True)


def _center_start_std():
    """Standard 4×12 with start/teleport at grid center (1, 6)."""
    return GeneralizedCliffWalkingEnv.center_start()


def _std_center_goal():
    """Standard 4×12 cliff but with goal at grid center (2, 6)."""
    return GeneralizedCliffWalkingEnv.center_goal()


def _windy_no_wind():
    """Plain 7×10 grid world (no wind). Same start/goal as classic windy."""
    return WindyGridWorldEnv.no_wind(nrows=7, ncols=10, start=(3, 0), goal=(3, 7))


def _windy_classic():
    """Classic 7×10 Windy GridWorld (S&B deterministic wind)."""
    return WindyGridWorldEnv.classic()


def _standard_windy():
    """Standard 4×12 cliff + per-cell random stochastic wind (p_wind=0.25, seed=42)."""
    return GeneralizedCliffWalkingEnv.standard_windy(rng_seed=42, p_wind=0.25)


VARIANTS = {
    "std_mirrored": {
        "train_fn"      : GeneralizedCliffWalkingEnv.standard,
        "test_fn"       : GeneralizedCliffWalkingEnv.mirrored,
        "goal_row_train": 3,
        "goal_row_test" : 0,
        "description"   : "Bottom cliff → Top cliff  [BASELINE — same as main project]",
    },
    "mirrored_std": {
        "train_fn"      : GeneralizedCliffWalkingEnv.mirrored,
        "test_fn"       : GeneralizedCliffWalkingEnv.standard,
        "goal_row_train": 0,
        "goal_row_test" : 3,
        "description"   : "Top cliff → Bottom cliff  [reverse direction — tests asymmetry]",
    },
    "std_vertical": {
        "train_fn"      : GeneralizedCliffWalkingEnv.standard,
        "test_fn"       : GeneralizedCliffWalkingEnv.vertical_cliff,
        "goal_row_train": 3,
        "goal_row_test" : 0,   # vertical_cliff: goal=(0,11)
        "description"   : "Bottom cliff → Vertical barrier col 6  [topology change]",
    },
    "std_narrow": {
        "train_fn"      : GeneralizedCliffWalkingEnv.standard,
        "test_fn"       : _narrow_mirrored,
        "goal_row_train": 3,
        "goal_row_test" : 0,
        "description"   : "Bottom cliff → Narrow top cliff cols 3-9  [fewer hazard cells]",
    },
    "std_tall": {
        "train_fn"      : GeneralizedCliffWalkingEnv.standard,
        "test_fn"       : _tall_mirrored,
        "goal_row_train": 3,
        "goal_row_test" : 2,   # tall: goal=(2,11)
        "description"   : "Bottom cliff → 2-row thick top cliff  [more hazard]",
    },
    "large_mirrored": {
        "train_fn"               : GeneralizedCliffWalkingEnv.standard,
        "test_fn"                : _large_mirrored,
        "test_during_training_fn": GeneralizedCliffWalkingEnv.standard,  # nS-compatible proxy
        "goal_row_train"         : 3,
        "goal_row_test"          : 0,
        "description"            : "Standard 4×12 → Large 8×24 mirrored  [scale transfer: 4× nS]",
    },
    "large_std": {
        "train_fn"               : GeneralizedCliffWalkingEnv.standard,
        "test_fn"                : lambda: GeneralizedCliffWalkingEnv.scaled(scale=2, mirrored=False),
        "test_during_training_fn": GeneralizedCliffWalkingEnv.standard,  # nS-compatible proxy
        "goal_row_train"         : 3,
        "goal_row_test"          : 7,  # bottom row of 8×24
        "description"            : "Standard 4×12 → Large 8×24 standard  [same topology, scale transfer]",
        "extra_unrolls_multipliers": [2, 4],
    },
    "large_std_K5U4": {
        "train_fn"               : GeneralizedCliffWalkingEnv.standard,
        "test_fn"                : lambda: GeneralizedCliffWalkingEnv.scaled(scale=2, mirrored=False),
        "test_during_training_fn": GeneralizedCliffWalkingEnv.standard,
        "goal_row_train"         : 3,
        "goal_row_test"          : 7,
        "description"            : "Standard 4×12 → Large 8×24 standard, K=5 U=4 focused  [scale transfer]",
        "extra_unrolls_multipliers": [2, 4],
        "K_values"               : [5],
        "num_unrolls"            : [4],
    },
    "center_start_std": {
        "train_fn"      : _center_start_std,
        "test_fn"       : _center_start_std,
        "goal_row_train": 3,
        "goal_row_test" : 3,
        "description"   : "Center-start 4×12 → same env  [same nS, non-corner start/teleport]",
        "extra_unrolls_multipliers": [2, 4],
        "K_values"      : [5, 10],
        "num_unrolls"   : [4, 10],
    },
    "windy_transfer": {
        "train_fn"      : _windy_no_wind,
        "test_fn"       : _windy_classic,
        "goal_row_train": 3,
        "goal_row_test" : 3,
        "description"   : "No-wind 7×10 → Windy-classic 7×10  [dynamics transfer, same nS]",
    },
    "std_center_goal": {
        "train_fn"      : GeneralizedCliffWalkingEnv.standard,
        "test_fn"       : _std_center_goal,
        "goal_row_train": 3,
        "goal_row_test" : 2,
        "description"   : "Standard 4×12 → same grid, goal at center (2,6)  [topology transfer]",
    },
    "large_std_K5U5_U150": {
        "train_fn"               : GeneralizedCliffWalkingEnv.standard,
        "test_fn"                : lambda: GeneralizedCliffWalkingEnv.scaled(scale=2, mirrored=False),
        "test_during_training_fn": GeneralizedCliffWalkingEnv.standard,
        "goal_row_train"         : 3,
        "goal_row_test"          : 7,
        "description"            : "Standard 4×12 → Large 8×24, K=5 U=5, test at U=150  [extreme extra-unrolls scale transfer]",
        "extra_unrolls_multipliers": [4, 6, 8, 10, 15, 20, 30, 40, 50, 75, 100],
        "K_values"               : [5],
        "num_unrolls"            : [5],
    },
    "large_std_K12U5": {
        "train_fn"               : GeneralizedCliffWalkingEnv.standard,
        "test_fn"                : lambda: GeneralizedCliffWalkingEnv.scaled(scale=2, mirrored=False),
        "test_during_training_fn": GeneralizedCliffWalkingEnv.standard,
        "goal_row_train"         : 3,
        "goal_row_test"          : 7,
        "description"            : "Standard 4×12 → Large 8×24, K=12 U=5, stability sweep  [extra-unrolls scale transfer]",
        "extra_unrolls_multipliers": [4, 6, 8, 10, 15, 20, 30, 40, 50, 75, 100],
        "K_values"               : [12],
        "num_unrolls"            : [5],
    },
    "large_std_K10U10": {
        "train_fn"               : GeneralizedCliffWalkingEnv.standard,
        "test_fn"                : lambda: GeneralizedCliffWalkingEnv.scaled(scale=2, mirrored=False),
        "test_during_training_fn": GeneralizedCliffWalkingEnv.standard,
        "goal_row_train"         : 3,
        "goal_row_test"          : 7,
        "description"            : "Standard 4×12 → Large 8×24, K=10 U=10, stability sweep  [extra-unrolls scale transfer]",
        "extra_unrolls_multipliers": [4, 6, 8, 10, 15, 20, 30, 40, 50],
        "K_values"               : [10],
        "num_unrolls"            : [10],
    },
    "std_windy_random": {
        "train_fn"                : GeneralizedCliffWalkingEnv.standard,
        "test_fn"                 : _standard_windy,
        "goal_row_train"          : 3,
        "goal_row_test"           : 3,
        "extra_unrolls_multipliers": [2, 4],
        "description"             : "Standard 4×12 cliff → same grid + random per-cell wind (p_wind≈0.25)  [dynamics transfer]",
    },
}

ALL_VARIANT_NAMES = list(VARIANTS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def eval_policy_extended(
    Pi: torch.Tensor, q_model: torch.Tensor, q_opt: torch.Tensor, env,
    gamma: float = 0.99, max_eval_iters: int = 200, device: str = "cpu",
):
    """Policy evaluation returning POG (squared), PVG, and Agreement.

    Works with any env that has .nS, .nA, .P, .r, .cliff_set, .goal_state.

    Returns:
        pog_joint, pog_sep   — Policy Optimality Gap (squared)
        pvg_joint, pvg_sep   — Policy Value Gap V_π vs V* (not squared)
        hard_agreement       — hard agreement % (greedy in A*)
        soft_agreement       — soft agreement % (prob mass on A*)
    """
    nS, nA = env.nS, env.nA
    P      = env.P.to(device)
    r      = env.r.to(device)
    q_opt_d = q_opt.to(device)
    Pi_d   = Pi.to(device)

    # Deterministic greedy policy
    max_vals = Pi_d.max(dim=1, keepdim=True).values
    is_max   = (Pi_d == max_vals)
    greedy   = torch.multinomial(is_max.float(), num_samples=1).squeeze(1)
    Pi_det   = torch.zeros_like(Pi_d)
    Pi_det[torch.arange(nS, device=device), greedy] = 1.0

    Pi_ext       = torch.zeros(nS, nS * nA, device=device)
    rows         = torch.arange(nS, device=device).repeat_interleave(nA)
    cols         = torch.arange(nS * nA, device=device)
    Pi_ext[rows, cols] = Pi_det.flatten()

    P_pi = P @ Pi_ext
    q_pi = torch.zeros(nS * nA, device=device)
    for _ in range(max_eval_iters):
        q_pi = r + gamma * (P_pi @ q_pi)

    # POG (squared)
    pog_joint = float((torch.norm(q_pi - q_opt_d) / torch.norm(q_opt_d)) ** 2)
    pog_sep   = float(torch.norm(q_pi / torch.norm(q_pi) - q_opt_d / torch.norm(q_opt_d)) ** 2)

    # PVG: V_π_greedy vs V* (not squared)
    V_pi  = q_pi.view(nS, nA).max(dim=1).values
    V_opt = q_opt_d.view(nS, nA).max(dim=1).values
    pvg_joint = float(torch.norm(V_pi - V_opt) / torch.norm(V_opt))
    pvg_sep   = float(torch.norm(V_pi / torch.norm(V_pi) - V_opt / torch.norm(V_opt)))

    # Agreement (uses model q and Pi; terminal states from env)
    Pi_np      = Pi.cpu().numpy()
    q_mod_np   = q_model.cpu().numpy()
    q_opt_np   = q_opt.cpu().numpy()
    terminal   = env.cliff_set | {env.goal_state}
    q_mat      = q_mod_np.reshape(nS, nA)
    q_opt_mat  = q_opt_np.reshape(nS, nA)
    is_opt     = (q_opt_mat >= q_opt_mat.max(axis=1, keepdims=True) - 1e-6)
    g_np       = q_mat.argmax(axis=1)
    valid      = np.array([s not in terminal for s in range(nS)])
    hard       = float(np.array([is_opt[s, g_np[s]] for s in range(nS)],
                                dtype=float)[valid].mean() * 100.0)
    soft       = float((Pi_np * is_opt).sum(axis=1)[valid].mean() * 100.0)

    return pog_joint, pog_sep, pvg_joint, pvg_sep, hard, soft


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_single_experiment(
    env, env_test,
    q_opt: torch.Tensor, q_opt_test: torch.Tensor,
    K: int, num_unrolls: int, architecture_type: int, run_idx: int,
    cfg: dict, results_dir: Path, label: str = "",
    env_test_posthoc=None, q_opt_test_posthoc=None,
    extra_unrolls_multipliers=None,
) -> dict:
    """Train one model and return all metrics as a result dict."""
    print(f"\n{'─'*70}")
    print(f"{label}  Arch={architecture_type}  K={K}  U={num_unrolls}  "
          f"run={run_idx + 1}/{cfg['n_runs']}")

    try:
        model = UnrollingPolicyIterationTrain(
            env=env, env_test=env_test,
            K=K, num_unrolls=num_unrolls,
            gamma=cfg["gamma"], tau=cfg["tau"], lr=cfg["lr"],
            N=1, init_q=cfg["init_q"], loss_type=cfg["loss_type"],
            architecture_type=architecture_type,
            normalize_separately=False,
            weight_sharing=True,
        )
        trainer = Trainer(
            max_epochs=cfg["max_epochs"],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )
        t0 = time.time()
        trainer.fit(model)
        elapsed = time.time() - t0

        device = str(model.device)
        q_tr   = model.q.detach()
        boe_tr  = float(model.bellman_error.cpu())
        boe_utr = float(model.bellman_error_unnormalized.cpu())

        if env_test_posthoc is not None:
            # Post-hoc: copy learned h weights to a model sized for the real test env
            from src import UnrolledPolicyIterationModel
            from src.algorithms.unrolling_policy_iteration import UnrollingDataset as _DS
            model_ph = UnrolledPolicyIterationModel(
                env_test_posthoc.P, env_test_posthoc.r,
                env_test_posthoc.nS, env_test_posthoc.nA,
                K, num_unrolls, cfg["tau"], 1.0, True, architecture_type,
                False, True, None,
            )
            with torch.no_grad():
                for p_src, p_dst in zip(model.model.parameters(), model_ph.parameters()):
                    p_dst.copy_(p_src)
                ds_ph = _DS(env_test_posthoc.nS, env_test_posthoc.nA, N=1,
                            init_q=cfg["init_q"])
                q_s, Pi_s = ds_ph[0]
                q_te, Pi_te = model_ph(q_s, Pi_s)
            q_te  = q_te.detach()
            Pi_te = Pi_te.detach()
            # Compute BOE for post-hoc test env manually
            nS_te, nA_te = env_test_posthoc.nS, env_test_posthoc.nA
            greedy_te = Pi_te.argmax(dim=1)
            Pi_det_te = torch.zeros_like(Pi_te)
            Pi_det_te[torch.arange(nS_te), greedy_te] = 1.0
            Pi_ext_te = torch.zeros(nS_te, nS_te * nA_te)
            rows_te   = torch.arange(nS_te).repeat_interleave(nA_te)
            cols_te   = torch.arange(nS_te * nA_te)
            Pi_ext_te[rows_te, cols_te] = Pi_det_te.flatten()
            P_pi_te   = env_test_posthoc.P @ Pi_ext_te
            target_te = env_test_posthoc.r + cfg["gamma"] * (P_pi_te @ q_te)
            diff_te   = q_te - target_te
            boe_ute   = float(torch.norm(diff_te))
            boe_te    = float(torch.norm(diff_te) / torch.norm(target_te))
            env_for_test = env_test_posthoc
            q_opt_te  = q_opt_test_posthoc
        else:
            q_te   = model.q_test.detach()
            Pi_te  = model.Pi_test.detach()
            boe_te  = float(model.bellman_error_test.cpu())
            boe_ute = float(model.bellman_error_unnormalized_test.cpu())
            env_for_test = env_test
            q_opt_te  = q_opt_test

        # ── Optimality gap (q)
        og_j_tr,  og_s_tr  = compute_optimality_gap(q_tr, q_opt,     device)
        og_j_te,  og_s_te  = compute_optimality_gap(q_te, q_opt_te,  device)

        # ── Optimality gap V
        ogV_j_tr, ogV_s_tr = compute_optimality_gap_V(q_tr, q_opt,     device)
        ogV_j_te, ogV_s_te = compute_optimality_gap_V(q_te, q_opt_te,  device)

        # ── Policy optimality gap + PVG + Agreement (uses env.P/r directly)
        pog_j_tr, pog_s_tr, pvg_j_tr, pvg_s_tr, hard_tr, soft_tr = eval_policy_extended(
            model.Pi, q_tr, q_opt, env, cfg["gamma"], device=device
        )
        pog_j_te, pog_s_te, pvg_j_te, pvg_s_te, hard_te, soft_te = eval_policy_extended(
            Pi_te, q_te, q_opt_te, env_for_test, cfg["gamma"], device=device
        )

        # ── Extra unrolls at test time (post-hoc, same h applied more times) ──
        # Use posthoc env (scale transfer) if available, else fall back to test env (same nS).
        env_for_extra  = env_test_posthoc if env_test_posthoc is not None else env_for_test
        q_opt_extra    = q_opt_test_posthoc if env_test_posthoc is not None else q_opt_te

        extra_metrics = {}
        if extra_unrolls_multipliers:
            from src import UnrolledPolicyIterationModel
            from src.algorithms.unrolling_policy_iteration import UnrollingDataset as _DS2
            for mult in extra_unrolls_multipliers:
                num_unrolls_ex = num_unrolls * mult
                model_ex = UnrolledPolicyIterationModel(
                    env_for_extra.P, env_for_extra.r,
                    env_for_extra.nS, env_for_extra.nA,
                    K, num_unrolls_ex, cfg["tau"], 1.0, True, architecture_type,
                    False, True, None,
                )
                with torch.no_grad():
                    for p_src, p_dst in zip(model.model.parameters(), model_ex.parameters()):
                        p_dst.copy_(p_src)
                    ds_ex = _DS2(env_for_extra.nS, env_for_extra.nA, N=1,
                                 init_q=cfg["init_q"])
                    q_s, Pi_s = ds_ex[0]
                    q_ex, Pi_ex = model_ex(q_s, Pi_s)
                q_ex  = q_ex.detach()
                Pi_ex = Pi_ex.detach()
                og_j_ex,  og_s_ex  = compute_optimality_gap(q_ex, q_opt_extra,  device)
                ogV_j_ex, ogV_s_ex = compute_optimality_gap_V(q_ex, q_opt_extra, device)
                pog_j_ex, pog_s_ex, pvg_j_ex, pvg_s_ex, hard_ex, soft_ex = \
                    eval_policy_extended(Pi_ex, q_ex, q_opt_extra, env_for_extra,
                                        cfg["gamma"], device=device)
                sfx = f"_{mult}x"
                extra_metrics.update({
                    f"optimality_gap_joint_test{sfx}":        og_j_ex,
                    f"optimality_gap_separate_test{sfx}":     og_s_ex,
                    f"optimality_gap_V_joint_test{sfx}":      ogV_j_ex,
                    f"optimality_gap_V_separate_test{sfx}":   ogV_s_ex,
                    f"policy_optimality_gap_joint_test{sfx}":    pog_j_ex,
                    f"policy_optimality_gap_separate_test{sfx}": pog_s_ex,
                    f"policy_value_gap_joint_test{sfx}":      pvg_j_ex,
                    f"policy_value_gap_separate_test{sfx}":   pvg_s_ex,
                    f"agreement_hard_test{sfx}": hard_ex,
                    f"agreement_soft_test{sfx}": soft_ex,
                })

        # ── Save policy data (compatible with visualize_unrolls_results_tfg.py)
        policy_fname = (
            f"policy_arch{architecture_type}_K{K}_"
            f"unrolls{num_unrolls}_run{run_idx}_temp.npz"
        )
        np.savez(
            results_dir / policy_fname,
            Pi_train=model.Pi.detach().cpu().numpy(),
            q_train=q_tr.cpu().numpy(),
            Pi_test=Pi_te.cpu().numpy(),
            q_test=q_te.cpu().numpy(),
        )

        result = {
            "architecture_type": architecture_type,
            "K": K, "num_unrolls": num_unrolls,
            "loss_type": cfg["loss_type"], "init_q": cfg["init_q"],
            "run_idx": run_idx, "training_time_sec": elapsed,
            # Bellman optimality error
            "bellman_optimality_error_train"             : boe_tr,
            "bellman_optimality_error_unnormalized_train": boe_utr,
            "bellman_optimality_error_test"              : boe_te,
            "bellman_optimality_error_unnormalized_test" : boe_ute,
            # Optimality gap (q)
            "optimality_gap_joint_train"    : og_j_tr,
            "optimality_gap_separate_train" : og_s_tr,
            "optimality_gap_joint_test"     : og_j_te,
            "optimality_gap_separate_test"  : og_s_te,
            # Optimality gap V
            "optimality_gap_V_joint_train"    : ogV_j_tr,
            "optimality_gap_V_separate_train" : ogV_s_tr,
            "optimality_gap_V_joint_test"     : ogV_j_te,
            "optimality_gap_V_separate_test"  : ogV_s_te,
            # Policy optimality gap (squared)
            "policy_optimality_gap_joint_train"    : pog_j_tr,
            "policy_optimality_gap_separate_train" : pog_s_tr,
            "policy_optimality_gap_joint_test"     : pog_j_te,
            "policy_optimality_gap_separate_test"  : pog_s_te,
            # Policy value gap (not squared)
            "policy_value_gap_joint_train"    : pvg_j_tr,
            "policy_value_gap_separate_train" : pvg_s_tr,
            "policy_value_gap_joint_test"     : pvg_j_te,
            "policy_value_gap_separate_test"  : pvg_s_te,
            # Agreement (%)
            "agreement_hard_train": hard_tr,
            "agreement_soft_train": soft_tr,
            "agreement_hard_test" : hard_te,
            "agreement_soft_test" : soft_te,
            # Misc
            "q_norm_squared_train": float(torch.norm(q_tr) ** 2),
            "q_norm_squared_test" : float(torch.norm(q_te) ** 2),
            "policy_file_temp"    : policy_fname,
            "success"             : True,
            **extra_metrics,
        }

        print(f"  ✓ {elapsed:.0f}s | "
              f"BOE: tr={boe_tr:.4f} te={boe_te:.4f} | "
              f"OG(q): tr={og_j_tr:.4f} te={og_j_te:.4f} | "
              f"POG: tr={pog_j_tr:.4f} te={pog_j_te:.4f}")
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "architecture_type": architecture_type,
            "K": K, "num_unrolls": num_unrolls,
            "loss_type": cfg["loss_type"], "run_idx": run_idx,
            "success": False, "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Representative run selection (mirrors unrolls_experiments_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────

def select_representative_runs(df: pd.DataFrame, cfg: dict, results_dir: Path,
                                K_values=None, num_unrolls_vals=None) -> pd.DataFrame:
    """Copy _temp.npz files to final names for the run closest to the median.

    Modifies df in place, adding 'policy_file' and 'policy_file_train' columns.
    Deletes all _temp.npz files afterwards.
    """
    K_values       = K_values       if K_values       is not None else cfg["K_values"]
    num_unrolls_vals = num_unrolls_vals if num_unrolls_vals is not None else cfg["num_unrolls"]

    df["policy_file"]       = None
    df["policy_file_train"] = None

    df_ok = df[df["success"] == True]

    for K in K_values:
        for U in num_unrolls_vals:
            for arch in cfg["architectures"]:
                sub = df_ok[
                    (df_ok["K"] == K) &
                    (df_ok["num_unrolls"] == U) &
                    (df_ok["architecture_type"] == arch)
                ]
                if len(sub) == 0:
                    continue

                # ── Representative for test metric
                med_te  = sub["policy_optimality_gap_joint_test"].median()
                run_te  = int(sub.loc[
                    (sub["policy_optimality_gap_joint_test"] - med_te).abs().idxmin(),
                    "run_idx"
                ])

                # ── Representative for train metric
                med_tr  = sub["policy_optimality_gap_joint_train"].median()
                run_tr  = int(sub.loc[
                    (sub["policy_optimality_gap_joint_train"] - med_tr).abs().idxmin(),
                    "run_idx"
                ])

                # ── Copy test representative
                src_te = results_dir / f"policy_arch{arch}_K{K}_unrolls{U}_run{run_te}_temp.npz"
                dst_te = results_dir / f"policy_arch{arch}_K{K}_unrolls{U}_run{run_te}.npz"
                if src_te.exists():
                    shutil.copy(src_te, dst_te)
                    mask_te = (
                        (df["K"] == K) & (df["num_unrolls"] == U) &
                        (df["architecture_type"] == arch) & (df["run_idx"] == run_te)
                    )
                    df.loc[mask_te, "policy_file"] = dst_te.name

                # ── Copy train representative (possibly same run)
                if run_tr != run_te:
                    src_tr = results_dir / f"policy_arch{arch}_K{K}_unrolls{U}_run{run_tr}_temp.npz"
                    dst_tr = results_dir / f"policy_arch{arch}_K{K}_unrolls{U}_run{run_tr}_train.npz"
                    if src_tr.exists():
                        shutil.copy(src_tr, dst_tr)
                        mask_tr = (
                            (df["K"] == K) & (df["num_unrolls"] == U) &
                            (df["architecture_type"] == arch) & (df["run_idx"] == run_tr)
                        )
                        df.loc[mask_tr, "policy_file_train"] = dst_tr.name
                else:
                    df.loc[mask_te, "policy_file_train"] = dst_te.name

    # Delete all temp files
    for f in results_dir.glob("*_temp.npz"):
        f.unlink()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Comparison visualisation (bar chart across variants)
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {1: "#0173B2", 2: "#DE8F05"}

METRICS_TO_PLOT = [
    ("policy_optimality_gap_joint_test",  "POG joint (test)"),
    ("policy_optimality_gap_joint_train", "POG joint (train)"),
    ("optimality_gap_V_joint_test",       "OG-V joint (test)"),
    ("bellman_optimality_error_test",     "Bellman err (test)"),
]


def plot_variant_comparison(summary_path: Path, cfg: dict):
    """Bar-chart comparison across all variants, aggregating over K/unrolls/runs."""
    df_all = pd.read_csv(summary_path)
    df_ok  = df_all[df_all["success"] == True]
    if len(df_ok) == 0:
        return

    variants = [v for v in ALL_VARIANT_NAMES if v in df_ok["variant_name"].unique()]
    archs    = cfg["architectures"]
    n_v      = len(variants)
    n_m      = len(METRICS_TO_PLOT)

    fig, axes = plt.subplots(1, n_m, figsize=(4.5 * n_m, 4))
    if n_m == 1:
        axes = [axes]

    x     = np.arange(n_v)
    w     = 0.35
    agg   = df_ok.groupby(["variant_name", "architecture_type"])[
        [m for m, _ in METRICS_TO_PLOT]
    ].mean().reset_index()

    for ax, (metric, title) in zip(axes, METRICS_TO_PLOT):
        for i, arch in enumerate(archs):
            vals = []
            for var in variants:
                sub = agg[(agg["variant_name"] == var) & (agg["architecture_type"] == arch)]
                vals.append(float(sub[metric].iloc[0]) if len(sub) > 0 else np.nan)
            offset = (i - 0.5) * w
            ax.bar(x + offset, vals, w, label=f"Arch {arch}",
                   color=COLORS[arch], alpha=0.85, edgecolor="white")

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=7)
        ax.set_ylabel("Error", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("Cliff Variations — Transfer Performance (mean over K, unrolls, runs)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = summary_path.parent / "comparison_plots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ comparison_plots.png")


def plot_kxunrolls_grid(summary_path: Path, cfg: dict):
    """2×2 grid: rows=K, cols=unrolls. Each cell: bar per variant×arch for POG_test."""
    df_all = pd.read_csv(summary_path)
    df_ok  = df_all[df_all["success"] == True]
    if len(df_ok) == 0:
        return

    variants = [v for v in ALL_VARIANT_NAMES if v in df_ok["variant_name"].unique()]
    K_vals   = sorted(df_ok["K"].unique())
    U_vals   = sorted(df_ok["num_unrolls"].unique())
    archs    = cfg["architectures"]
    metric   = "policy_optimality_gap_joint_test"

    n_K = len(K_vals)
    n_U = len(U_vals)
    fig, axes = plt.subplots(n_K, n_U, figsize=(5 * n_U, 4 * n_K), squeeze=False)

    x   = np.arange(len(variants))
    w   = 0.35

    for ri, K in enumerate(K_vals):
        for ci, U in enumerate(U_vals):
            ax  = axes[ri][ci]
            sub_all = df_ok[(df_ok["K"] == K) & (df_ok["num_unrolls"] == U)]

            for i, arch in enumerate(archs):
                vals = []
                for var in variants:
                    sub = sub_all[
                        (sub_all["variant_name"] == var) &
                        (sub_all["architecture_type"] == arch)
                    ]
                    vals.append(sub[metric].mean() if len(sub) > 0 else np.nan)
                offset = (i - 0.5) * w
                ax.bar(x + offset, vals, w, label=f"Arch {arch}",
                       color=COLORS[arch], alpha=0.85, edgecolor="white")

            ax.set_title(f"K={K}  U={U}", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=6)
            ax.set_ylabel("POG joint (test)", fontsize=7)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(bottom=0)

    fig.suptitle("POG joint (test) — mean over runs", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = summary_path.parent / "comparison_KxU_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ comparison_KxU_grid.png")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(df_all: pd.DataFrame, cfg: dict):
    """Print a compact comparison table to stdout."""
    df_ok = df_all[df_all["success"] == True]
    if len(df_ok) == 0:
        return

    variants = [v for v in ALL_VARIANT_NAMES if v in df_ok["variant_name"].unique()]

    hdr = (f"{'Variant':<20} {'Arch':>4}  "
           f"{'POG_tr':>8} {'POG_te':>8} {'OG-V_te':>8} {'BOE_te':>8}")
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (mean across K, unrolls, runs)")
    print("=" * 70)
    print(hdr)
    print("-" * 70)

    for var in variants:
        for arch in cfg["architectures"]:
            sub = df_ok[
                (df_ok["variant_name"] == var) &
                (df_ok["architecture_type"] == arch)
            ]
            if len(sub) == 0:
                continue
            pog_tr = sub["policy_optimality_gap_joint_train"].mean()
            pog_te = sub["policy_optimality_gap_joint_test"].mean()
            ogv_te = sub["optimality_gap_V_joint_test"].mean()
            boe_te = sub["bellman_optimality_error_test"].mean()
            print(f"{var:<20} {arch:>4}  "
                  f"{pog_tr:>8.4f} {pog_te:>8.4f} {ogv_te:>8.4f} {boe_te:>8.4f}")
        print()

    # Detailed breakdown by K and unrolls
    print("\n" + "=" * 70)
    print("BREAKDOWN BY K AND UNROLLS (POG joint test, mean over runs)")
    print("=" * 70)
    hdr2 = f"{'Variant':<20} {'Arch':>4} {'K':>4} {'U':>4}  {'POG_tr':>8} {'POG_te':>8}"
    print(hdr2)
    print("-" * 70)
    for var in variants:
        for arch in cfg["architectures"]:
            for K in sorted(df_ok["K"].unique()):
                for U in sorted(df_ok["num_unrolls"].unique()):
                    sub = df_ok[
                        (df_ok["variant_name"] == var) &
                        (df_ok["architecture_type"] == arch) &
                        (df_ok["K"] == K) &
                        (df_ok["num_unrolls"] == U)
                    ]
                    if len(sub) == 0:
                        continue
                    pog_tr = sub["policy_optimality_gap_joint_train"].mean()
                    pog_te = sub["policy_optimality_gap_joint_test"].mean()
                    print(f"{var:<20} {arch:>4} {K:>4} {U:>4}  "
                          f"{pog_tr:>8.4f} {pog_te:>8.4f}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cliff variations experiment — fast iteration script"
    )
    parser.add_argument(
        "--variants", nargs="+", default=None, choices=ALL_VARIANT_NAMES,
        help=f"Variants to run (default: all). Options: {ALL_VARIANT_NAMES}",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run and overwrite existing results",
    )
    parser.add_argument(
        "--epochs", type=int, default=QUICK_CONFIG["max_epochs"],
        help=f"Max training epochs per model (default {QUICK_CONFIG['max_epochs']})",
    )
    parser.add_argument(
        "--runs", type=int, default=QUICK_CONFIG["n_runs"],
        help=f"Number of runs per config (default {QUICK_CONFIG['n_runs']})",
    )
    args = parser.parse_args()

    cfg = {**QUICK_CONFIG, "max_epochs": args.epochs, "n_runs": args.runs}
    variants_to_run = args.variants or ALL_VARIANT_NAMES

    n_models = (
        len(cfg["K_values"]) * len(cfg["num_unrolls"]) *
        len(cfg["architectures"]) * cfg["n_runs"]
    )

    print("=" * 70)
    print("CLIFF VARIATIONS EXPERIMENT")
    print("=" * 70)
    print(f"Variants   : {variants_to_run}")
    print(f"K values   : {cfg['K_values']}")
    print(f"Num unrolls: {cfg['num_unrolls']}")
    print(f"Runs/config: {cfg['n_runs']}")
    print(f"Max epochs : {cfg['max_epochs']}")
    print(f"Models/var : {n_models}  "
          f"(≈ {n_models * 0.6:.0f}–{n_models * 1.2:.0f} min/variant)")
    print()

    all_dfs = []

    for var_name in variants_to_run:
        var         = VARIANTS[var_name]
        results_dir = RESULTS_BASE / var_name
        results_dir.mkdir(exist_ok=True)
        csv_path    = results_dir / "all_experiments_results.csv"

        print(f"\n{'='*70}")
        print(f"VARIANT: {var_name}")
        print(f"  {var['description']}")
        print(f"{'='*70}")

        if csv_path.exists() and not args.force:
            print(f"  ⚡ Already exists — skipping  (--force to re-run)")
            df_var = pd.read_csv(csv_path)
            df_var["variant_name"] = var_name
            all_dfs.append(df_var)
            continue

        # ── Build environments
        env           = var["train_fn"]()
        env_test_real = var["test_fn"]()
        test_during_fn = var.get("test_during_training_fn")
        env_test_for_train = test_during_fn() if test_during_fn else env_test_real
        print(f"  Train: {env}")
        print(f"  Test : {env_test_real}")
        if test_during_fn:
            print(f"  Test (during training): {env_test_for_train}  [post-hoc eval on real test]")

        # ── Compute optimal Q for both environments
        print("\n  Computing optimal Q (train) ...")
        q_opt = get_optimal_q_for_env(env, var["goal_row_train"], cfg["gamma"])
        print(f"    ||q*_train|| = {torch.norm(q_opt):.3f}")

        print("  Computing optimal Q (test) ...")
        q_opt_test = get_optimal_q_for_env(env_test_real, var["goal_row_test"], cfg["gamma"])
        print(f"    ||q*_test||  = {torch.norm(q_opt_test):.3f}")

        # ── Per-variant K/U overrides (or fall back to global config)
        K_values_var    = var.get("K_values",    cfg["K_values"])
        num_unrolls_var = var.get("num_unrolls", cfg["num_unrolls"])

        # ── Inner loop — same order as unrolls_experiments_analysis.py
        results    = []
        t_var_start = time.time()

        for K in K_values_var:
            for U in num_unrolls_var:
                for arch in cfg["architectures"]:
                    for run_idx in range(cfg["n_runs"]):
                        res = run_single_experiment(
                            env, env_test_for_train,
                            q_opt, q_opt_test,
                            K, U, arch, run_idx,
                            cfg, results_dir,
                            label=f"[{var_name}]",
                            env_test_posthoc=env_test_real if test_during_fn else None,
                            q_opt_test_posthoc=q_opt_test if test_during_fn else None,
                            extra_unrolls_multipliers=var.get("extra_unrolls_multipliers"),
                        )
                        results.append(res)

        # ── Build and save DataFrame
        df = pd.DataFrame(results)
        df = select_representative_runs(df, cfg, results_dir,
                                        K_values=K_values_var,
                                        num_unrolls_vals=num_unrolls_var)
        df = df.sort_values(
            ["architecture_type", "K", "num_unrolls", "run_idx"]
        ).reset_index(drop=True)
        df.to_csv(csv_path, index=False)

        t_var = time.time() - t_var_start
        n_ok  = int((df["success"] == True).sum())
        print(f"\n  ✓ Variant done in {t_var/60:.1f} min | "
              f"{n_ok}/{len(df)} successful | {csv_path}")

        # Quick per-variant summary
        df_ok = df[df["success"] == True]
        if len(df_ok) > 0:
            print("  ── Quick summary ─────────────────────────────────────")
            for arch in cfg["architectures"]:
                sub = df_ok[df_ok["architecture_type"] == arch]
                if len(sub) == 0:
                    continue
                print(f"    Arch {arch}: "
                      f"POG_train={sub['policy_optimality_gap_joint_train'].mean():.4f}  "
                      f"POG_test={sub['policy_optimality_gap_joint_test'].mean():.4f}  "
                      f"BOE_test={sub['bellman_optimality_error_test'].mean():.4f}")

        df["variant_name"] = var_name
        all_dfs.append(df)

    # ── Cross-variant summary
    if all_dfs:
        df_all       = pd.concat(all_dfs, ignore_index=True)
        summary_path = RESULTS_BASE / "comparison_summary.csv"
        df_all.to_csv(summary_path, index=False)
        print(f"\n✓ Combined summary: {summary_path}")

        print("\nGenerating comparison plots ...")
        try:
            plot_variant_comparison(summary_path, cfg)
            plot_kxunrolls_grid(summary_path, cfg)
        except Exception as e:
            print(f"  ⚠ Plot generation failed: {e}")

        print_summary_table(df_all, cfg)

    print("\n✓ All done.")


if __name__ == "__main__":
    main()
