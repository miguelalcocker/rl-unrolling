"""
K_2 Sweep Experiments - Varying w Filter Order in Architecture 2
=================================================================

Experimentos con dos configuraciones fijas, variando el orden K_2
del filtro w en la Arquitectura 2.

CONFIGURACIONES:
- Config 1: num_unrolls=4,  K=5  → K_2 ∈ {0, 1, …, 6}  (7 variantes Arch2)
- Config 2: num_unrolls=5,  K=5  → K_2 ∈ {0, 1, …, 6}  (7 variantes Arch2)

ARQUITECTURAS:
- Arquitectura 1: variante única (sin K_2 relevante; siempre 1 término Q)
- Arquitectura 2: todos los K_2 posibles (0 hasta K+1)

CUANDO K_2=0, Arch2 es matemáticamente equivalente a Arch1:
  q̂ = Σ(k=0..K) h_k P^k r  +  w_0 P^{K+1} q_0

MÉTRICAS: idénticas a unrolls_experiments_analysis.py

Uso:
    python k2_sweep_experiments.py              # Ejecuta todo (reanuda automáticamente)
    python k2_sweep_experiments.py --force      # Fuerza re-ejecución completa

Author: Miguel Alcocer
Date: 2025
"""

import sys
import os
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from pytorch_lightning import Trainer
import time
import argparse
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.algorithms.unrolling_policy_iteration import UnrollingPolicyIterationTrain
from src.environments import CliffWalkingEnv, MirroredCliffWalkingEnv
from src.utils import (get_optimal_q, test_pol_err,
                       compute_optimality_gap, compute_optimality_gap_V)


# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = Path("k2_sweep_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Two fixed configurations
CONFIGS = [
    {"num_unrolls": 6,  "K": 5},
    {"num_unrolls": 5, "K": 5},
]

CONFIG = {
    "n_runs":     15,
    "max_epochs": 3000,
    "tau":        5.0,
    "lr":         5e-3,
    "gamma":      0.99,
    "loss_type":  "original_no_detach",
    "init_q":     "random",
}


def get_k2_values(K: int):
    """All valid K_2 values for Architecture 2: 0 to K+1 (inclusive)."""
    return list(range(K + 2))


# ============================================================================
# SINGLE EXPERIMENT
# ============================================================================

def run_single_experiment(env, env_test, q_opt, q_opt_test,
                          K, num_unrolls, architecture_type, K_2,
                          run_idx, run_name=""):
    """Run one training experiment and collect all metrics."""
    k2_str = f"K_2={K_2}" if K_2 is not None else "K_2=N/A (Arch1)"
    print(f"\n{'─'*80}")
    print(f"{run_name} | Arch={architecture_type} | K={K} | "
          f"Unrolls={num_unrolls} | {k2_str} | Run={run_idx+1}/{CONFIG['n_runs']}")
    print(f"{'─'*80}")

    try:
        model = UnrollingPolicyIterationTrain(
            env=env,
            env_test=env_test,
            K=K,
            num_unrolls=num_unrolls,
            gamma=CONFIG['gamma'],
            tau=CONFIG['tau'],
            lr=CONFIG['lr'],
            N=1,
            init_q=CONFIG['init_q'],
            loss_type=CONFIG['loss_type'],
            architecture_type=architecture_type,
            weight_sharing=True,
            use_legacy_init=True,
            K_2=K_2,          # None for Arch1 (ignored); int for Arch2
        )

        trainer = Trainer(
            max_epochs=CONFIG['max_epochs'],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )

        t_start = time.time()
        trainer.fit(model, ckpt_path=None)
        t_elapsed = time.time() - t_start

        device = model.device

        # ── Bellman Optimality Error ──────────────────────────────────────
        bellman_opt_err_train       = float(model.bellman_error.cpu())
        bellman_opt_err_unnorm_train = float(model.bellman_error_unnormalized.cpu())
        bellman_opt_err_test        = float(model.bellman_error_test.cpu())
        bellman_opt_err_unnorm_test  = float(model.bellman_error_unnormalized_test.cpu())

        # ── Optimality Gap (q from model) ─────────────────────────────────
        q_train = model.q.detach()
        q_test  = model.q_test.detach()

        og_joint_train, og_sep_train = compute_optimality_gap(q_train, q_opt, device)
        og_joint_test,  og_sep_test  = compute_optimality_gap(q_test,  q_opt_test, device)

        # ── Optimality Gap V (V = max_a q) ───────────────────────────────
        og_V_joint_train, og_V_sep_train = compute_optimality_gap_V(q_train, q_opt, device)
        og_V_joint_test,  og_V_sep_test  = compute_optimality_gap_V(q_test,  q_opt_test, device)

        # ── Policy Optimality Gap (q_π via policy evaluation) ─────────────
        pog_joint_train, pog_sep_train = test_pol_err(
            model.Pi, q_opt, mirror_env=False, device=device)
        pog_joint_test,  pog_sep_test  = test_pol_err(
            model.Pi_test, q_opt_test, mirror_env=True, device=device)

        result = {
            "num_unrolls":      num_unrolls,
            "K":                K,
            "architecture_type": architecture_type,
            # K_2: stored as NaN for Arch1, int value for Arch2
            "K_2":              K_2 if K_2 is not None else np.nan,
            # w_order = K_2 + 1  (1 for Arch1 = same as Arch2 with K_2=0)
            "w_order":          (K_2 + 1) if K_2 is not None else 1,
            "run_idx":          run_idx,
            "training_time_sec": t_elapsed,
            "loss_type":        CONFIG['loss_type'],
            "init_q":           CONFIG['init_q'],
            # Bellman error
            "bellman_optimality_error_train":              bellman_opt_err_train,
            "bellman_optimality_error_unnormalized_train": bellman_opt_err_unnorm_train,
            "bellman_optimality_error_test":               bellman_opt_err_test,
            "bellman_optimality_error_unnormalized_test":  bellman_opt_err_unnorm_test,
            # Optimality gap (q)
            "optimality_gap_joint_train":    og_joint_train,
            "optimality_gap_separate_train": og_sep_train,
            "optimality_gap_joint_test":     og_joint_test,
            "optimality_gap_separate_test":  og_sep_test,
            # Optimality gap (V)
            "optimality_gap_V_joint_train":    og_V_joint_train,
            "optimality_gap_V_separate_train": og_V_sep_train,
            "optimality_gap_V_joint_test":     og_V_joint_test,
            "optimality_gap_V_separate_test":  og_V_sep_test,
            # Policy Optimality Gap (q_π, squared)
            "policy_optimality_gap_joint_train":    float(pog_joint_train),
            "policy_optimality_gap_separate_train": float(pog_sep_train),
            "policy_optimality_gap_joint_test":     float(pog_joint_test),
            "policy_optimality_gap_separate_test":  float(pog_sep_test),
            "success": True,
        }

        print(f"  ✓ {t_elapsed:.1f}s | "
              f"POG_train={pog_joint_train:.4f} | POG_test={pog_joint_test:.4f}")
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "num_unrolls":       num_unrolls,
            "K":                 K,
            "architecture_type": architecture_type,
            "K_2":               K_2 if K_2 is not None else np.nan,
            "w_order":           (K_2 + 1) if K_2 is not None else 1,
            "run_idx":           run_idx,
            "success":           False,
            "error":             str(e),
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='K_2 sweep experiments')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run of all experiments (ignores existing results)')
    args = parser.parse_args()

    print("=" * 80)
    print("K_2 SWEEP EXPERIMENTS  –  VARYING w FILTER ORDER IN ARCHITECTURE 2")
    print("=" * 80)

    # ── Build full experiment list ──────────────────────────────────────────
    experiment_list = []
    for cfg in CONFIGS:
        K, num_unrolls = cfg["K"], cfg["num_unrolls"]
        # Arch 1: one variant
        for run_idx in range(CONFIG['n_runs']):
            experiment_list.append(dict(num_unrolls=num_unrolls, K=K,
                                        arch=1, K_2=None, run_idx=run_idx))
        # Arch 2: all K_2 values
        for K_2 in get_k2_values(K):
            for run_idx in range(CONFIG['n_runs']):
                experiment_list.append(dict(num_unrolls=num_unrolls, K=K,
                                            arch=2, K_2=K_2, run_idx=run_idx))

    total_experiments = len(experiment_list)

    print(f"\nConfiguration summary:")
    for cfg in CONFIGS:
        K, u = cfg["K"], cfg["num_unrolls"]
        n_k2 = len(get_k2_values(K))
        n_variants = 1 + n_k2   # Arch1 + all Arch2 K_2 variants
        print(f"  unrolls={u}, K={K}: Arch1 (1) + Arch2 ({n_k2} K_2 values) "
              f"= {n_variants} variants × {CONFIG['n_runs']} runs = "
              f"{n_variants * CONFIG['n_runs']} experiments")
    print(f"  TOTAL: {total_experiments} experiments")
    print(f"  Max epochs per run: {CONFIG['max_epochs']}")
    print(f"  Force re-run: {args.force}")
    print()

    # ── Load existing results ───────────────────────────────────────────────
    csv_path = RESULTS_DIR / "all_experiments_results.csv"
    existing_keys = set()
    existing_rows = []

    if csv_path.exists() and not args.force:
        df_existing = pd.read_csv(csv_path)
        df_ok = df_existing[df_existing['success'] == True]
        for _, row in df_ok.iterrows():
            k2_val = None if pd.isna(row.get('K_2', np.nan)) else int(row['K_2'])
            key = (int(row['num_unrolls']), int(row['K']),
                   int(row['architecture_type']), k2_val, int(row['run_idx']))
            existing_keys.add(key)
        # Only keep successful runs to avoid duplicates if a failed run is re-executed
        existing_rows = df_ok.to_dict('records')
        print(f"Resuming: found {len(existing_keys)} already-completed experiments "
              f"(discarding {len(df_existing) - len(df_ok)} failed records).\n")
    elif args.force:
        print("Force mode: discarding existing results.\n")

    # ── Compute optimal Q-values ────────────────────────────────────────────
    print("Computing optimal Q-values...")
    env      = CliffWalkingEnv()
    env_test = MirroredCliffWalkingEnv()

    q_opt      = get_optimal_q(mirror_env=False, use_logger=False,
                               max_eval_iters=50, max_epochs=50)
    q_opt_test = get_optimal_q(mirror_env=True,  use_logger=False,
                               max_eval_iters=50, max_epochs=50)

    print(f"  ✓ Train q*: ‖q*‖ = {torch.norm(q_opt):.4f}")
    print(f"  ✓ Test  q*: ‖q*‖ = {torch.norm(q_opt_test):.4f}\n")

    # ── Main loop ───────────────────────────────────────────────────────────
    results = list(existing_rows)   # start from already-done results
    new_count = 0
    skip_count = 0
    start_time = time.time()

    for exp_idx, exp in enumerate(experiment_list, start=1):
        num_unrolls = exp['num_unrolls']
        K           = exp['K']
        arch        = exp['arch']
        K_2         = exp['K_2']
        run_idx     = exp['run_idx']

        key = (num_unrolls, K, arch, K_2, run_idx)
        if key in existing_keys:
            skip_count += 1
            continue

        run_name = f"U{num_unrolls}_K{K}_A{arch}_K2={K_2}"
        result = run_single_experiment(
            env=env, env_test=env_test,
            q_opt=q_opt, q_opt_test=q_opt_test,
            K=K, num_unrolls=num_unrolls,
            architecture_type=arch,
            K_2=K_2,
            run_idx=run_idx,
            run_name=run_name,
        )
        results.append(result)
        new_count += 1

        # ── Progress & ETA ──────────────────────────────────────────────
        elapsed   = time.time() - start_time
        avg_per_run = elapsed / new_count if new_count > 0 else 0
        remaining_exps = total_experiments - exp_idx - skip_count
        remaining_new  = max(0, remaining_exps - skip_count)
        eta_secs  = remaining_new * avg_per_run
        eta_time  = datetime.now() + timedelta(seconds=eta_secs)

        print(f"\n  ── Progress: {exp_idx}/{total_experiments} exps scheduled "
              f"({skip_count} skipped, {new_count} run so far) ──")
        print(f"     Elapsed: {elapsed/60:.1f} min | "
              f"Avg/run: {avg_per_run:.1f}s | "
              f"Remaining (new): ~{eta_secs/60:.1f} min | "
              f"ETA: {eta_time.strftime('%H:%M:%S')}")

        # ── Save intermediate results ───────────────────────────────────
        df_interim = pd.DataFrame(results)
        df_interim.to_csv(csv_path, index=False)

    # ── Final save ──────────────────────────────────────────────────────────
    df_final = pd.DataFrame(results)
    df_final = df_final.sort_values(
        ['num_unrolls', 'K', 'architecture_type', 'K_2', 'run_idx']
    ).reset_index(drop=True)
    df_final.to_csv(csv_path, index=False)

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"  Results saved to : {csv_path}")
    print(f"  Total records    : {len(df_final)}")
    print(f"  Successful       : {df_final['success'].sum()}")
    print(f"  Failed           : {(~df_final['success'].astype(bool)).sum()}")
    print(f"  New runs this session: {new_count}")
    print(f"  Skipped (already done): {skip_count}")
    print(f"  Total wall time  : {total_elapsed/60:.1f} min")
    print("=" * 80)

    # ── Summary statistics ───────────────────────────────────────────────────
    df_ok = df_final[df_final['success'] == True]
    if len(df_ok) > 0:
        print("\nPOLICY OPTIMALITY GAP SUMMARY (successful new runs)")
        print("-" * 60)
        for cfg in CONFIGS:
            K, u = cfg["K"], cfg["num_unrolls"]
            df_cfg = df_ok[(df_ok['K'] == K) & (df_ok['num_unrolls'] == u)]
            print(f"\n  unrolls={u}, K={K}:")
            for arch in [1, 2]:
                df_a = df_cfg[df_cfg['architecture_type'] == arch]
                if len(df_a) == 0:
                    continue
                med_tr = df_a['policy_optimality_gap_joint_train'].median()
                med_te = df_a['policy_optimality_gap_joint_test'].median()
                print(f"    Arch{arch}: POG_train median={med_tr:.6f} | "
                      f"POG_test median={med_te:.6f}")


if __name__ == "__main__":
    main()
