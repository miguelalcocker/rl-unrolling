"""
Unrolls Experiments Analysis - Varying Number of Unrolls
==========================================================

Experimentos variando el número de unrolls para analizar su efecto.

MÉTRICAS CALCULADAS:
--------------------
1. BELLMAN OPTIMALITY ERROR (usa q de la salida del modelo, NO q_π):
   - bellman_optimality_error = ||q - (r + γ P_π q)|| / ||r + γ P_π q||
   - bellman_optimality_error_unnormalized = ||q - (r + γ P_π q)||

2. OPTIMALITY GAP (usa q de la salida del modelo directamente):
   - optimality_gap_joint = ||q - q*|| / ||q*||
   - optimality_gap_separate = ||q/||q|| - q*/||q*|||

3. POLICY OPTIMALITY GAP (usa q_π obtenida via policy evaluation):
   - policy_optimality_gap_joint = (||q_π - q*|| / ||q*||)²
   - policy_optimality_gap_separate = (||q_π/||q_π|| - q*/||q*|||)²

4. MAPAS DE POLÍTICAS (guardados en archivos .npz)

Configuración:
- Arquitecturas: 1 y 2
- Loss: original_no_detach únicamente
- Inicialización: random
- K: [3, 5, 10, 12]
- num_unrolls: configurable (por defecto: todos)
- normalize_separately: False (SIEMPRE)

Uso:
    python unrolls_experiments_analysis.py                    # Ejecuta todos los unrolls
    python unrolls_experiments_analysis.py --unrolls 5 15     # Solo ejecuta unrolls 5 y 15
    python unrolls_experiments_analysis.py --force            # Fuerza re-ejecución de todo

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

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.algorithms.unrolling_policy_iteration import UnrollingPolicyIterationTrain
from src.environments import CliffWalkingEnv, MirroredCliffWalkingEnv
from src.utils import (get_optimal_q, test_pol_err,
                       compute_optimality_gap, compute_optimality_gap_V)


# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = Path("unrolls_results_v3_3")
RESULTS_DIR.mkdir(exist_ok=True)

# All possible num_unrolls values
ALL_NUM_UNROLLS = [2, 4, 5, 6, 8, 10, 15]

CONFIG = {
    "n_runs": 15,  # Múltiples runs para estadística robusta
    "max_epochs": 3000,
    "K_values": [3, 5, 10, 12],
    "tau": 5.0,
    "lr": 5e-3,
    "gamma": 0.99,
    "loss_type": "original_no_detach",  # detach variant produces different results
    "init_q": "random",
    "architectures": [1, 2],
}


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def run_single_experiment(
    env,
    env_test,
    q_opt,
    q_opt_test,
    K,
    num_unrolls,
    architecture_type,
    run_idx,
    run_name="",
):
    """Run a single training experiment and collect all metrics."""

    print(f"\n{'─'*80}")
    print(f"Config: {run_name} | Arch={architecture_type} | K={K} | "
          f"Unrolls={num_unrolls} | Run={run_idx+1}/{CONFIG['n_runs']}")
    print(f"{'─'*80}")

    try:
        # Create model (normalize_separately ALWAYS False)
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
        )

        # Training
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

        # ====================================================================
        # COMPUTE ALL METRICS (TRAIN and TEST)
        # ====================================================================

        device = model.device

        # -----------------------------------------------------------------
        # BELLMAN OPTIMALITY ERROR (usa q de la salida del modelo, NO q_π)
        # -----------------------------------------------------------------
        bellman_optimality_error_train = float(model.bellman_error.cpu())
        bellman_optimality_error_unnormalized_train = float(model.bellman_error_unnormalized.cpu())
        bellman_optimality_error_test = float(model.bellman_error_test.cpu())
        bellman_optimality_error_unnormalized_test = float(model.bellman_error_unnormalized_test.cpu())

        # -----------------------------------------------------------------
        # OPTIMALITY GAP (usa q de la salida del modelo directamente)
        # -----------------------------------------------------------------
        q_train_tensor = model.q.detach()
        optimality_gap_joint_train, optimality_gap_separate_train = compute_optimality_gap(
            q_train_tensor, q_opt, device
        )

        q_test_tensor = model.q_test.detach()
        optimality_gap_joint_test, optimality_gap_separate_test = compute_optimality_gap(
            q_test_tensor, q_opt_test, device
        )

        # -----------------------------------------------------------------
        # OPTIMALITY GAP V (usa V = max_a q(s,a) del modelo)
        # -----------------------------------------------------------------
        optimality_gap_V_joint_train, optimality_gap_V_separate_train = compute_optimality_gap_V(
            q_train_tensor, q_opt, device
        )

        optimality_gap_V_joint_test, optimality_gap_V_separate_test = compute_optimality_gap_V(
            q_test_tensor, q_opt_test, device
        )

        # -----------------------------------------------------------------
        # POLICY OPTIMALITY GAP (usa q_π via policy evaluation)
        # -----------------------------------------------------------------
        # TRAIN: q_π from policy evaluation of learned policy
        policy_opt_gap_joint_train, policy_opt_gap_separate_train = test_pol_err(
            model.Pi, q_opt, mirror_env=False, device=device
        )

        # TEST: q_π from policy evaluation of learned policy
        policy_opt_gap_joint_test, policy_opt_gap_separate_test = test_pol_err(
            model.Pi_test, q_opt_test, mirror_env=True, device=device
        )

        # -----------------------------------------------------------------
        # ADDITIONAL METRICS
        # -----------------------------------------------------------------
        q_norm_squared_train = float(torch.norm(model.q) ** 2)
        q_norm_squared_test = float(torch.norm(model.q_test) ** 2)

        # -----------------------------------------------------------------
        # SAVE POLICY FOR VISUALIZATION (mapas de políticas)
        # -----------------------------------------------------------------
        Pi_train = model.Pi.detach().cpu().numpy()
        q_train = model.q.detach().cpu().numpy()
        Pi_test = model.Pi_test.detach().cpu().numpy()
        q_test = model.q_test.detach().cpu().numpy()

        policy_data = {
            "Pi_train": Pi_train,
            "q_train": q_train,
            "Pi_test": Pi_test,
            "q_test": q_test,
        }
        policy_filename = (
            f"policy_arch{architecture_type}_K{K}_"
            f"unrolls{num_unrolls}_run{run_idx}_temp.npz"
        )
        np.savez(RESULTS_DIR / policy_filename, **policy_data)

        # -----------------------------------------------------------------
        # BUILD RESULT DICTIONARY
        # -----------------------------------------------------------------
        result = {
            "architecture_type": architecture_type,
            "K": K,
            "num_unrolls": num_unrolls,
            "loss_type": CONFIG['loss_type'],
            "init_q": CONFIG['init_q'],
            "run_idx": run_idx,
            "training_time_sec": t_elapsed,
            # BELLMAN OPTIMALITY ERROR (q from model, NOT q_π)
            "bellman_optimality_error_train": bellman_optimality_error_train,
            "bellman_optimality_error_unnormalized_train": bellman_optimality_error_unnormalized_train,
            "bellman_optimality_error_test": bellman_optimality_error_test,
            "bellman_optimality_error_unnormalized_test": bellman_optimality_error_unnormalized_test,
            # OPTIMALITY GAP (q from model directly)
            "optimality_gap_joint_train": optimality_gap_joint_train,
            "optimality_gap_separate_train": optimality_gap_separate_train,
            "optimality_gap_joint_test": optimality_gap_joint_test,
            "optimality_gap_separate_test": optimality_gap_separate_test,
            # OPTIMALITY GAP V (V = max_a q(s,a) from model)
            "optimality_gap_V_joint_train": optimality_gap_V_joint_train,
            "optimality_gap_V_separate_train": optimality_gap_V_separate_train,
            "optimality_gap_V_joint_test": optimality_gap_V_joint_test,
            "optimality_gap_V_separate_test": optimality_gap_V_separate_test,
            # POLICY OPTIMALITY GAP (q_π via policy evaluation) - squared
            "policy_optimality_gap_joint_train": float(policy_opt_gap_joint_train),
            "policy_optimality_gap_separate_train": float(policy_opt_gap_separate_train),
            "policy_optimality_gap_joint_test": float(policy_opt_gap_joint_test),
            "policy_optimality_gap_separate_test": float(policy_opt_gap_separate_test),
            # ADDITIONAL
            "q_norm_squared_train": q_norm_squared_train,
            "q_norm_squared_test": q_norm_squared_test,
            "policy_file_temp": policy_filename,
            "success": True,
        }

        print(f"✓ Success | Time: {t_elapsed:.1f}s")
        print(f"  BELLMAN OPT ERR:     train={bellman_optimality_error_train:.4f} | test={bellman_optimality_error_test:.4f}")
        print(f"  OPT GAP (q):         joint_train={optimality_gap_joint_train:.4f} | joint_test={optimality_gap_joint_test:.4f}")
        print(f"  OPT GAP (V):         joint_train={optimality_gap_V_joint_train:.4f} | joint_test={optimality_gap_V_joint_test:.4f}")
        print(f"  POLICY OPT GAP (q_π): joint_train={policy_opt_gap_joint_train:.4f} | joint_test={policy_opt_gap_joint_test:.4f}")

        return result

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "architecture_type": architecture_type,
            "K": K,
            "num_unrolls": num_unrolls,
            "loss_type": CONFIG['loss_type'],
            "run_idx": run_idx,
            "success": False,
            "error": str(e),
        }


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run unrolls experiments')
    parser.add_argument('--unrolls', type=int, nargs='+', default=None,
                        help='Specific num_unrolls values to run (e.g., --unrolls 5 15)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if results exist (overwrites all)')
    args = parser.parse_args()

    print("="*80)
    print("UNROLLS EXPERIMENTS - VARYING NUMBER OF UNROLLS")
    print("="*80)

    # Determine which num_unrolls to run
    if args.unrolls:
        num_unrolls_to_run = sorted(args.unrolls)
    else:
        num_unrolls_to_run = ALL_NUM_UNROLLS

    print(f"\nConfiguration:")
    print(f"  Loss type: {CONFIG['loss_type']}")
    print(f"  Initialization: {CONFIG['init_q']}")
    print(f"  K values: {CONFIG['K_values']}")
    print(f"  num_unrolls to run: {num_unrolls_to_run}")
    print(f"  Architectures: {CONFIG['architectures']}")
    print(f"  normalize_separately: ALWAYS False")
    print(f"  Runs per config: {CONFIG['n_runs']}")
    print(f"  Max epochs: {CONFIG['max_epochs']}")
    print(f"  Force re-run: {args.force}")
    print()

    # Load existing results if they exist (to preserve other unrolls)
    csv_path = RESULTS_DIR / "all_experiments_results.csv"
    if csv_path.exists() and not args.force:
        print("Loading existing results...")
        df_existing = pd.read_csv(csv_path)
        print(f"  Found {len(df_existing)} existing experiment records")

        # Preserve results from unrolls that we're NOT going to run
        df_preserved = df_existing[~df_existing['num_unrolls'].isin(num_unrolls_to_run)]
        print(f"  Preserving {len(df_preserved)} records from other num_unrolls values")
        print(f"  Will overwrite records for num_unrolls: {num_unrolls_to_run}")
    else:
        df_preserved = pd.DataFrame()
        if args.force:
            print("Force mode: will overwrite all existing results")

    # Compute optimal Q-values once
    print("\nComputing optimal Q-values...")
    env = CliffWalkingEnv()
    env_test = MirroredCliffWalkingEnv()

    q_opt = get_optimal_q(
        mirror_env=False,
        use_logger=False,
        max_eval_iters=50,
        max_epochs=50
    )

    q_opt_test = get_optimal_q(
        mirror_env=True,
        use_logger=False,
        max_eval_iters=50,
        max_epochs=50
    )

    print(f"✓ Optimal Q computed")
    print(f"  Train env: ||q*|| = {torch.norm(q_opt):.2f}")
    print(f"  Test env:  ||q*|| = {torch.norm(q_opt_test):.2f}")

    # Run experiments for specified num_unrolls
    results = []
    total_experiments = (
        len(CONFIG['K_values']) *
        len(num_unrolls_to_run) *
        len(CONFIG['architectures']) *
        CONFIG['n_runs']
    )

    exp_count = 0
    start_time = time.time()

    for K in CONFIG['K_values']:
        for num_unrolls in num_unrolls_to_run:
            for arch in CONFIG['architectures']:
                for run_idx in range(CONFIG['n_runs']):
                    exp_count += 1

                    run_name = f"Arch{arch}_K{K}_U{num_unrolls}"

                    result = run_single_experiment(
                        env=env,
                        env_test=env_test,
                        q_opt=q_opt,
                        q_opt_test=q_opt_test,
                        K=K,
                        num_unrolls=num_unrolls,
                        architecture_type=arch,
                        run_idx=run_idx,
                        run_name=run_name,
                    )

                    results.append(result)

                    # Progress update
                    elapsed = time.time() - start_time
                    avg_time = elapsed / exp_count
                    remaining = (total_experiments - exp_count) * avg_time

                    print(f"\nProgress: {exp_count}/{total_experiments} "
                          f"({100*exp_count/total_experiments:.1f}%)")
                    print(f"Elapsed: {elapsed/60:.1f}min | "
                          f"Remaining: {remaining/60:.1f}min")

    # Combine new results with preserved results
    df_new = pd.DataFrame(results)

    if len(df_preserved) > 0:
        # Ensure columns match (add missing columns with NaN)
        all_columns = set(df_new.columns) | set(df_preserved.columns)
        for col in all_columns:
            if col not in df_new.columns:
                df_new[col] = np.nan
            if col not in df_preserved.columns:
                df_preserved[col] = np.nan

        df = pd.concat([df_preserved, df_new], ignore_index=True)
        print(f"\nCombined {len(df_preserved)} preserved + {len(df_new)} new = {len(df)} total records")
    else:
        df = df_new

    # Sort by architecture, K, num_unrolls, run_idx
    df = df.sort_values(['architecture_type', 'K', 'num_unrolls', 'run_idx']).reset_index(drop=True)

    # Save results
    df.to_csv(csv_path, index=False)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {csv_path}")
    print(f"Total experiments in file: {len(df)}")
    print(f"New experiments run: {len(results)}")
    print(f"Successful: {df_new['success'].sum()}")
    print(f"Failed: {(~df_new['success']).sum()}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")

    # Summary statistics for new results
    df_success = df_new[df_new['success'] == True]
    if len(df_success) > 0:
        print("\n" + "-"*80)
        print("SUMMARY STATISTICS (new runs)")
        print("-"*80)

        metrics = [
            'bellman_optimality_error_train', 'bellman_optimality_error_test',
            'optimality_gap_joint_train', 'optimality_gap_joint_test',
            'optimality_gap_V_joint_train', 'optimality_gap_V_joint_test',
            'policy_optimality_gap_joint_train', 'policy_optimality_gap_joint_test',
        ]

        for metric in metrics:
            if metric in df_success.columns:
                print(f"\n{metric}:")
                print(f"  Mean: {df_success[metric].mean():.6f}")
                print(f"  Std:  {df_success[metric].std():.6f}")
                print(f"  Min:  {df_success[metric].min():.6f}")
                print(f"  Max:  {df_success[metric].max():.6f}")

    # ============================================================================
    # SELECT REPRESENTATIVE RUNS (closest to median) - for newly run num_unrolls
    # ============================================================================

    print("\n" + "="*80)
    print("SELECTING REPRESENTATIVE RUNS (closest to median)")
    print("="*80)
    print()

    # Metrics for selection
    metric_for_test = 'policy_optimality_gap_joint_test'
    metric_for_train = 'policy_optimality_gap_joint_train'

    for K in CONFIG['K_values']:
        for num_unrolls in num_unrolls_to_run:
            for arch in CONFIG['architectures']:
                # Filter data for this configuration
                df_config = df_success[
                    (df_success['K'] == K) &
                    (df_success['num_unrolls'] == num_unrolls) &
                    (df_success['architecture_type'] == arch)
                ]

                if len(df_config) == 0:
                    print(f"⚠ No successful runs for K={K}, Unrolls={num_unrolls}, Arch={arch}")
                    continue

                print(f"K={K}, Unrolls={num_unrolls}, Arch={arch}:")

                # ================================================================
                # SELECT REPRESENTATIVE RUN FOR TEST (based on TEST metric)
                # ================================================================
                median_test = df_config[metric_for_test].median()
                df_config_copy = df_config.copy()
                df_config_copy['diff_from_median'] = abs(df_config_copy[metric_for_test] - median_test)
                best_idx_test = df_config_copy['diff_from_median'].idxmin()
                best_run_test = df_config_copy.loc[best_idx_test]
                best_run_idx_test = int(best_run_test['run_idx'])
                best_value_test = best_run_test[metric_for_test]

                print(f"  TEST:  Median POG={median_test:.6f} -> run_idx={best_run_idx_test} (value={best_value_test:.6f})")

                # ================================================================
                # SELECT REPRESENTATIVE RUN FOR TRAIN (based on TRAIN metric)
                # ================================================================
                median_train = df_config[metric_for_train].median()
                df_config_copy['diff_from_median'] = abs(df_config_copy[metric_for_train] - median_train)
                best_idx_train = df_config_copy['diff_from_median'].idxmin()
                best_run_train = df_config_copy.loc[best_idx_train]
                best_run_idx_train = int(best_run_train['run_idx'])
                best_value_train = best_run_train[metric_for_train]

                print(f"  TRAIN: Median POG={median_train:.6f} -> run_idx={best_run_idx_train} (value={best_value_train:.6f})")

                # ================================================================
                # SAVE FILES
                # ================================================================
                import shutil

                # Save TEST representative file
                temp_filename_test = f"policy_arch{arch}_K{K}_unrolls{num_unrolls}_run{best_run_idx_test}_temp.npz"
                final_filename_test = f"policy_arch{arch}_K{K}_unrolls{num_unrolls}_run{best_run_idx_test}.npz"
                temp_path_test = RESULTS_DIR / temp_filename_test
                final_path_test = RESULTS_DIR / final_filename_test

                if temp_path_test.exists():
                    shutil.copy(str(temp_path_test), str(final_path_test))
                    print(f"  ✓ Saved (TEST):  {final_filename_test}")

                    mask = (
                        (df['K'] == K) &
                        (df['num_unrolls'] == num_unrolls) &
                        (df['architecture_type'] == arch) &
                        (df['run_idx'] == best_run_idx_test)
                    )
                    df.loc[mask, 'policy_file'] = final_filename_test
                else:
                    print(f"  ✗ Warning: {temp_filename_test} not found")

                # Save TRAIN representative file (if different from TEST)
                if best_run_idx_train != best_run_idx_test:
                    temp_filename_train = f"policy_arch{arch}_K{K}_unrolls{num_unrolls}_run{best_run_idx_train}_temp.npz"
                    final_filename_train = f"policy_arch{arch}_K{K}_unrolls{num_unrolls}_run{best_run_idx_train}_train.npz"
                    temp_path_train = RESULTS_DIR / temp_filename_train
                    final_path_train = RESULTS_DIR / final_filename_train

                    if temp_path_train.exists():
                        shutil.copy(str(temp_path_train), str(final_path_train))
                        print(f"  ✓ Saved (TRAIN): {final_filename_train}")

                        mask_train = (
                            (df['K'] == K) &
                            (df['num_unrolls'] == num_unrolls) &
                            (df['architecture_type'] == arch) &
                            (df['run_idx'] == best_run_idx_train)
                        )
                        df.loc[mask_train, 'policy_file_train'] = final_filename_train
                    else:
                        print(f"  ✗ Warning: {temp_filename_train} not found")
                else:
                    # Same run for both - use same file
                    print(f"  (Same run selected for both TRAIN and TEST)")
                    mask = (
                        (df['K'] == K) &
                        (df['num_unrolls'] == num_unrolls) &
                        (df['architecture_type'] == arch) &
                        (df['run_idx'] == best_run_idx_test)
                    )
                    df.loc[mask, 'policy_file_train'] = final_filename_test

                # Delete all temporary files for this configuration
                for run_idx in range(CONFIG['n_runs']):
                    temp_file = RESULTS_DIR / f"policy_arch{arch}_K{K}_unrolls{num_unrolls}_run{run_idx}_temp.npz"
                    if temp_file.exists():
                        temp_file.unlink()

                print()

    # Save updated DataFrame
    df.to_csv(csv_path, index=False)
    print("✓ Updated CSV with representative policy files")

    # Print metric definitions
    print("\n" + "="*80)
    print("METRIC DEFINITIONS")
    print("="*80)
    print("""
BELLMAN OPTIMALITY ERROR (usa q de la salida del modelo, NO q_π):
  - bellman_optimality_error = ||q - (r + γ P_π q)|| / ||r + γ P_π q||
  - bellman_optimality_error_unnormalized = ||q - (r + γ P_π q)||

OPTIMALITY GAP (usa q de la salida del modelo directamente):
  - optimality_gap_joint = ||q - q*|| / ||q*||
  - optimality_gap_separate = ||q/||q|| - q*/||q*|||

OPTIMALITY GAP V (usa V = max_a q(s,a) del modelo):
  - optimality_gap_V_joint = ||V - V*|| / ||V*||
  - optimality_gap_V_separate = ||V/||V|| - V*/||V*|||

POLICY OPTIMALITY GAP (usa q_π obtenida via policy evaluation):
  - policy_optimality_gap_joint = (||q_π - q*|| / ||q*||)²
  - policy_optimality_gap_separate = (||q_π/||q_π|| - q*/||q*|||)²
""")


if __name__ == "__main__":
    main()
