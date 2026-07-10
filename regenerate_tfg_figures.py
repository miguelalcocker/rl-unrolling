"""
Regenerates Cap4 TFG figures using existing experimental data.
No retraining — only visualization.

DEPENDENCY NOTE:
  Part 1 (spectral figures) requires freq_response_per_unroll.py (lost — never committed).
  Part 2 (6-metric unrolls figures) requires visualize_unrolls_results_v2.py (lost — never committed).
  Part 3 (cliff variation policy maps) uses experiments/visualize_cliff_variations.py (available).

  The generated figures are already present in TFG/memoria/images/cap4/.
  To regenerate only Part 3 (cliff maps), run this script and comment out Parts 1 and 2.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from src.utils import get_optimal_q

# ── Output dir
CAP4 = Path("TFG/memoria/images/cap4")
FREQ_DIR = Path("freq_per_unroll_results")
UNROLLS_DIR = Path("unrolls_results_v3_3")
VIZ_DIR = UNROLLS_DIR / "visualizations_v2"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# PART 1: Composition figures (from freq_response_per_unroll)
# ============================================================
print("=" * 60)
print("PART 1: Composition vs resolvent figures")
print("=" * 60)

import freq_response_per_unroll as frq

abs_eigs = frq.get_eigenvalues()
csv_path = FREQ_DIR / "all_results.csv"
df_freq  = pd.read_csv(csv_path)
df_freq = frq.select_representatives(df_freq, FREQ_DIR, frq.CFG)

u_groups = sorted({c["U_max"] for c in frq.CONFIGS})
for U_group in u_groups:
    cfg_sub = [c for c in frq.CONFIGS if c["U_max"] == U_group]
    suf = f"_U{U_group}"
    print(f"\nGenerating composition figures for U_max={U_group}...")
    frq.plot_composition_and_resolvent(
        FREQ_DIR, df_freq, abs_eigs, configs_list=cfg_sub, fname_suffix=suf)
    frq.plot_composition_and_resolvent_indexed(
        FREQ_DIR, df_freq, abs_eigs, configs_list=cfg_sub, fname_suffix=suf)

# Copy to images/cap4
for fname in [
    "composition_vs_resolvent_U4.png",
    "composition_vs_resolvent_U5.png",
    "composition_vs_resolvent_indexed_U4.png",
]:
    src = FREQ_DIR / fname
    dst = CAP4 / fname
    if src.exists():
        shutil.copy2(src, dst)
        print(f"✓ Copied {fname} to images/cap4/")
    else:
        print(f"✗ NOT FOUND: {src}")

# ============================================================
# PART 2: Policy maps (from visualize_unrolls_results_v2)
# ============================================================
print()
print("=" * 60)
print("PART 2: Policy maps (main unrolls, 2-column Arch1 vs Arch2)")
print("=" * 60)

import visualize_unrolls_results_v2 as viz

_orig_output = viz.OUTPUT_DIR
viz.OUTPUT_DIR = VIZ_DIR

df_viz = viz.load_results()

print("\nComputing optimal Q-values...")
q_opt = get_optimal_q(mirror_env=False, use_logger=False,
                      max_eval_iters=50, max_epochs=50)
q_opt_test = get_optimal_q(mirror_env=True, use_logger=False,
                            max_eval_iters=50, max_epochs=50)

TARGET_UNROLLS = [4, 5, 10, 15]

for num_unrolls in TARGET_UNROLLS:
    matches = [v for v in df_viz['num_unrolls'].unique()
               if abs(float(v) - num_unrolls) < 0.1]
    if not matches:
        print(f"  ✗ No data for num_unrolls={num_unrolls}")
        continue
    nu = matches[0]
    print(f"\nGenerating policy maps for num_unrolls={nu}...")
    viz.plot_policy_maps_train_universal_scale(df_viz, nu, q_opt)
    viz.plot_policy_maps_test_universal_scale(df_viz, nu, q_opt_test)

    for suffix in ["train", "test"]:
        src = VIZ_DIR / f"unrolls{nu}_v2_policy_maps_{suffix}_universal_scale.png"
        dst = CAP4 / f"unrolls{num_unrolls}_v2_policy_maps_{suffix}_universal_scale.png"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"✓ Copied → {dst.name}")
        else:
            print(f"✗ NOT FOUND: {src}")

# ============================================================
# PART 3: Transfer experiment policy maps
# ============================================================
print()
print("=" * 60)
print("PART 3: Transfer experiment policy maps")
print("=" * 60)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))
import visualize_cliff_variations as vcv

CLIFF_RESULTS = Path("cliff_variations_results")

# (variant, results_subdir, U_to_show, cap4_output_name)
TRANSFER_VARIANTS = [
    ("center_start_std", "center_start_std", 4,  "center_start_std_policy_maps"),
    ("std_vertical",     "std_vertical",     5,  "std_vertical_policy_maps"),
    ("large_std",        "large_std",        5,  "large_std_policy_maps"),
    ("std_windy_random", "std_windy_random", 5,  "std_windy_random_policy_maps"),
]

for variant_name, results_subdir, U_show, out_stem in TRANSFER_VARIANTS:
    results_dir = CLIFF_RESULTS / results_subdir
    if not results_dir.exists():
        print(f"  ✗ Results dir not found: {results_dir}")
        continue

    print(f"\nGenerating policy maps for {variant_name} (U={U_show})...")
    env_cfg = vcv.VARIANT_ENV_CONFIG.get(variant_name)
    if env_cfg is None:
        print(f"  ✗ No env config for {variant_name}")
        continue

    env_tr_type, env_te_type, goal_row_tr, goal_row_te = env_cfg

    try:
        env_tr  = vcv._make_env(env_tr_type)
        env_te  = vcv._make_env(env_te_type)
        q_opt_v = vcv.get_optimal_q(env=env_tr, use_logger=False,
                                    max_eval_iters=50, max_epochs=50)
        q_opt_te_v = vcv.get_optimal_q(env=env_te, use_logger=False,
                                       max_eval_iters=50, max_epochs=50)
    except Exception as e:
        print(f"  ✗ Error building env/q_opt for {variant_name}: {e}")
        continue

    vcv.plot_policy_maps(
        variant_name, results_dir,
        q_opt_v, q_opt_te_v,
        env_tr, env_te,
        goal_row_tr, goal_row_te,
        num_unrolls=U_show,
    )

    # Copy the generated test map to cap4 with expected name
    vis_dir = results_dir / "visualizations"
    # prefer test (shows transfer result); fall back to train
    for suffix in ["test", "train"]:
        src = vis_dir / f"{variant_name}_U{U_show}_policy_maps_{suffix}_universal_scale.png"
        if src.exists():
            dst = CAP4 / f"{out_stem}.png"
            shutil.copy2(src, dst)
            print(f"✓ Copied {src.name} → {dst.name} (mode={suffix})")
            break
    else:
        print(f"  ✗ No output found for {variant_name}")

print()
print("All done.")
