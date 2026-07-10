"""
Regenera las figuras del Cap.4 del TFG a partir de los datos experimentales.
No reentrena — solo visualización.

Estado de cada parte:
  Parte 1 (T_comparison): FUNCIONAL — usa experiments/frequency_response_analysis.py
                           y los datos de freq_analysis_results/.
  Parte 2 (6-métricas):   FUNCIONAL — usa experiments/visualize_unrolls_results_tfg.py
                           y los datos de unrolls_results/.
  Parte 3 (mapas cliff):  FUNCIONAL — usa experiments/visualize_cliff_variations.py.
  OMITIDA — espectrales:  freq_response_per_unroll.py nunca commiteado (PERDIDO).
                           Figuras composition/resolvent ya presentes en images/cap4/.
  OMITIDA — k2_sweep:     Requiere reruns de k2_sweep_experiments.py + visualize_k2_sweep.py.

Uso:
    source .venv/bin/activate
    python regenerate_tfg_figures.py          # ejecuta partes 1, 2 y 3
    python experiments/regen_6metrics.py      # solo parte 2 (más rápido)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))

import shutil
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

from src.utils import get_optimal_q

CAP4      = Path("TFG/memoria/images/cap4")
FREQ_DIR  = Path("freq_per_unroll_results")
VIZ_DIR   = Path("unrolls_results") / "visualizations_tfg"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# PARTE 1: T_comparison (operador de transferencia espectral)
# ============================================================
print("=" * 60)
print("PARTE 1: T_comparison (frequency_response_analysis.py)")
print("=" * 60)

try:
    import experiments.frequency_response_analysis as fra
    fra.RESULTS_DIR = Path("freq_analysis_results")
    abs_eigs = fra.get_eigenvalues()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, cfg in zip(axes, fra.CONFIGS):
        fra.plot_T_comparison(ax, cfg["K"], cfg["num_unrolls"], abs_eigs)
    axes[1].set_ylabel("")
    fig.suptitle(
        "Comparación de operadores de transferencia $T(\\lambda)$\n"
        r"Ideal vs Práctica ($\gamma^t$) vs Aprendida ($h_t$)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out_tcomp = fra.RESULTS_DIR / "T_comparison.png"
    fig.savefig(out_tcomp, bbox_inches="tight", dpi=200)
    plt.close(fig)
    shutil.copy2(out_tcomp, CAP4 / "T_comparison.png")
    print(f"  ✓ T_comparison.png")
except Exception as e:
    print(f"  ✗ Error en Parte 1: {e}")
    print("    Verifica que freq_analysis_results/ contiene policy_arch*.npz")

# ============================================================
# PARTE 2: Figuras 6-métricas + mapas de política (unrolls)
# ============================================================
print()
print("=" * 60)
print("PARTE 2: Figuras 6-métricas y mapas de política")
print("=" * 60)

import visualize_unrolls_results_tfg as viz

viz.OUTPUT_DIR = VIZ_DIR

print("Cargando resultados de unrolls_results/ ...")
df_viz = viz.load_results()
print(f"  {len(df_viz)} filas cargadas.")

print("\nGenerando figuras 6-métricas ...")
viz.plot_comprehensive_6metrics(df_viz)
for src in VIZ_DIR.glob("comprehensive_6metrics*.png"):
    shutil.copy2(src, CAP4 / src.name)
    print(f"  ✓ {src.name}")

print("\nCalculando Q óptimas ...")
q_opt      = get_optimal_q(mirror_env=False, use_logger=False, max_eval_iters=50, max_epochs=50)
q_opt_test = get_optimal_q(mirror_env=True,  use_logger=False, max_eval_iters=50, max_epochs=50)

for num_unrolls in [4, 5, 10, 15]:
    matches = [v for v in df_viz['num_unrolls'].unique()
               if abs(float(v) - num_unrolls) < 0.1]
    if not matches:
        print(f"  ✗ Sin datos para num_unrolls={num_unrolls}")
        continue
    nu = matches[0]
    print(f"\n  num_unrolls={nu} ...")
    viz.plot_policy_maps_train_universal_scale(df_viz, nu, q_opt)
    viz.plot_policy_maps_test_universal_scale(df_viz, nu, q_opt_test)
    for suffix in ["train", "test"]:
        src = VIZ_DIR / f"unrolls{nu}_v2_policy_maps_{suffix}_universal_scale.png"
        dst = CAP4   / f"unrolls{num_unrolls}_v2_policy_maps_{suffix}_universal_scale.png"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"    ✓ {dst.name}")
        else:
            print(f"    ✗ No encontrado: {src}")

# ============================================================
# PARTE 3: Mapas de política — variantes cliff
# ============================================================
print()
print("=" * 60)
print("PARTE 3: Mapas de política — variantes cliff")
print("=" * 60)

import visualize_cliff_variations as vcv

CLIFF_RESULTS = Path("cliff_variations_results")

TRANSFER_VARIANTS = [
    ("center_start_std", 4, "center_start_std_policy_maps"),
    ("std_vertical",     5, "std_vertical_policy_maps"),
    ("large_std",        5, "large_std_policy_maps"),
    ("std_windy_random", 5, "std_windy_random_policy_maps"),
]

for variant_name, U_show, out_stem in TRANSFER_VARIANTS:
    results_dir = CLIFF_RESULTS / variant_name
    if not results_dir.exists():
        print(f"  ✗ Sin resultados para {variant_name}")
        continue

    print(f"\n  {variant_name} (U={U_show}) ...")
    env_cfg = vcv.VARIANT_ENV_CONFIG.get(variant_name)
    if env_cfg is None:
        print(f"  ✗ Sin configuración de entorno para {variant_name}")
        continue

    env_tr_type, env_te_type, goal_row_tr, goal_row_te = env_cfg
    try:
        env_tr     = vcv._make_env(env_tr_type)
        env_te     = vcv._make_env(env_te_type)
        q_opt_v    = vcv.get_optimal_q(env=env_tr, use_logger=False,
                                       max_eval_iters=50, max_epochs=50)
        q_opt_te_v = vcv.get_optimal_q(env=env_te, use_logger=False,
                                       max_eval_iters=50, max_epochs=50)
    except Exception as e:
        print(f"  ✗ Error construyendo entorno para {variant_name}: {e}")
        continue

    vcv.plot_policy_maps(
        variant_name, results_dir,
        q_opt_v, q_opt_te_v,
        env_tr, env_te,
        goal_row_tr, goal_row_te,
        num_unrolls=U_show,
    )

    vis_dir = results_dir / "visualizations"
    for suffix in ["test", "train"]:
        src = vis_dir / f"{variant_name}_U{U_show}_policy_maps_{suffix}_universal_scale.png"
        if src.exists():
            shutil.copy2(src, CAP4 / f"{out_stem}.png")
            print(f"    ✓ {out_stem}.png (modo={suffix})")
            break
    else:
        print(f"  ✗ Sin salida para {variant_name}")

print()
print("Hecho.")
