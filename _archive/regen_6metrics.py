"""
Regenera las figuras de 6 métricas del Cap.4 del TFG (Figs 4.x unrolls).

Lee los datos de unrolls_results/ (experimentos originales del TFG, init_q='ones')
y genera las figuras mediante visualize_unrolls_results_tfg.py.

Uso:
    cd /home/malco/rl-unrolling
    source .venv/bin/activate
    python experiments/regen_6metrics.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import shutil
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

import visualize_unrolls_results_tfg as viz
from src.utils import get_optimal_q

CAP4 = Path("TFG/memoria/images/cap4")
VIZ_DIR = Path("unrolls_results") / "visualizations_v2"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

viz.OUTPUT_DIR = VIZ_DIR

print("Cargando resultados de unrolls_results/...")
df = viz.load_results()
print(f"  {len(df)} filas cargadas.")

print("Generando figuras 6-métricas...")
viz.plot_comprehensive_6metrics(df)

src_fig = VIZ_DIR / "comprehensive_6metrics.png"
if src_fig.exists():
    dst = CAP4 / "comprehensive_6metrics.png"
    shutil.copy2(src_fig, dst)
    print(f"  Copiado → {dst}")

print("\nGenerando mapas de política (train + test, escalas universales)...")
q_opt      = get_optimal_q(mirror_env=False, use_logger=False, max_eval_iters=50, max_epochs=50)
q_opt_test = get_optimal_q(mirror_env=True,  use_logger=False, max_eval_iters=50, max_epochs=50)

for num_unrolls in [4, 5, 10, 15]:
    matches = [v for v in df['num_unrolls'].unique() if abs(float(v) - num_unrolls) < 0.1]
    if not matches:
        print(f"  ✗ Sin datos para num_unrolls={num_unrolls}")
        continue
    nu = matches[0]
    print(f"\n  num_unrolls={nu}...")
    viz.plot_policy_maps_train_universal_scale(df, nu, q_opt)
    viz.plot_policy_maps_test_universal_scale(df, nu, q_opt_test)

    for suffix in ["train", "test"]:
        src = VIZ_DIR / f"unrolls{nu}_v2_policy_maps_{suffix}_universal_scale.png"
        dst = CAP4 / f"unrolls{num_unrolls}_v2_policy_maps_{suffix}_universal_scale.png"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"    ✓ {dst.name}")
        else:
            print(f"    ✗ No encontrado: {src}")

print("\nHecho. Figuras en TFG/memoria/images/cap4/")
