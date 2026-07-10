"""
Empirical Amplification Analysis — SCRIPT INOPERATIVO
=======================================================

Este script medía empíricamente la amplificación espectral de cada modelo
entrenado (K, arch, U_max) proyectando los autovectores de P_π sobre la red:

    amp_U_i = ||q_U(v_i)||_2 / ||v_i||_2

y comparaba el resultado con la resolvente teórica:

    T_res(λ_i) = |H_r(λ_i) / (1 - G_q(λ_i))|

ESTADO: No puede ejecutarse. Dependía de `freq_response_per_unroll.py`,
que nunca fue commiteado y no puede recuperarse.

DATOS DISPONIBLES:
  - freq_per_unroll_results/        datos experimentales (CSV + .npz por run)
  - TFG/memoria/images/cap4/        figuras finales ya generadas e incluidas en el PDF

PARA REPRODUCIR LAS FIGURAS DEL TFG:
  Las figuras espectrales (composición vs resolvente, resolvente por unroll)
  están en TFG/memoria/images/cap4/ y son las originales generadas durante
  el desarrollo. No hay forma de regenerarlas sin freq_response_per_unroll.py.
"""

raise SystemExit(
    "empirical_amplification.py no puede ejecutarse: falta freq_response_per_unroll.py.\n"
    "Consulta la documentación en el docstring de este fichero."
)
