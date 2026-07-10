# BellNet: Iteración de Política Desenrollada mediante Filtros Polinómicos sobre Grafos

Código fuente del Trabajo Fin de Grado *«Unrolling de filtros polinómicos sobre grafos para iteración de política en programación dinámica»*, desarrollado por Miguel Alcocer Pérez en la Universidad Rey Juan Carlos (Escuela de Ingeniería de Fuenlabrada, Grado en Ciencia e Ingeniería de Datos, curso 2025/2026).

El repositorio implementa **Unrolled Policy Iteration (UPI)**: un método que obtiene, como identidad algebraica exacta, un filtro polinómico sobre grafos al aplicar el paradigma de *algorithm unrolling* al paso de evaluación de política. Se analizan dos variantes arquitectónicas del método, sus propiedades espectrales y su capacidad de generalización a entornos no vistos durante el entrenamiento.

---

## Publicaciones de referencia

El marco teórico y las dos arquitecturas implementadas fueron propuestos en los siguientes trabajos del grupo de investigación:

**[A]** S. Rozada, S. Rey, G. Mateos y A. G. Marques, «Unrolling Dynamic Programming via Graph Filters», *IEEE International Workshop on Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP)*, 2025.

**[B]** S. Rozada, S. Rey, **M. Alcocer Pérez**, G. Mateos y A. G. Marques, «Unrolled Policy Iteration Via Graph Filters», *NeurIPS 2025 Workshop: New Perspectives in Advancing Graph Machine Learning*, 2025.

```bibtex
@inproceedings{rozada2025camsap,
  author    = {Rozada, Sergio and Rey, Samuel and Mateos, Gonzalo and Marques, Antonio G.},
  title     = {Unrolling Dynamic Programming via Graph Filters},
  booktitle = {IEEE International Workshop on Computational Advances in
               Multi-Sensor Adaptive Processing (CAMSAP)},
  year      = {2025}
}

@inproceedings{rozada2025neurips,
  author    = {Rozada, Sergio and Rey, Samuel and {Alcocer P\'erez}, Miguel and
               Mateos, Gonzalo and Marques, Antonio G.},
  title     = {Unrolled Policy Iteration Via Graph Filters},
  booktitle = {39th Conference on Neural Information Processing Systems
               (NeurIPS 2025) Workshop: New Perspectives in Advancing
               Graph Machine Learning},
  year      = {2025}
}
```

---

## Estructura del proyecto

```
rl-unrolling/
├── src/                            # Biblioteca principal
│   ├── __init__.py
│   ├── environments.py             # CliffWalkingEnv, MirroredCliffWalkingEnv,
│   │                               #   GeneralizedCliffWalkingEnv, WindyGridWorldEnv
│   ├── models.py                   # PolicyEvaluationLayer (Arch. 1 y 2),
│   │                               #   PolicyImprovementLayer, UnrolledPolicyIterationModel
│   ├── plots.py                    # Funciones de visualización compartidas; ARCH_COLORS
│   ├── utils.py                    # get_optimal_q, compute_optimality_gap,
│   │                               #   eval_policy_extended, plot_errors
│   └── algorithms/
│       ├── generalized_policy_iteration.py   # PolicyIterationTrain (pl.LightningModule)
│       └── unrolling_policy_iteration.py     # UnrollingPolicyIterationTrain
│
├── experiments/
│   ├── unrolls_experiments_analysis.py  # Exp. 1 y 2: efecto de U y K (Pipeline B)
│   ├── k2_sweep_experiments.py          # Exp. 3: barrido del orden K₂ en Arq. 2
│   ├── cliff_variations.py              # Exp. 5: generalización estructural
│   ├── frequency_response_analysis.py   # Exp. 4: análisis de respuesta en frecuencia
│   ├── visualize_unrolls_results_tfg.py # Figuras del Cap. 4 (Exp. 1 y 2)
│   ├── visualize_cliff_variations.py    # Figuras del Cap. 4 (Exp. 5)
│   └── visualize_k2_sweep.py            # Figuras del Cap. 4 (Exp. 3)
│
├── regenerate_tfg_figures.py       # Orquestador: regenera todas las figuras del Cap. 4
├── main.py                         # Punto de entrada rápido (entrenamiento individual)
└── requirements.txt
```

---

## Instalación

```bash
git clone https://github.com/miguelalcocker/rl-unrolling.git
cd rl-unrolling
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Reproducción de experimentos

Todos los scripts se ejecutan desde la raíz del repositorio con el entorno virtual activado.

### Regenerar todas las figuras del TFG (Cap. 4) de una sola vez

```bash
python regenerate_tfg_figures.py
```

Lee los datos experimentales ya guardados y vuelca las figuras en `TFG/memoria/images/cap4/`. No requiere reentrenar los modelos.

---

### Experimento 1 y 2 — Influencia del número de unrolls U y del orden del filtro K

Corresponde a la Sección 4.2 de la memoria (apartado «Número de capas U y orden del filtro K»).

```bash
# Entrenar (todas las configuraciones de U y K)
python experiments/unrolls_experiments_analysis.py

# Solo algunos valores de num_unrolls
python experiments/unrolls_experiments_analysis.py --unrolls 5 10 15

# Visualizar resultados (figuras 6-métricas + mapas de política)
python experiments/visualize_unrolls_results_tfg.py
```

Resultados en `unrolls_results/`.

---

### Experimento 3 — Barrido del orden K₂ en Arquitectura 2

Corresponde a la Sección 4.2 de la memoria (apartado «Orden del filtro K₂ en la Arquitectura 2»).

```bash
# Entrenar
python experiments/k2_sweep_experiments.py

# Visualizar
python experiments/visualize_k2_sweep.py
```

Resultados en `k2_sweep_results/`.

---

### Experimento 4 — Análisis de respuesta en frecuencia

Corresponde a la Sección 4.2 de la memoria (apartado «Análisis espectral»). Lee coeficientes guardados en `freq_analysis_results/`; no requiere reentrenamiento.

```bash
python experiments/frequency_response_analysis.py
```

Figuras en `freq_analysis_results/` (`T_comparison.png`, `T_composition.png`).

---

### Experimento 5 — Generalización estructural (transferibilidad)

Corresponde a las Secciones 4.3–4.5 de la memoria. Compara el comportamiento de los filtros aprendidos en entornos con topología, escala y estocasticidad distintas.

```bash
# Entrenar todas las variantes
python experiments/cliff_variations.py

# Solo una variante
python experiments/cliff_variations.py --variants std_mirrored

# Visualizar (mapas de política, curvas de métricas, figura comprehensive)
python experiments/visualize_cliff_variations.py
```

Resultados en `cliff_variations_results/{variante}/`.

---

## Variantes arquitectónicas

### Arquitectura 1 — retroalimentación monomio (K+2 parámetros)

```
q̂ = Σ_{k=0}^{K} h_k P_π^k r  +  h_{K+1} P_π^{K+1} q_0
```

El término de retroalimentación es un único monomio de grado K+1. Su respuesta espectral decae naturalmente para |λ| < 1, lo que preserva el orden relativo de los Q-valores y favorece la convergencia con pocos unrolls.

### Arquitectura 2 — retroalimentación polinómica (2K+3 parámetros)

```
q̂ = Σ_{k=0}^{K} h_k P_π^k r  +  Σ_{k=0}^{K+1} w_k P_π^k q_0
```

El término de retroalimentación es un polinomio completo de grado K+1 con K+2 coeficientes libres w_k. Mayor expresividad que Arq. 1, pero requiere más unrolls para estabilizarse debido a la posible amplificación no uniforme de modos intermedios.

Ambas arquitecturas usan *weight sharing*: los mismos coeficientes (h_k y w_k) se reutilizan en todas las capas del unrolling.
