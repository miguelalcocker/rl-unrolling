"""
BellNet: Unrolling Dynamic Programming via Graph Filters

A research library implementing learnable policy iteration
using graph signal processing for efficient reinforcement
learning.
"""

from .environments import (
    CliffWalkingEnv,
    MirroredCliffWalkingEnv,
    GeneralizedCliffWalkingEnv,
    WindyGridWorldEnv,
    ChainMDP,
    RandomGraphMDP,
)
from .models import (
    PolicyEvaluationLayer,
    PolicyImprovementLayer,
    UnrolledPolicyIterationModel,
)
from .utils import (get_optimal_q, get_optimal_q_for_env,
                    compute_optimality_gap, compute_optimality_gap_V,
                    eval_policy_extended,
                    test_pol_err, plot_errors, save_error_matrix_to_csv)
from .plots import plot_policy_and_value, plot_Pi, plot_filter_coefs, ARCH_COLORS

__version__ = "1.0.0"
__author__ = "Sergio Rozada, Samuel Rey, Gonzalo Mateos, Antonio G. Marques; extended by Miguel Alcocer Pérez"

__all__ = [
    "CliffWalkingEnv",
    "MirroredCliffWalkingEnv",
    "GeneralizedCliffWalkingEnv",
    "WindyGridWorldEnv",
    "PolicyEvaluationLayer",
    "PolicyImprovementLayer",
    "UnrolledPolicyIterationModel",
    "get_optimal_q",
    "get_optimal_q_for_env",
    "compute_optimality_gap",
    "compute_optimality_gap_V",
    "eval_policy_extended",
    "test_pol_err",
    "plot_errors",
    "save_error_matrix_to_csv",
    "plot_policy_and_value",
    "plot_Pi",
    "plot_filter_coefs",
    "ARCH_COLORS",
]