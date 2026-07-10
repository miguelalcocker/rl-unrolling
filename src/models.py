"""Neural network models for UPI (Unrolled Policy Iteration).

Implements two architecture variants of the graph-filter policy evaluation layer:

  Architecture 1  — single monomial feedback term  h_{K+1} P^{K+1} q_0.
  Architecture 2  — polynomial feedback of degree K_2  Σ w_k P^k q_0.

Both share the same reward polynomial  Σ h_k P^k r  (k = 0..K).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PolicyEvaluationLayer(nn.Module):
    """One policy-evaluation step implemented as a trainable graph filter.

    Args:
        P: Transition matrix (nS*nA, nS).
        r: Reward vector (nS*nA,).
        nS, nA: State/action counts.
        K: Filter order (reward polynomial degree).
        beta: Scaling factor on the feedback term.
        shared_h: Pre-created shared parameter (weight-sharing mode).
        architecture_type: 1 or 2.
        use_legacy_init: If True use randn*0.1 (reproduces original paper results);
                         otherwise Xavier uniform.
        K_2: Feedback polynomial degree for Architecture 2 (default K+1).
    """

    def __init__(self, P: torch.Tensor, r: torch.Tensor, nS: int, nA: int,
                 K: int, beta: float, shared_h: Optional[nn.Parameter] = None,
                 architecture_type: int = 1, use_legacy_init: bool = False,
                 K_2: Optional[int] = None):
        super().__init__()
        self.nS = nS
        self.nA = nA
        self.K = K
        self.beta = beta
        self.architecture_type = architecture_type
        self.K_2 = K_2 if K_2 is not None else K + 1

        self.register_buffer("P", P)
        self.register_buffer("r", r)

        if shared_h is None:
            if architecture_type == 1:
                self.h = nn.Parameter(torch.randn(K + 2))
            elif architecture_type == 2:
                self.h = nn.Parameter(torch.randn(K + self.K_2 + 2))
            else:
                raise ValueError(f"Unsupported architecture_type={architecture_type}. Use 1 or 2.")

            if use_legacy_init:
                self.h.data *= 0.1
            else:
                nn.init.xavier_uniform_(self.h.unsqueeze(0))
                self.h.data = self.h.data.squeeze(0)
        else:
            self.h = shared_h

    def lift_policy_matrix(self, Pi: torch.Tensor) -> torch.Tensor:
        """Lift policy matrix to state-action space."""
        Pi_ext = torch.zeros(self.nS, self.nS * self.nA, device=Pi.device)
        rows = torch.arange(self.nS, device=Pi.device).repeat_interleave(self.nA)
        cols = torch.arange(self.nS * self.nA, device=Pi.device)
        Pi_ext[rows, cols] = Pi.flatten()
        return Pi_ext

    def compute_transition_matrix(self, Pi: torch.Tensor) -> torch.Tensor:
        """Compute P_π = P · Π_ext  of shape (nS*nA, nS*nA)."""
        return self.P @ self.lift_policy_matrix(Pi)

    def forward_architecture_1(self, q: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        """q̂ = Σ_{k=0}^{K} h_k P_π^k r  +  h_{K+1} P_π^{K+1} q_0"""
        P_pi = self.compute_transition_matrix(Pi)

        q_prime = self.h[0] * self.r
        r_power = self.r.clone()
        for k in range(1, self.K + 1):
            r_power = P_pi @ r_power
            q_prime = q_prime + self.h[k] * r_power

        q_power = q.clone()
        for _ in range(self.K + 1):
            q_power = P_pi @ q_power
        return q_prime + self.beta * self.h[self.K + 1] * q_power

    def forward_architecture_2(self, q: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        """q̂ = Σ_{k=0}^{K} h_k P_π^k r  +  Σ_{k=K-K_2+1}^{K+1} w_k P_π^k q_0"""
        P_pi = self.compute_transition_matrix(Pi)

        q_prime = self.h[0] * self.r
        r_power = self.r.clone()
        for k in range(1, self.K + 1):
            r_power = P_pi @ r_power
            q_prime = q_prime + self.h[k] * r_power

        q_power = q.clone()
        k_start = self.K - self.K_2 + 1
        for _ in range(k_start):
            q_power = P_pi @ q_power

        q_term = self.h[self.K + 1] * q_power
        for i in range(1, self.K_2 + 1):
            q_power = P_pi @ q_power
            q_term = q_term + self.h[self.K + 1 + i] * q_power

        return q_prime + self.beta * q_term

    def forward(self, q: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        if self.architecture_type == 1:
            return self.forward_architecture_1(q, Pi)
        elif self.architecture_type == 2:
            return self.forward_architecture_2(q, Pi)
        else:
            raise ValueError(f"Unsupported architecture_type={self.architecture_type}")


class PolicyImprovementLayer(nn.Module):
    """Softmax policy improvement with temperature τ."""

    def __init__(self, nS: int, nA: int, tau: float):
        super().__init__()
        self.nS = nS
        self.nA = nA
        self.tau = tau

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return F.softmax(q.view(self.nS, self.nA) / self.tau, dim=1)


class UnrolledPolicyIterationModel(nn.Module):
    """Full UPI network: alternating PolicyEvaluation / PolicyImprovement layers.

    Args:
        P, r, nS, nA: Environment tensors and dimensions.
        K: Filter order.
        num_unrolls: Number of (eval + improve) pairs.
        tau: Softmax temperature.
        beta: Feedback scaling.
        weight_sharing: Share h coefficients across evaluation layers.
        architecture_type: 1 or 2.
        use_legacy_init: Initialization strategy (see PolicyEvaluationLayer).
        K_2: Feedback degree for Architecture 2.
    """

    def __init__(self, P: torch.Tensor, r: torch.Tensor, nS: int, nA: int,
                 K: int = 3, num_unrolls: int = 5, tau: float = 1,
                 beta: float = 1.0, weight_sharing: bool = False,
                 architecture_type: int = 1, use_legacy_init: bool = False,
                 K_2: Optional[int] = None):
        super().__init__()
        self.nS = nS
        self.nA = nA

        if weight_sharing:
            if architecture_type == 1:
                self.h = nn.Parameter(torch.randn(K + 2))
            elif architecture_type == 2:
                K_2_eff = K_2 if K_2 is not None else K + 1
                self.h = nn.Parameter(torch.randn(K + K_2_eff + 2))
            else:
                raise ValueError(f"Unsupported architecture_type={architecture_type}")

            if use_legacy_init:
                self.h.data *= 0.1
            else:
                nn.init.xavier_uniform_(self.h.unsqueeze(0))
                self.h.data = self.h.data.squeeze(0)
        else:
            self.h = None

        self.layers = nn.ModuleList()
        for _ in range(num_unrolls):
            self.layers.append(
                PolicyEvaluationLayer(P, r, nS, nA, K, beta, self.h,
                                      architecture_type, use_legacy_init, K_2)
            )
            self.layers.append(PolicyImprovementLayer(nS, nA, tau))

    def forward(self, q_init: torch.Tensor,
                Pi_init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = q_init.squeeze()
        Pi = Pi_init
        for layer in self.layers:
            if isinstance(layer, PolicyEvaluationLayer):
                q = layer(q, Pi)
            else:
                Pi = layer(q)
        return q, Pi
