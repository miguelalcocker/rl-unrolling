"""Neural network models for BellNet.

This module implements the core neural network components for unrolled
policy iteration including policy evaluation and improvement layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PolicyEvaluationLayer(nn.Module):
    """Policy evaluation layer using graph filters.
    
    This layer implements the policy evaluation step of policy iteration
    using graph filters to parameterize the Bellman operator.
    
    Args:
        P: Transition probability tensor of shape (nS * nA, nS)
        r: Reward tensor of shape (nS * nA,)
        nS: Number of states
        nA: Number of actions  
        K: Graph filter order
        beta: Bellman operator parameter
        shared_h: Optional shared filter coefficients
        architecture_type: Architecture type (1, 2, 3, or 5)
        K_2: For architecture 2, controls the range of the second summation (optional)
    """
    def __init__(self, P: torch.Tensor, r: torch.Tensor, nS: int, nA: int, 
                 K: int, beta: float, shared_h: Optional[nn.Parameter] = None, 
                 architecture_type: int = 1, use_legacy_init: bool = False, K_2: Optional[int] = None):
        super().__init__()
        self.nS = nS
        self.nA = nA
        self.K = K
        self.beta = beta
        self.architecture_type = architecture_type
        self.K_2 = K_2 if K_2 is not None else K + 1  # Default to original behavior

        self.register_buffer("P", P)  # shape: (nS * nA, nS)
        self.register_buffer("r", r)  # shape: (nS * nA,)

        if shared_h is None:
            # Architecture-specific coefficient sizes
            if architecture_type == 1:
                # Architecture 1: Σ(k=0 to K) h_k P^k r + h_{K+1} P^{K+1} q_0 → needs K+2 coefficients
                self.h = nn.Parameter(torch.randn(K + 2))
            elif architecture_type == 2:
                # Architecture 2: Use optimal parameter sizing for full expressivity
                # Total params needed: (K+1) for h_k + (K_2+1) for w_k = K + K_2 + 2
                total_params = K + self.K_2 + 2
                self.h = nn.Parameter(torch.randn(total_params))
            elif architecture_type == 3:
                # Architecture 3: Σ(k=0 to K) h_k P^k X → needs K+1 coefficients
                self.h = nn.Parameter(torch.randn(K + 1))
            else:
                # Default case (should not reach here for arch 5)
                self.h = nn.Parameter(torch.randn(K + 2))
            
            if use_legacy_init:
                # Original initialization: randn * 0.1
                self.h.data *= 0.1
            else:
                # Xavier initialization for h coefficients
                nn.init.xavier_uniform_(self.h.unsqueeze(0))
                self.h.data = self.h.data.squeeze(0)
        else:
            self.h = shared_h

        # Additional parameters for different architectures
        if architecture_type == 2:
            # Architecture 2: Now uses unified h vector, no separate w needed
            # The w parameters are taken from appropriate indices in h
            pass  # All parameters are in self.h
        elif architecture_type == 3:
            # Architecture 3: joint filter for concatenated [r; q_0]
            # h is already initialized above with correct size (K+1)
            pass
        elif architecture_type == 5:
            # Architecture 5: matrix filters H_k and final linear layer
            # Note: Weight sharing for Architecture 5 is complex and not theoretically justified
            d = 2
            if shared_h is None:  # No weight sharing
                self.H = nn.ParameterList([nn.Parameter(torch.randn(2, d)) for _ in range(K + 1)])
                self.w_final = nn.Parameter(torch.randn(d, 1))  # final linear weights
                if use_legacy_init:
                    for h_k in self.H:
                        h_k.data *= 0.1
                    self.w_final.data *= 0.1
                else:
                    for h_k in self.H:
                        nn.init.xavier_uniform_(h_k)
                    nn.init.xavier_uniform_(self.w_final)
            else:
                # Weight sharing for Arch 5 is not well-defined - disable it
                raise ValueError("Weight sharing is not supported for Architecture 5 due to its matrix-based nature")

    def lift_policy_matrix(self, Pi: torch.Tensor) -> torch.Tensor:
        """Lift policy matrix to state-action space.
        
        Args:
            Pi: Policy matrix of shape (nS, nA)
            
        Returns:
            Extended policy matrix of shape (nS, nS * nA)
        """
        Pi_ext = torch.zeros(self.nS, self.nS * self.nA, device=Pi.device)
        rows = torch.arange(self.nS, device=Pi.device).repeat_interleave(self.nA)
        cols = torch.arange(self.nS * self.nA, device=Pi.device)
        Pi_ext[rows, cols] = Pi.flatten()
        return Pi_ext

    def compute_transition_matrix(self, Pi: torch.Tensor) -> torch.Tensor:
        """Compute transition matrix under policy Pi.
        
        Args:
            Pi: Policy matrix of shape (nS, nA)
            
        Returns:
            Transition matrix P_π of shape (nS*nA, nS*nA)
        """
        Pi_ext = self.lift_policy_matrix(Pi)  # shape: (nS, nS*nA)
        return self.P @ Pi_ext  # shape: (nS*nA, nS*nA)

    def forward_architecture_1(self, q: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        """CORRECTED architecture 1: q̂ = Σ(k=0 to K) h_k P^k r + h_{K+1} P^{K+1} q_0"""
        P_pi = self.compute_transition_matrix(Pi)

        # Reward terms: Σ(k=0 to K) h_k P^k r
        q_prime = self.h[0] * self.r
        r_power = self.r.clone()
        for k in range(1, self.K + 1):
            r_power = P_pi @ r_power
            q_prime += self.h[k] * r_power

        # Q-value term: h_{K+1} P^{K+1} q_0 (CORRECTED: proper power)
        q_power = q.clone()
        for k in range(self.K + 1):  # CORRECTED: apply P exactly K+1 times
            q_power = P_pi @ q_power
        q_term = self.h[self.K + 1] * q_power

        return q_prime + self.beta * q_term

    def forward_architecture_2(self, q: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        """Architecture 2: q̂ = Σ(k=0 to K) h_k P^k r + Σ(k=K-K_2+1 to K+1) w_k P^k q_0"""
        P_pi = self.compute_transition_matrix(Pi)

        # Reward terms: Σ(k=0 to K) h_k P^k r
        q_prime = self.h[0] * self.r
        r_power = self.r.clone()
        for k in range(1, self.K + 1):
            r_power = P_pi @ r_power
            q_prime += self.h[k] * r_power

        # Q-value terms: Σ(k=K-K_2+1 to K+1) w_k P^k q_0
        # Using unified parameter vector: w parameters are in h[K+1:K+2+K_2]
        # But for compatibility and simplicity, we map them differently:
        # - For K_2=0: use h[K+1] (like Architecture 1)  
        # - For K_2>0: use remaining parameters from unified vector
        
        # First apply P^(K-K_2+1) times to q to get the starting power
        q_power = q.clone()
        k_start = self.K - self.K_2 + 1
        for k in range(k_start):
            q_power = P_pi @ q_power
        
        # Now sum from k=K-K_2+1 to k=K+1 using optimal parameter allocation
        # h[0:K+1] are for reward terms h_k, k=0..K
        # h[K+1:K+K_2+2] are for q-value terms w_k, mapped to k=K-K_2+1..K+1
        q_term = self.h[self.K + 1] * q_power  # First q-value term (k=K-K_2+1)
        for i in range(1, self.K_2 + 1):  # Remaining q-value terms
            q_power = P_pi @ q_power
            q_term += self.h[self.K + 1 + i] * q_power  # Unique parameter for each term

        return q_prime + self.beta * q_term

    def forward_architecture_3(self, q: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        """Architecture 3: q̂ = Σ(k=0 to K+1) h_k P^k X 1, with X = [r; q_0]"""
        P_pi = self.compute_transition_matrix(Pi)

        # Concatenate X = [r; q_0]
        X = torch.stack([self.r, q], dim=1)  # shape: (nS *nA, 2)

        ones = torch.ones(2, 1, device=X.device)

        # k=0
        result = self.h[0] * (X @ ones)  # shape: (nS * nA, 1)
        X_power = X.clone()

        # k=1..K+1
        for k in range(1, self.K + 2):
            X_power = P_pi @ X_power            # shape: (nS * nA, 2)
            result += self.h[k] * (X_power @ ones)

        return result.squeeze(-1)  # shape: (nS * nA,)

    def forward_architecture_5(self, q: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        """Architecture 5: Q̂ = Σ(k=0 to K+1) P^k X H_k, with X = [r; q_0], q̂ = σ(Q̂ w)"""
        P_pi = self.compute_transition_matrix(Pi)

        # Concatenate X = [r; q_0]
        X = torch.stack([self.r, q], dim=1)  # shape: (nS * nA, 2)

        # k=0
        Q_hat = X @ self.H[0]  # (nS * nA, d)
        X_power = X.clone()

        # k=1..K+1
        for k in range(1, self.K + 2):
            X_power = P_pi @ X_power          # (nS * nA, 2)
            Q_hat += X_power @ self.H[k]      # (nS * nA, d)

        # q̂ = σ(Q̂ w)
        q_prime = torch.sigmoid(Q_hat @ self.w_final).squeeze(-1)  # (nS * nA,)
        return q_prime

    def forward(self, q: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        """Forward pass for policy evaluation.
        
        Args:
            q: Current Q-values of shape (nS * nA,)
            Pi: Current policy of shape (nS, nA)
            
        Returns:
            Updated Q-values of shape (nS * nA,)
        """
        if self.architecture_type == 1:
            return self.forward_architecture_1(q, Pi)
        elif self.architecture_type == 2:
            return self.forward_architecture_2(q, Pi)
        elif self.architecture_type == 3:
            return self.forward_architecture_3(q, Pi)
        elif self.architecture_type == 5:
            return self.forward_architecture_5(q, Pi)
        else:
            raise ValueError(f"Unsupported architecture type: {self.architecture_type}")


class PolicyImprovementLayer(nn.Module):
    """Policy improvement layer using softmax.
    
    This layer implements the policy improvement step using softmax
    with temperature scaling.
    
    Args:
        nS: Number of states
        nA: Number of actions
        tau: Temperature parameter for softmax
    """
    def __init__(self, nS: int, nA: int, tau: float):
        super().__init__()
        self.nS = nS
        self.nA = nA
        self.tau = tau

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Forward pass for policy improvement.
        
        Args:
            q: Q-values of shape (nS * nA,)
            
        Returns:
            Improved policy of shape (nS, nA)
        """
        q_reshaped = q.view(self.nS, self.nA)
        Pi = F.softmax(q_reshaped / self.tau, dim=1)
        return Pi


class UnrolledPolicyIterationModel(nn.Module):
    """Unrolled policy iteration neural network.
    
    This model implements unrolled policy iteration by stacking
    alternating policy evaluation and improvement layers.
    
    Args:
        P: Transition probability tensor of shape (nS * nA, nS)
        r: Reward tensor of shape (nS * nA,)
        nS: Number of states
        nA: Number of actions
        K: Graph filter order
        num_unrolls: Number of unrolling steps
        tau: Temperature parameter
        beta: Bellman operator parameter  
        weight_sharing: Whether to share weights across layers
        architecture_type: Architecture type (1, 2, 3, or 5)
        use_residual: Whether to use residual connections
        K_2: For architecture 2, controls the range of the second summation (optional)
    """
    def __init__(self, P: torch.Tensor, r: torch.Tensor, nS: int, nA: int, 
                 K: int = 3, num_unrolls: int = 5, tau: float = 1, 
                 beta: float = 1.0, weight_sharing: bool = False,
                 architecture_type: int = 1, use_residual: bool = False, 
                 use_legacy_init: bool = False, K_2: Optional[int] = None):
        super().__init__()
        self.nS = nS
        self.nA = nA
        self.use_residual = use_residual

        if weight_sharing:
            # Check if weight sharing is compatible with architecture
            if architecture_type == 5:
                raise ValueError("Weight sharing is not supported for Architecture 5 due to its matrix-based nature")
            
            # Architecture-specific shared coefficient sizes
            if architecture_type == 1:
                # Architecture 1: needs K+2 shared coefficients
                self.h = nn.Parameter(torch.randn(K + 2))
            elif architecture_type == 2:
                # Architecture 2: needs optimal sizing for full expressivity  
                K_2_param = K_2 if K_2 is not None else K + 1  # Default reasonable K_2
                total_params = K + K_2_param + 2
                self.h = nn.Parameter(torch.randn(total_params))
            elif architecture_type == 3:
                # Architecture 3: needs K+1 shared coefficients
                self.h = nn.Parameter(torch.randn(K + 1))
            else:
                # Default fallback
                self.h = nn.Parameter(torch.randn(K + 1))
                
            if use_legacy_init:
                self.h.data *= 0.1
            else:
                # Xavier initialization for shared parameters
                nn.init.xavier_uniform_(self.h.unsqueeze(0))
                self.h.data = self.h.data.squeeze(0)
            
            # For Architecture 2, w parameters are now part of unified h vector
            # No separate w parameter needed
            self.w = None
        else:
            self.h = None
            self.w = None

        self.layers = nn.ModuleList()
        for _ in range(num_unrolls):
            layer = PolicyEvaluationLayer(P, r, nS, nA, K, beta, self.h, architecture_type, use_legacy_init, K_2)
            # Architecture 2 now uses unified h vector, no separate w parameter needed
            self.layers.append(layer)
            self.layers.append(PolicyImprovementLayer(nS, nA, tau))

    def forward(self, q_init: torch.Tensor, Pi_init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through unrolled policy iteration.
        
        Args:
            q_init: Initial Q-values of shape (nS * nA,)
            Pi_init: Initial policy of shape (nS, nA)
            
        Returns:
            Tuple of (final_q_values, final_policy)
        """
        q = q_init.squeeze()
        Pi = Pi_init
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, PolicyEvaluationLayer):
                q_new = layer(q, Pi)
                # Apply residual connection if enabled (only for policy evaluation layers)
                if self.use_residual and i > 0:
                    q = q_new + q
                else:
                    q = q_new
            elif isinstance(layer, PolicyImprovementLayer):
                Pi = layer(q)
        return q, Pi
