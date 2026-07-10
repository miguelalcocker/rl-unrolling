"""Environment implementations for BellNet.

This module contains custom environment implementations:

  Legacy (backward-compatible wrappers around gymnasium):
    - CliffWalkingEnv          Standard 4×12 CliffWalking, modified goal.
    - MirroredCliffWalkingEnv  Cliff moved to top row.

  New (built from scratch, no gymnasium dependency):
    - GeneralizedCliffWalkingEnv  Fully configurable grid-world with cliff hazards.
    - WindyGridWorldEnv           Grid world with column-wise deterministic wind.
    - ChainMDP                    1-D chain MDP for spectral analysis and scaling.

All environments expose the same interface:
    nS  (int)            Number of states.
    nA  (int)            Number of actions.
    P   (Tensor nS*nA×nS) Transition probability matrix.
    r   (Tensor nS*nA)    Expected immediate reward vector.
"""

import gymnasium as gym
import torch
from typing import Dict, List, Optional, Tuple


# ─── Action constants (shared by all grid environments) ──────────────────────
ACT_UP    = 0
ACT_RIGHT = 1
ACT_DOWN  = 2
ACT_LEFT  = 3


# ─────────────────────────────────────────────────────────────────────────────
# Legacy environments  (kept for full backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class CliffWalkingEnv:
    """Standard CliffWalking environment with modified goal state.

    This environment wraps the OpenAI Gym CliffWalking environment
    and modifies the goal state to be absorbing with zero reward.

    Attributes:
        nS: Number of states
        nA: Number of actions
        P: Transition probability tensor of shape (nS * nA, nS)
        r: Reward tensor of shape (nS * nA,)
    """
    def __init__(self) -> None:
        self.env = gym.make("CliffWalking-v1", render_mode="rgb_array").unwrapped
        self.nS = self.env.observation_space.n
        self.nA = self.env.action_space.n

        goal_state = 47
        for a in range(self.nA):
            self.env.P[goal_state][a] = [(1.0, goal_state, 0.0, True)]

        self.P = torch.zeros(self.nS * self.nA, self.nS)
        self.r = torch.zeros(self.nS * self.nA)
        for s in range(self.nS):
            for a in range(self.nA):
                idx = s * self.nA + a
                for prob, next_s, reward, done in self.env.P[s][a]:
                    self.P[idx, next_s] += prob
                    self.r[idx] += prob * reward

    def reset(self) -> int:
        """Reset environment to initial state.

        Returns:
            Initial state
        """
        return self.env.reset()

    def step(self, action: int) -> tuple:
        """Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        return self.env.step(action)

    def render(self):
        """Render the environment.

        Returns:
            RGB array of the rendered environment
        """
        return self.env.render()

    def close(self) -> None:
        """Close the environment."""
        self.env.close()


class MirroredCliffWalkingEnv:
    """Modified CliffWalking environment with mirrored cliff.

    This environment modifies the standard CliffWalking by:
    1. Removing the cliff from the bottom row
    2. Adding a cliff to the top row (mirrored)
    3. Changing start/goal positions accordingly

    The cliff is now in the top row (states 1-10) instead of bottom row.
    Start state is at (0,0) and goal state is at (0,11).

    Attributes:
        nS: Number of states
        nA: Number of actions
        P: Transition probability tensor of shape (nS * nA, nS)
        r: Reward tensor of shape (nS * nA,)
        start_state: Starting state (top-left)
        goal_state: Goal state (top-right)
    """
    def __init__(self) -> None:
        self.env = gym.make("CliffWalking-v1", render_mode="rgb_array").unwrapped
        self.nS = self.env.observation_space.n  # Number of states
        self.nA = self.env.action_space.n       # Number of actions

        cliff_states = [c for c in range(1, 11)]

        self.start_state = 0        # Top-left corner (0, 0)
        self.goal_state = 11        # Top-right corner (0, 11)

        # Make goal state absorbing with 0 reward
        for a in range(self.nA):
            self.env.P[self.goal_state][a] = [(1.0, self.goal_state, 0.0, True)]

        # Remove the original cliff in the bottom row (row 3, columns 1 to 10)
        ## Removing transitions from last row
        for col in range(0, 12):
            state = 3 * 12 + col  # row 3, col = col
            for a in range(self.nA):
                # Compute geometric next state directly (do NOT read from env.P,
                # which already has cliff redirects embedded by gym)
                if a == ACT_UP:
                    next_state = 2 * 12 + col          # move to row 2
                elif a == ACT_RIGHT:
                    next_state = 3 * 12 + min(col + 1, 11)  # boundary at col 11
                elif a == ACT_DOWN:
                    next_state = 3 * 12 + col          # stay (bottom boundary)
                else:  # ACT_LEFT
                    next_state = 3 * 12 + max(col - 1, 0)   # boundary at col 0
                self.env.P[state][a] = [(1.0, next_state, -1.0, False)]

        ## Removing transitions from states above the cliff
        for col in range(1, 11):
            state_above_cliff = 2 * 12 + col  # row 2, col = col → states 25 to 34
            for i, (prob, next_s, reward, done) in enumerate(self.env.P[state_above_cliff][ACT_DOWN]):
                if reward == -100.0:
                    # Compute geometric next state directly (do NOT use next_s from gym,
                    # which already contains the cliff redirect to s=36, not 3*12+col)
                    next_state = state_above_cliff + 12  # = 3*12 + col (row 3, same column)
                    self.env.P[state_above_cliff][ACT_DOWN][i] = (1.0, next_state, -1.0, False)

        # Add mirrored cliff in the top row (row 0, columns 1 to 10)
        for col in range(0, 11):
            from_state = col
            for a in range(self.nA):
                for i, (prob, next_s, reward, done) in enumerate(self.env.P[from_state][a]):
                    if next_s in cliff_states:
                        self.env.P[from_state][a][i] = (1.0, self.start_state, -100.0, False)
                    elif next_s == self.goal_state:
                        self.env.P[from_state][a][i] = (1.0, self.goal_state, -1, True)

        # Modify transitions INTO the new cliff from below
        for col in range(1, 12):
            from_state = 1 * 12 + col  # row 1
            for i, (prob, next_s, reward, done) in enumerate(self.env.P[from_state][ACT_UP]):
                if next_s in cliff_states:
                    self.env.P[from_state][ACT_UP][i] = (1.0, self.start_state, -100.0, False)
                elif next_s == self.goal_state:
                    self.env.P[from_state][ACT_UP][i] = (1.0, self.goal_state, -1, True)

        # Convert transition dynamics and rewards into torch tensors
        self.P = torch.zeros(self.nS * self.nA, self.nS)
        self.r = torch.zeros(self.nS * self.nA)
        for s in range(self.nS):
            for a in range(self.nA):
                idx = s * self.nA + a
                for prob, next_s, reward, done in self.env.P[s][a]:
                    self.P[idx, next_s] += prob
                    self.r[idx] += prob * reward


    def reset(self) -> int:
        """Reset environment to initial state.

        Returns:
            Initial state
        """
        self.env.s = self.start_state
        self.env.lastaction = None  # inicializar
        return self.env.s

    def step(self, action: int) -> tuple:
        """Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.env.lastaction = action
        return self.env.step(action)

    def render(self):
        """Render the environment.

        Returns:
            RGB array of the rendered environment
        """
        return self.env.render()

    def close(self) -> None:
        """Close the environment."""
        self.env.close()


# ─────────────────────────────────────────────────────────────────────────────
# GeneralizedCliffWalkingEnv
# ─────────────────────────────────────────────────────────────────────────────

class GeneralizedCliffWalkingEnv:
    """Highly configurable grid-world with cliff hazards.

    Generalizes CliffWalkingEnv and MirroredCliffWalkingEnv.
    Built entirely from scratch – no gymnasium dependency.

    Grid layout
    -----------
    Rows   : 0 (top) … nrows-1 (bottom).
    Columns: 0 (left) … ncols-1 (right).
    State  : s = row * ncols + col.
    Actions: UP=0, RIGHT=1, DOWN=2, LEFT=3.

    Dynamics
    --------
    Normal step   → next_s,            reward = step_reward.
    Boundary hit  → stay (same state), reward = step_reward.
    Cliff hit     → cliff_teleport,    reward = cliff_reward.
    At goal state → absorbing self-loop, reward = 0.

    The goal reward on ARRIVAL (last step before absorbing) equals step_reward,
    consistent with CliffWalkingEnv / MirroredCliffWalkingEnv.

    Parameters
    ----------
    nrows, ncols : Grid dimensions.
    cliff_cells  : List of (row, col) tuples that are cliff cells.
                   Default: entire bottom row except start and goal corners.
    start        : (row, col) start position. Default: (nrows-1, 0).
    goal         : (row, col) goal position.  Default: (nrows-1, ncols-1).
    cliff_reward : Reward for stepping into a cliff.  Default: -100.
    step_reward  : Reward for every other transition.  Default: -1.
    cliff_teleport : (row, col) to land on after hitting a cliff.
                     Defaults to start.

    Examples
    --------
    Standard 4×12 (equivalent to CliffWalkingEnv):
        env = GeneralizedCliffWalkingEnv.standard()

    Mirrored 4×12 (equivalent to MirroredCliffWalkingEnv):
        env = GeneralizedCliffWalkingEnv.mirrored()

    Canonical train/test pair:
        train_env, test_env = GeneralizedCliffWalkingEnv.make_train_test_pair()
    """

    def __init__(
        self,
        nrows: int = 4,
        ncols: int = 12,
        cliff_cells: Optional[List[Tuple[int, int]]] = None,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
        cliff_reward: float = -100.0,
        step_reward: float = -1.0,
        cliff_teleport: Optional[Tuple[int, int]] = None,
        wind_map: Optional[Dict[int, Tuple[float, int]]] = None,
        isotropic_wind_p: float = 0.0,
    ) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.nS    = nrows * ncols
        self.nA    = 4

        # ── Defaults ──────────────────────────────────────────────────────────
        self.start = tuple(start) if start is not None else (nrows - 1, 0)
        self.goal  = tuple(goal)  if goal  is not None else (nrows - 1, ncols - 1)

        if cliff_cells is None:
            cliff_cells = [(nrows - 1, c) for c in range(1, ncols - 1)]
        self.cliff_cells = list(cliff_cells)

        self.cliff_teleport = (
            tuple(cliff_teleport) if cliff_teleport is not None else self.start
        )
        self.cliff_reward = cliff_reward
        self.step_reward  = step_reward
        self.wind_map         = wind_map or {}
        self.isotropic_wind_p = float(isotropic_wind_p)

        # ── Derived indices ───────────────────────────────────────────────────
        self.start_state    = self.start[0]          * ncols + self.start[1]
        self.goal_state     = self.goal[0]           * ncols + self.goal[1]
        self.teleport_state = self.cliff_teleport[0] * ncols + self.cliff_teleport[1]
        self.cliff_set      = frozenset(r * ncols + c for r, c in self.cliff_cells)

        # ── Validation ────────────────────────────────────────────────────────
        if self.goal_state in self.cliff_set:
            raise ValueError("Goal cell cannot be a cliff cell.")
        if self.start_state in self.cliff_set:
            raise ValueError("Start cell cannot be a cliff cell.")
        if self.teleport_state in self.cliff_set:
            raise ValueError("Cliff teleport target cannot be a cliff cell.")

        self.P, self.r = self._build_dynamics()

    # ── Dynamics builder ──────────────────────────────────────────────────────

    def _build_dynamics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        nS, nA = self.nS, self.nA
        ncols, nrows = self.ncols, self.nrows
        P = torch.zeros(nS * nA, nS)
        r = torch.zeros(nS * nA)

        for s in range(nS):
            row, col = divmod(s, ncols)

            for a in range(nA):
                idx = s * nA + a

                # Absorbing goal: self-loop with zero reward.
                if s == self.goal_state:
                    P[idx, s] = 1.0
                    # r[idx] remains 0.0
                    continue

                # Intended next position with boundary clamping.
                if a == ACT_UP:
                    nr, nc = max(row - 1, 0), col
                elif a == ACT_RIGHT:
                    nr, nc = row, min(col + 1, ncols - 1)
                elif a == ACT_DOWN:
                    nr, nc = min(row + 1, nrows - 1), col
                else:  # ACT_LEFT
                    nr, nc = row, max(col - 1, 0)

                next_s = nr * ncols + nc

                if next_s in self.cliff_set:
                    P[idx, self.teleport_state] = 1.0
                    r[idx] = self.cliff_reward
                else:
                    P[idx, next_s] = 1.0
                    r[idx] = self.step_reward

        # Wind blending: for each cell with wind, mix its action rows with the
        # wind-direction row — equivalent to "intended action replaced by wind
        # direction with probability p_wind[s]".
        if self.wind_map:
            P_base = P.clone()
            r_base = r.clone()
            for s, (p_wind, wind_dir) in self.wind_map.items():
                if p_wind <= 0.0 or s == self.goal_state:
                    continue
                for a in range(nA):
                    idx_a = s * nA + a
                    idx_w = s * nA + wind_dir
                    P[idx_a] = (1.0 - p_wind) * P_base[idx_a] + p_wind * P_base[idx_w]
                    r[idx_a] = (1.0 - p_wind) * r_base[idx_a] + p_wind * r_base[idx_w]

        # Isotropic wind: with probability p, intended action is replaced by a
        # uniformly random action over all 4 directions.
        # P(s,a) = (1-p)*P_base(s,a) + (p/nA)*Σ_d P_base(s,d)
        if self.isotropic_wind_p > 0.0:
            p = self.isotropic_wind_p
            P_base = P.clone()
            r_base = r.clone()
            for s in range(nS):
                if s == self.goal_state:
                    continue
                mean_P = P_base[s * nA:(s + 1) * nA].mean(dim=0)
                mean_r = r_base[s * nA:(s + 1) * nA].mean(dim=0)
                for a in range(nA):
                    idx_a = s * nA + a
                    P[idx_a] = (1.0 - p) * P_base[idx_a] + p * mean_P
                    r[idx_a] = (1.0 - p) * r_base[idx_a] + p * mean_r

        return P, r

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def standard(cls) -> "GeneralizedCliffWalkingEnv":
        """Standard 4×12 cliff (bottom row, cols 1-10).

        Produces the same P, r as CliffWalkingEnv.
        """
        return cls(
            nrows=4, ncols=12,
            cliff_cells=[(3, c) for c in range(1, 11)],
            start=(3, 0), goal=(3, 11),
        )

    @classmethod
    def mirrored(cls) -> "GeneralizedCliffWalkingEnv":
        """Mirrored 4×12 cliff (top row, cols 1-10).

        Produces the same P, r as MirroredCliffWalkingEnv.
        """
        return cls(
            nrows=4, ncols=12,
            cliff_cells=[(0, c) for c in range(1, 11)],
            start=(0, 0), goal=(0, 11),
        )

    @classmethod
    def scaled(
        cls,
        scale: int = 2,
        mirrored: bool = False,
        **kwargs,
    ) -> "GeneralizedCliffWalkingEnv":
        """Scaled version of the standard grid (scale × 4 rows, scale × 12 cols).

        Keeps the same relative structure (cliff fills the bottom/top row except
        corners).  Useful for testing whether learned polynomial filters transfer
        to larger state spaces.

        Parameters
        ----------
        scale    : Multiplier applied to both dimensions (default 2 → 8×24).
        mirrored : If True, cliff is on the top row instead of bottom.
        """
        nrows, ncols = 4 * scale, 12 * scale
        if mirrored:
            return cls(
                nrows=nrows, ncols=ncols,
                cliff_cells=[(0, c) for c in range(1, ncols - 1)],
                start=(0, 0), goal=(0, ncols - 1),
                **kwargs,
            )
        return cls(
            nrows=nrows, ncols=ncols,
            cliff_cells=[(nrows - 1, c) for c in range(1, ncols - 1)],
            start=(nrows - 1, 0), goal=(nrows - 1, ncols - 1),
            **kwargs,
        )

    @classmethod
    def center_goal(cls) -> "GeneralizedCliffWalkingEnv":
        """Standard 4×12 cliff but with goal at the grid center (2, 6).

        Cliff stays at bottom row (cols 1-10). Start/teleport at (3, 0).
        Goal at row 2, col 6 — the interior center of the 4×12 grid.
        """
        return cls(
            nrows=4, ncols=12,
            cliff_cells=[(3, c) for c in range(1, 11)],
            start=(3, 0), goal=(2, 6),
        )

    @classmethod
    def center_start(cls) -> "GeneralizedCliffWalkingEnv":
        """Standard 4×12 cliff but with start (and cliff teleport) at grid center.

        Cliff stays at bottom row (cols 1-10). Goal at (3,11).
        Start/teleport: (1, 6) — row 1, col 6 (center of the grid).
        Tests whether the learned operator handles a non-corner restart point.
        """
        return cls(
            nrows=4, ncols=12,
            cliff_cells=[(3, c) for c in range(1, 11)],
            start=(1, 6), goal=(3, 11),
        )

    @classmethod
    def vertical_cliff(cls) -> "GeneralizedCliffWalkingEnv":
        """4×12 grid with a vertical cliff strip in the middle column.

        Forces the agent to navigate around a vertical barrier.
        Train: standard bottom cliff.  Test variant: use this configuration.
        Cliff: rows 1-2, col 6.  Start: bottom-left.  Goal: top-right.
        """
        return cls(
            nrows=4, ncols=12,
            cliff_cells=[(r, 6) for r in range(1, 3)],
            start=(3, 0), goal=(0, 11),
        )

    @classmethod
    def standard_windy(
        cls,
        rng_seed: int = 42,
        p_wind: float = 0.25,
    ) -> "GeneralizedCliffWalkingEnv":
        """Standard 4×12 cliff + per-cell random stochastic wind.

        Each non-goal cell gets an independently sampled wind direction
        (UP/RIGHT/DOWN/LEFT) and wind probability uniformly in [0, 2·p_wind].
        Wind model: with probability p_wind[s], the intended action is replaced
        by the wind direction assigned to that cell.  Mean probability = p_wind.

        Parameters
        ----------
        rng_seed : Seed for reproducible wind assignment.
        p_wind   : Mean per-cell wind probability (default 0.25).
        """
        import numpy as _np
        rng      = _np.random.default_rng(rng_seed)
        goal_s   = 3 * 12 + 11          # standard goal: row 3, col 11
        wind_map: Dict[int, Tuple[float, int]] = {}
        for s in range(4 * 12):
            if s == goal_s:
                continue
            prob      = float(rng.uniform(0.0, 2.0 * p_wind))
            direction = int(rng.integers(0, 4))
            wind_map[s] = (prob, direction)
        return cls(
            nrows=4, ncols=12,
            cliff_cells=[(3, c) for c in range(1, 11)],
            start=(3, 0), goal=(3, 11),
            wind_map=wind_map,
        )

    @classmethod
    def standard_uniform_wind(
        cls,
        p_wind: float = 0.01,
        wind_dir: int = ACT_DOWN,
    ) -> "GeneralizedCliffWalkingEnv":
        """Standard 4×12 cliff + uniform stochastic wind in a single direction.

        Every non-goal cell has the same wind direction and the same probability
        p_wind.  With probability p_wind the agent's intended action is replaced
        by wind_dir; otherwise the intended action executes.  This produces a
        P matrix with a clean, homogeneous stochastic structure that is
        analytically tractable — unlike standard_windy whose per-cell random
        assignment mixes stochasticity with heterogeneity.

        The optimal policy transitions from the 'risky' row-2 path to safer
        higher rows as p_wind increases.  The break-even point is approximately
        p_wind ≈ 0.002 for the default cliff/step rewards (-100/-1, γ=0.99).

        Parameters
        ----------
        p_wind   : Probability that the intended action is replaced by wind_dir.
        wind_dir : Wind action (default: ACT_DOWN=2 — downward push).
        """
        goal_s   = 3 * 12 + 11
        wind_map: Dict[int, Tuple[float, int]] = {}
        for s in range(4 * 12):
            if s != goal_s:
                wind_map[s] = (p_wind, wind_dir)
        return cls(
            nrows=4, ncols=12,
            cliff_cells=[(3, c) for c in range(1, 11)],
            start=(3, 0), goal=(3, 11),
            wind_map=wind_map,
        )

    @classmethod
    def standard_isotropic_wind(
        cls,
        p_wind: float = 0.01,
    ) -> "GeneralizedCliffWalkingEnv":
        """Standard 4×12 cliff + isotropic stochastic wind in all 4 directions.

        With probability p_wind the agent's intended action is replaced by a
        uniformly random action (UP/RIGHT/DOWN/LEFT equally likely).  This adds
        symmetric, direction-neutral noise — unlike standard_uniform_wind which
        has a directional bias.

        Transition: P(s,a) = (1-p)*P_base(s,a) + (p/4)*Σ_d P_base(s,d)

        Parameters
        ----------
        p_wind : Probability that the intended action is replaced by a random one.
        """
        return cls(
            nrows=4, ncols=12,
            cliff_cells=[(3, c) for c in range(1, 11)],
            start=(3, 0), goal=(3, 11),
            isotropic_wind_p=p_wind,
        )

    @classmethod
    def double_cliff(cls) -> "GeneralizedCliffWalkingEnv":
        """4×12 grid with cliffs on BOTH bottom and top row interior cells.

        The agent must navigate through a narrow interior corridor.
        Start: (1, 0).  Goal: (1, 11).
        """
        cliff_cells = (
            [(3, c) for c in range(1, 11)]
            + [(0, c) for c in range(1, 11)]
        )
        return cls(
            nrows=4, ncols=12,
            cliff_cells=cliff_cells,
            start=(1, 0), goal=(1, 11),
        )

    @classmethod
    def l_shaped_cliff(cls) -> "GeneralizedCliffWalkingEnv":
        """4×12 grid with an L-shaped cliff (bottom row + left column interior).

        Introduces a qualitatively different topology from the standard case.
        Start: (3, 0) shifted right to (3, 1) to avoid cliff.  Goal: (0, 11).

        Note: start and goal are in safe cells.
        """
        cliff_cells = (
            [(3, c) for c in range(2, 11)]       # bottom row (partial)
            + [(r, 0) for r in range(1, 3)]       # left column interior
        )
        return cls(
            nrows=4, ncols=12,
            cliff_cells=cliff_cells,
            start=(3, 11), goal=(0, 11),
        )

    # ── Train / test pair factories ───────────────────────────────────────────

    @classmethod
    def make_train_test_pair(
        cls,
        nrows: int = 4,
        ncols: int = 12,
        cliff_reward: float = -100.0,
        step_reward: float = -1.0,
    ) -> Tuple["GeneralizedCliffWalkingEnv", "GeneralizedCliffWalkingEnv"]:
        """Canonical train/test pair.

        train_env : cliff on bottom row (standard CliffWalking).
        test_env  : cliff on top row    (mirrored CliffWalking).

        This is the pair used throughout the project, now configurable for any
        grid size.
        """
        train_env = cls(
            nrows=nrows, ncols=ncols,
            cliff_cells=[(nrows - 1, c) for c in range(1, ncols - 1)],
            start=(nrows - 1, 0), goal=(nrows - 1, ncols - 1),
            cliff_reward=cliff_reward, step_reward=step_reward,
        )
        test_env = cls(
            nrows=nrows, ncols=ncols,
            cliff_cells=[(0, c) for c in range(1, ncols - 1)],
            start=(0, 0), goal=(0, ncols - 1),
            cliff_reward=cliff_reward, step_reward=step_reward,
        )
        return train_env, test_env

    @classmethod
    def make_scaled_transfer_pair(
        cls,
        train_scale: int = 1,
        test_scale: int = 2,
        **kwargs,
    ) -> Tuple["GeneralizedCliffWalkingEnv", "GeneralizedCliffWalkingEnv"]:
        """Train on a smaller grid, test on a proportionally larger grid.

        Tests whether the learned polynomial filter coefficients h_k transfer
        across state-space sizes.  Because the filter is parameterised by
        scalars independent of nS, the same h_k values are applied to a
        different (larger) P matrix.

        train_scale=1, test_scale=2 → train on 4×12, test on 8×24.
        """
        return (
            cls.scaled(train_scale, **kwargs),
            cls.scaled(test_scale,  **kwargs),
        )

    @classmethod
    def make_topology_transfer_pair(
        cls,
    ) -> Tuple["GeneralizedCliffWalkingEnv", "GeneralizedCliffWalkingEnv"]:
        """Train on standard cliff, test on vertical cliff.

        Tests transfer across qualitatively different cliff topologies
        (horizontal → vertical barrier).
        """
        return cls.standard(), cls.vertical_cliff()

    # ── Info ──────────────────────────────────────────────────────────────────

    def info(self) -> dict:
        """Returns a dictionary with environment configuration."""
        return {
            "nrows"         : self.nrows,
            "ncols"         : self.ncols,
            "nS"            : self.nS,
            "nA"            : self.nA,
            "start"         : self.start,
            "goal"          : self.goal,
            "cliff_cells"   : self.cliff_cells,
            "n_cliff_cells" : len(self.cliff_cells),
            "cliff_reward"  : self.cliff_reward,
            "step_reward"   : self.step_reward,
            "cliff_teleport": self.cliff_teleport,
        }

    def __repr__(self) -> str:
        return (
            f"GeneralizedCliffWalkingEnv("
            f"{self.nrows}×{self.ncols}, "
            f"cliff={len(self.cliff_cells)} cells, "
            f"start={self.start}, goal={self.goal})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# WindyGridWorldEnv
# ─────────────────────────────────────────────────────────────────────────────

class WindyGridWorldEnv:
    """Grid world with column-wise deterministic upward wind.

    The classic Windy GridWorld from Sutton & Barto (Example 6.5), extended
    to arbitrary grid sizes and wind profiles.  No cliff cells – only walls
    (boundary = stay in place).

    Wind model
    ----------
    In column c, after the agent's action moves it to (nr, nc), wind pushes
    the agent a further ``wind_strengths.get(nc, 0)`` cells upward (row
    decreases), clamped at row 0.

    Stochastic wind variant
    -----------------------
    Pass ``wind_probs: Dict[int, float]`` to get a stochastic push of exactly
    1 cell upward with the given probability per column.  This produces a
    stochastic P matrix (non-integer entries) and is useful for testing
    transfer under dynamic uncertainty.

    At most one of ``wind_strengths`` or ``wind_probs`` should be provided.
    If both are None the result is a plain deterministic grid world.

    Parameters
    ----------
    nrows, ncols  : Grid dimensions (default 7×10, classic size).
    wind_strengths: {col: cells_pushed_up}.  Deterministic integer push.
    wind_probs    : {col: probability_of_1-cell_push}.  Stochastic wind.
    start         : (row, col).  Default: (nrows-1, 0).
    goal          : (row, col).  Default: (nrows//2, ncols-2) – classic target.
    step_reward   : Default: -1.0.

    Factory methods
    ---------------
    WindyGridWorldEnv.classic()                       Classic S&B 7×10.
    WindyGridWorldEnv.classic_stochastic(p=0.5)       Stochastic wind version.
    WindyGridWorldEnv.no_wind(nrows, ncols)           Pure grid world.
    WindyGridWorldEnv.make_train_test_pair()          No-wind → windy transfer.
    """

    def __init__(
        self,
        nrows: int = 7,
        ncols: int = 10,
        wind_strengths: Optional[Dict[int, int]]   = None,
        wind_probs:     Optional[Dict[int, float]] = None,
        start: Optional[Tuple[int, int]] = None,
        goal:  Optional[Tuple[int, int]] = None,
        step_reward: float = -1.0,
    ) -> None:
        if wind_strengths is not None and wind_probs is not None:
            raise ValueError(
                "Provide at most one of wind_strengths or wind_probs."
            )

        self.nrows = nrows
        self.ncols = ncols
        self.nS    = nrows * ncols
        self.nA    = 4

        self.wind_strengths = wind_strengths or {}
        self.wind_probs     = wind_probs     or {}
        self.step_reward    = step_reward

        self.start = tuple(start) if start is not None else (nrows - 1, 0)
        self.goal  = tuple(goal)  if goal  is not None else (nrows // 2, ncols - 2)

        self.start_state = self.start[0] * ncols + self.start[1]
        self.goal_state  = self.goal[0]  * ncols + self.goal[1]

        # No cliff cells in windy grid world — present for API compatibility
        # with GeneralizedCliffWalkingEnv (eval_policy_extended, _render_policy_map).
        self.cliff_cells = []
        self.cliff_set   = frozenset()

        self.P, self.r = self._build_dynamics()

    def _build_dynamics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        nS, nA = self.nS, self.nA
        ncols, nrows = self.ncols, self.nrows
        P = torch.zeros(nS * nA, nS)
        r = torch.zeros(nS * nA)

        for s in range(nS):
            row, col = divmod(s, ncols)

            for a in range(nA):
                idx = s * nA + a

                # Absorbing goal.
                if s == self.goal_state:
                    P[idx, s] = 1.0
                    continue

                # Intended next position (no wind yet).
                if a == ACT_UP:
                    nr, nc = max(row - 1, 0), col
                elif a == ACT_RIGHT:
                    nr, nc = row, min(col + 1, ncols - 1)
                elif a == ACT_DOWN:
                    nr, nc = min(row + 1, nrows - 1), col
                else:  # ACT_LEFT
                    nr, nc = row, max(col - 1, 0)

                r[idx] = self.step_reward

                # ── Deterministic wind ─────────────────────────────────────
                if self.wind_strengths:
                    w = self.wind_strengths.get(nc, 0)
                    final_row = max(nr - w, 0)
                    final_s   = final_row * ncols + nc
                    P[idx, final_s] = 1.0

                # ── Stochastic wind ────────────────────────────────────────
                elif self.wind_probs:
                    p_wind = self.wind_probs.get(nc, 0.0)
                    base_s = nr * ncols + nc
                    if p_wind > 0.0:
                        wind_row = max(nr - 1, 0)
                        wind_s   = wind_row * ncols + nc
                        if wind_s == base_s:
                            # Already at top row – wind has no effect.
                            P[idx, base_s] = 1.0
                        else:
                            P[idx, base_s] += (1.0 - p_wind)
                            P[idx, wind_s] += p_wind
                    else:
                        P[idx, base_s] = 1.0

                # ── No wind ────────────────────────────────────────────────
                else:
                    P[idx, nr * ncols + nc] = 1.0

        return P, r

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def classic(cls) -> "WindyGridWorldEnv":
        """Classic Sutton & Barto 7×10 Windy GridWorld.

        Wind strengths per column: [0,0,0,1,1,1,2,2,1,0].
        Start: (3,0).  Goal: (3,7).
        """
        return cls(
            nrows=7, ncols=10,
            wind_strengths={3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 1},
            start=(3, 0), goal=(3, 7),
        )

    @classmethod
    def classic_stochastic(cls, p: float = 0.5) -> "WindyGridWorldEnv":
        """Stochastic version of the classic 7×10 Windy GridWorld.

        In the same 'windy' columns as the classic environment each step has
        probability p of being pushed 1 cell upward instead of the deterministic
        integer push.  This creates a non-integer P matrix.

        Use as the test environment after training on classic() to measure
        transfer under stochastic dynamics.
        """
        return cls(
            nrows=7, ncols=10,
            wind_probs={3: p, 4: p, 5: p, 6: p, 7: p, 8: p},
            start=(3, 0), goal=(3, 7),
        )

    @classmethod
    def no_wind(
        cls,
        nrows: int = 7,
        ncols: int = 10,
        start: Optional[Tuple[int, int]] = None,
        goal:  Optional[Tuple[int, int]] = None,
    ) -> "WindyGridWorldEnv":
        """Plain deterministic grid world (no wind, no cliffs)."""
        return cls(nrows=nrows, ncols=ncols, start=start, goal=goal)

    @classmethod
    def make_train_test_pair(
        cls,
        nrows: int = 7,
        ncols: int = 10,
        wind_p: float = 0.6,
        start: Optional[Tuple[int, int]] = None,
        goal:  Optional[Tuple[int, int]] = None,
    ) -> Tuple["WindyGridWorldEnv", "WindyGridWorldEnv"]:
        """Train/test pair: no-wind (deterministic) → stochastic wind.

        Tests transfer when the agent trained on a deterministic MDP is
        evaluated on a stochastic version with the same grid geometry.
        The polynomial filter h_k is identical; only P changes.

        wind_p : Probability of 1-cell upward push in windy columns.
        """
        windy_cols = list(range(ncols // 3, 2 * ncols // 3))
        train_env = cls(nrows=nrows, ncols=ncols, start=start, goal=goal)
        test_env  = cls(
            nrows=nrows, ncols=ncols,
            wind_probs={c: wind_p for c in windy_cols},
            start=start, goal=goal,
        )
        return train_env, test_env

    @classmethod
    def make_wind_transfer_pair(
        cls,
        p_train: float = 0.3,
        p_test:  float = 0.7,
    ) -> Tuple["WindyGridWorldEnv", "WindyGridWorldEnv"]:
        """Train with weak wind, test with stronger wind.

        Both environments are stochastic; this tests robustness to changes in
        the wind intensity parameter while keeping the grid geometry fixed.
        """
        cols = {3, 4, 5, 6, 7, 8}
        train_env = cls(
            nrows=7, ncols=10,
            wind_probs={c: p_train for c in cols},
            start=(3, 0), goal=(3, 7),
        )
        test_env = cls(
            nrows=7, ncols=10,
            wind_probs={c: p_test for c in cols},
            start=(3, 0), goal=(3, 7),
        )
        return train_env, test_env

    def info(self) -> dict:
        return {
            "nrows"          : self.nrows,
            "ncols"          : self.ncols,
            "nS"             : self.nS,
            "nA"             : self.nA,
            "start"          : self.start,
            "goal"           : self.goal,
            "wind_strengths" : self.wind_strengths,
            "wind_probs"     : self.wind_probs,
            "step_reward"    : self.step_reward,
        }

    def __repr__(self) -> str:
        wind_info = (
            f"wind_strengths={self.wind_strengths}"
            if self.wind_strengths
            else f"wind_probs={self.wind_probs}"
            if self.wind_probs
            else "no_wind"
        )
        return (
            f"WindyGridWorldEnv({self.nrows}×{self.ncols}, "
            f"{wind_info}, start={self.start}, goal={self.goal})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ChainMDP
# ─────────────────────────────────────────────────────────────────────────────

class ChainMDP:
    """One-dimensional chain MDP.

    N states arranged in a line: 0 – 1 – 2 – … – N-1.
    Two actions: LEFT=0, RIGHT=1.

    Dynamics
    --------
    - RIGHT from s < N-1  → s+1, reward = step_reward.
    - LEFT  from s > 0    → s-1, reward = step_reward.
    - At goal (state N-1) → absorbing self-loop, reward = 0.
    - Boundary at s=0:
        If cliff_left=True : LEFT → teleport to start_state, reward = cliff_reward.
        If cliff_left=False: LEFT → stay at 0, reward = step_reward.

    Spectral properties
    -------------------
    For a 1-D chain the eigenvalues of P_π (under the optimal policy) are
    approximately cos(kπ/N) for k=0,…,N-1, spanning [-1, 1] uniformly.
    Increasing N densifies the spectrum without changing its support –
    an ideal testbed for studying how the learned polynomial filter scales.

    Transfer scenarios
    ------------------
    make_train_test_pair(N_train, N_test):   Different chain lengths.
    make_mirrored_pair(N):                   Goal on left vs right (mirrored).
    make_stochastic_pair(N, p_slip):         Deterministic → slippery chain.

    Parameters
    ----------
    N            : Number of states (chain length).
    start        : Starting state index.  Default: N // 4.
    cliff_left   : Whether stepping off the left end is a cliff.  Default: True.
    cliff_reward : Reward for hitting the left cliff.  Default: -100.
    step_reward  : Reward for all other transitions.  Default: -1.
    p_slip       : Probability of slipping (action reversed).  Default: 0.0.
                   If > 0, produces a stochastic P matrix.
    goal_left    : If True, goal is at state 0 (left end); otherwise N-1.
                   Default: False.
    """

    def __init__(
        self,
        N: int,
        start: Optional[int] = None,
        cliff_left: bool = True,
        cliff_reward: float = -100.0,
        step_reward:  float = -1.0,
        p_slip: float = 0.0,
        goal_left: bool = False,
    ) -> None:
        if N < 3:
            raise ValueError("Chain must have at least 3 states.")

        self.N           = N
        self.nS          = N
        self.nA          = 2
        self.cliff_left  = cliff_left
        self.cliff_reward = cliff_reward
        self.step_reward  = step_reward
        self.p_slip       = p_slip
        self.goal_left    = goal_left

        # Goal and start
        self.goal_state  = 0 if goal_left else N - 1
        if start is None:
            # Place start at opposite end from goal, slightly inward.
            start = (N - 1) if goal_left else 0
        self.start_state = start

        # Teleport target when cliff_left is True (never the goal state).
        self.teleport_state = self.start_state

        self.P, self.r = self._build_dynamics()

    def _build_dynamics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        N, nA = self.N, self.nA
        P = torch.zeros(N * nA, N)
        r = torch.zeros(N * nA)

        ACT_LEFT_CHAIN  = 0
        ACT_RIGHT_CHAIN = 1

        # Directions for the goal_left variant: invert LEFT/RIGHT semantics.
        # When goal_left=True, 'toward goal' is LEFT; mirror so the
        # difficulty is structurally symmetric.
        if self.goal_left:
            toward_goal, away_goal = ACT_LEFT_CHAIN, ACT_RIGHT_CHAIN
        else:
            toward_goal, away_goal = ACT_RIGHT_CHAIN, ACT_LEFT_CHAIN

        for s in range(N):
            for a in range(nA):
                idx = s * nA + a

                # Absorbing goal.
                if s == self.goal_state:
                    P[idx, s] = 1.0
                    continue

                # Intended transition (before slip).
                intended = a
                slip_a   = ACT_RIGHT_CHAIN if a == ACT_LEFT_CHAIN else ACT_LEFT_CHAIN

                def _next(state: int, action: int) -> Tuple[int, float]:
                    """Compute (next_state, reward) for a deterministic step."""
                    if action == ACT_RIGHT_CHAIN:
                        ns = min(state + 1, N - 1)
                    else:  # ACT_LEFT
                        if state == 0:
                            if self.cliff_left and not self.goal_left:
                                return self.teleport_state, self.cliff_reward
                            else:
                                return state, self.step_reward  # boundary
                        ns = state - 1
                    if ns == self.goal_state:
                        return ns, self.step_reward
                    return ns, self.step_reward

                # Also handle goal_left cliff (right boundary is hazard).
                def _next_gl(state: int, action: int) -> Tuple[int, float]:
                    if action == ACT_LEFT_CHAIN:
                        ns = max(state - 1, 0)
                    else:  # ACT_RIGHT
                        if state == N - 1:
                            if self.cliff_left:  # cliff_left interpreted as cliff_far_end
                                return self.teleport_state, self.cliff_reward
                            else:
                                return state, self.step_reward
                        ns = state + 1
                    if ns == self.goal_state:
                        return ns, self.step_reward
                    return ns, self.step_reward

                next_fn = _next_gl if self.goal_left else _next

                ns_int, rew_int   = next_fn(s, intended)
                ns_slip, rew_slip = next_fn(s, slip_a)

                if self.p_slip > 0.0:
                    # Stochastic: action reverses with probability p_slip.
                    pi, ps = 1.0 - self.p_slip, self.p_slip
                    P[idx, ns_int]  += pi
                    P[idx, ns_slip] += ps
                    # Expected reward.
                    r[idx] = pi * rew_int + ps * rew_slip
                else:
                    P[idx, ns_int] = 1.0
                    r[idx] = rew_int

        return P, r

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def standard(cls, N: int = 20) -> "ChainMDP":
        """Standard chain: goal at right end, cliff at left end."""
        return cls(N=N, cliff_left=True)

    @classmethod
    def mirrored(cls, N: int = 20) -> "ChainMDP":
        """Mirrored chain: goal at left end, cliff at right end."""
        return cls(N=N, cliff_left=True, goal_left=True, start=N - 1)

    @classmethod
    def no_cliff(cls, N: int = 20) -> "ChainMDP":
        """Chain without cliff penalty (just boundary bounce)."""
        return cls(N=N, cliff_left=False)

    @classmethod
    def make_train_test_pair(
        cls,
        N_train: int = 20,
        N_test: int  = 48,
        **kwargs,
    ) -> Tuple["ChainMDP", "ChainMDP"]:
        """Train on shorter chain, test on longer chain.

        The polynomial filter coefficients h_k are invariant to N; this pair
        tests whether they transfer to a larger state space with a denser
        eigenvalue spectrum.

        Note: N_test=48 matches the number of states in CliffWalkingEnv,
        making cross-environment spectral comparisons straightforward.
        """
        return cls.standard(N_train, **kwargs), cls.standard(N_test, **kwargs)

    @classmethod
    def make_mirrored_pair(cls, N: int = 20) -> Tuple["ChainMDP", "ChainMDP"]:
        """Train on standard chain, test on mirrored chain.

        Analogous to CliffWalkingEnv → MirroredCliffWalkingEnv.
        Same N, same cliff penalty; goal and hazard swap ends.
        """
        return cls.standard(N), cls.mirrored(N)

    @classmethod
    def make_stochastic_pair(
        cls,
        N: int = 20,
        p_slip: float = 0.2,
    ) -> Tuple["ChainMDP", "ChainMDP"]:
        """Train on deterministic chain, test on slippery chain.

        With probability p_slip the intended action is reversed.
        This creates a stochastic P matrix (non-integer entries), testing
        whether the polynomial filter is robust to dynamic uncertainty.
        """
        return cls.standard(N), cls(N=N, cliff_left=True, p_slip=p_slip)

    def info(self) -> dict:
        return {
            "N"           : self.N,
            "nS"          : self.nS,
            "nA"          : self.nA,
            "start"       : self.start_state,
            "goal"        : self.goal_state,
            "cliff_left"  : self.cliff_left,
            "cliff_reward": self.cliff_reward,
            "step_reward" : self.step_reward,
            "p_slip"      : self.p_slip,
            "goal_left"   : self.goal_left,
        }

    def __repr__(self) -> str:
        hazard = "cliff" if self.cliff_left else "boundary"
        goal_side = "left" if self.goal_left else "right"
        slip = f", p_slip={self.p_slip}" if self.p_slip > 0 else ""
        return f"ChainMDP(N={self.N}, goal={goal_side}, {hazard}{slip})"


# ─────────────────────────────────────────────────────────────────────────────
# RandomGraphMDP
# ─────────────────────────────────────────────────────────────────────────────

class RandomGraphMDP:
    """Randomly generated tabular MDP for out-of-distribution transfer testing.

    Each (state, action) pair transitions to ``b`` randomly chosen next states
    with Dirichlet-random probabilities, plus a guaranteed small escape
    probability toward the goal to ensure reachability.

    Exposes the same interface as GeneralizedCliffWalkingEnv (nS, nA, P, r,
    goal_state, cliff_set) so it drops in anywhere an env is expected.

    Parameters
    ----------
    nS         : Number of states.
    nA         : Number of actions.
    b          : Branching factor — distinct successors per (s, a).  Clamped
                 to min(b, nS-1) to exclude the goal from the random draw.
    seed       : RNG seed for reproducibility.
    goal_state : Absorbing goal state index (default: nS - 1).
    r_range    : (min, max) reward range for non-goal transitions.
    p_goal     : Guaranteed minimum probability of reaching goal from any
                 non-goal state on any action (ensures reachability).
    """

    def __init__(
        self,
        nS: int,
        nA: int,
        b: int = 4,
        seed: int = 0,
        goal_state: Optional[int] = None,
        r_range: Tuple[float, float] = (-1.0, 0.0),
        p_goal: float = 0.05,
    ) -> None:
        import numpy as _np
        rng = _np.random.default_rng(seed)

        self.nS         = nS
        self.nA         = nA
        self.goal_state = int(nS - 1) if goal_state is None else int(goal_state)
        self.cliff_set  = frozenset()   # no cliff cells

        non_goal = [s for s in range(nS) if s != self.goal_state]
        b_eff    = min(b, len(non_goal))

        P = torch.zeros(nS * nA, nS)
        r = torch.zeros(nS * nA)

        for s in range(nS):
            for a in range(nA):
                idx = s * nA + a
                if s == self.goal_state:
                    P[idx, self.goal_state] = 1.0
                    continue
                # b random non-goal successors + guaranteed goal escape
                nexts   = rng.choice(non_goal, size=b_eff, replace=False).tolist()
                weights = rng.dirichlet(_np.ones(b_eff)).tolist()
                for ns, w in zip(nexts, weights):
                    P[idx, ns] = float(w) * (1.0 - p_goal)
                P[idx, self.goal_state] += p_goal
                r[idx] = float(rng.uniform(r_range[0], r_range[1]))

        self.P = P
        self.r = r

    def __repr__(self) -> str:
        return f"RandomGraphMDP(nS={self.nS}, nA={self.nA}, goal={self.goal_state})"
