"""Quick entry point for training a single UPI model.

For systematic experiments (sweeping K, N_unrolls, or transfer), use the
scripts in experiments/ instead.
"""

import time
import wandb
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from src import CliffWalkingEnv, MirroredCliffWalkingEnv
from src.algorithms import PolicyIterationTrain, UnrollingPolicyIterationTrain


def policy_iteration(max_eval_iters: int = 10, max_epochs: int = 20) -> None:
    env = CliffWalkingEnv()
    model = PolicyIterationTrain(env, gamma=0.99, max_eval_iters=max_eval_iters)
    trainer = Trainer(max_epochs=max_epochs, log_every_n_steps=1, accelerator="cpu",
                      logger=False)
    trainer.fit(model)


def unrolled_policy_iteration(
    K: int = 5,
    num_unrolls: int = 5,
    tau: float = 5.0,
    beta: float = 1.0,
    lr: float = 5e-3,
    N: int = 1,
    weight_sharing: bool = True,
    init_q: str = "random",
    architecture_type: int = 1,
    max_epochs: int = 3000,
    use_wandb: bool = False,
) -> None:
    """Train a single UPI model on CliffWalking with optional W&B logging.

    Args:
        K: Graph filter order.
        num_unrolls: Number of (eval + improve) pairs.
        tau: Softmax temperature for policy improvement.
        beta: Scaling on the feedback term.
        lr: Adam learning rate.
        N: Number of samples in the dataset.
        weight_sharing: Share h coefficients across eval layers.
        init_q: Q initialisation — 'zeros', 'ones', or 'random'.
        architecture_type: 1 (monomial feedback) or 2 (polynomial feedback).
        max_epochs: Training epochs.
        use_wandb: Log to Weights & Biases.
    """
    env = CliffWalkingEnv()
    env_test = MirroredCliffWalkingEnv()

    model = UnrollingPolicyIterationTrain(
        env=env, env_test=env_test,
        K=K, num_unrolls=num_unrolls, tau=tau, beta=beta,
        lr=lr, N=N, weight_sharing=weight_sharing,
        init_q=init_q, architecture_type=architecture_type,
        use_legacy_init=True,
    )

    logger = WandbLogger(
        project="rl-unrolling",
        name=f"upi-K{K}-U{num_unrolls}-arch{architecture_type}",
    ) if use_wandb else False

    trainer = Trainer(max_epochs=max_epochs, log_every_n_steps=1,
                      accelerator="cpu", logger=logger)
    t0 = time.perf_counter()
    trainer.fit(model)
    print(f"Done in {time.perf_counter() - t0:.1f}s  |  "
          f"Bellman err: {model.bellman_error:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    unrolled_policy_iteration(K=5, num_unrolls=5, architecture_type=1)
