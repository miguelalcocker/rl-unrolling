import torch
import pytorch_lightning as pl

import wandb
import matplotlib.pyplot as plt

def safe_wandb_log(*args, **kwargs):
    if wandb.run is not None:
        wandb.log(*args, **kwargs)

class PolicyIterationTrain(pl.LightningModule):
    def __init__(self, env, goal_row=3, gamma=0.99, max_eval_iters=1000, Pi_init=None, test=False, log=False):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.nS, self.nA = env.nS, env.nA
        self.goal_row = goal_row
        self.register_buffer("P",  env.P.clone())
        self.register_buffer("r",  env.r.clone())
        
        self.Pi = Pi_init
        self.q  = None
        self.gamma = gamma
        self.max_eval_iters = max_eval_iters
        self.test = test

    def lift_policy_matrix(self, Pi):
        Pi_ext = torch.zeros(self.nS, self.nS * self.nA, device=self.device)
        for s in range(self.nS):
            for a in range(self.nA):
                Pi_ext[s, s * self.nA + a] = Pi[s, a]
        return Pi_ext

    def compute_transition_matrix(self, P, Pi):
        Pi_ext = self.lift_policy_matrix(Pi)
        return P @ Pi_ext

    def policy_evaluation(self, P_pi, r):
        q = self.q[::]
        for _ in range(self.max_eval_iters):
            q = r + self.gamma * (P_pi @ q)
        return q

    def policy_improvement(self, q):
        q_reshaped = q.view(self.nS, self.nA)
        # greedy_actions = torch.argmax(q_reshaped, dim=1)
        max_vals = q_reshaped.max(dim=1, keepdim=True).values
        is_max = q_reshaped == max_vals
        greedy_actions = torch.multinomial(is_max.float(), num_samples=1).squeeze(1)
        Pi_new = torch.zeros((self.nS, self.nA), device=self.device)
        Pi_new[torch.arange(self.nS), greedy_actions] = 1.0
        return Pi_new

    def on_fit_start(self):
        from src.plots import plot_policy_and_value
        if self.Pi is None:
            # self.Pi = torch.full((self.nS, self.nA), 1 / self.nA, device=self.device)
            self.Pi = torch.rand(self.nS, self.nA, device=self.device)
            self.Pi = self.Pi / self.Pi.sum(dim=1, keepdim=True) 

        self.q  = torch.zeros(self.nS * self.nA, device=self.device)

        r = self.r.view(self.nS, self.nA)
        try:
            fig = plot_policy_and_value(r, self.Pi, highlight_cliffs=False, goal_row=self.goal_row)
            safe_wandb_log({"policy_plot_init": wandb.Image(fig)})
            plt.close(fig)
        except (AssertionError, Exception):
            pass

    def on_fit_end(self):
        from src.plots import plot_policy_and_value
        q = self.q.view(self.nS, self.nA)
        try:
            fig = plot_policy_and_value(q, self.Pi, goal_row=self.goal_row)
            safe_wandb_log({"policy_plot": wandb.Image(fig)})
            plt.close(fig)
        except (AssertionError, Exception):
            pass

    def training_step(self, batch, batch_idx):
        P_pi = self.compute_transition_matrix(self.P, self.Pi)
        q = self.policy_evaluation(P_pi, self.r)

        target = self.r + self.gamma * P_pi @ q
        self.bellman_error = ( torch.norm(q - target) / torch.norm(target)).detach()
        policy_diff = torch.norm(self.Pi - self.policy_improvement(q))
        q_norm = torch.norm(q)

        self.log("bellman_error", self.bellman_error, on_step=True, on_epoch=False, prog_bar=True)
        self.log("policy_diff", policy_diff, on_step=True, on_epoch=False, prog_bar=True)
        self.log("q_norm", q_norm, on_step=True, on_epoch=False, prog_bar=True)

        self.Pi = self.policy_improvement(q)
        self.q = q

    def configure_optimizers(self):
        return None

    def train_dataloader(self):
        return torch.utils.data.DataLoader([0])
