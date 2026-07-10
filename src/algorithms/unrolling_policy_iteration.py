import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import mse_loss, smooth_l1_loss
from src.plots import plot_policy_and_value, plot_Pi, plot_filter_coefs
from src import UnrolledPolicyIterationModel, PolicyEvaluationLayer


def rew_smoothness(P_pi, r):
    diff = r.unsqueeze(1) - r.unsqueeze(0)
    smoothness = (P_pi * diff.square()).sum() / r.square().sum()
    return smoothness


def safe_wandb_log(*args, **kwargs):
    if wandb.run is not None:
        wandb.log(*args, **kwargs)


class UnrollingDataset(Dataset):
    def __init__(self, nS, nA, N=1, init_q="zeros"):
        self.policies = torch.ones(N, nS, nA)
        self.policies = self.policies / self.policies.sum(dim=-1, keepdim=True)
        
        if init_q == "zeros":
            self.qs = torch.zeros(N, nS * nA)
        elif init_q == "ones":
            self.qs = torch.ones(N, nS * nA)
        elif init_q == "random":
            self.qs = torch.randn(N, nS * nA)
        else:
            raise ValueError(f"Invalid init_q: {init_q}. Must be 'zeros', 'ones', or 'random'")

    def __len__(self):
        return len(self.policies)

    def __getitem__(self, idx):
        return self.qs[idx], self.policies[idx]


class UnrollingPolicyIterationTrain(pl.LightningModule):
    def __init__(self, env, env_test, K=3, num_unrolls=5, gamma=0.99, lr=1e-3, tau=1.0, beta=1.0, freq_plots=10, N=1, weight_sharing=True, init_q="zeros", loss_type="original_with_detach", architecture_type=1, use_huber_loss=False, use_legacy_init=True, K_2=None):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.nS, self.nA = env.nS, env.nA
        self.register_buffer("P", env.P.clone())
        self.register_buffer("r", env.r.clone())
        self.register_buffer("P_test", env_test.P.clone())
        self.register_buffer("r_test", env_test.r.clone())
        self.gamma = gamma
        self.lr = lr
        self.N = N
        self.init_q = init_q
        self.loss_type = loss_type
        self.use_huber_loss = use_huber_loss

        self.freq_plots = freq_plots
        self.Pi_train = []

        self.model = UnrolledPolicyIterationModel(self.P, self.r, self.nS, self.nA, K, num_unrolls, tau, beta, weight_sharing, architecture_type, use_legacy_init, K_2)
        self.model_test = UnrolledPolicyIterationModel(self.P_test, self.r_test, self.nS, self.nA, K, num_unrolls, tau, beta, weight_sharing, architecture_type, use_legacy_init, K_2)

    def training_step(self, batch, batch_idx):
        q_in, Pi_in = batch
        q_pred, Pi_pred = self.model(q_in, Pi_in)

        P_pi = self.model.layers[-2].compute_transition_matrix(Pi_pred)
        
        if self.loss_type == "original_with_detach":
            P_pi_detached = self.model.layers[-2].compute_transition_matrix(Pi_pred.detach())
            target = self.r + self.gamma * (P_pi_detached @ q_pred.detach())
        elif self.loss_type == "original_no_detach":
            target = self.r + self.gamma * (P_pi @ q_pred)
        elif self.loss_type == "max_with_detach":
            v = torch.max(q_pred.view(self.nS, self.nA), dim=1)[0]
            target = (self.r + self.gamma * (self.P @ v.detach())).view(-1)
        elif self.loss_type == "max_no_detach":
            v = torch.max(q_pred.view(self.nS, self.nA), dim=1)[0]
            target = (self.r + self.gamma * (self.P @ v)).view(-1)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        q_reshaped = q_pred.view(self.nS, self.nA)
        target_reshaped = target.view(self.nS, self.nA)

        # Use Huber loss if specified, otherwise MSE
        if self.use_huber_loss:
            loss = smooth_l1_loss(q_reshaped, target_reshaped)
        else:
            loss = mse_loss(q_reshaped, target_reshaped)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        smoothness = rew_smoothness(P_pi, self.r)
        self.log("reward_smoothness", smoothness, on_step=True, on_epoch=False, prog_bar=True)
        
        # For debug
        q_pred = q_pred.detach()
        bellman_error = torch.norm(q_pred - target)
        self.log("bellman_error", bellman_error, on_step=True, on_epoch=False, prog_bar=True)

        if batch_idx == 0 and self.current_epoch % self.freq_plots == 0:
            self.Pi_train.append(Pi_pred.detach().cpu().numpy())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(UnrollingDataset(self.nS, self.nA, N=self.N, init_q=self.init_q), batch_size=1, shuffle=True)

    def _evaluate_env(self, model, r):
        """Run one forward pass and return (q, Pi, bellman_error, bellman_error_unnormalized)."""
        dataset = UnrollingDataset(self.nS, self.nA, N=self.N, init_q=self.init_q)
        q_sample, Pi_sample = dataset[0]
        q_sample, Pi_sample = q_sample.to(self.device), Pi_sample.to(self.device)
        q, Pi_out = model(q_sample, Pi_sample)

        nS, _ = Pi_out.shape
        greedy_actions = Pi_out.argmax(dim=1)
        Pi_det = torch.zeros_like(Pi_out)
        Pi_det[torch.arange(nS), greedy_actions] = 1.0

        P_pi = model.layers[-2].compute_transition_matrix(Pi_det).detach()
        target = r + self.gamma * (P_pi @ q.detach())
        diff = q - target

        bellman_error_unnormalized = torch.norm(diff)
        bellman_error = torch.norm(diff) / torch.norm(target)

        return q.detach(), Pi_out.detach(), bellman_error.detach(), bellman_error_unnormalized.detach()

    def on_fit_end(self):
        self.q, self.Pi, self.bellman_error, self.bellman_error_unnormalized = \
            self._evaluate_env(self.model, self.r)

        try:
            fig_policy = plot_policy_and_value(self.q.view(self.nS, self.nA), self.Pi)
            fig_policy_full = plot_policy_and_value(self.q.view(self.nS, self.nA), self.Pi, plot_all_trans=True)
            fig_P = plot_Pi(self.Pi.detach().cpu().numpy())
            safe_wandb_log({
                "policy_plot": wandb.Image(fig_policy),
                "full_policy_plot": wandb.Image(fig_policy_full),
                "Pi_plot": wandb.Image(fig_P)})
            plt.close(fig_policy_full)
            plt.close(fig_policy)
            plt.close(fig_P)
        except (AssertionError, Exception):
            pass

        if self.model.h is not None:
            fig_h = plot_filter_coefs(self.model.h.detach().cpu().numpy())
            safe_wandb_log({"shared_h_coefficients": wandb.Image(fig_h)})
            plt.close(fig_h)

        with torch.no_grad():
            for layer1, layer2 in zip(self.model.layers, self.model_test.layers):
                if isinstance(layer1, PolicyEvaluationLayer) and isinstance(layer2, PolicyEvaluationLayer):
                    layer2.h.copy_(layer1.h)

        self.q_test, self.Pi_test, self.bellman_error_test, self.bellman_error_unnormalized_test = \
            self._evaluate_env(self.model_test, self.r_test)

        try:
            fig_policy = plot_policy_and_value(self.q_test.view(self.nS, self.nA), self.Pi_test, goal_row=0)
            fig_policy_full = plot_policy_and_value(self.q_test.view(self.nS, self.nA), self.Pi_test, goal_row=0, plot_all_trans=True)
            fig_P = plot_Pi(self.Pi_test.detach().numpy())
            safe_wandb_log({
                "policy_transf_plot": wandb.Image(fig_policy),
                "full_policy_transf_plot": wandb.Image(fig_policy_full),
                "Pi_transf_plot": wandb.Image(fig_P)})
            plt.close(fig_policy_full)
            plt.close(fig_policy)
            plt.close(fig_P)
        except (AssertionError, Exception):
            pass