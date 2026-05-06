"""Soft Actor-Critic with adaptive discount (SAC3). Hyperparameters only via JSON (--config)."""
import os
import csv
import time
import random
from collections import deque

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

try:
    import safety_gymnasium
    HAS_SAFETY_GYM = True
except ImportError:
    HAS_SAFETY_GYM = False
    print("WARNING: safety-gymnasium not installed. Install with: pip install safety-gymnasium")



import argparse
from load_config import load_config

cfg = None
device = None
directory = None
script_name = "sac"
LOG_STD_MIN = None
LOG_STD_MAX = None
EPSILON = None


def init_runtime(c):
    """All runtime globals come from JSON config (no hard-coded hyperparameters)."""
    global cfg, device, directory, LOG_STD_MIN, LOG_STD_MAX, EPSILON
    cfg = c
    dev = getattr(cfg, "device", None)
    if dev is None:
        raise ValueError("config must set device (e.g. cuda:0 or cpu)")
    device = dev if isinstance(dev, str) else str(dev)
    out = getattr(cfg, "output_dir", None)
    if not out:
        raise ValueError("config must set output_dir")
    directory = os.path.abspath(out)
    directory = directory if directory.endswith(os.sep) else directory + os.sep
    LOG_STD_MIN = float(cfg.log_std_min)
    LOG_STD_MAX = float(cfg.log_std_max)
    EPSILON = float(cfg.epsilon)


def set_experiment_directory(path):
    global directory
    path = os.path.abspath(path)
    directory = path if path.endswith(os.sep) else path + os.sep


def ensure_run_directories():
    os.makedirs(directory, exist_ok=True)
    os.makedirs(directory + "param/", exist_ok=True)
    os.makedirs(directory + "img/", exist_ok=True)




def make_env(env_name, render_mode=None):
    if HAS_SAFETY_GYM:
        return safety_gymnasium.make(env_name, render_mode=render_mode)
    return gym.make(env_name, render_mode=render_mode)


def env_step(env, action):
    result = env.step(action)
    if len(result) == 6:
        obs, reward, cost, terminated, truncated, info = result
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result
        cost = info.get("cost", float(cfg.default_env_cost_when_missing))
    else:
        raise ValueError(f"Unexpected step() returns: {len(result)} values")
    return obs, reward, float(cost), terminated, truncated, info


class CSVLogger:
    def __init__(self, log_dir, file1="episode_rewards.csv", file2="training_loss.csv"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_file = os.path.join(log_dir, file1)
        self.loss_file = os.path.join(log_dir, file2)
        with open(self.episode_file, "w", newline="") as f:
            csv.writer(f).writerow([
                "episode", "reward", "cost", "running_reward", "running_cost",
                "steps", "gamma_mean", "alpha", "lagrange", "total_steps", "timestamp"])
        with open(self.loss_file, "w", newline="") as f:
            csv.writer(f).writerow([
                "episode", "update_step", "policy_loss", "q1_loss", "q2_loss",
                "alpha_loss", "gamma_info", "gamma_mean", "alpha", "entropy",
                "lagrange", "timestamp"])
        self.start_time = time.time()

    def log_reward(self, episode, reward, cost, running_reward, running_cost,
                   steps, gamma_mean, alpha, lagrange, total_steps):
        t = time.time() - self.start_time
        with open(self.episode_file, "a", newline="") as f:
            csv.writer(f).writerow([
                episode, f"{reward:.2f}", f"{cost:.2f}", f"{running_reward:.2f}",
                f"{running_cost:.2f}", steps, f"{gamma_mean:.6f}", f"{alpha:.6f}",
                f"{lagrange:.6f}", total_steps, f"{t:.2f}"])

    def log_loss(self, episode, step, ploss, q1, q2, aloss, ginfo, gmean, alpha, entropy, lagrange):
        t = time.time() - self.start_time
        with open(self.loss_file, "a", newline="") as f:
            csv.writer(f).writerow([
                episode, step, f"{ploss:.6f}", f"{q1:.6f}", f"{q2:.6f}",
                f"{aloss:.6f}", f"{ginfo:.6f}", f"{gmean:.6f}",
                f"{alpha:.6f}", f"{entropy:.6f}", f"{lagrange:.6f}", f"{t:.2f}"])


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.costs = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.episode_ids = np.zeros(capacity, dtype=np.int64)
        self.current_episode_id = 0
        self.current_step = 0

    def push(self, state, action, reward, cost, next_state, done):
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.costs[i] = cost
        self.next_states[i] = next_state
        self.dones[i] = done
        self.episode_ids[i] = self.current_episode_id
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.current_step += 1
        if done:
            self.current_episode_id += 1
            self.current_step = 0

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.FloatTensor(self.states[idx]).to(device),
            torch.FloatTensor(self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).to(device),
            torch.FloatTensor(self.costs[idx]).to(device),
            torch.FloatTensor(self.next_states[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).to(device),
        )

    def sample_two_batches(self, batch_size):
        idx = np.random.choice(self.size, 2 * batch_size, replace=False)
        a, b = idx[:batch_size], idx[batch_size:]
        def _batch(i):
            return (
                torch.FloatTensor(self.states[i]).to(device),
                torch.FloatTensor(self.actions[i]).to(device),
                torch.FloatTensor(self.rewards[i]).to(device),
                torch.FloatTensor(self.costs[i]).to(device),
                torch.FloatTensor(self.next_states[i]).to(device),
                torch.FloatTensor(self.dones[i]).to(device),
            )
        return _batch(a), _batch(b)

    def sample_sequences(self, batch_size, seq_len):
        seq_starts = []
        attempts = 0
        max_attempts = batch_size * int(cfg.sequence_sample_attempts_multiplier)
        while len(seq_starts) < batch_size and attempts < max_attempts:
            i = np.random.randint(0, max(1, self.size - seq_len))
            if i + seq_len > self.capacity:
                attempts += 1
                continue
            ep = self.episode_ids[i]
            ok = all(self.episode_ids[i + k] == ep for k in range(1, seq_len))
            if ok:
                seq_starts.append(i)
            attempts += 1
        if len(seq_starts) < batch_size:
            return None

        B = len(seq_starts)
        sd = self.states.shape[1]
        ad = self.actions.shape[1]
        s_seq = np.zeros((B, seq_len, sd), np.float32)
        a_seq = np.zeros((B, seq_len, ad), np.float32)
        r_seq = np.zeros((B, seq_len), np.float32)
        c_seq = np.zeros((B, seq_len), np.float32)
        d_seq = np.zeros((B, seq_len), np.float32)
        ns_last = np.zeros((B, sd), np.float32)

        for b, start in enumerate(seq_starts):
            for k in range(seq_len):
                j = start + k
                s_seq[b, k] = self.states[j]
                a_seq[b, k] = self.actions[j]
                r_seq[b, k] = self.rewards[j, 0]
                c_seq[b, k] = self.costs[j, 0]
                d_seq[b, k] = self.dones[j, 0]
            ns_last[b] = self.next_states[start + seq_len - 1]

        return (
            torch.FloatTensor(s_seq).to(device),
            torch.FloatTensor(a_seq).to(device),
            torch.FloatTensor(r_seq).to(device),
            torch.FloatTensor(c_seq).to(device),
            torch.FloatTensor(d_seq).to(device),
            torch.FloatTensor(ns_last).to(device),
        )

    def __len__(self):
        return self.size


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=float(cfg.linear_xavier_gain))
        nn.init.constant_(m.bias, 0)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        self.apply(weights_init_)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        mu = self.mean(x)
        ls = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mu, ls

    def sample(self, s):
        mu, ls = self.forward(s)
        std = ls.exp()
        n = Normal(mu, std)
        xt = n.rsample()
        yt = torch.tanh(xt)
        a = yt * self.max_action
        lp = n.log_prob(xt) - torch.log(self.max_action * (1 - yt.pow(2)) + EPSILON)
        lp = lp.sum(-1, keepdim=True)
        return a, lp, torch.tanh(mu) * self.max_action

    def get_action(self, s, evaluate=False):
        s = torch.FloatTensor(s).unsqueeze(0).to(device)
        if evaluate:
            _, _, a = self.sample(s)
        else:
            a, _, _ = self.sample(s)
        return a.detach().cpu().numpy().flatten()


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        d = state_dim + action_dim
        self.l1 = nn.Linear(d, hidden_dim); self.l2 = nn.Linear(hidden_dim, hidden_dim); self.l3 = nn.Linear(hidden_dim, 1)
        self.l4 = nn.Linear(d, hidden_dim); self.l5 = nn.Linear(hidden_dim, hidden_dim); self.l6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, s, a):
        xu = torch.cat([s, a], -1)
        x1 = F.relu(self.l1(xu)); x1 = F.relu(self.l2(x1)); x1 = self.l3(x1)
        x2 = F.relu(self.l4(xu)); x2 = F.relu(self.l5(x2)); x2 = self.l6(x2)
        return x1, x2


class GammaNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, gamma_min, gamma_max, init_gamma):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self._init_to(init_gamma)

    def _init_to(self, target):
        t = (target - self.gamma_min) / (self.gamma_max - self.gamma_min)
        t = np.clip(t, float(cfg.gamma_init_clip_low), float(cfg.gamma_init_clip_high))
        logit = np.log(t / (1 - t + float(cfg.logit_stability_epsilon)))
        with torch.no_grad():
            self.net[-2].weight.fill_(float(cfg.gamma_net_last_weight_fill))
            self.net[-2].bias.fill_(logit)

    def forward(self, s):
        return self.gamma_min + (self.gamma_max - self.gamma_min) * self.net(s)


class AdaptiveGammaModule:
    def __init__(self, state_dim, action_dim, hidden_dim, approach):
        self.approach = approach
        self.gamma_min = cfg.gamma_min
        self.gamma_max = cfg.gamma_max
        self.update_count = 0
        if approach == "uncertainty":
            self.log_beta = torch.tensor(np.log(cfg.uncertainty_beta_init),
                                         requires_grad=cfg.uncertainty_beta_learnable,
                                         device=device, dtype=torch.float32)
            if cfg.uncertainty_beta_learnable:
                self.beta_optimizer = optim.Adam([self.log_beta], lr=cfg.uncertainty_beta_lr)
        elif approach in ("return_consistency", "cross_validated"):
            self.gamma_net = GammaNetwork(
                state_dim, cfg.gamma_hidden_dim, self.gamma_min, self.gamma_max, cfg.gamma_default
            ).to(device)
            self.gamma_optimizer = optim.Adam(self.gamma_net.parameters(), lr=cfg.gamma_lr)

    def compute_gamma(self, states, q1=None, q2=None):
        B = states.shape[0]
        if self.approach == "fixed":
            return torch.full((B, 1), cfg.gamma_default, device=device)
        if self.approach == "uncertainty":
            beta = self.log_beta.exp()
            dis = torch.abs(q1 - q2)
            return self.gamma_max - (self.gamma_max - self.gamma_min) * torch.sigmoid(beta * dis)
        return self.gamma_net(states)

    def compute_gamma_scalar(self, state, critic=None, policy=None):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device) if not isinstance(state, torch.Tensor) else state.unsqueeze(0)
            if self.approach == "fixed":
                return cfg.gamma_default
            if self.approach == "uncertainty":
                if critic is None or policy is None:
                    return (self.gamma_min + self.gamma_max) / 2
                a, _, _ = policy.sample(s)
                q1, q2 = critic(s, a)
                return self.compute_gamma(s, q1, q2).item()
            return self.gamma_net(s).item()

    def update_beta(self, gammas):
        if self.approach != "uncertainty" or not cfg.uncertainty_beta_learnable:
            return 0.0
        loss = (gammas.mean() - cfg.uncertainty_target_gamma).pow(2)
        self.beta_optimizer.zero_grad()
        loss.backward()
        self.beta_optimizer.step()
        return loss.item()

    def update_return_consistency(self, buffer, critic_target, policy, alpha):
        if self.approach != "return_consistency":
            return 0.0
        self.update_count += 1
        if self.update_count % cfg.gamma_update_freq != 0:
            return 0.0
        n = cfg.gamma_n_step
        seq = buffer.sample_sequences(cfg.batch_size, n)
        if seq is None:
            return 0.0
        s_seq, _, r_seq, _, d_seq, ns_last = seq

        with torch.no_grad():
            a_last, lp_last, _ = policy.sample(ns_last)
            q1l, q2l = critic_target(ns_last, a_last)
            v_last = torch.min(q1l, q2l) - alpha * lp_last
            G_n = torch.zeros(cfg.batch_size, 1, device=device)
            disc = torch.ones(cfg.batch_size, 1, device=device)
            alive = torch.ones(cfg.batch_size, 1, device=device)
            for k in range(n):
                G_n += disc * alive * r_seq[:, k:k + 1]
                alive *= (1 - d_seq[:, k:k + 1])
                disc *= cfg.gamma_ref
            G_n += disc * alive * v_last

        s0 = s_seq[:, 0]
        r0 = r_seq[:, 0:1]
        d0 = d_seq[:, 0:1]
        s1 = s_seq[:, 1] if n > 1 else ns_last
        with torch.no_grad():
            a1, lp1, _ = policy.sample(s1)
            q11, q21 = critic_target(s1, a1)
            v1 = torch.min(q11, q21) - alpha * lp1

        gamma_pred = self.gamma_net(s0)
        one_step = r0 + (1 - d0) * gamma_pred * v1.detach()
        loss_main = (one_step - G_n).pow(2).mean() * cfg.consistency_loss_weight

        loss_reg = cfg.gamma_deviation_coef * (gamma_pred - cfg.gamma_default).pow(2).mean()
        loss_var = cfg.gamma_variance_coef * gamma_pred.var() if gamma_pred.numel() > 1 else 0.0
        loss_bnd = cfg.gamma_boundary_coef * (
            F.relu(self.gamma_min + float(cfg.gamma_boundary_epsilon) - gamma_pred).mean() +
            F.relu(gamma_pred - self.gamma_max + float(cfg.gamma_boundary_epsilon)).mean()
        )
        total = loss_main + loss_reg + loss_var + loss_bnd
        self.gamma_optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(self.gamma_net.parameters(), cfg.max_grad_norm)
        self.gamma_optimizer.step()
        return total.item()

    def update_cross_validated(self, buffer, critic, critic_target, policy, alpha, state_dim, action_dim, hidden_dim):
        if self.approach != "cross_validated":
            return 0.0
        self.update_count += 1
        if self.update_count % cfg.gamma_update_freq != 0:
            return 0.0
        if len(buffer) < 2 * cfg.batch_size:
            return 0.0

        batch_a, batch_b = buffer.sample_two_batches(cfg.batch_size)
        sa, aa, ra, _, nsa, da = batch_a
        sb, ab, rb, _, nsb, db = batch_b
        temp_critic = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        temp_critic.load_state_dict(critic.state_dict())
        temp_opt = optim.Adam(temp_critic.parameters(), lr=cfg.learning_rate)

        with torch.no_grad():
            na_a, nlp_a, _ = policy.sample(nsa)
            tq1a, tq2a = critic_target(nsa, na_a)
            tqa = torch.min(tq1a, tq2a) - alpha * nlp_a
            gamma_a = self.gamma_net(sa)
            target_a = ra + (1 - da) * gamma_a * tqa
        cq1a, cq2a = temp_critic(sa, aa)
        closs_a = F.mse_loss(cq1a, target_a) + F.mse_loss(cq2a, target_a)
        temp_opt.zero_grad()
        closs_a.backward()
        temp_opt.step()

        with torch.no_grad():
            na_b, nlp_b, _ = policy.sample(nsb)
            tq1b, tq2b = critic_target(nsb, na_b)
            tqb = torch.min(tq1b, tq2b) - alpha * nlp_b
        gamma_b = self.gamma_net(sb)
        target_b = rb + (1 - db) * gamma_b * tqb.detach()
        with torch.no_grad():
            cq1b, cq2b = temp_critic(sb, ab)
            cur_qb = torch.min(cq1b, cq2b)
        td_err = (cur_qb - target_b).pow(2).mean() * cfg.cv_loss_weight
        reg = cfg.gamma_deviation_coef * (gamma_b - cfg.gamma_default).pow(2).mean()
        var_ = cfg.gamma_variance_coef * gamma_b.var() if gamma_b.numel() > 1 else 0.0
        bnd = cfg.gamma_boundary_coef * (
            F.relu(self.gamma_min + float(cfg.gamma_boundary_epsilon) - gamma_b).mean() +
            F.relu(gamma_b - self.gamma_max + float(cfg.gamma_boundary_epsilon)).mean()
        )
        total = td_err + reg + var_ + bnd
        self.gamma_optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(self.gamma_net.parameters(), cfg.max_grad_norm)
        self.gamma_optimizer.step()
        return total.item()

    def refresh_gamma_ref_from_net(self, buffer):
        """EMA-update cfg.gamma_ref using batch mean of gamma_net(s). Used for G_n discounting."""
        if self.approach != "return_consistency" or not cfg.gamma_ref_adaptive:
            return None
        if len(buffer) < cfg.batch_size:
            return None
        states, _, _, _, _, _ = buffer.sample(cfg.batch_size)
        with torch.no_grad():
            g_batch = self.gamma_net(states).mean().item()
        g_batch = float(np.clip(g_batch, cfg.gamma_min, cfg.gamma_max))
        tau = float(cfg.gamma_ref_ema_tau)
        tau = max(0.0, min(1.0, tau))
        cfg.gamma_ref = (1.0 - tau) * float(cfg.gamma_ref) + tau * g_batch
        return cfg.gamma_ref

    def state_dict(self):
        d = {"approach": self.approach}
        if self.approach == "uncertainty":
            d["log_beta"] = self.log_beta.detach().cpu()
            if cfg.uncertainty_beta_learnable:
                d["beta_opt"] = self.beta_optimizer.state_dict()
        elif self.approach in ("return_consistency", "cross_validated"):
            d["gamma_net"] = self.gamma_net.state_dict()
            d["gamma_opt"] = self.gamma_optimizer.state_dict()
        return d

    def load_state_dict(self, d):
        if self.approach == "uncertainty" and "log_beta" in d:
            self.log_beta = d["log_beta"].to(device).requires_grad_(cfg.uncertainty_beta_learnable)
            if cfg.uncertainty_beta_learnable and "beta_opt" in d:
                self.beta_optimizer = optim.Adam([self.log_beta], lr=cfg.uncertainty_beta_lr)
                self.beta_optimizer.load_state_dict(d["beta_opt"])
        elif self.approach in ("return_consistency", "cross_validated"):
            if "gamma_net" in d:
                self.gamma_net.load_state_dict(d["gamma_net"])
            if "gamma_opt" in d:
                self.gamma_optimizer.load_state_dict(d["gamma_opt"])


class SAC:
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        H = cfg.hidden_dim
        self.critic = QNetwork(state_dim, action_dim, H).to(device)
        self.critic_target = QNetwork(state_dim, action_dim, H).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.policy = GaussianPolicy(state_dim, action_dim, H, max_action).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        self.gamma_module = AdaptiveGammaModule(state_dim, action_dim, H, approach=cfg.gamma_approach)

        if cfg.automatic_entropy_tuning:
            te = getattr(cfg, "target_entropy", None)
            if te is None:
                raise ValueError("config must set target_entropy (e.g. negative action_dim for SAC)")
            self.target_entropy = float(te)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = cfg.alpha

        if cfg.use_cost_constraint:
            self.cost_critic = QNetwork(state_dim, action_dim, H).to(device)
            self.cost_critic_target = QNetwork(state_dim, action_dim, H).to(device)
            self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
            self.cost_critic_optimizer = optim.Adam(self.cost_critic.parameters(), lr=cfg.learning_rate)
            init_log = np.log(max(cfg.lagrange_init, float(cfg.lagrange_init_floor)))
            self.log_lagrange = torch.tensor(init_log, requires_grad=True, device=device, dtype=torch.float32)
            self.lagrange_optimizer = optim.Adam([self.log_lagrange], lr=cfg.lagrange_lr)
            self.lagrange = self.log_lagrange.exp().item()
        else:
            self.lagrange = 0.0

        self.buffer = ReplayBuffer(cfg.capacity, state_dim, action_dim)
        self.num_updates = 0
        self.total_steps = 0
        self.current_gamma_mean = cfg.gamma_default
        self.last_losses = dict(policy=0, q1=0, q2=0, alpha=0, gamma=0, entropy=0)

    def select_action(self, state, evaluate=False):
        return self.policy.get_action(state, evaluate)

    def store(self, state, action, reward, cost, next_state, done):
        self.buffer.push(state, action, reward, cost, next_state, float(done))
        self.total_steps += 1

    def get_gamma_for_state(self, state):
        return self.gamma_module.compute_gamma_scalar(state, self.critic, self.policy)

    def update(self, logger=None, episode=0):
        if len(self.buffer) < cfg.min_buffer_size:
            return False
        for _ in range(cfg.gradient_steps):
            self._update_step(logger, episode)
        return True

    def _update_step(self, logger, episode):
        # Implementation 1: warmup stage for learned-gamma approaches.
        original_approach = cfg.gamma_approach
        learned_gamma = original_approach in ("return_consistency", "cross_validated")
        if learned_gamma and self.total_steps < int(max(0, cfg.gamma_warmup_steps)):
            cfg.gamma_approach = "fixed"
        try:
            self._update_step_core(logger, episode)
        finally:
            cfg.gamma_approach = original_approach

    def _update_step_core(self, logger, episode):
        self.num_updates += 1
        states, actions, rewards, costs, next_states, dones = self.buffer.sample(cfg.batch_size)
        with torch.no_grad():
            if cfg.gamma_approach == "uncertainty":
                pa, _, _ = self.policy.sample(states)
                q1g, q2g = self.critic(states, pa)
                gammas = self.gamma_module.compute_gamma(states, q1g, q2g)
            elif cfg.gamma_approach in ("return_consistency", "cross_validated"):
                gammas = self.gamma_module.compute_gamma(states)
            else:
                gammas = cfg.gamma_default
        self.current_gamma_mean = gammas.mean().item() if torch.is_tensor(gammas) else gammas

        if cfg.use_cost_constraint:
            with torch.no_grad():
                na, nlp, _ = self.policy.sample(next_states)
                ctq1, ctq2 = self.cost_critic_target(next_states, na)
                ct_target = costs + (1 - dones) * cfg.gamma_default * torch.max(ctq1, ctq2)
            cc1, cc2 = self.cost_critic(states, actions)
            cc_loss = F.mse_loss(cc1, ct_target) + F.mse_loss(cc2, ct_target)
            self.cost_critic_optimizer.zero_grad()
            cc_loss.backward()
            nn.utils.clip_grad_norm_(self.cost_critic.parameters(), cfg.max_grad_norm)
            self.cost_critic_optimizer.step()

        with torch.no_grad():
            na, nlp, _ = self.policy.sample(next_states)
            tq1, tq2 = self.critic_target(next_states, na)
            tq = torch.min(tq1, tq2) - self.alpha * nlp
            target = rewards + (1 - dones) * gammas * tq
        cq1, cq2 = self.critic(states, actions)
        q1_loss = F.mse_loss(cq1, target)
        q2_loss = F.mse_loss(cq2, target)
        c_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
        self.critic_optimizer.step()

        new_a, lp, _ = self.policy.sample(states)
        q1n, q2n = self.critic(states, new_a)
        policy_loss = (self.alpha * lp - torch.min(q1n, q2n)).mean()
        if cfg.use_cost_constraint:
            cq1n, cq2n = self.cost_critic(states, new_a)
            policy_loss += self.lagrange * torch.max(cq1n, cq2n).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.policy_optimizer.step()

        alpha_loss = torch.tensor(0.0)
        if cfg.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        gamma_loss = 0.0
        if cfg.gamma_approach == "uncertainty" and cfg.uncertainty_beta_learnable and self.num_updates % cfg.gamma_update_freq == 0:
            with torch.no_grad():
                pa2, _, _ = self.policy.sample(states)
                q1b, q2b = self.critic(states, pa2)
            gammas_grad = self.gamma_module.compute_gamma(states, q1b, q2b)
            gamma_loss = self.gamma_module.update_beta(gammas_grad)
        elif cfg.gamma_approach == "return_consistency":
            gamma_loss = self.gamma_module.update_return_consistency(self.buffer, self.critic_target, self.policy, self.alpha)
        elif cfg.gamma_approach == "cross_validated":
            gamma_loss = self.gamma_module.update_cross_validated(
                self.buffer, self.critic, self.critic_target, self.policy,
                self.alpha, self.state_dim, self.action_dim, cfg.hidden_dim
            )

        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data * (1 - cfg.tau) + p.data * cfg.tau)
        if cfg.use_cost_constraint:
            for tp, p in zip(self.cost_critic_target.parameters(), self.cost_critic.parameters()):
                tp.data.copy_(tp.data * (1 - cfg.tau) + p.data * cfg.tau)

        self.last_losses["policy"] = policy_loss.item()
        self.last_losses["q1"] = q1_loss.item()
        self.last_losses["q2"] = q2_loss.item()
        self.last_losses["alpha"] = alpha_loss.item() if torch.is_tensor(alpha_loss) else 0.0
        self.last_losses["gamma"] = gamma_loss if isinstance(gamma_loss, float) else 0.0
        self.last_losses["entropy"] = -lp.mean().item()
        if logger and self.num_updates % int(cfg.loss_log_interval_updates) == 0:
            logger.log_loss(episode, self.num_updates, self.last_losses["policy"], self.last_losses["q1"],
                            self.last_losses["q2"], self.last_losses["alpha"], self.last_losses["gamma"],
                            self.current_gamma_mean, self.alpha, self.last_losses["entropy"], self.lagrange)

    def update_lagrange(self, episode_cost):
        if not cfg.use_cost_constraint:
            return
        violation = episode_cost - cfg.cost_limit
        loss = -self.log_lagrange * violation
        self.lagrange_optimizer.zero_grad()
        loss.backward()
        self.lagrange_optimizer.step()
        with torch.no_grad():
            self.log_lagrange.clamp_(float(cfg.lagrange_log_clamp_min), float(cfg.lagrange_log_clamp_max))
        self.lagrange = self.log_lagrange.exp().item()

    def save(self, suffix=""):
        p = directory + "param/"
        torch.save(self.policy.state_dict(), p + f"policy{suffix}.pth")
        torch.save(self.critic.state_dict(), p + f"critic{suffix}.pth")
        torch.save(self.critic_target.state_dict(), p + f"critic_target{suffix}.pth")
        torch.save(self.gamma_module.state_dict(), p + f"gamma_module{suffix}.pth")
        if cfg.automatic_entropy_tuning:
            torch.save({"log_alpha": self.log_alpha}, p + f"alpha{suffix}.pth")
        if cfg.use_cost_constraint:
            torch.save(self.cost_critic.state_dict(), p + f"cost_critic{suffix}.pth")
            torch.save(self.cost_critic_target.state_dict(), p + f"cost_critic_target{suffix}.pth")
            torch.save({"log_lagrange": self.log_lagrange}, p + f"lagrange{suffix}.pth")
        print(f"  [saved {suffix}]")

    def load(self, suffix=""):
        p = directory + "param/"
        try:
            self.policy.load_state_dict(torch.load(p + f"policy{suffix}.pth", map_location=device))
            self.critic.load_state_dict(torch.load(p + f"critic{suffix}.pth", map_location=device))
            self.critic_target.load_state_dict(torch.load(p + f"critic_target{suffix}.pth", map_location=device))
            gp = p + f"gamma_module{suffix}.pth"
            if os.path.exists(gp):
                self.gamma_module.load_state_dict(torch.load(gp, map_location=device))
            if cfg.automatic_entropy_tuning:
                ap = p + f"alpha{suffix}.pth"
                if os.path.exists(ap):
                    c = torch.load(ap, map_location=device)
                    self.log_alpha = c["log_alpha"]
                    self.alpha = self.log_alpha.exp().item()
            if cfg.use_cost_constraint:
                cp = p + f"cost_critic{suffix}.pth"
                if os.path.exists(cp):
                    self.cost_critic.load_state_dict(torch.load(cp, map_location=device))
                ctp = p + f"cost_critic_target{suffix}.pth"
                if os.path.exists(ctp):
                    self.cost_critic_target.load_state_dict(torch.load(ctp, map_location=device))
                lp = p + f"lagrange{suffix}.pth"
                if os.path.exists(lp):
                    c = torch.load(lp, map_location=device)
                    self.log_lagrange = c["log_lagrange"]
                    self.lagrange = self.log_lagrange.exp().item()
            print(f"  [loaded {suffix}]")
            return True
        except FileNotFoundError as e:
            print(f"  [load failed: {e}]")
            return False


def train(agent, env, logger):
    ensure_run_directories()
    running_reward = None
    running_cost = None
    best_reward = -float("inf")
    ep_rewards = deque(maxlen=int(cfg.running_stats_window))
    ep_costs = deque(maxlen=int(cfg.running_stats_window))
    use_step_budget = cfg.max_total_steps is not None
    next_eval_at = cfg.eval_interval if cfg.eval_interval > 0 else None

    print("=" * 75)
    if cfg.run_name:
        print(f"  run_name: {cfg.run_name}")
    print(f"  SAC + Adaptive γ ({cfg.gamma_approach})  on  {cfg.env_name}")
    print(f"  Safety: {'ON' if cfg.use_cost_constraint else 'OFF'}  (limit {cfg.cost_limit})")
    print(f"  Device: {device}   State: {agent.state_dim}   Action: {agent.action_dim}")
    print(f"  γ range: [{cfg.gamma_min}, {cfg.gamma_max}]   Hidden: {cfg.hidden_dim}   Batch: {cfg.batch_size}")
    print(f"  Stop at env steps: {cfg.max_total_steps}" if use_step_budget else f"  Episodes: {cfg.num_episodes}")
    if next_eval_at:
        print(f"  Eval every {cfg.eval_interval} steps ({cfg.eval_episodes} episodes)")
    if cfg.gamma_approach == "return_consistency" and getattr(Config, "gamma_ref_adaptive", False):
        print(f"  γ_ref adaptive: every {cfg.gamma_ref_update_episodes} ep, EMA τ={cfg.gamma_ref_ema_tau} "
              f"(after warmup={getattr(Config, 'gamma_ref_update_after_warmup', True)})")
    print("=" * 75)

    episode = 0
    stop_training = False
    while True:
        if use_step_budget and agent.total_steps >= cfg.max_total_steps:
            break
        if (not use_step_budget) and episode >= cfg.num_episodes:
            break
        state, _ = env.reset(seed=cfg.random_seed + episode if cfg.seed else None)
        ep_reward, ep_cost, ep_steps = 0.0, 0.0, 0

        for _ in range(cfg.max_steps):
            if use_step_budget and agent.total_steps >= cfg.max_total_steps:
                stop_training = True
                break
            action = agent.select_action(state)
            next_state, reward, cost, terminated, truncated, _ = env_step(env, action)
            done = terminated or truncated
            agent.store(state, action, reward, cost, next_state, terminated)
            if agent.total_steps % cfg.update_every == 0:
                agent.update(logger=logger, episode=episode)

            while next_eval_at is not None and agent.total_steps >= next_eval_at:
                tester = TestModule(agent)
                s = tester.test(num_episodes=cfg.eval_episodes, deterministic=True, seed=cfg.random_seed + int(cfg.eval_seed_offset), verbose=False)["statistics"]
                print(f"  [eval @ {agent.total_steps}] R={s['mean_reward']:.1f}±{s['std_reward']:.1f} C={s['mean_cost']:.1f}±{s['std_cost']:.1f}")
                next_eval_at += cfg.eval_interval

            ep_reward += reward
            ep_cost += cost
            ep_steps += 1
            state = next_state
            if done:
                break

        if cfg.use_cost_constraint and ep_steps > 0:
            agent.update_lagrange(ep_cost)
        ep_rewards.append(ep_reward)
        ep_costs.append(ep_cost)
        if running_reward is None:
            running_reward, running_cost = ep_reward, ep_cost
        else:
            b = float(cfg.running_return_ema_beta)
            running_reward = b * running_reward + (1.0 - b) * ep_reward
            running_cost = b * running_cost + (1.0 - b) * ep_cost

        logger.log_reward(episode, ep_reward, ep_cost, running_reward, running_cost,
                          ep_steps, agent.current_gamma_mean, agent.alpha, agent.lagrange, agent.total_steps)
        if episode % cfg.print_interval == 0:
            avg_r, avg_c = np.mean(ep_rewards), np.mean(ep_costs)
            buf = f"Buf:{len(agent.buffer)}/{cfg.min_buffer_size}" if len(agent.buffer) < cfg.min_buffer_size else f"Upd:{agent.num_updates}"
            gi = f"γ:{agent.current_gamma_mean:.4f}"
            if cfg.gamma_approach == "uncertainty":
                gi += f"(β={agent.gamma_module.log_beta.exp().item():.2f})"
            li = f"λ:{agent.lagrange:.3f}" if cfg.use_cost_constraint else ""
            print(f"Ep{episode:5d} | R:{ep_reward:7.1f} AvgR:{avg_r:7.1f} | C:{ep_cost:5.1f} AvgC:{avg_c:5.1f} | "
                  f"St:{ep_steps:4d} | {gi} | α:{agent.alpha:.4f} | {li} | {buf} | steps:{agent.total_steps}")

        if running_reward > best_reward and agent.num_updates > int(cfg.min_updates_best_checkpoint):
            best_reward = running_reward
            agent.save(suffix="_best")
        if episode % cfg.log_interval == 0 and episode > 0:
            agent.save(suffix="_latest")
        episode += 1
        if (
            cfg.gamma_approach == "return_consistency"
            and getattr(Config, "gamma_ref_adaptive", False)
            and episode > 0
            and episode % int(max(1, cfg.gamma_ref_update_episodes)) == 0
        ):
            if (not getattr(Config, "gamma_ref_update_after_warmup", True)) or (
                agent.total_steps >= int(max(0, cfg.gamma_warmup_steps))
            ):
                new_ref = agent.gamma_module.refresh_gamma_ref_from_net(agent.buffer)
                if new_ref is not None:
                    print(
                        f"  [γ_ref EMA → {new_ref:.4f}]  ep={episode}  steps={agent.total_steps}",
                        flush=True,
                    )
        if stop_training:
            break

    agent.save(suffix="_final")
    print("\nTraining completed!")
    return agent


class TestModule:
    def __init__(self, agent, env_name=None, log_dir=None):
        self.agent = agent
        self.env_name = env_name if env_name is not None else cfg.env_name
        self.log_dir = log_dir if log_dir is not None else directory
        self.results = {}

    def run_episode(self, env, deterministic=True):
        state, _ = env.reset()
        ep_r, ep_c, ep_steps = 0.0, 0.0, 0
        gammas = []
        for _ in range(cfg.max_steps):
            action = self.agent.select_action(state, evaluate=deterministic)
            gammas.append(self.agent.get_gamma_for_state(state))
            next_state, reward, cost, terminated, truncated, _ = env_step(env, action)
            ep_r += reward
            ep_c += cost
            ep_steps += 1
            state = next_state
            if terminated or truncated:
                break
        return dict(reward=ep_r, cost=ep_c, steps=ep_steps,
                    avg_gamma=np.mean(gammas) if gammas else cfg.gamma_default,
                    gammas=gammas)

    def test(self, num_episodes=None, deterministic=True, render=False, seed=None, verbose=True):
        if num_episodes is None:
            num_episodes = cfg.test_episodes
        env = make_env(self.env_name, render_mode="human" if render else None)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        if verbose:
            print("\n" + "=" * 75)
            print(f"  TESTING  |  {self.env_name}  |  γ-approach: {cfg.gamma_approach}")
            print("=" * 75)
        all_r, all_c, all_s, all_g, all_data = [], [], [], [], []
        for i in range(num_episodes):
            if seed is not None:
                env.reset(seed=seed + i)
            d = self.run_episode(env, deterministic)
            all_r.append(d["reward"]); all_c.append(d["cost"]); all_s.append(d["steps"]); all_g.append(d["avg_gamma"]); all_data.append(d)
            if verbose:
                safe = "✓" if d["cost"] <= cfg.cost_limit else "✗"
                print(f"  {i+1:3d} | R:{d['reward']:8.2f} | C:{d['cost']:6.1f} | St:{d['steps']:4d} | γ:{d['avg_gamma']:.4f} | {safe}")
        env.close()
        stats = dict(
            mean_reward=np.mean(all_r), std_reward=np.std(all_r),
            mean_cost=np.mean(all_c), std_cost=np.std(all_c),
            mean_steps=np.mean(all_s), mean_gamma=np.mean(all_g),
            cost_violation_rate=np.mean(np.array(all_c) > cfg.cost_limit) * float(cfg.percent_scale)
        )
        self.results = dict(rewards=all_r, costs=all_c, steps=all_s, gammas=all_g, data=all_data, statistics=stats)
        if verbose:
            self._print_summary(stats)
        return self.results

    def _print_summary(self, s):
        print("\n" + "-" * 75)
        print("  SUMMARY")
        print("-" * 75)
        print(f"  Mean Reward : {s['mean_reward']:8.2f} ± {s['std_reward']:.2f}")
        print(f"  Mean Cost   : {s['mean_cost']:8.2f} ± {s['std_cost']:.2f}")
        print(f"  Mean Steps  : {s['mean_steps']:8.1f}")
        print(f"  Mean γ      : {s['mean_gamma']:8.4f}")
        print(f"  Cost Viol.  : {s['cost_violation_rate']:.1f}%")
        print("-" * 75)
        print("  ✓ SAFE — average cost within budget" if s["mean_cost"] <= cfg.cost_limit
              else "  ✗ UNSAFE — average cost exceeds budget")
        print("=" * 75 + "\n")

    def save_results(self, filename=None):
        if not self.results:
            print("No results.")
            return
        if filename is None:
            filename = f"test_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        fp = os.path.join(self.log_dir, filename)
        with open(fp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ep", "reward", "cost", "steps", "avg_gamma"])
            for i, (r, c, s, g) in enumerate(zip(self.results["rewards"], self.results["costs"], self.results["steps"], self.results["gammas"])):
                w.writerow([i + 1, f"{r:.2f}", f"{c:.2f}", s, f"{g:.4f}"])
        print(f"  Saved: {fp}")

    def plot_results(self, save=True, show=True):
        if not self.results:
            print("No results.")
            return
        fig, axes = plt.subplots(2, 2, figsize=(float(cfg.test_plot_figsize_w), float(cfg.test_plot_figsize_h)))
        fig.suptitle(f"SAC + Adaptive γ ({cfg.gamma_approach}) — {self.env_name}", fontsize=float(cfg.test_plot_title_fontsize), fontweight="bold")
        eps = range(1, len(self.results["rewards"]) + 1)
        ax = axes[0, 0]
        ax.bar(eps, self.results["rewards"], color="steelblue", alpha=0.7)
        ax.axhline(self.results["statistics"]["mean_reward"], color="red", ls="--",
                   label=f"Mean: {self.results['statistics']['mean_reward']:.2f}")
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax.set_title("Episode Rewards"); ax.legend(); ax.grid(True, alpha=0.3)
        ax = axes[0, 1]
        colors = ["green" if c <= cfg.cost_limit else "red" for c in self.results["costs"]]
        ax.bar(eps, self.results["costs"], color=colors, alpha=0.7)
        ax.axhline(cfg.cost_limit, color="orange", ls=":", lw=2, label=f"Limit: {cfg.cost_limit}")
        ax.axhline(self.results["statistics"]["mean_cost"], color="red", ls="--",
                   label=f"Mean: {self.results['statistics']['mean_cost']:.2f}")
        ax.set_xlabel("Episode"); ax.set_ylabel("Cost"); ax.set_title("Episode Costs"); ax.legend(); ax.grid(True, alpha=0.3)
        ax = axes[1, 0]
        ax.bar(eps, self.results["steps"], color="coral", alpha=0.7)
        ax.set_xlabel("Episode"); ax.set_ylabel("Steps"); ax.set_title("Episode Steps"); ax.grid(True, alpha=0.3)
        ax = axes[1, 1]
        ax.bar(eps, self.results["gammas"], color="forestgreen", alpha=0.7)
        ax.axhline(self.results["statistics"]["mean_gamma"], color="red", ls="--",
                   label=f"Mean: {self.results['statistics']['mean_gamma']:.4f}")
        ax.axhline(cfg.gamma_min, color="gray", ls=":", alpha=0.5)
        ax.axhline(cfg.gamma_max, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("Episode"); ax.set_ylabel("γ"); ax.set_title("Average γ per Episode"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            path = os.path.join(self.log_dir, "img", f"test_{time.strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(path, dpi=int(cfg.test_plot_save_dpi), bbox_inches="tight")
            print(f"  Plot saved: {path}")
        if show:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="SAC — all hyperparameters via JSON.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config.")
    args = parser.parse_args()
    init_runtime(load_config(args.config))
    ensure_run_directories()

    env = make_env(cfg.env_name)
    sd = env.observation_space.shape[0]
    ad = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SAC(sd, ad, max_action)
    logger = CSVLogger(directory)
    train(agent, env, logger)
    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
