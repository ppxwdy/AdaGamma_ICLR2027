import os

import pickle
import csv
import time
import random
from collections import namedtuple, deque

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from PIL import Image

try:
    import safety_gymnasium
    HAS_SAFETY_GYM = True
except ImportError:
    HAS_SAFETY_GYM = False
    print("WARNING: safety_gymnasium not installed. Install via: pip install safety-gymnasium")

# Hyperparameters: JSON only (see init_runtime).

# --------------------------------------
# 2. System Setup
# --------------------------------------

import argparse
from load_config import load_config

cfg = None
device = None
directory = None
script_name = 'ppo2_submission'


def init_runtime(c):
    global cfg, device, directory
    cfg = c
    if getattr(cfg, 'mujoco_gl', None):
        os.environ['MUJOCO_GL'] = str(cfg.mujoco_gl)
    dev = getattr(cfg, 'device', None)
    if dev is None:
        raise ValueError('config must set device (e.g. cuda:0 or cpu)')
    device = torch.device(dev)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    out = getattr(cfg, 'output_dir', None)
    if not out:
        raise ValueError('config must set output_dir')
    directory = os.path.abspath(out)
    directory = directory if directory.endswith(os.sep) else directory + os.sep


def set_experiment_directory(path):
    """Redirect checkpoints / logs to ``path``."""
    global directory
    path = os.path.abspath(path)
    directory = path if path.endswith(os.sep) else path + os.sep


def ensure_run_directories():
    os.makedirs(directory, exist_ok=True)
    os.makedirs(directory + 'param/', exist_ok=True)
    os.makedirs(directory + 'img/', exist_ok=True)
    os.makedirs(directory + 'videos/', exist_ok=True)

# --------------------------------------
# 3. Data Structures
# --------------------------------------

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])

# --------------------------------------
# 4. CSV Logger (extended with cost tracking)
# --------------------------------------

class CSVLogger:
    def __init__(self, log_dir=directory, file1='episode_rewards.csv', file2='training_loss.csv'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_file = os.path.join(log_dir, file1)
        self.loss_file = os.path.join(log_dir, file2)

        with open(self.episode_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'score', 'running_reward', 'episode_cost',
                             'running_cost', 'avg_gamma', 'timestamp'])

        with open(self.loss_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'update_step', 'actor_loss', 'critic_loss',
                             'gamma_loss', 'avg_gamma', 'entropy', 'timestamp'])

        self.start_time = time.time()

    def log_reward(self, episode, score, running_reward, episode_cost=0.0,
                   running_cost=0.0, avg_gamma=None):
        elapsed_time = time.time() - self.start_time
        gamma_str = f'{avg_gamma:.4f}' if avg_gamma is not None else 'N/A'
        with open(self.episode_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, f'{score:.2f}', f'{running_reward:.2f}',
                             f'{episode_cost:.2f}', f'{running_cost:.2f}',
                             gamma_str, f'{elapsed_time:.2f}'])

    def log_loss(self, episode, update_step, actor_loss, critic_loss, gamma_loss,
                 avg_gamma, entropy=0.0):
        elapsed_time = time.time() - self.start_time
        with open(self.loss_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, update_step, f'{actor_loss:.6f}', f'{critic_loss:.6f}',
                             f'{gamma_loss:.6f}', f'{avg_gamma:.4f}', f'{entropy:.4f}',
                             f'{elapsed_time:.2f}'])

# --------------------------------------
# 5. Gamma Network (Paper Section 4.1, Eq. 6)
#    For SafetyPointGoal1: hidden_dim=256
# --------------------------------------

class GammaNet(nn.Module):
    """Gamma MLP."""
    def __init__(self, state_dim, hidden_dim, gamma_min, gamma_max, output_bias):
        super(GammaNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        nn.init.orthogonal_(self.fc1.weight, gain=float(cfg.gamma_net_orthogonal_gain))
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=float(cfg.gamma_net_orthogonal_gain))
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, float(output_bias))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        raw = self.fc3(x)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(raw)
        return gamma  # shape: (..., 1)

# --------------------------------------
# 6. Rollout Buffer (with cost and next_state)
# --------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.logprobs = []
        self.rewards = []
        self.costs = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.costs[:]
        del self.state_values[:]
        del self.is_terminals[:]

# --------------------------------------
# 7. Actor-Critic Network
# --------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, hidden_dim):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        dim = int(hidden_dim)

        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, dim), nn.Tanh(),
                nn.Linear(dim, dim), nn.Tanh(),
                nn.Linear(dim, action_dim), nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, dim), nn.Tanh(),
                nn.Linear(dim, dim), nn.Tanh(),
                nn.Linear(dim, action_dim), nn.Softmax(dim=-1)
            )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, dim), nn.Tanh(),
            nn.Linear(dim, dim), nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

# --------------------------------------
# 8. PPO Agent with AdaGamma + Danger-Aware Loss
# --------------------------------------

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init):

        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space,
                                  action_std_init, cfg.actor_hidden_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space,
                                      action_std_init, cfg.actor_hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # --- AdaGamma: gamma network (Paper Section 4.1, 5.1) ---
        if cfg.use_adagamma:
            self.gamma_net = GammaNet(
                state_dim,
                cfg.gamma_hidden_dim,
                cfg.gamma_min,
                cfg.gamma_max,
                cfg.gamma_net_output_bias,
            ).to(device)
            self.optimizer_gamma = optim.Adam(
                self.gamma_net.parameters(), lr=cfg.gamma_lr
            )

        self.training_step = 0
        self.last_avg_gamma = gamma
        self.last_entropy = 0.0
        self.current_episode = 0

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
            self.set_action_std(self.action_std)

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state_tensor)

        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()

    def store_transition(self, next_state, reward, cost, done):
        """Store reward, cost, terminal flag, and next_state."""
        self.buffer.rewards.append(reward)
        self.buffer.costs.append(cost)
        self.buffer.is_terminals.append(done)
        self.buffer.next_states.append(torch.FloatTensor(next_state).to(device))

    def select_action_for_eval(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            action, _, _ = self.policy_old.act(state_tensor)

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()

    # -------------------------------------------------
    # Modified GAE with state-dependent gamma
    # Paper Eq. 9-10, Algorithm 2 lines 9-14
    # -------------------------------------------------
    def compute_gae_adaptive(self, rewards, values, next_values, is_terminals, gamma_values):
        """
        Eq. 9:  delta_t = r_t + gamma_phi(s_t) * V(s_{t+1}) - V(s_t)
        Eq. 10: A_hat_t = delta_t + gamma_phi(s_t) * lambda * A_hat_{t+1}

        Produces product-of-gammas weighting (Proposition 1, Eq. 11).
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=device)
        gae = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - is_terminals[t]
            gamma_t = gamma_values[t]
            delta = rewards[t] + gamma_t * next_values[t] * mask - values[t]
            gae = delta + gamma_t * cfg.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def compute_gae_fixed(self, rewards, values, next_values, is_terminals, gamma):
        T = len(rewards)
        advantages = torch.zeros(T, device=device)
        gae = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - is_terminals[t]
            delta = rewards[t] + gamma * next_values[t] * mask - values[t]
            gae = delta + gamma * cfg.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    # -------------------------------------------------
    # Danger-Aware Auxiliary Loss
    # Paper lines 308-310:
    #   "danger-aware auxiliary loss that regresses gamma_phi(s)
    #    toward a risk-conditioned target in [0.92, 0.995]
    #    computed from short-horizon cost accumulation"
    # -------------------------------------------------
    def _compute_danger_target(self, states, costs, is_terminals):
        """
        Compute risk-conditioned gamma target per state based on
        short-horizon cost accumulation.

        For each timestep, accumulate costs in a short forward window.
        High cost => lower gamma target (more myopic near danger).
        Low cost => higher gamma target (longer horizon when safe).

        Parameters from paper lines 309-310:
          scale=2.0, threshold=0.25, temperature=0.10
          target range: [0.92, 0.995]
        """
        T = len(costs)
        # Short-horizon cost accumulation (look ahead ~10 steps)
        horizon = min(10, T)
        cost_accum = torch.zeros(T, device=device)

        # Running backward sum of costs in short window
        running_cost = 0.0
        count = 0
        for t in reversed(range(T)):
            if is_terminals[t] > 0.5:
                running_cost = 0.0
                count = 0
            running_cost += costs[t]
            count += 1
            if count > horizon:
                running_cost -= costs[min(t + horizon, T - 1)]
                count = horizon
            cost_accum[t] = running_cost

        # Normalize cost accumulation
        cost_normed = cost_accum * cfg.danger_scale

        # Sigmoid mapping: high cost => low gamma target
        # danger_score in [0, 1], 1 = very dangerous
        danger_score = torch.sigmoid(
            (cost_normed - cfg.danger_threshold)
            / max(cfg.danger_temperature, float(cfg.danger_temperature_floor))
        )

        # Map danger_score to gamma target: [danger_gamma_high, danger_gamma_low]
        # danger=0 => gamma_high (safe, long horizon)
        # danger=1 => gamma_low (dangerous, short horizon)
        gamma_target = (cfg.danger_gamma_high
                        - (cfg.danger_gamma_high - cfg.danger_gamma_low) * danger_score)

        return gamma_target

    # -------------------------------------------------
    # Return-Consistency + Danger-Aware + Regularization
    # Paper Section 4.3.2 (Eq. 13), 4.3.4 (Eq. 15-16)
    # + SafetyPointGoal1-specific danger-aware loss
    # -------------------------------------------------
    def _compute_gamma_loss(self, states, next_states, rewards, costs, is_terminals):
        """
        Full gamma network training objective (Eq. 15 + danger-aware):
          J_gamma(phi) = rc_weight * L^RC(phi)
                       + danger_weight * L_danger(phi)
                       + lambda_dev * E[(gamma_phi(s) - gamma_target)^2]
                       + lambda_var * Var[gamma_phi(s)]
                       + lambda_bound * L_boundary
        """
        T = len(rewards)
        n = cfg.rc_horizon
        gamma_bar = cfg.rc_ref_gamma

        # --- Predict gamma with gradients ---
        gamma_pred = self.gamma_net(states).squeeze(-1)  # (T,)

        with torch.no_grad():
            v_next = self.policy.critic(next_states).squeeze(-1)
            v_states = self.policy.critic(states).squeeze(-1)
            v_last = self.policy.critic(next_states[-1:]).squeeze(-1)
            v_extended = torch.cat([v_states, v_last])  # (T+1,)

            # --- n-step returns under fixed reference gamma_bar (Eq. 13) ---
            G_n = torch.zeros(T, device=device)
            for t in range(T):
                g = 0.0
                gamma_power = 1.0
                terminated = False
                k = 0
                while k < n:
                    if t + k >= T:
                        break
                    g += gamma_power * rewards[t + k]
                    gamma_power *= gamma_bar
                    if is_terminals[t + k] > 0.5:
                        terminated = True
                        break
                    k += 1

                if not terminated:
                    bootstrap_idx = t + k
                    if bootstrap_idx <= T:
                        g += gamma_power * v_extended[bootstrap_idx]
                G_n[t] = g

        # --- One-step bootstrap under learned gamma (Eq. 13) ---
        mask = 1.0 - is_terminals
        v_hat_1 = rewards + gamma_pred * v_next.detach() * mask

        # --- Return-consistency loss (Eq. 13) ---
        L_rc = ((v_hat_1 - G_n.detach()) ** 2).mean()

        # --- Danger-aware auxiliary loss (Paper lines 308-310) ---
        L_danger = torch.tensor(0.0, device=device)
        if cfg.use_danger_aware:
            danger_gamma_target = self._compute_danger_target(states, costs, is_terminals)
            L_danger = ((gamma_pred - danger_gamma_target.detach()) ** 2).mean()

        # --- Deviation penalty (Eq. 15) ---
        L_dev = ((gamma_pred - cfg.gamma_target) ** 2).mean()

        # --- Variance penalty (Eq. 15) ---
        L_var = gamma_pred.var()

        # --- Boundary penalty (Eq. 16) ---
        L_boundary = (
            F.relu(cfg.gamma_min + cfg.epsilon_bound - gamma_pred) +
            F.relu(gamma_pred - cfg.gamma_max + cfg.epsilon_bound)
        ).mean()

        # --- Full objective ---
        total_loss = (cfg.rc_weight * L_rc
                      + cfg.danger_weight * L_danger
                      + cfg.lambda_dev * L_dev
                      + cfg.lambda_var * L_var
                      + cfg.lambda_bound * L_boundary)

        return total_loss, gamma_pred.detach()

    def refresh_rc_ref_from_gamma_net(self, states):
        """EMA-update reference discount for G_n (SAC3-style), clip mean to [gamma_min, gamma_max]."""
        if not cfg.rc_ref_adaptive or not cfg.use_adagamma:
            return None
        with torch.no_grad():
            g_batch = self.gamma_net(states).mean().item()
        g_batch = float(np.clip(g_batch, cfg.gamma_min, cfg.gamma_max))
        tau = float(np.clip(cfg.rc_ref_ema_tau, 0.0, 1.0))
        cfg.rc_ref_gamma = (1.0 - tau) * float(cfg.rc_ref_gamma) + tau * g_batch
        if self.training_step % int(cfg.rc_ref_log_interval_updates) == 0:
            print(f"  [rc_ref EMA → {cfg.rc_ref_gamma:.4f}]  ppo_upd={self.training_step}", flush=True)
        return cfg.rc_ref_gamma

    # -------------------------------------------------
    # Main PPO Update (Algorithm 2 adapted for Safety)
    # -------------------------------------------------
    def update(self, logger=None, episode=0):
        self.training_step += 1
        self.current_episode = episode

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(
            torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        next_states = torch.stack(self.buffer.next_states, dim=0).detach().to(device)

        rewards_raw = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(device)
        costs = torch.tensor(self.buffer.costs, dtype=torch.float32).to(device)
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(device)

        # Optional: penalize cost in reward
        rewards = rewards_raw - cfg.cost_coef * costs

        # Normalize rewards for GAE
        rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + float(cfg.normalize_std_epsilon))

        # ============================================================
        # Step 1 & 2: Frozen gamma + Modified GAE
        # ============================================================
        with torch.no_grad():
            next_values = self.policy.critic(next_states).squeeze(-1)
            next_values = next_values * (1.0 - is_terminals)

            if cfg.use_adagamma:
                if episode >= cfg.gamma_warmup_episodes:
                    gamma_values = self.gamma_net(old_states).squeeze(-1)
                else:
                    gamma_values = torch.full_like(rewards, cfg.gamma_max)

                advantages, returns = self.compute_gae_adaptive(
                    rewards_normalized, old_state_values, next_values,
                    is_terminals, gamma_values
                )
                self.last_avg_gamma = gamma_values.mean().item()
            else:
                advantages, returns = self.compute_gae_fixed(
                    rewards_normalized, old_state_values, next_values,
                    is_terminals, self.gamma
                )
                self.last_avg_gamma = self.gamma

            advantages = (advantages - advantages.mean()) / (advantages.std() + float(cfg.normalize_std_epsilon))

        # ============================================================
        # Step 3: PPO epochs (frozen gamma)
        # ============================================================
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        update_count = 0

        for epoch in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = (-torch.min(surr1, surr2).mean()
                          - cfg.entropy_coef * dist_entropy.mean())

            value_pred_clipped = old_state_values + torch.clamp(
                state_values - old_state_values, -self.eps_clip, self.eps_clip
            )
            value_loss1 = self.MseLoss(state_values, returns)
            value_loss2 = self.MseLoss(value_pred_clipped, returns)
            critic_loss = torch.max(value_loss1, value_loss2)

            loss = actor_loss + float(cfg.critic_loss_coef) * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            self.optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += dist_entropy.mean().item()
            update_count += 1

        # ============================================================
        # Step 4: Update gamma network (return-consistency + danger-aware)
        # ============================================================
        gamma_loss_val = 0.0
        if cfg.use_adagamma and episode >= cfg.gamma_warmup_episodes:
            gamma_loss, gamma_pred = self._compute_gamma_loss(
                old_states, next_states, rewards_normalized, costs, is_terminals
            )
            self.optimizer_gamma.zero_grad()
            gamma_loss.backward()
            nn.utils.clip_grad_norm_(self.gamma_net.parameters(), cfg.max_grad_norm)
            self.optimizer_gamma.step()
            gamma_loss_val = gamma_loss.item()
            self.last_avg_gamma = gamma_pred.mean().item()
            if (cfg.rc_ref_adaptive
                    and self.training_step % max(1, cfg.rc_ref_update_every_ppo_updates) == 0):
                if (not cfg.rc_ref_update_after_warmup) or (episode >= cfg.gamma_warmup_episodes):
                    self.refresh_rc_ref_from_gamma_net(old_states)

        # --- Logging ---
        if logger and update_count > 0:
            avg_actor_loss = total_actor_loss / update_count
            avg_critic_loss = total_critic_loss / update_count
            avg_entropy = total_entropy / update_count
            self.last_entropy = avg_entropy
            logger.log_loss(episode, self.training_step, avg_actor_loss, avg_critic_loss,
                            gamma_loss_val, self.last_avg_gamma, avg_entropy)

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    # -------------------------------------------------
    # Save / Load
    # -------------------------------------------------
    def save(self, checkpoint_path):
        model_info = {
            'policy_state_dict': self.policy_old.state_dict(),
            'use_adagamma': cfg.use_adagamma,
            'gamma': self.gamma,
            'gamma_min': cfg.gamma_min if cfg.use_adagamma else None,
            'gamma_max': cfg.gamma_max if cfg.use_adagamma else None,
            'seed': cfg.random_seed,
            'env_name': cfg.env_name,
        }
        torch.save(model_info, checkpoint_path)

        if cfg.use_adagamma:
            gamma_path = checkpoint_path.replace('.pth', '_gamma.pth')
            torch.save(self.gamma_net.state_dict(), gamma_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
            self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            print(f"Model Info:")
            print(f"  - Env: {checkpoint.get('env_name', 'Unknown')}")
            print(f"  - AdaGamma: {checkpoint.get('use_adagamma', 'Unknown')}")
            print(f"  - Gamma: {checkpoint.get('gamma', 'Unknown')}")
            print(f"  - Seed: {checkpoint.get('seed', 'Unknown')}")
        else:
            self.policy_old.load_state_dict(checkpoint)
            self.policy.load_state_dict(checkpoint)
            print("Old format model loaded (no metadata)")

        if cfg.use_adagamma:
            gamma_path = checkpoint_path.replace('.pth', '_gamma.pth')
            if os.path.exists(gamma_path):
                self.gamma_net.load_state_dict(
                    torch.load(gamma_path, map_location=lambda storage, loc: storage))
                print(f"Gamma network loaded from: {gamma_path}")
            else:
                print("Warning: Gamma network file not found!")

# --------------------------------------
# 9. Environment Wrapper for Safety Gymnasium
# --------------------------------------

def make_env(env_name, render_mode=None):
    """Same pattern as SAC: prefer safety_gymnasium when available."""
    if HAS_SAFETY_GYM:
        return safety_gymnasium.make(env_name, render_mode=render_mode)
    return gym.make(env_name, render_mode=render_mode)


def make_safety_env(env_name, render_mode=None):
    """Alias for backward compatibility."""
    return make_env(env_name, render_mode=render_mode)


def extract_cost(info, raw_outputs=None):
    """
    Extract cost from safety_gymnasium step output.
    safety_gymnasium may return cost as:
      - 3rd return value: obs, reward, cost, terminated, truncated, info
      - or inside info dict: info['cost']
    """
    if raw_outputs is not None and len(raw_outputs) == 6:
        # (obs, reward, cost, terminated, truncated, info)
        return raw_outputs[2]
    if isinstance(info, dict) and 'cost' in info:
        return float(info['cost'])
    return 0.0

# --------------------------------------
# 10. Training Function
# --------------------------------------

def eval_policy(ppo_agent, env, num_episodes, seed_base):
    """Deterministic rollout stats for periodic eval (no gradient)."""
    rewards = []
    costs = []
    for i in range(num_episodes):
        state, info = env.reset(seed=seed_base + i)
        ep_r = 0.0
        ep_c = 0.0
        for _ in range(cfg.max_steps):
            action = ppo_agent.select_action_for_eval(state)
            step_result = env.step(action)
            if len(step_result) == 6:
                state, reward, cost, terminated, truncated, info = step_result
            else:
                state, reward, terminated, truncated, info = step_result
                cost = float(info.get('cost', float(cfg.default_env_cost_when_missing)))
            ep_r += reward
            ep_c += cost
            if terminated or truncated:
                break
        rewards.append(ep_r)
        costs.append(ep_c)
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_cost': float(np.mean(costs)),
        'std_cost': float(np.std(costs)),
    }


def train(ppo_agent, env, csv_logger):
    ensure_run_directories()
    training_records = []
    running_reward = 0
    running_cost = 0
    best_running_reward = -float('inf')
    time_step = 0
    i_episode = 0

    use_step_budget = cfg.max_total_steps is not None
    next_eval_at = cfg.eval_interval if cfg.eval_interval > 0 else None
    stop_training = False

    print("=" * 70)
    if cfg.run_name:
        print(f"run_name: {cfg.run_name}")
    print(f"PPO + AdaGamma Training on {cfg.env_name}")
    print(f"Use AdaGamma: {cfg.use_adagamma}")
    if cfg.use_adagamma:
        print(f"Gamma Range: [{cfg.gamma_min}, {cfg.gamma_max}]")
        print(f"Gamma Hidden Dim: {cfg.gamma_hidden_dim}")
        print(f"Gamma Target: {cfg.gamma_target}")
        print(f"Return-Consistency: n={cfg.rc_horizon}, gamma_bar(init)={cfg.rc_ref_init}, "
              f"adaptive_ref={getattr(Config, 'rc_ref_adaptive', False)}")
        if getattr(Config, 'rc_ref_adaptive', False):
            print(f"  rc_ref EMA: tau={cfg.rc_ref_ema_tau}, "
                  f"every {max(1, cfg.rc_ref_update_every_ppo_updates)} PPO update(s), "
                  f"after_warmup={cfg.rc_ref_update_after_warmup}")
        print(f"Danger-Aware Loss: {cfg.use_danger_aware}")
        if cfg.use_danger_aware:
            print(f"  weight={cfg.danger_weight}, scale={cfg.danger_scale}, "
                  f"threshold={cfg.danger_threshold}, temp={cfg.danger_temperature}")
            print(f"  gamma range: [{cfg.danger_gamma_low}, {cfg.danger_gamma_high}]")
        print(f"Regularization: lambda_dev={cfg.lambda_dev}, "
              f"lambda_var={cfg.lambda_var}, lambda_bound={cfg.lambda_bound}")
        print(f"Warmup Episodes: {cfg.gamma_warmup_episodes}")
    print(f"Cost Limit: {cfg.cost_limit}")
    print(f"GAE Lambda: {cfg.gae_lambda}")
    print(f"Device: {device}")
    if use_step_budget:
        print(f"Stop at env steps: {cfg.max_total_steps}")
    else:
        print(f"Max episodes: {cfg.max_episode}")
    if next_eval_at:
        print(f"Eval every {cfg.eval_interval} steps ({cfg.eval_episodes} episodes)")
    print("=" * 70)

    while True:
        if use_step_budget and time_step >= cfg.max_total_steps:
            break
        if not use_step_budget and i_episode >= cfg.max_episode:
            break

        state, info = env.reset(
            seed=cfg.random_seed if cfg.seed and i_episode == 0 else None)
        current_ep_reward = 0
        current_ep_cost = 0

        for t in range(1, cfg.max_steps + 1):
            if use_step_budget and time_step >= cfg.max_total_steps:
                stop_training = True
                break

            action = ppo_agent.select_action(state)

            # safety_gymnasium step: may return 5 or 6 values
            step_result = env.step(action)

            if len(step_result) == 6:
                # (obs, reward, cost, terminated, truncated, info)
                next_state, reward, cost, terminated, truncated, info = step_result
            elif len(step_result) == 5:
                # (obs, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = step_result
                cost = float(info.get('cost', float(cfg.default_env_cost_when_missing)))
            else:
                raise ValueError(f"Unexpected step output length: {len(step_result)}")

            done = terminated or truncated

            ppo_agent.store_transition(next_state, reward, cost, done)

            time_step += 1
            current_ep_reward += reward
            current_ep_cost += cost

            if time_step % cfg.buffer_capacity == 0:
                ppo_agent.update(logger=csv_logger, episode=i_episode)

            while (next_eval_at is not None and time_step >= next_eval_at):
                stats = eval_policy(
                    ppo_agent, env, cfg.eval_episodes,
                    cfg.random_seed + 999)
                print(f"  [eval @ {time_step}] "
                      f"R={stats['mean_reward']:.1f}±{stats['std_reward']:.1f} "
                      f"C={stats['mean_cost']:.1f}±{stats['std_cost']:.1f}")
                next_eval_at += cfg.eval_interval

            if (cfg.has_continuous_action_space and
                    time_step % cfg.action_std_decay_freq == 0):
                ppo_agent.decay_action_std(cfg.action_std_decay_rate, cfg.min_action_std)

            state = next_state

            if done:
                break

        if stop_training:
            break

        # Episode bookkeeping
        if i_episode == 0:
            running_reward = current_ep_reward
            running_cost = current_ep_cost
        else:
            br = float(cfg.running_return_ema_beta)
            running_reward = running_reward * br + current_ep_reward * (1.0 - br)
            running_cost = running_cost * br + current_ep_cost * (1.0 - br)

        training_records.append(TrainingRecord(i_episode, running_reward))

        current_gamma = ppo_agent.last_avg_gamma
        csv_logger.log_reward(i_episode, current_ep_reward, running_reward,
                              current_ep_cost, running_cost, current_gamma)

        if i_episode % cfg.log_interval == 0:
            warmup_str = (" (warmup)" if cfg.use_adagamma
                          and i_episode < cfg.gamma_warmup_episodes else "")
            gamma_str = (f"γ: {current_gamma:.4f}{warmup_str}"
                         if cfg.use_adagamma else f"γ: {cfg.gamma}")
            cost_flag = " ⚠" if current_ep_cost > cfg.cost_limit else ""
            print(f'Ep {i_episode:4d} | R: {current_ep_reward:7.2f} | '
                  f'Avg R: {running_reward:7.2f} | '
                  f'Cost: {current_ep_cost:5.1f} (avg {running_cost:5.1f}){cost_flag} | '
                  f'{gamma_str} | Ent: {ppo_agent.last_entropy:.3f}')

        if running_reward > best_running_reward:
            best_running_reward = running_reward
            if i_episode > 50:
                checkpoint_path = directory + 'param/PPO_best.pth'
                ppo_agent.save(checkpoint_path)

        if i_episode % cfg.save_interval == 0 and i_episode > 0:
            checkpoint_path = directory + f'param/PPO_{i_episode}.pth'
            ppo_agent.save(checkpoint_path)

        if not use_step_budget and running_reward > cfg.solved_reward:
            print("=" * 70)
            print(f"Solved! Moving average score is now {running_reward:.2f}!")
            print(f"Average cost: {running_cost:.2f} (limit: {cfg.cost_limit})")
            print("=" * 70)
            checkpoint_path = directory + 'param/PPO_solved.pth'
            ppo_agent.save(checkpoint_path)
            break

        i_episode += 1

    # Final save
    checkpoint_path = directory + 'param/PPO_final.pth'
    ppo_agent.save(checkpoint_path)

    with open(directory + 'training_records.pkl', 'wb') as f:
        pickle.dump(training_records, f)

    return training_records

# --------------------------------------
# 11. Testing Function
# --------------------------------------

def test(ppo_agent, env, num_episodes=10, seed=None):
    print("\n" + "=" * 60)
    print(f"Testing on {cfg.env_name}...")
    print("=" * 60)

    total_rewards = []
    total_costs = []

    for i in range(num_episodes):
        reset_seed = (seed + i) if seed is not None else None
        state, info = env.reset(seed=reset_seed)
        episode_reward = 0
        episode_cost = 0
        steps = 0

        for t in range(cfg.max_steps):
            action = ppo_agent.select_action_for_eval(state)
            step_result = env.step(action)

            if len(step_result) == 6:
                state, reward, cost, terminated, truncated, info = step_result
            else:
                state, reward, terminated, truncated, info = step_result
                cost = float(info.get('cost', float(cfg.default_env_cost_when_missing)))

            episode_reward += reward
            episode_cost += cost
            steps += 1

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        total_costs.append(episode_cost)
        cost_flag = " ⚠" if episode_cost > cfg.cost_limit else " ✓"
        print(f"Test Ep {i + 1:2d} | Reward: {episode_reward:7.2f} | "
              f"Cost: {episode_cost:5.1f}{cost_flag} | Steps: {steps:4d}")

    print("-" * 60)
    print(f"Avg Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Avg Cost:   {np.mean(total_costs):.2f} ± {np.std(total_costs):.2f}")
    print(f"Cost Limit: {cfg.cost_limit}")

# --------------------------------------
# 12. Video Recording
# --------------------------------------

def record_video(ppo_agent, env_name, checkpoint_path, output_dir,
                 num_episodes, max_steps, video_eval_action_std):
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make(env_name, render_mode='rgb_array')
    try:
        from gymnasium.wrappers import RecordVideo
        env = RecordVideo(env, output_dir, episode_trigger=lambda x: True)
    except ImportError:
        print("Warning: RecordVideo wrapper not available")

    ppo_agent.load(checkpoint_path)

    if ppo_agent.has_continuous_action_space:
        ppo_agent.set_action_std(float(video_eval_action_std))

    total_rewards = []
    total_costs = []

    for ep in range(num_episodes):
        state, info = env.reset(seed=cfg.random_seed + ep)
        episode_reward = 0
        episode_cost = 0

        for step in range(max_steps):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                action_mean = ppo_agent.policy_old.actor(state_tensor)
                action = action_mean.cpu().numpy().flatten()

            step_result = env.step(action)

            if len(step_result) == 6:
                state, reward, cost, terminated, truncated, info = step_result
            else:
                state, reward, terminated, truncated, info = step_result
                cost = float(info.get('cost', float(cfg.default_env_cost_when_missing)))

            episode_reward += reward
            episode_cost += cost

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        total_costs.append(episode_cost)
        print(f"Episode {ep + 1}: Reward = {episode_reward:.2f}, "
              f"Cost = {episode_cost:.1f}, Steps = {step + 1}")

    env.close()
    print(f"\nAvg Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Avg Cost:   {np.mean(total_costs):.2f} ± {np.std(total_costs):.2f}")

# --------------------------------------
# 13. GIF Creation
# --------------------------------------

def create_gif(ppo_agent, env_name, checkpoint_path, output_path, max_steps, fps, frame_duration_ms_base):
    print("\n" + "=" * 60)
    print("Creating GIF...")
    print("=" * 60)

    ppo_agent.load(checkpoint_path)

    env = gym.make(env_name, render_mode='rgb_array')

    frames = []
    state, info = env.reset()
    episode_reward = 0
    episode_cost = 0

    for step in range(max_steps):
        frame = env.render()
        frames.append(Image.fromarray(frame))

        action = ppo_agent.select_action_for_eval(state)
        step_result = env.step(action)

        if len(step_result) == 6:
            state, reward, cost, terminated, truncated, info = step_result
        else:
            state, reward, terminated, truncated, info = step_result
            cost = float(info.get('cost', float(cfg.default_env_cost_when_missing)))

        episode_reward += reward
        episode_cost += cost

        if terminated or truncated:
            break

    env.close()

    print(f"Frames: {len(frames)}, Reward: {episode_reward:.2f}, Cost: {episode_cost:.1f}")

    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(frame_duration_ms_base) // int(fps),
            loop=0
        )
        print(f"GIF saved to: {output_path}")

# --------------------------------------
# 14. Plotting
# --------------------------------------

def plot_training(training_records, save_path=None, show=None):
    episodes = [r.ep for r in training_records]
    rewards = [r.reward for r in training_records]

    plt.figure(figsize=(float(cfg.plot_figsize_w), float(cfg.plot_figsize_h)))
    plt.plot(episodes, rewards, color=str(cfg.training_plot_line_color), linestyle='-', alpha=float(cfg.training_plot_line_alpha))
    plt.xlabel('Episode')
    plt.ylabel('Running Reward')
    plt.title(f'PPO + AdaGamma on {cfg.env_name}')
    plt.grid(True, alpha=float(cfg.training_plot_grid_alpha))

    if save_path:
        plt.savefig(save_path, dpi=int(cfg.training_plot_save_dpi), bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show is None:
        show = save_path is None
    if show:
        plt.show()
    else:
        plt.close()

# --------------------------------------
# 15. Main
# --------------------------------------

def main():
    if cfg.seed:
        torch.manual_seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        random.seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)
            torch.backends.cudnn.deterministic = True

    if not HAS_SAFETY_GYM:
        print("ERROR: safety_gymnasium required. Install: pip install safety-gymnasium")
        print("Attempting to continue (may fail)...")

    render_mode = 'human' if cfg.render else None
    env = make_safety_env(cfg.env_name, render_mode=render_mode)

    state_dim = env.observation_space.shape[0]
    if cfg.has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    print(f"\nEnvironment: {cfg.env_name}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}\n")

    ppo_agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=cfg.learning_rate_actor,
        lr_critic=cfg.learning_rate_critic,
        gamma=cfg.gamma,
        K_epochs=cfg.ppo_epoch,
        eps_clip=cfg.clip_param,
        has_continuous_action_space=cfg.has_continuous_action_space,
        action_std_init=cfg.action_std_init
    )

    if cfg.load:
        checkpoint_path = directory + 'param/PPO_best.pth'
        ppo_agent.load(checkpoint_path)
        print(f"Loaded model from {checkpoint_path}")

    # =====================
    # TRAIN MODE
    # =====================
    if cfg.mode == 'train':
        suffix = '_adagamma' if cfg.use_adagamma else '_fixed_gamma'
        csv_logger = CSVLogger(
            log_dir=directory,
            file1=f'episode_rewards{suffix}_seed_{cfg.random_seed}.csv',
            file2=f'training_loss{suffix}_seed_{cfg.random_seed}.csv'
        )
        training_records = train(ppo_agent, env, csv_logger)

        plot_path = directory + f'img/training_curve{suffix}.png'
        plot_training(training_records, save_path=plot_path)

        print("\nRunning evaluation after training...")
        test(ppo_agent, env, num_episodes=int(cfg.post_train_eval_episodes))

        env.close()

        print("\nRecording video of trained agent...")
        checkpoint_path = directory + 'param/PPO_best.pth'
        if os.path.exists(checkpoint_path):
            video_dir = directory + 'videos/'
            record_video(
                ppo_agent=ppo_agent,
                env_name=cfg.env_name,
                checkpoint_path=checkpoint_path,
                output_dir=video_dir,
                num_episodes=int(cfg.video_episodes),
                max_steps=int(cfg.video_max_steps),
                video_eval_action_std=float(cfg.video_eval_action_std),
            )

            gif_path = directory + 'safety_point_goal.gif'
            create_gif(
                ppo_agent=ppo_agent,
                env_name=cfg.env_name,
                checkpoint_path=checkpoint_path,
                output_path=gif_path,
                max_steps=int(cfg.gif_max_steps),
                fps=int(cfg.gif_fps),
                frame_duration_ms_base=int(cfg.gif_frame_duration_ms_base),
            )

    # =====================
    # TEST MODE
    # =====================
    elif cfg.mode == 'test':
        checkpoint_path = directory + 'param/PPO_best.pth'
        ppo_agent.load(checkpoint_path)
        test_env = make_safety_env(cfg.env_name, render_mode='human')
        test(ppo_agent, test_env, num_episodes=int(cfg.post_train_eval_episodes))
        test_env.close()
        env.close()

    # =====================
    # RECORD MODE
    # =====================
    elif cfg.mode == 'record':
        env.close()
        checkpoint_path = directory + 'param/PPO_best.pth'
        video_dir = directory + 'videos/'

        record_video(
            ppo_agent=ppo_agent,
            env_name=cfg.env_name,
            checkpoint_path=checkpoint_path,
            output_dir=video_dir,
            num_episodes=int(cfg.video_episodes),
            max_steps=int(cfg.video_max_steps),
            video_eval_action_std=float(cfg.video_eval_action_std),
        )

        gif_path = directory + 'safety_point_goal.gif'
        create_gif(
            ppo_agent=ppo_agent,
            env_name=cfg.env_name,
            checkpoint_path=checkpoint_path,
            output_path=gif_path,
            max_steps=int(cfg.gif_max_steps),
            fps=int(cfg.gif_fps),
            frame_duration_ms_base=int(cfg.gif_frame_duration_ms_base),
        )


def _entry_main():
    parser = argparse.ArgumentParser(description='PPO — JSON config only.')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    init_runtime(load_config(args.config))
    ensure_run_directories()
    main()


if __name__ == '__main__':
    _entry_main()