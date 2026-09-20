"""Small dependency-free PPO (numpy) for the SEA quadruped.

Trains an MLP Gaussian policy + value network with GAE, clipped surrogate,
entropy bonus, observation normalization, and Adam.  No PyTorch required.

API::

    agent = Agent(obs_dim, act_dim)
    mu, logp, v = agent.act(obs, sample=True)   # obs: (N, obs_dim)
    agent.update_obs_stats(obs)                 # rolling obs normalization
    agent.train(obs, acts, rets, advs, oldlogp) # update after each rollout
    agent.save(path) / agent.load(path)
"""

from __future__ import annotations

import numpy as np


def _mlp_init(dims, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    p = {}
    for k in range(len(dims) - 1):
        fan_in = dims[k]
        p[f"w{k}"] = rng.normal(0.0, scale / np.sqrt(fan_in), (fan_in, dims[k + 1]))
        p[f"b{k}"] = np.zeros(dims[k + 1])
    return p


def _mlp_forward(p, x, keep=False):
    h = x
    hs = [x]
    n = sum(1 for k in p if k.startswith("w"))
    for k in range(n - 1):
        h = np.tanh(h @ p[f"w{k}"] + p[f"b{k}"])
        hs.append(h)
    out = h @ p[f"w{n-1}"] + p[f"b{n-1}"]
    return (out, hs) if keep else out


def _mlp_backward(p, x, hs, d_out):
    """Backprop through a tanh MLP whose output gradient is d_out (…, outdim)."""
    n = sum(1 for k in p if k.startswith("w"))
    # pre-activation of the last affine layer is hs[-1] (tanh output) except the
    # input layer; output layer is linear so d_h_last = d_out.
    d_h = d_out
    g = {}
    g[f"w{n-1}"] = hs[-1].T @ d_h
    g[f"b{n-1}"] = d_h.sum(0)
    for k in range(n - 2, -1, -1):
        d_h = d_h @ p[f"w{k+1}"].T
        d_h = d_h * (1 - hs[k + 1] * hs[k + 1])   # d(tanh)
        g[f"w{k}"] = hs[k].T @ d_h
        g[f"b{k}"] = d_h.sum(0)
    return g


class Agent:
    def __init__(self, obs_dim, act_dim, hidden=64, lr=3e-4, seed=0,
                 gamma=0.99, lam=0.95, clip=0.2, ent_coef=1e-2, vf_coef=0.5,
                 epochs=4, minibatch=64, max_grad_norm=0.5):
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.epochs, self.minibatch = epochs, minibatch
        self.max_grad_norm = max_grad_norm

        self.policy = _mlp_init([obs_dim, hidden, hidden, act_dim], seed=seed)
        self.value = _mlp_init([obs_dim, hidden, hidden, 1], seed=seed + 1)
        self.log_std = np.full(act_dim, -0.7)

        self.lr = lr
        self._adam = {n: {"m": np.zeros_like(v), "v": np.zeros_like(v), "t": 0}
                      for n, v in self._params().items()}

        self.obs_mean = np.zeros(obs_dim)
        self.obs_var = np.ones(obs_dim)

    # ---- params / obs norm --------------------------------------------------
    # Policy and value nets both use "w{k}"/"b{k}" keys, so they are merged
    # under distinct namespaces ("pi_" / "vf_") to avoid silent key clashes.
    def _params(self):
        p = {f"pi_{k}": v for k, v in self.policy.items()}
        p.update({f"vf_{k}": v for k, v in self.value.items()})
        p["log_std"] = self.log_std
        return p

    def set_params(self, p):
        for k, v in p.items():
            if k.startswith("pi_"):
                self.policy[k[3:]] = v
            elif k.startswith("vf_"):
                self.value[k[3:]] = v
        self.log_std = p["log_std"]

    def _norm(self, obs):
        return (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-4)

    def update_obs_stats(self, obs):
        x = np.asarray(obs, dtype=np.float64).reshape(-1, self.obs_dim)
        mean = x.mean(0)
        var = x.var(0) + 1e-4
        # EMA keeps normalization tracking the policy's own distribution
        self.obs_mean = 0.995 * self.obs_mean + 0.005 * mean
        self.obs_var = 0.995 * self.obs_var + 0.005 * var

    # ---- inference ----------------------------------------------------------
    def act(self, obs, sample=True):
        x = self._norm(np.asarray(obs, dtype=np.float64))
        mu = _mlp_forward(self.policy, x)
        n = mu.shape[0]
        std = np.exp(self.log_std) * np.ones((n, 1))
        a = mu + (std * np.random.default_rng().normal(size=mu.shape)) if sample else mu
        logp = (-0.5 * ((a - mu) / std) ** 2 - self.log_std - 0.5 * np.log(2 * np.pi)).sum(1)
        v = _mlp_forward(self.value, x)[:, 0]
        return a, logp, v

    def act_eval(self, obs):
        x = self._norm(np.asarray(obs, dtype=np.float64))
        return _mlp_forward(self.policy, x)

    # ---- GAE ---------------------------------------------------------------
    def gae(self, rewards, dones, truncs, values, v_last):
        """(H, N) arrays; v_last (N,) value after the final step."""
        H, N = rewards.shape
        advs = np.zeros((H, N))
        gae_t = np.zeros(N)
        for t in reversed(range(H)):
            nv = np.where(truncs[t], v_last, 0.0)   # no bootstrap on crash
            delta = rewards[t] + self.gamma * nv - values[t]
            gae_t = delta + self.gamma * self.lam * np.where(dones[t], 0.0, gae_t)
            advs[t] = gae_t
        rets = advs + values
        return advs, rets

    # ---- training -----------------------------------------------------------
    def _adam_update(self):
        grad_norm = np.sqrt(sum(
            np.sum(np.asarray(g) ** 2) for g in self._grad.values()))
        scale = min(1.0, self.max_grad_norm / (grad_norm + 1e-8))
        for name, g in self._grad.items():
            g *= scale
            st = self._adam[name]
            st["t"] += 1
            st["m"] = 0.9 * st["m"] + 0.1 * g
            st["v"] = 0.999 * st["v"] + 0.001 * g * g
            mhat = st["m"] / (1 - 0.9 ** st["t"])
            vhat = st["v"] / (1 - 0.999 ** st["t"])
            p = self._params()[name]          # live array; updates the real net
            p -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)

    def train(self, obs, acts, rets, advs, old_logp):
        obs = np.asarray(obs, dtype=np.float64)
        acts = np.asarray(acts, dtype=np.float64)
        rets = np.asarray(rets, dtype=np.float64)
        advs = (np.asarray(advs) - np.asarray(advs).mean()) / (np.asarray(advs).std() + 1e-6)
        old_logp = np.asarray(old_logp, dtype=np.float64)
        x = self._norm(obs)
        n = obs.shape[0]

        rng = np.random.default_rng(0)
        for _ in range(self.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, self.minibatch):
                b = idx[s:s + self.minibatch]
                self._update_step(x[b], acts[b], rets[b], advs[b], old_logp[b])

    def _update_step(self, xb, ab, rb, advb, oldb):
        mb = xb.shape[0]
        g = {}
        ls = self.log_std
        sigma2 = np.exp(2 * ls)[None, :]

        # ---- policy forward ----
        mu, hs = _mlp_forward(self.policy, xb, keep=True)
        logp = (-0.5 * (ab - mu) ** 2 / sigma2 - ls - 0.5 * np.log(2 * np.pi)).sum(1)
        ratio = np.exp(logp - oldb)

        # clipped surrogate objective (loss = -surr.mean())
        clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip)
        surr = np.minimum(ratio * advb, clipped * advb)
        mask = ratio * advb <= clipped * advb          # active branch (unclipped)
        pol_loss = -surr.mean()

        # dL/d logp_i = -adv_i * ratio_i * mask_i ;  dlogp/dmu and dlogp/dlog_std
        dL_dlogp = -advb * ratio * mask
        d_logp_d_mu = (ab - mu) / sigma2
        d_out = dL_dlogp[:, None] * d_logp_d_mu            # grad wrt mu (linear out)
        for k, v in _mlp_backward(self.policy, xb, hs, d_out).items():
            g[f"pi_{k}"] = v
        # log_std gradient: surrogate part + entropy bonus (-ent_coef*H)
        d_surr_d_ls = np.sum(
            dL_dlogp[:, None] * ((ab - mu) ** 2 / sigma2 - 1.0), axis=0) / mb
        g["log_std"] = d_surr_d_ls - self.ent_coef

        # ---- value forward + loss ----
        vpred, vhs = _mlp_forward(self.value, xb, keep=True)
        vd = vpred[:, 0] - rb
        v_loss = 0.5 * (vd ** 2).mean()
        for k, v in _mlp_backward(self.value, xb, vhs,
                                  (self.vf_coef * vd)[:, None]).items():
            g[f"vf_{k}"] = v

        self._grad = g
        self._adam_update()
        return pol_loss, v_loss