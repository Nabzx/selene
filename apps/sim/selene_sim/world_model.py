"""Shared 5-MLP ensemble world model and spatial uncertainty map in PyTorch.

Implements a PETS-style ensemble of 5 DynamicsMLP networks with Randomised Prior
Functions (Osband et al., 2018) so that in-distribution transitions converge to
low ensemble variance while unvisited craters and out-of-distribution actions
exhibit high epistemic disagreement. Exposes per-crater and 20x20 spatial error
and uncertainty maps for the Argon orchestrator.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from selene_sim.sim import TerrainGrid


_CACHED_FASTEST_DEVICE: torch.device | None = None


def select_device(prefer_fastest: bool = True) -> torch.device:
    """Micro-benchmark 50 forward/backward steps of a 2-layer 128-unit MLP on CPU vs MPS.

    Returns whichever device achieves lower wall-clock latency.
    """
    global _CACHED_FASTEST_DEVICE
    if not prefer_fastest or not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return torch.device("cpu")
    if _CACHED_FASTEST_DEVICE is not None:
        return _CACHED_FASTEST_DEVICE

    def _bench(dev: torch.device) -> float:
        torch.manual_seed(42)
        net = nn.Sequential(
            nn.Linear(48, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 34),
        ).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
        x = torch.randn(512, 48, device=dev)
        y = torch.randn(512, 34, device=dev)

        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(net(x), y)
            loss.backward()
            opt.step()
        if dev.type == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(50):
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(net(x), y)
            loss.backward()
            opt.step()
        if dev.type == "mps":
            torch.mps.synchronize()
        return time.perf_counter() - t0

    cpu_time = _bench(torch.device("cpu"))
    try:
        mps_time = _bench(torch.device("mps"))
    except Exception:
        _CACHED_FASTEST_DEVICE = torch.device("cpu")
        return _CACHED_FASTEST_DEVICE

    _CACHED_FASTEST_DEVICE = (
        torch.device("mps") if mps_time < cpu_time else torch.device("cpu")
    )
    return _CACHED_FASTEST_DEVICE


class TransitionBuffer:
    """Pooled replay buffer storing fleet transitions across hoppers."""

    def __init__(self, max_size: int = 100_000) -> None:
        self.max_size = int(max_size)
        self._obs: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._next_obs: list[np.ndarray] = []
        self._rewards: list[np.ndarray] = []
        self._crashed: list[np.ndarray] = []
        self._crater_ids: list[np.ndarray] = []
        self._pos_xy: list[np.ndarray] = []
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        next_obs: np.ndarray,
        rewards: np.ndarray,
        crashed: np.ndarray,
        crater_ids: np.ndarray | None = None,
        pos_xy: np.ndarray | None = None,
    ) -> None:
        """Add single or batched transitions `(obs, actions, next_obs, rewards, crashed, ...)`."""
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[None, :]
        else:
            obs_arr = obs_arr.reshape(-1, obs_arr.shape[-1])

        batch_n = obs_arr.shape[0]
        next_obs_arr = np.asarray(next_obs, dtype=np.float32).reshape(batch_n, -1)
        act_arr = np.asarray(actions, dtype=np.int32).reshape(batch_n)
        rew_arr = np.asarray(rewards, dtype=np.float32).reshape(batch_n)
        crash_arr = np.asarray(crashed, dtype=bool).reshape(batch_n)

        if crater_ids is None:
            cid_arr = np.where(act_arr >= 3, act_arr - 3, -1).astype(np.int32)
        else:
            cid_arr = np.asarray(crater_ids, dtype=np.int32).reshape(batch_n)

        if pos_xy is None:
            # First two observation dimensions store normalised (x / half_extent, y / half_extent)
            pos_arr = (next_obs_arr[:, :2] * 10000.0).astype(np.float32)
        else:
            pos_arr = np.asarray(pos_xy, dtype=np.float32).reshape(batch_n, 2)

        self._obs.append(obs_arr)
        self._actions.append(act_arr)
        self._next_obs.append(next_obs_arr)
        self._rewards.append(rew_arr)
        self._crashed.append(crash_arr)
        self._crater_ids.append(cid_arr)
        self._pos_xy.append(pos_arr)
        self._size += batch_n

        if self._size > self.max_size:
            full = self.as_dict()
            keep = slice(-self.max_size, None)
            self._obs = [full["obs"][keep]]
            self._actions = [full["actions"][keep]]
            self._next_obs = [full["next_obs"][keep]]
            self._rewards = [full["rewards"][keep]]
            self._crashed = [full["crashed"][keep]]
            self._crater_ids = [full["crater_ids"][keep]]
            self._pos_xy = [full["pos_xy"][keep]]
            self._size = self.max_size

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return all stored transitions as a dictionary of contiguous NumPy arrays."""
        if self._size == 0:
            raise ValueError("TransitionBuffer is empty.")
        return {
            "obs": np.concatenate(self._obs, axis=0),
            "actions": np.concatenate(self._actions, axis=0),
            "next_obs": np.concatenate(self._next_obs, axis=0),
            "rewards": np.concatenate(self._rewards, axis=0),
            "crashed": np.concatenate(self._crashed, axis=0),
            "crater_ids": np.concatenate(self._crater_ids, axis=0),
            "pos_xy": np.concatenate(self._pos_xy, axis=0),
        }

    def sample_split(
        self, val_fraction: float = 0.1, seed: int = 42
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Deterministically split pooled transitions into training and validation dictionaries."""
        data = self.as_dict()
        total = data["obs"].shape[0]
        val_n = max(1, int(round(total * float(val_fraction))))
        train_n = max(1, total - val_n)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(total)
        train_idx = perm[:train_n]
        val_idx = perm[train_n:]
        if val_idx.size == 0:
            val_idx = perm[-1:]

        train_dict = {k: v[train_idx] for k, v in data.items()}
        val_dict = {k: v[val_idx] for k, v in data.items()}
        return train_dict, val_dict


class DynamicsMLP(nn.Module):
    """2-layer dynamics MLP predicting `(delta_obs, reward, crash_logit)` with a frozen prior."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        prior_scale: float = 0.65,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_dim = int(hidden_dim)
        self.prior_scale = float(prior_scale)
        in_dim = self.obs_dim + self.n_actions
        out_dim = self.obs_dim + 2

        if seed is not None:
            torch.manual_seed(seed)

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, out_dim),
        )

        # Untrainable Randomised Prior network (Osband et al.) ensures distinct predictions
        # on unvisited actions/craters while cancelling out on visited training data.
        self.prior_net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, out_dim),
        )
        self.prior_action_head = nn.Linear(self.n_actions, out_dim, bias=False)
        nn.init.normal_(self.prior_action_head.weight, mean=0.0, std=1.2)
        for param in self.prior_net.parameters():
            param.requires_grad_(False)
        for param in self.prior_action_head.parameters():
            param.requires_grad_(False)

    def forward(
        self, x: torch.Tensor, actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning `(delta_obs, reward, crash_logit)`."""
        if actions is not None:
            one_hot = F.one_hot(actions.long(), num_classes=self.n_actions).to(x.dtype)
            x = torch.cat([x, one_hot], dim=-1)

        out = self.net(x)
        if self.prior_scale > 0.0:
            act_slice = x[..., self.obs_dim :]
            prior_out = self.prior_net(x) + self.prior_action_head(act_slice)
            out = out + self.prior_scale * prior_out

        delta_obs = out[..., : self.obs_dim]
        reward = out[..., self.obs_dim]
        crash_logit = out[..., self.obs_dim + 1]
        return delta_obs, reward, crash_logit


class WorldModelEnsemble:
    """5-MLP ensemble world model with input/target normalisation and spatial error mapping."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        ensemble_size: int = 5,
        hidden_dim: int = 128,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        device: torch.device | str | None = None,
        seed: int = 42,
        prior_scale: float = 0.65,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.ensemble_size = int(ensemble_size)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.device = (
            select_device(prefer_fastest=True)
            if device is None
            else torch.device(device)
        )

        self.models: list[DynamicsMLP] = []
        self.optimizers: list[torch.optim.Optimizer] = []
        for k in range(self.ensemble_size):
            member_seed = self.seed + 1009 * (k + 1)
            model = DynamicsMLP(
                obs_dim=self.obs_dim,
                n_actions=self.n_actions,
                hidden_dim=self.hidden_dim,
                prior_scale=prior_scale,
                seed=member_seed,
            ).to(self.device)
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            self.models.append(model)
            self.optimizers.append(opt)

        self.obs_mean = torch.zeros(self.obs_dim, device=self.device)
        self.obs_std = torch.ones(self.obs_dim, device=self.device)
        self.delta_mean = torch.zeros(self.obs_dim, device=self.device)
        self.delta_std = torch.ones(self.obs_dim, device=self.device)
        self.rew_mean = torch.tensor(0.0, device=self.device)
        self.rew_std = torch.tensor(1.0, device=self.device)
        self.last_val_mse: float = 0.0

    @staticmethod
    def _coerce_split(
        buffer_or_dict: TransitionBuffer | dict[str, np.ndarray],
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        if isinstance(buffer_or_dict, TransitionBuffer):
            full_dict = buffer_or_dict.as_dict()
            train_dict, val_dict = buffer_or_dict.sample_split(
                val_fraction=val_fraction, seed=seed
            )
            return full_dict, train_dict, val_dict

        full_dict = {k: np.asarray(v) for k, v in buffer_or_dict.items()}
        temp_buf = TransitionBuffer(max_size=max(100_000, full_dict["obs"].shape[0]))
        temp_buf.add(
            obs=full_dict["obs"],
            actions=full_dict["actions"],
            next_obs=full_dict["next_obs"],
            rewards=full_dict["rewards"],
            crashed=full_dict["crashed"],
            crater_ids=full_dict.get("crater_ids"),
            pos_xy=full_dict.get("pos_xy"),
        )
        train_dict, val_dict = temp_buf.sample_split(val_fraction=val_fraction, seed=seed)
        return temp_buf.as_dict(), train_dict, val_dict

    def _fit_normaliser(self, train_dict: dict[str, np.ndarray]) -> None:
        obs_t = torch.as_tensor(train_dict["obs"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(
            train_dict["next_obs"], dtype=torch.float32, device=self.device
        )
        delta_t = next_obs_t - obs_t
        rew_t = torch.as_tensor(
            train_dict["rewards"], dtype=torch.float32, device=self.device
        )

        self.obs_mean = obs_t.mean(dim=0)
        self.obs_std = torch.clamp(obs_t.std(dim=0), min=1e-4)
        self.delta_mean = delta_t.mean(dim=0)
        self.delta_std = torch.clamp(delta_t.std(dim=0), min=1e-4)
        self.rew_mean = rew_t.mean()
        self.rew_std = torch.clamp(rew_t.std(), min=1e-4)

    def _eval_val_mse(
        self,
        val_x: torch.Tensor,
        val_delta_norm: torch.Tensor,
        val_rew_norm: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            pred_deltas = []
            pred_rews = []
            for model in self.models:
                model.eval()
                d_obs, rew, _ = model(val_x)
                pred_deltas.append(d_obs)
                pred_rews.append(rew)
            mean_delta = torch.stack(pred_deltas, dim=0).mean(dim=0)
            mean_rew = torch.stack(pred_rews, dim=0).mean(dim=0)
            mse_delta = F.mse_loss(mean_delta, val_delta_norm)
            mse_rew = F.mse_loss(mean_rew, val_rew_norm)
            return float((mse_delta + 0.1 * mse_rew).item())

    def train_epochs(
        self,
        buffer_or_dict: TransitionBuffer | dict[str, np.ndarray],
        epochs: int = 25,
        batch_size: int = 512,
        seed: int = 42,
    ) -> dict[str, list[float]]:
        """Train the 5 ensemble members on bootstrapped subsets and log `train_loss` and `val_mse`."""
        _, train_dict, val_dict = self._coerce_split(
            buffer_or_dict, val_fraction=0.1, seed=seed
        )
        self._fit_normaliser(train_dict)

        train_obs = torch.as_tensor(
            train_dict["obs"], dtype=torch.float32, device=self.device
        )
        train_next_obs = torch.as_tensor(
            train_dict["next_obs"], dtype=torch.float32, device=self.device
        )
        train_act = torch.as_tensor(
            train_dict["actions"], dtype=torch.long, device=self.device
        )
        train_rew = torch.as_tensor(
            train_dict["rewards"], dtype=torch.float32, device=self.device
        )
        train_crash = torch.as_tensor(
            train_dict["crashed"].astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        train_obs_norm = (train_obs - self.obs_mean) / self.obs_std
        train_one_hot = F.one_hot(train_act, num_classes=self.n_actions).to(torch.float32)
        train_x = torch.cat([train_obs_norm, train_one_hot], dim=-1)
        train_delta_norm = ((train_next_obs - train_obs) - self.delta_mean) / self.delta_std
        train_rew_norm = (train_rew - self.rew_mean) / self.rew_std

        val_obs = torch.as_tensor(
            val_dict["obs"], dtype=torch.float32, device=self.device
        )
        val_next_obs = torch.as_tensor(
            val_dict["next_obs"], dtype=torch.float32, device=self.device
        )
        val_act = torch.as_tensor(
            val_dict["actions"], dtype=torch.long, device=self.device
        )
        val_rew = torch.as_tensor(
            val_dict["rewards"], dtype=torch.float32, device=self.device
        )

        val_obs_norm = (val_obs - self.obs_mean) / self.obs_std
        val_one_hot = F.one_hot(val_act, num_classes=self.n_actions).to(torch.float32)
        val_x = torch.cat([val_obs_norm, val_one_hot], dim=-1)
        val_delta_norm = ((val_next_obs - val_obs) - self.delta_mean) / self.delta_std
        val_rew_norm = (val_rew - self.rew_mean) / self.rew_std

        n_train = train_x.shape[0]
        rng = np.random.default_rng(seed)

        # Generate fixed per-member bootstrap sample indices (with replacement)
        bootstrap_indices = [
            torch.as_tensor(
                rng.integers(0, n_train, size=n_train),
                dtype=torch.long,
                device=self.device,
            )
            for _ in range(self.ensemble_size)
        ]

        train_losses: list[float] = []
        val_mses: list[float] = []

        for epoch in range(int(epochs)):
            if epoch == 0:
                # Record pre-step baseline validation MSE at epoch 0 before first weight update
                val_mses.append(self._eval_val_mse(val_x, val_delta_norm, val_rew_norm))

            epoch_loss_sum = 0.0
            n_batches = 0

            for k, (model, opt) in enumerate(zip(self.models, self.optimizers, strict=True)):
                model.train()
                perm = torch.as_tensor(
                    rng.permutation(n_train), dtype=torch.long, device=self.device
                )
                member_idx = bootstrap_indices[k][perm]

                for start in range(0, n_train, batch_size):
                    batch_idx = member_idx[start : start + batch_size]
                    bx = train_x[batch_idx]
                    b_delta = train_delta_norm[batch_idx]
                    b_rew = train_rew_norm[batch_idx]
                    b_crash = train_crash[batch_idx]

                    opt.zero_grad(set_to_none=True)
                    pred_delta, pred_rew, pred_crash_logit = model(bx)
                    loss_delta = F.mse_loss(pred_delta, b_delta)
                    loss_rew = F.mse_loss(pred_rew, b_rew)
                    loss_crash = F.binary_cross_entropy_with_logits(
                        pred_crash_logit, b_crash
                    )
                    loss = loss_delta + 0.2 * loss_rew + 0.1 * loss_crash
                    loss.backward()
                    opt.step()

                    epoch_loss_sum += float(loss.item())
                    n_batches += 1

            train_losses.append(epoch_loss_sum / max(1, n_batches))
            if epoch > 0:
                val_mses.append(self._eval_val_mse(val_x, val_delta_norm, val_rew_norm))

        # Ensure final post-training validation MSE is captured at the end of training
        final_mse = self._eval_val_mse(val_x, val_delta_norm, val_rew_norm)
        val_mses[-1] = final_mse
        self.last_val_mse = final_mse

        return {"train_loss": train_losses, "val_mse": val_mses}

    def predict(
        self, obs_batch: np.ndarray, act_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict `(mean_next_obs, mean_reward, mean_crash_prob, epistemic_uncertainty)`."""
        obs_np = np.asarray(obs_batch, dtype=np.float32)
        single = obs_np.ndim == 1
        if single:
            obs_np = obs_np[None, :]
        act_np = np.asarray(act_batch, dtype=np.int64).reshape(-1)

        with torch.no_grad():
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
            act_t = torch.as_tensor(act_np, dtype=torch.long, device=self.device)
            obs_norm = (obs_t - self.obs_mean) / self.obs_std
            one_hot = F.one_hot(act_t, num_classes=self.n_actions).to(torch.float32)
            x = torch.cat([obs_norm, one_hot], dim=-1)

            member_next_obs = []
            member_norm_deltas = []
            member_rews = []
            member_crash_probs = []

            for model in self.models:
                model.eval()
                d_norm, r_norm, c_logit = model(x)
                d_unnorm = d_norm * self.delta_std + self.delta_mean
                next_obs_k = obs_t + d_unnorm
                rew_k = r_norm * self.rew_std + self.rew_mean
                crash_prob_k = torch.sigmoid(c_logit)

                member_next_obs.append(next_obs_k)
                member_norm_deltas.append(d_norm)
                member_rews.append(rew_k)
                member_crash_probs.append(crash_prob_k)

            stack_next_obs = torch.stack(member_next_obs, dim=0)  # (K, B, obs_dim)
            stack_norm_deltas = torch.stack(member_norm_deltas, dim=0)  # (K, B, obs_dim)
            stack_rews = torch.stack(member_rews, dim=0)  # (K, B)
            stack_crash = torch.stack(member_crash_probs, dim=0)  # (K, B)

            mean_next_obs = stack_next_obs.mean(dim=0)
            mean_reward = stack_rews.mean(dim=0)
            mean_crash_prob = stack_crash.mean(dim=0)

            # Epistemic uncertainty: ensemble standard deviation across the K=5 members
            delta_var = stack_norm_deltas.var(dim=0, unbiased=False).mean(dim=-1)
            rew_var = (stack_rews / self.rew_std).var(dim=0, unbiased=False)
            epistemic_std = torch.sqrt(delta_var + 0.1 * rew_var + 1e-8)

        out_next_obs = mean_next_obs.cpu().numpy().astype(np.float32)
        out_rew = mean_reward.cpu().numpy().astype(np.float32)
        out_crash = mean_crash_prob.cpu().numpy().astype(np.float32)
        out_unc = epistemic_std.cpu().numpy().astype(np.float32)

        if single:
            return out_next_obs[0], out_rew[:1], out_crash[:1], out_unc[:1]
        return out_next_obs, out_rew, out_crash, out_unc

    def world_model_error_map(
        self,
        terrain: TerrainGrid,
        buffer_or_dict: TransitionBuffer | dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Compute JSON-serialisable per-crater and 20x20 spatial error and uncertainty maps."""
        full_dict, _, val_dict = self._coerce_split(buffer_or_dict, val_fraction=0.1, seed=42)

        obs_all = full_dict["obs"]
        act_all = full_dict["actions"]
        next_obs_all = full_dict["next_obs"]
        crater_ids_all = full_dict["crater_ids"]
        pos_xy_all = full_dict["pos_xy"]

        pred_next_obs, _, _, unc_all = self.predict(obs_all, act_all)
        per_sample_error = np.linalg.norm(pred_next_obs - next_obs_all, axis=-1).astype(
            np.float32
        )

        # Validation MSE on held-out split
        val_pred_next, _, _, _ = self.predict(val_dict["obs"], val_dict["actions"])
        overall_val_mse = float(
            np.mean((val_pred_next - val_dict["next_obs"]) ** 2)
        )

        # Build synthetic probe observations for each crater so unvisited craters are also probed
        half_ext = terrain.half_extent_m
        probe_obs_base = np.median(obs_all, axis=0).astype(np.float32)
        crater_metrics: list[dict[str, Any]] = []

        for c_idx, crater in enumerate(terrain.craters):
            mask = crater_ids_all == c_idx
            visit_count = int(np.sum(mask))
            if visit_count > 0:
                mean_err = float(np.mean(per_sample_error[mask]))
                mean_unc = float(np.mean(unc_all[mask]))
            else:
                # Probe the ensemble with hop actions targeting this unvisited crater
                n_probes = 32
                rng = np.random.default_rng(1000 + c_idx)
                probes = np.repeat(probe_obs_base[None, :], n_probes, axis=0)
                angles = np.linspace(0.0, 2.0 * np.pi, n_probes, endpoint=False)
                px = np.clip(crater.x_m + 1200.0 * np.cos(angles), -half_ext, half_ext)
                py = np.clip(crater.y_m + 1200.0 * np.sin(angles), -half_ext, half_ext)
                probes[:, 0] = (px / half_ext).astype(np.float32)
                probes[:, 1] = (py / half_ext).astype(np.float32)
                probes += rng.normal(0.0, 0.02, size=probes.shape).astype(np.float32)
                probe_acts = np.full(n_probes, min(3 + c_idx, self.n_actions - 1), dtype=np.int32)
                _, _, _, probe_unc = self.predict(probes, probe_acts)
                mean_unc = float(np.mean(probe_unc))
                mean_err = float(mean_unc * 0.5)

            crater_metrics.append(
                {
                    "id": str(crater.id),
                    "crater_index": int(c_idx),
                    "x_m": float(crater.x_m),
                    "y_m": float(crater.y_m),
                    "visit_count": visit_count,
                    "mean_error": mean_err,
                    "epistemic_uncertainty": mean_unc,
                }
            )

        # 20 x 20 coarse grid over [-10000, 10000] m (cell width = 1000 m)
        grid_n = 20
        cell_span_m = (2.0 * half_ext) / grid_n
        uncertainty_grid: list[list[float]] = []
        mean_error_grid: list[list[float]] = []
        visit_count_grid: list[list[int]] = []
        all_cells: list[dict[str, Any]] = []

        crater_xy = np.array([[c.x_m, c.y_m] for c in terrain.craters], dtype=np.float64)

        gx_idx = np.clip(
            np.floor((pos_xy_all[:, 0] + half_ext) / cell_span_m).astype(np.int32),
            0,
            grid_n - 1,
        )
        gy_idx = np.clip(
            np.floor((pos_xy_all[:, 1] + half_ext) / cell_span_m).astype(np.int32),
            0,
            grid_n - 1,
        )

        # Precompute probe uncertainty for all 400 cell centres in a single batched pass
        cell_coords_x = (np.arange(grid_n, dtype=np.float64) + 0.5) * cell_span_m - half_ext
        cell_coords_y = (np.arange(grid_n, dtype=np.float64) + 0.5) * cell_span_m - half_ext
        cxx, cyy = np.meshgrid(cell_coords_x, cell_coords_y, indexing="xy")
        flat_x = cxx.reshape(-1)
        flat_y = cyy.reshape(-1)

        dists = np.hypot(flat_x[:, None] - crater_xy[:, 0], flat_y[:, None] - crater_xy[:, 1])
        nearest_c_idx = np.argmin(dists, axis=-1).astype(np.int32)

        cell_probes = np.repeat(probe_obs_base[None, :], grid_n * grid_n, axis=0)
        cell_probes[:, 0] = (flat_x / half_ext).astype(np.float32)
        cell_probes[:, 1] = (flat_y / half_ext).astype(np.float32)
        cell_acts = np.minimum(3 + nearest_c_idx, self.n_actions - 1).astype(np.int32)
        _, _, _, cell_probe_unc = self.predict(cell_probes, cell_acts)
        cell_probe_unc_grid = cell_probe_unc.reshape(grid_n, grid_n)

        for i in range(grid_n):
            unc_row: list[float] = []
            err_row: list[float] = []
            cnt_row: list[int] = []
            y_center = float(cell_coords_y[i])
            for j in range(grid_n):
                x_center = float(cell_coords_x[j])
                cell_mask = (gy_idx == i) & (gx_idx == j)
                cnt = int(np.sum(cell_mask))
                c_idx = int(nearest_c_idx[i * grid_n + j])
                nearest_cid = str(terrain.craters[c_idx].id)

                if cnt > 0:
                    c_err = float(np.mean(per_sample_error[cell_mask]))
                    c_unc = float(np.mean(unc_all[cell_mask]))
                else:
                    c_unc = float(cell_probe_unc_grid[i, j])
                    c_err = float(c_unc * 0.5)

                unc_row.append(c_unc)
                err_row.append(c_err)
                cnt_row.append(cnt)
                all_cells.append(
                    {
                        "grid_ij": [int(i), int(j)],
                        "x_m": x_center,
                        "y_m": y_center,
                        "mean_error": c_err,
                        "uncertainty": c_unc,
                        "visit_count": cnt,
                        "nearest_crater_id": nearest_cid,
                    }
                )
            uncertainty_grid.append(unc_row)
            mean_error_grid.append(err_row)
            visit_count_grid.append(cnt_row)

        all_cells.sort(key=lambda item: (item["uncertainty"] + item["mean_error"]), reverse=True)
        top_error_cells = all_cells[:5]

        return {
            "overall_val_mse": overall_val_mse,
            "crater_metrics": crater_metrics,
            "uncertainty_grid": uncertainty_grid,
            "mean_error_grid": mean_error_grid,
            "visit_count_grid": visit_count_grid,
            "top_error_cells": top_error_cells,
        }


SharedWorldModel = WorldModelEnsemble
