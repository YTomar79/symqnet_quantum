from __future__ import annotations

import numpy as np
import torch

from .analysis.fisher import classical_fisher_for_actions
from .math_utils import gaussian_entropy_from_cov
from .smc import SMCParticleFilter


KNOWN_BASELINES = (
    "random",
    "fixed",
    "fixed_optimized",
    "fisher_greedy_fast",
    "fisher_greedy",
    "greedy_bed",
    "smc_adaptive",
    "bald_1step",
    "bald_2step_fast",
    "bald_2step",
)


def _bounded_action_candidates(
    n_actions: int,
    max_actions: int | None,
    *,
    seed: int = 0,
    step: int = 0,
    salt: int = 0,
) -> list[int]:
    if max_actions is None:
        return list(range(int(n_actions)))
    n_actions = int(n_actions)
    limit = max(1, min(n_actions, int(max_actions)))
    if n_actions <= limit:
        return list(range(n_actions))

    anchor_count = min(limit, max(1, int(np.ceil(np.sqrt(limit)))))
    anchors = np.linspace(0, n_actions - 1, num=anchor_count, dtype=int).tolist()
    rng = np.random.default_rng(int(seed) + 1009 * int(step) + 9173 * int(salt))
    pool = np.arange(n_actions)
    rng.shuffle(pool)

    out: list[int] = []
    seen: set[int] = set()
    for action in [*anchors, *pool.tolist()]:
        action = int(action)
        if action in seen:
            continue
        out.append(action)
        seen.add(action)
        if len(out) >= limit:
            break
    return out


class ActionPolicy:
    name = "policy"

    def reset(self) -> None:
        pass

    def metadata(self) -> dict[str, object]:
        return {"name": self.name}

    def select_action(self, obs: np.ndarray, smc: SMCParticleFilter, info: dict[str, object] | None = None) -> int:
        raise NotImplementedError


class RandomPolicy(ActionPolicy):
    name = "random"

    def __init__(self, n_actions: int, rng: np.random.Generator | None = None):
        self.n_actions = int(n_actions)
        self.rng = rng or np.random.default_rng()

    def metadata(self) -> dict[str, object]:
        return {"name": self.name, "n_actions": self.n_actions}

    def select_action(self, obs: np.ndarray, smc: SMCParticleFilter, info: dict[str, object] | None = None) -> int:
        return int(self.rng.integers(self.n_actions))


class FixedProtocolPolicy(ActionPolicy):
    name = "fixed"

    def __init__(self, n_qubits: int, m_evo: int, basis_order: tuple[int, ...] = (0, 1, 2)):
        self.n_qubits = int(n_qubits)
        self.m_evo = int(m_evo)
        self.basis_order = tuple(basis_order)
        self.ptr = 0
        self.sequence = [
            (q * 3 + b) * self.m_evo + t
            for t in range(self.m_evo)
            for b in self.basis_order
            for q in range(self.n_qubits)
        ]

    def reset(self) -> None:
        self.ptr = 0

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "m_evo": self.m_evo,
            "basis_order": self.basis_order,
        }

    def select_action(self, obs: np.ndarray, smc: SMCParticleFilter, info: dict[str, object] | None = None) -> int:
        action = int(self.sequence[self.ptr % len(self.sequence)])
        self.ptr += 1
        return action


class FixedOptimizedProtocolPolicy(ActionPolicy):
    """Non-adaptive Fisher-greedy protocol optimized once at the prior center."""

    name = "fixed_optimized"

    def __init__(self, n_actions: int, ridge: float = 1e-4):
        self.n_actions = int(n_actions)
        self.ridge = float(ridge)
        self.ptr = 0
        self.sequence: list[int] | None = None

    def reset(self) -> None:
        self.ptr = 0

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_actions": self.n_actions,
            "objective": "prior_center_fisher_logdet",
            "ridge": self.ridge,
            "non_adaptive": True,
        }

    def _prior_center_theta(self, smc: SMCParticleFilter) -> torch.Tensor:
        env = smc.env
        j_mid = 0.5 * (float(env.J_range[0]) + float(env.J_range[1]))
        h_mid = 0.5 * (float(env.h_range[0]) + float(env.h_range[1]))
        values = [j_mid] * (env.N - 1) + [h_mid] * env.N
        return torch.tensor(values, device=smc.device, dtype=torch.float32)

    def _score_fim(self, fim: torch.Tensor) -> float:
        eye = torch.eye(fim.shape[0], device=fim.device, dtype=fim.dtype)
        sign, logabsdet = torch.linalg.slogdet(fim + self.ridge * eye)
        if float(sign.item()) <= 0:
            return -float("inf")
        return float(logabsdet.item())

    def _build_sequence(self, smc: SMCParticleFilter) -> list[int]:
        theta = self._prior_center_theta(smc)
        dim = theta.numel()
        fim = torch.zeros(dim, dim, device=smc.device, dtype=torch.float32)
        sequence: list[int] = []
        shots = int(smc.env.current_shots)
        with torch.enable_grad():
            action_fims = [
                classical_fisher_for_actions(smc.env, theta, [action], shots).detach()
                for action in range(self.n_actions)
            ]
        for _ in range(smc.env.T):
            scores = [self._score_fim(fim + candidate_fim) for candidate_fim in action_fims]
            action = int(np.argmax(scores))
            sequence.append(action)
            fim = fim + action_fims[action]
        return sequence

    def select_action(self, obs: np.ndarray, smc: SMCParticleFilter, info: dict[str, object] | None = None) -> int:
        if self.sequence is None:
            self.sequence = self._build_sequence(smc)
        action = int(self.sequence[self.ptr % len(self.sequence)])
        self.ptr += 1
        return action


class FisherGreedyPolicy(ActionPolicy):
    """Adaptive non-Bayesian design using a particle-linearized Fisher score."""

    name = "fisher_greedy"

    def __init__(self, n_actions: int, ridge: float = 1e-4):
        self.n_actions = int(n_actions)
        self.ridge = float(ridge)
        self.fim: torch.Tensor | None = None

    def reset(self) -> None:
        self.fim = None

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_actions": self.n_actions,
            "objective": "particle_linearized_fisher_logdet",
            "ridge": self.ridge,
        }

    def _score_fim(self, fim: torch.Tensor) -> float:
        eye = torch.eye(fim.shape[0], device=fim.device, dtype=fim.dtype)
        sign, logabsdet = torch.linalg.slogdet(fim + self.ridge * eye)
        if float(sign.item()) <= 0:
            return -float("inf")
        return float(logabsdet.item())

    def _action_increment(self, smc: SMCParticleFilter, action: int, posterior) -> torch.Tensor:
        q, b, t = smc.env.decode_action(action)
        means = smc.predicted_means(q, b, t).float()
        mean_y = torch.sum(smc.w * means)
        centered_theta = smc.particles.float() - posterior.mean[None, :]
        centered_y = means - mean_y
        cov_theta_y = torch.einsum("p,pi,p->i", smc.w, centered_theta, centered_y)
        eye = torch.eye(posterior.cov.shape[0], device=smc.device, dtype=torch.float32)
        grad = torch.linalg.solve(posterior.cov + self.ridge * eye, cov_theta_y)
        obs_var = ((1.0 - mean_y * mean_y).clamp_min(1e-6) / max(1, int(smc.env.current_shots))).float()
        return torch.outer(grad, grad) / obs_var

    def select_action(self, obs: np.ndarray, smc: SMCParticleFilter, info: dict[str, object] | None = None) -> int:
        posterior = smc.posterior()
        if self.fim is None:
            dim = posterior.mean.numel()
            self.fim = torch.zeros(dim, dim, device=smc.device, dtype=torch.float32)
        best_action = 0
        best_score = -float("inf")
        best_increment: torch.Tensor | None = None
        for action in range(self.n_actions):
            increment = self._action_increment(smc, action, posterior).detach()
            score = self._score_fim(self.fim + increment)
            if score > best_score:
                best_score = score
                best_action = action
                best_increment = increment
        if best_increment is not None:
            self.fim = self.fim + best_increment
        return int(best_action)


class BoundedFisherGreedyPolicy(FisherGreedyPolicy):
    """Adaptive Fisher comparator with capped per-step action search."""

    name = "fisher_greedy_fast"

    def __init__(self, n_actions: int, ridge: float = 1e-4, max_actions: int = 4, seed: int = 0):
        super().__init__(n_actions, ridge)
        self.max_actions = int(max_actions)
        self.seed = int(seed)
        self.decision_index = 0

    def reset(self) -> None:
        super().reset()
        self.decision_index = 0

    def metadata(self) -> dict[str, object]:
        out = super().metadata()
        out.update(
            {
                "name": self.name,
                "max_actions": self.max_actions,
                "seed": self.seed,
                "bounded_candidates": True,
            }
        )
        return out

    def select_action(self, obs: np.ndarray, smc: SMCParticleFilter, info: dict[str, object] | None = None) -> int:
        posterior = smc.posterior()
        if self.fim is None:
            dim = posterior.mean.numel()
            self.fim = torch.zeros(dim, dim, device=smc.device, dtype=torch.float32)

        best_action = 0
        best_score = -float("inf")
        best_increment: torch.Tensor | None = None
        candidates = _bounded_action_candidates(
            self.n_actions,
            self.max_actions,
            seed=self.seed,
            step=self.decision_index,
        )
        for action in candidates:
            increment = self._action_increment(smc, action, posterior).detach()
            score = self._score_fim(self.fim + increment)
            if score > best_score:
                best_score = score
                best_action = action
                best_increment = increment
        if best_increment is not None:
            self.fim = self.fim + best_increment
        self.decision_index += 1
        return int(best_action)


class GreedyBEDPolicy(ActionPolicy):
    name = "smc_adaptive"

    def __init__(self, n_actions: int, predictive_samples: int = 5):
        self.n_actions = int(n_actions)
        self.predictive_samples = int(predictive_samples)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_actions": self.n_actions,
            "predictive_samples": self.predictive_samples,
        }

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, smc: SMCParticleFilter, info: dict[str, object] | None = None) -> int:
        prior = smc.posterior()
        h_prior = gaussian_entropy_from_cov(prior.cov)
        best_action = 0
        best_score = -float("inf")
        shots = int(smc.env.current_shots)

        for action in range(self.n_actions):
            q, b, t = smc.env.decode_action(action)
            means = smc.predicted_means(q, b, t)
            predictive_mean = torch.sum(smc.w * means)
            predictive_std = torch.sqrt(torch.sum(smc.w * (means - predictive_mean) ** 2).clamp_min(1e-6))
            candidates = torch.linspace(-1.0, 1.0, self.predictive_samples, device=smc.device)
            candidates = (predictive_mean + predictive_std * candidates).clamp(-1.0, 1.0)

            score = 0.0
            for y in candidates:
                trial = smc.clone()
                post = trial.update_pseudo_observation(y, q, b, t, shots)
                score += float(torch.clamp(h_prior - gaussian_entropy_from_cov(post.cov), min=0.0).item())
            score /= max(1, self.predictive_samples)
            if score > best_score:
                best_score = score
                best_action = action
        return int(best_action)


class BALDOneStepPolicy(ActionPolicy):
    """One-step Bayesian experimental design using predictive KL on particle weights."""

    name = "bald_1step"

    def __init__(self, n_actions: int, predictive_samples: int = 7, seed: int = 0, max_actions: int | None = None):
        self.n_actions = int(n_actions)
        self.predictive_samples = int(predictive_samples)
        self.seed = int(seed)
        self.max_actions = None if max_actions is None else int(max_actions)
        self.decision_index = 0
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def reset(self) -> None:
        self.decision_index = 0

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_actions": self.n_actions,
            "predictive_samples": self.predictive_samples,
            "seed": self.seed,
            "max_actions": self.max_actions,
        }

    @torch.no_grad()
    def _candidate_observations(self, smc: SMCParticleFilter, means: torch.Tensor, shots: int) -> torch.Tensor:
        particle_idx = torch.multinomial(smc.w.detach().cpu(), self.predictive_samples, replacement=True, generator=self.generator).to(smc.device)
        chosen_means = means[particle_idx]
        var = ((1.0 - chosen_means * chosen_means).clamp_min(1e-6) / max(1, shots)).sqrt()
        eps = torch.randn(self.predictive_samples, generator=self.generator).to(smc.device)
        return (chosen_means + eps * var).clamp(-1.0, 1.0)

    @torch.no_grad()
    def _posterior_weight_kl(self, smc: SMCParticleFilter, y: torch.Tensor, q: int, b: int, t: int, shots: int) -> float:
        means = smc.predicted_means(q, b, t)
        var = (1.0 - means * means).clamp_min(1e-6) / max(1, shots)
        ll = -0.5 * ((y - means) ** 2) / var - 0.5 * torch.log(var)
        log_prior = torch.log(smc.w.clamp_min(1e-12))
        log_post = log_prior + ll
        log_post = log_post - torch.logsumexp(log_post, dim=0)
        post = torch.exp(log_post)
        kl = torch.sum(post * (log_post - log_prior))
        return float(kl.clamp_min(0.0).item())

    @torch.no_grad()
    def score_action(self, smc: SMCParticleFilter, action: int) -> float:
        q, b, t = smc.env.decode_action(action)
        shots = int(smc.env.current_shots)
        means = smc.predicted_means(q, b, t)
        ys = self._candidate_observations(smc, means, shots)
        return float(np.mean([self._posterior_weight_kl(smc, y, q, b, t, shots) for y in ys]))

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, smc: SMCParticleFilter, info: dict[str, object] | None = None) -> int:
        candidates = _bounded_action_candidates(
            self.n_actions,
            self.max_actions,
            seed=self.seed,
            step=self.decision_index,
        )
        scores = [self.score_action(smc, action) for action in candidates]
        self.decision_index += 1
        return int(candidates[int(np.argmax(scores))])


class BALDNStepPolicy(BALDOneStepPolicy):
    """Shallow lookahead BED. Depth 2 is the intended paper baseline."""

    name = "bald_nstep"

    def __init__(
        self,
        n_actions: int,
        depth: int = 2,
        predictive_samples: int = 3,
        top_k: int | None = None,
        seed: int = 0,
        max_actions: int | None = None,
        future_max_actions: int | None = None,
    ):
        super().__init__(n_actions, predictive_samples, seed, max_actions=max_actions)
        self.depth = int(depth)
        self.top_k = top_k
        self.future_max_actions = None if future_max_actions is None else int(future_max_actions)

    def metadata(self) -> dict[str, object]:
        out = super().metadata()
        out.update(
            {
                "name": self.name,
                "depth": self.depth,
                "top_k": self.top_k,
                "future_max_actions": self.future_max_actions,
            }
        )
        return out

    @torch.no_grad()
    def _condition_without_resampling(self, smc: SMCParticleFilter, y: torch.Tensor, q: int, b: int, t: int, shots: int) -> SMCParticleFilter:
        trial = smc.clone(share_cache=True)
        means = trial.predicted_means(q, b, t)
        var = (1.0 - means * means).clamp_min(1e-6) / max(1, shots)
        ll = -0.5 * ((y - means) ** 2) / var - 0.5 * torch.log(var)
        logw = torch.log(trial.w.clamp_min(1e-12)) + ll
        trial.w = torch.softmax(logw, dim=0)
        return trial

    @torch.no_grad()
    def _recursive_score(self, smc: SMCParticleFilter, depth: int) -> float:
        candidates_now = _bounded_action_candidates(
            self.n_actions,
            self.future_max_actions,
            seed=self.seed,
            step=self.decision_index,
            salt=depth,
        )
        one_step = [(action, BALDOneStepPolicy.score_action(self, smc, action)) for action in candidates_now]
        if depth <= 1:
            return max(score for _, score in one_step)
        candidates = sorted(one_step, key=lambda item: item[1], reverse=True)
        if self.top_k is not None:
            candidates = candidates[: self.top_k]

        best = -float("inf")
        shots = int(smc.env.current_shots)
        for action, immediate in candidates:
            q, b, t = smc.env.decode_action(action)
            means = smc.predicted_means(q, b, t)
            ys = self._candidate_observations(smc, means, shots)
            future_scores = []
            for y in ys:
                trial = self._condition_without_resampling(smc, y, q, b, t, shots)
                future_scores.append(self._recursive_score(trial, depth - 1))
            best = max(best, immediate + float(np.mean(future_scores)))
        return best

    @torch.no_grad()
    def score_action(self, smc: SMCParticleFilter, action: int) -> float:
        if self.depth <= 1:
            return BALDOneStepPolicy.score_action(self, smc, action)
        q, b, t = smc.env.decode_action(action)
        shots = int(smc.env.current_shots)
        immediate = BALDOneStepPolicy.score_action(self, smc, action)
        means = smc.predicted_means(q, b, t)
        ys = self._candidate_observations(smc, means, shots)
        future_scores = []
        for y in ys:
            trial = self._condition_without_resampling(smc, y, q, b, t, shots)
            future_scores.append(self._recursive_score(trial, self.depth - 1))
        return immediate + float(np.mean(future_scores))


def make_baseline(name: str, n_qubits: int, m_evo: int, n_actions: int, seed: int = 0) -> ActionPolicy:
    if name == "random":
        return RandomPolicy(n_actions, np.random.default_rng(seed))
    if name == "fixed":
        return FixedProtocolPolicy(n_qubits, m_evo)
    if name == "fixed_optimized":
        return FixedOptimizedProtocolPolicy(n_actions)
    if name == "fisher_greedy_fast":
        return BoundedFisherGreedyPolicy(n_actions, max_actions=4, seed=seed)
    if name == "fisher_greedy":
        return FisherGreedyPolicy(n_actions)
    if name in {"greedy_bed", "smc_adaptive"}:
        return GreedyBEDPolicy(n_actions)
    if name == "bald_1step":
        return BALDOneStepPolicy(n_actions, seed=seed)
    if name == "bald_2step_fast":
        return BALDNStepPolicy(
            n_actions,
            depth=2,
            predictive_samples=1,
            top_k=2,
            seed=seed,
            max_actions=3,
            future_max_actions=2,
        )
    if name == "bald_2step":
        return BALDNStepPolicy(n_actions, depth=2, predictive_samples=5, top_k=10, seed=seed)
    raise ValueError(f"Unknown baseline: {name}")
