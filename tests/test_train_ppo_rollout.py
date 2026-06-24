from __future__ import annotations

import torch

from symqnet.env import SpinChainEnv
from symqnet.math_utils import gaussian_entropy_from_cov, set_seed
from symqnet.metadata import build_metadata
from symqnet.models.agent import SymQNetAgent
from symqnet.models.vae import VariationalAutoencoder
from symqnet.smc import SMCParticleFilter


def test_immediate_reward_loop_emits_one_transition_per_env_step() -> None:
    set_seed(123)
    device = torch.device("cpu")
    env = SpinChainEnv(n_qubits=3, m_evo=2, horizon=5, seed=123, device=device, shots_set=None)
    smc = SMCParticleFilter(env, n_particles=8, device=device)
    vae = VariationalAutoencoder(env.N, latent_dim=4)
    agent = SymQNetAgent(
        vae,
        env.N,
        latent_dim=4,
        history=4,
        n_actions=env.n_actions,
        m_evo=env.M_evo,
        use_vae=False,
        device=device,
    )

    obs = env.reset()
    smc.reset()
    posterior = smc.posterior()
    prev_info = None
    transitions = []
    done = False

    while not done:
        obs_t = torch.from_numpy(obs).float()
        metadata = build_metadata(
            env.N,
            env.M_evo,
            agent.theta_dim,
            agent.cov_feat_dim,
            agent.use_smc_feedback,
            agent.belief_mode,
            device,
            prev_info,
            posterior,
            env.shots_max,
        )
        dist, _ = agent(obs_t, metadata)
        action = int(torch.argmax(dist.probs).item())
        obs, _, done, info = env.step(action)
        h_prior = gaussian_entropy_from_cov(posterior.cov)
        posterior = smc.update(torch.from_numpy(obs).float(), info)
        reward = torch.clamp(h_prior - gaussian_entropy_from_cov(posterior.cov), min=0.0)
        transitions.append((action, float(reward.item()), done))
        prev_info = info

    assert len(transitions) == env.T
    assert transitions[-1][2] is True
    assert sum(1 for _, _, step_done in transitions if step_done) == 1
