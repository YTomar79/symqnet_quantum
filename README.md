# SymQNet: Amortized Acquisition for Low-Latency Adaptive Hamiltonian Learning

**Yash Vardhan Tomar**¹, **Dheeraj Peddireddy**¹

¹ Purdue University

[![arXiv](https://img.shields.io/badge/arXiv-2606.12808-b31b1b.svg)](https://arxiv.org/abs/2606.12808)

---

This repo contains the code, raw CSVs, and results for "SymQNet: Amortized Acquisition for Low-Latency Adaptive Hamiltonian Learning" (arXiv:2606.12808v3). 

Submitted to IEEE Quantum Week International Workshop on Quantum Computing and Reinforcement Learning 2026.

## Abstract

Adaptive Hamiltonian learning is central to calibrating and characterizing quantum devices. In an adaptive controller, choosing the next experiment is itself a computation. Bayesian design rules are recomputed after every posterior update, and that step can take seconds. Across hundreds of shots, those seconds become a significant wall-clock cost for adaptivity. We introduce SymQNet, an amortized reinforcement-learning approach for low-latency adaptive Hamiltonian learning. SymQNet learns a posterior-conditioned acquisition policy offline, then uses a fast policy forward pass online while retaining Bayesian posterior feedback. On transverse-field Ising benchmarks, SymQNet substantially reduces acquisition latency relative to bounded Fisher-information search and bounded two-step Bayesian active learning by disagreement (BALD). At five qubits, it reduces acquisition-only decision latency by 47.1× and 72.6× relative to these online baselines; at twelve qubits, full simulated steps take 1.02 seconds for SymQNet versus 13.27 seconds for bounded two-step BALD. Overall, we show that learned acquisition can make adaptive Hamiltonian learning practical for repeated low-latency workloads.

## Repository layout

```
symqnet/            Core package
  env.py            Spin-chain simulator (statevector / MPS backends, shot noise)
  smc.py            Sequential Monte Carlo particle filter
  models/           Agent, graph encoder, temporal encoder, belief VAE
  train_ppo.py      PPO training entry point
  pretrain_vae.py   Belief-VAE pretraining
  eval.py           Evaluation entry point
  baselines.py      random / fixed / fixed_optimized / Fisher / BALD comparators
  cross_eval.py     Zero-shot transfer / out-of-distribution evaluation
  analysis/         Statistics, figures, tables, and result-validation tools
configs/            Experiment, ablation, and smoke configurations
scripts/            Sweep drivers for the main result, scaling, and ablations
runs/               Committed experiment outputs (summaries, traces, manifests)
artifacts/          Pretrained belief-VAE checkpoints
tests/              Pipeline and readiness tests
```

## Installation

Requires Python 3.12+.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Quick start

Wiring-level smoke checks run in seconds:

```bash
.venv/bin/python -m symqnet.eval --config configs/smoke.json --task-bank runs/smoke/task_bank.npz
.venv/bin/python -m symqnet.train_ppo --config configs/smoke.json --updates 1
```

## Training and evaluation

```bash
# 1. Pretrain the belief VAE
.venv/bin/python -m symqnet.pretrain_vae --config configs/default.json

# 2. Train the policy with PPO
.venv/bin/python -m symqnet.train_ppo --config configs/default.json

# 3. Evaluate against the baselines on a shared task bank
.venv/bin/python -m symqnet.eval --config configs/default.json \
  --task-bank runs/main_result/task_bank.npz
```

Training with `model.use_vae=true` requires the configured VAE checkpoint to exist;
pass `--allow-random-vae` only for smoke/debug runs. The best checkpoint is selected
by held-out validation-task MSE, not by training-rollout MSE.

## Reproducing the full result set

The complete benchmark sweep — main result, N-scaling, native-noise evaluation,
wider-prior transfer, ablations, and the reward-alignment diagnostic — is driven by
a single script:

```bash
bash reproduce.sh
```

Individual stages can be run directly:

```bash
bash scripts/run_main_result.sh     # N=5 Pareto benchmark, five seeds
bash scripts/run_scaling.sh         # N=8/10/12 MPS-backed scaling
bash scripts/run_ablations.sh       # architecture ablations
```

Before treating a run as final, validate it:

```bash
.venv/bin/python -m symqnet.analysis.paper_readiness \
  --run-root runs/main_result --config configs/default.json
```

## Citation

If you use this code or find this work helpful, please cite:

```bibtex
@article{tomar2026symqnet,
  title={SymQNet: Amortized Acquisition for Low-Latency Adaptive Hamiltonian Learning},
  author={Tomar, Yash Vardhan and Peddireddy, Dheeraj},
  journal={arXiv preprint arXiv:2606.12808},
  year={2026}
}
```

## License

Released under the [MIT License](LICENSE).


