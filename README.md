# SymQNet: Amortized Acquisition for Low-Latency Adaptive Hamiltonian Learning

This repo contains the code, raw CSVs, and results for "SymQNet: Amortized Acquisition for Low-Latency Adaptive Hamiltonian Learning" (arXiv:2606.12808v3). Submitted to QCE QCRL Workshop 2026. 

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

## License

Released under the MIT License. See [LICENSE](LICENSE).
