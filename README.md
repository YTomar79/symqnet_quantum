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
symqnet/                       Core package
├── env.py                     TFIM spin-chain simulator (statevector / MPS backends, finite-shot + noise)
├── smc.py                     Sequential Monte Carlo particle filter (belief over Hamiltonian params)
├── backends.py                Statevector and MPS/TEBD simulation backends
├── task_bank.py               Sampling and serialization of shared evaluation task banks
├── baselines.py               random / fixed / fixed_optimized / Fisher-greedy / BALD comparators
├── models/                    Policy network components
│   ├── agent.py               SymQNetAgent: belief-conditioned actor-critic
│   ├── graph.py               Graph encoder over the spin chain
│   ├── temporal.py            Transformer encoder over measurement history
│   └── vae.py                 Belief VAE for measurement-outcome embedding
├── train_ppo.py               PPO training entry point
├── pretrain_vae.py            Belief-VAE pretraining entry point
├── behavior_clone.py          Behavior-cloning warm start
├── eval.py                    Evaluation entry point (CLI)
├── evaluation.py              Episode rollout and metric collection
├── cross_eval.py              Zero-shot transfer / out-of-distribution evaluation
├── config.py                  Experiment configuration schema and loading
├── math_utils.py              Fisher information, CRLB, and numerical helpers
├── metadata.py                Run metadata capture
├── provenance.py              Seed / config-hash / hardware provenance records
├── manifest.py                Per-run manifest writing and validation
├── plot_shot_budget.py        Shot-budget figure helper
├── paper_cpu_cluster.py       Parallel multi-seed sweep runner
└── analysis/                  Post-processing: statistics, figures, tables, gating
    ├── paired_main.py         Paired MSE/latency comparison with bootstrap CIs
    ├── stats.py               Wilcoxon, Cliff's delta, multiple-comparison correction
    ├── claim_gate.py          Pass/fail gate for the scaling claim
    ├── complexity.py          Action-space / latency complexity analysis
    ├── latency_scaling.py     Decision-latency scaling fits
    ├── mps_validation.py      MPS-vs-statevector agreement checks
    ├── paper_figures.py       Figure generation
    ├── tables.py              LaTeX/CSV table generation
    ├── paper_readiness.py     Fail-fast readiness check over a run directory
    └── ...                    Additional per-metric analyses

configs/                       Experiment configurations (JSON)
├── default.json               Primary N=5 experiment
├── smoke.json                 Fast wiring checks
├── noisy_native.json          Native decoherence/readout-noise model
├── ood_wide.json              Wider-prior transfer evaluation
├── dad_transformer.json       Neural BED baseline config
├── ablations_paper/           Headline ablations (full, no_vae, no_graph, no_smc_feedback, mlp_only)
├── ablations/                 Extended exploratory ablations
└── scaling/                   Per-chain-size configs (n4 … n12)

scripts/                       Sweep drivers
├── run_main_result.sh         N=5 Pareto benchmark, five seeds
├── run_scaling.sh             N=8/10/12 MPS-backed scaling
├── run_ablations.sh           Architecture ablation sweep
├── run_qcrl2026_repositioned.sh   Full end-to-end suite
└── ...                        Sensitivity, smoke, and learning-curve runners

runs/                          Committed experiment outputs
├── main_result/               N=5 benchmark: per-seed checkpoints, traces, tables, manifest
│   ├── paired_main.csv        Headline paired comparison
│   ├── shot_budget.{csv,svg}  MSE vs. shot-budget result
│   ├── symqnet_seed_*/        Trained policy checkpoints per seed
│   └── manifest.json          Inputs, file hashes, provenance
├── scaling/                   N-scaling: claim_gate.json, scaling_summary.csv, n8/n10/n12
├── ablations_paper/           Architecture ablation results
├── noisy_native/              Native-noise evaluation
├── ood_wide/                  Out-of-distribution transfer
├── reward_mse_delta/          Reward-alignment diagnostic
└── smoke*/                    Smoke-test outputs

artifacts/                     Pretrained belief-VAE checkpoints (vae_n{5,8,10,12}_l16.pt)
tests/                         Pipeline and readiness tests
reproduce.sh                   One-command full reproduction
requirements.txt               Pinned dependencies
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


