# SymQNet

**Amortized adaptive Hamiltonian learning for transverse-field Ising models.**

SymQNet is a reinforcement-learning policy that chooses adaptive quantum
measurement actions to estimate Hamiltonian parameters. It amortizes the cost of
adaptive experimental design: once trained, it selects near-optimal measurements
in a single forward pass, where classical online Bayesian design (BALD, Fisher-greedy)
must solve an expensive optimization at every step. The policy targets the
scaling-relevant regime where that per-step optimization becomes the dominant
bottleneck.

The system has three parts:

- **`SymQNetAgent`** chooses adaptive measurement actions from a belief-conditioned
  graph + temporal encoder.
- **`SMCParticleFilter`** maintains a posterior over Hamiltonian parameters and
  supplies belief feedback to the policy.
- **`SpinChainEnv`** applies finite-shot sampling plus either readout-flip noise or
  a lightweight native decoherence/readout model, with a dense statevector backend
  by default and an opt-in MPS/TEBD backend for larger chains.

## Key result

On TFIM chains of increasing size, SymQNet matches or beats the parameter MSE of
adaptive online baselines while keeping per-decision latency almost flat, whereas
online Bayesian design grows steeply with the action space.

| Chain size | MSE ratio vs. online BALD (95% CI hi) | Decision-latency speedup (95% CI lo) |
|-----------:|--------------------------------------:|-------------------------------------:|
| N = 10     | 0.82 (lower MSE)                      | ~4,300×                              |
| N = 12     | 0.81 (lower MSE)                      | ~6,300×                              |

Decision-latency log-slope in the action space is **0.30** for SymQNet versus
**1.52** for the online BALD comparator. Source data and the pass/fail criteria
are in `runs/scaling/claim_gate.json`.

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

## Baselines

Every baseline runs in the same environment and reports final parameter MSE and
decision latency on a shared task bank: `random`, `fixed`, `fixed_optimized`
(non-adaptive optimized schedule), `fisher_greedy_fast` (bounded adaptive Fisher),
`bald_2step_fast` (bounded two-step BALD), and `dad_transformer` (a neural
Bayesian-experimental-design comparator).

## Ablations

The headline architecture ablations live in `configs/ablations_paper/`: `full`,
`no_vae`, `no_graph`, `no_smc_feedback`, and `mlp_only`. A larger exploratory set is
in `configs/ablations/`.

## Results provenance

Result CSVs carry explicit `train_seed`, `eval_seed`, `task_id`, checkpoint path,
config hash, device, and training wall-clock provenance. Each run directory includes
a `manifest.json` recording inputs and file hashes.

## Large files

The largest per-episode traces are tracked with **Git LFS** (see `.gitattributes`).
Install Git LFS before cloning to retrieve them:

```bash
git lfs install
git clone <repository-url>
```

## Testing

```bash
.venv/bin/python -m pytest tests
```

## Scope and limitations

Results use statevector/MPS simulation rather than hardware, local single-qubit
measurements, and bounded online Fisher/BALD comparators. PPO trains on entropy
reduction while evaluation reports final parameter MSE; `reward_objective.csv` is a
required diagnostic of this proxy, not a settled validation. The SMC estimator is
biased, so CRLB ratios are reported as diagnostics only, not as a headline
efficiency claim.

## License

Released under the MIT License. See [LICENSE](LICENSE).
