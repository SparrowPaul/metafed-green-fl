# MetaFed: Carbon-Aware Client Selection and Auditable Orchestration for Federated Learning
 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
 
Under review at IEEE Transactions on Neural Networks and Learning Systems (TNNLS).
Double-anonymous submission — author information omitted.
 
---
 
## What is MetaFed?
 
MetaFed is a federated learning framework that closes three practical gaps present in most existing systems.
 
**1. Correct BatchNorm buffer aggregation.**
Standard FedAvg aggregates only weight and bias parameters, silently ignoring BatchNorm running statistics. This leaves test accuracy near random chance — in our experiments, MNIST accuracy stayed near 10% without the fix. MetaFed aggregates float buffers by weighted averaging and integer buffers by first-client value, recovering 98–99% accuracy.
 
**2. Carbon-aware client selection.**
A registry-based orchestrator scores each client using a priority function P(i,t) that combines capability score, regional carbon intensity, and time-of-day grid intensity. Selection prefers clients that are simultaneously capable and located in lower-carbon regions at lower-carbon times.
 
**3. Auditability and participation fairness.**
Each training round is committed to a tamper-evident append-only log recording selected clients, carbon cost, loss, and accuracy. Jain's fairness index and minimum participation rate are reported to quantify the trade-off between carbon efficiency and fair client participation.
 
---
 
## Relationship to Prior Work
 
This codebase extends the open-source framework of
[Yagiz et al. (2025)](https://arxiv.org/abs/2508.17341)
(MIT licence), which provides the base FL training loop, experiment
runners, and client–server scaffolding. MetaFed replaces the MARL
orchestration engine of that framework with a registry-based carbon-aware
selection policy and adds:
 
- Correct BatchNorm buffer aggregation (`src/metafed/core/aggregation_bn.py`, `server.py`)
- OSMD bandit orchestration baseline (`src/metafed/orchestration/osmd_orchestrator.py`)
- Tamper-evident round commits (`src/metafed/blockchain/federated_registry.py`)
- Participation fairness metrics (Jain's index, minimum participation rate)
None of these appear in Yagiz et al. (2025).
 
---
 
## Key Results
 
All reproduced baselines: non-IID Dirichlet α=0.5, 50 clients, 10/round,
100 rounds, ResNet-18, 3 seeds (42, 43, 44).
 
| Method | MNIST Acc (%) | CIFAR-10 Acc (%) | Jain Index | Min. Part. |
|--------|:---:|:---:|:---:|:---:|
| MetaFed (random orch., 3 seeds) | 99.32 ± 0.06 | 72.88 ± 0.35 | 0.965 | 0.130 |
| MetaFed (carbon-aware orch.) | 98.82 | — | 0.20 | 0.00 |
| HybridBN — Chen et al. (2025) | 99.33 ± 0.02 | 72.82 ± 0.73 | 0.966 | 0.130 |
| OSMD — Zhao et al. (2025) | 99.31 ± 0.09 | 73.96 ± 1.08 | 0.952 | 0.113 |
 
MetaFed matches 2025 baselines on accuracy while additionally providing
carbon-aware selection, tamper-evident auditability, and participation
fairness reporting — none of which appear in either baseline.
 
---
 
## Installation
 
```bash
pip install -e .
```
 
Requirements: Python 3.8+, PyTorch 2.0+. See `requirements.txt`.
 
Verify installation with a quick smoke test (5 rounds, 20 clients):
 
```bash
python run_smoke_test.py
```
 
---
 
## Running Experiments
 
**MNIST FedAvg — random selection, 100 rounds:**
```bash
python -m experiments.mnist.run_experiment \
  --algorithm fedavg \
  --num-rounds 100 --num-clients 50 --clients-per-round 10 \
  --output-dir ./results/mnist_fedavg
```
 
**MNIST with carbon-aware registry — 50 rounds:**
```bash
python -m experiments.mnist.run_experiment \
  --algorithm fedavg --orchestrator registry \
  --num-rounds 50 --num-clients 50 --clients-per-round 10 \
  --output-dir ./results/mnist_registry
```
 
**CIFAR-10 FedProx — 100 rounds:**
```bash
python -m experiments.cifar10.run_experiment \
  --algorithm fedprox --fedprox-mu 0.01 \
  --num-rounds 100 --num-clients 50 --clients-per-round 10 \
  --output-dir ./results/cifar10_fedprox
```
 
**HybridBN baseline — MNIST, 3 seeds:**
```bash
for SEED in 42 43 44; do
  python -m experiments.mnist.run_experiment \
    --algorithm fedavg --bn-aggregation hybridbn \
    --num-rounds 100 --num-clients 50 --clients-per-round 10 \
    --local-epochs 5 --batch-size 32 --learning-rate 0.01 \
    --non-iid-alpha 0.5 --seed $SEED \
    --output-dir ./results/hybridbn_mnist_seed${SEED}
done
```
 
**OSMD baseline — MNIST, 3 seeds:**
```bash
for SEED in 42 43 44; do
  python -m experiments.mnist.run_experiment \
    --algorithm fedavg --orchestrator osmd --osmd-alpha 0.4 \
    --num-rounds 100 --num-clients 50 --clients-per-round 10 \
    --local-epochs 5 --batch-size 32 --learning-rate 0.01 \
    --non-iid-alpha 0.5 --seed $SEED \
    --output-dir ./results/osmd_mnist_seed${SEED}
done
```
 
Full commands for all experiments including CIFAR-10 baselines are in `RUN.md`.
 
---
 
## Project Structure
 
```
src/metafed/
├── core/
│   ├── server.py              # Federated server + BatchNorm buffer aggregation
│   ├── aggregation_bn.py      # HybridBN and MetaFed BN aggregation strategies
│   ├── aggregation.py         # FedAvg / FedProx parameter aggregation
│   └── client.py              # Local training (FedAvg, FedProx)
├── orchestration/
│   ├── green_registry_orchestrator.py  # Carbon-aware priority selection
│   ├── osmd_orchestrator.py            # OSMD bandit baseline
│   ├── random_orchestrator.py          # Random selection baseline
│   └── base.py
├── blockchain/
│   └── federated_registry.py  # Append-only tamper-evident round log
├── green/
│   └── carbon_tracking.py     # Carbon intensity model I(t)
experiments/
├── mnist/run_experiment.py
├── cifar10/run_experiment.py
└── configs/
```
 
---
 
## Acknowledgements
 
Base framework by Yagiz, Cengiz, and Goktas (2025), MIT licence.