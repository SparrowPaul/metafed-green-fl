# How to Run MetaFed-FL and Improve Results

This guide gets the project running and gives practical tips to improve accuracy and efficiency.

---

## 1. One-time setup

**From the project root** (`MetaFed-FL/`):

```bash
# Optional: use a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package and dependencies (editable so code changes apply immediately)
pip install -e .

# If you hit dependency issues, install core deps first:
pip install -r requirements.txt
pip install -e .
```

- **Python**: 3.9 or newer.
- **Device**: CPU works; use `--device cuda` (or `auto`) if you have a GPU for faster runs.

---

## 2. Run experiments (from project root)

All commands below must be run from the **project root** so `experiments` and `src` are on the path.

### Quick run (MNIST, few rounds – for sanity check)

```bash
python -m experiments.mnist.run_experiment \
  --num-rounds 5 \
  --num-clients 20 \
  --clients-per-round 5 \
  --output-dir ./results/quick
```

### Full MNIST run (default config)

```bash
python -m experiments.mnist.run_experiment \
  --algorithm fedavg \
  --num-rounds 100 \
  --num-clients 50 \
  --clients-per-round 10 \
  --output-dir ./results/mnist_fedavg
```

### CIFAR-10 (FedProx, more rounds)

```bash
python -m experiments.cifar10.run_experiment \
  --algorithm fedprox \
  --fedprox-mu 0.01 \
  --num-rounds 100 \
  --num-clients 50 \
  --clients-per-round 10 \
  --output-dir ./results/cifar10_fedprox
```

### Using a YAML config

```bash
python -m experiments.mnist.run_experiment --config experiments/configs/mnist_fedavg.yaml
python -m experiments.cifar10.run_experiment --config experiments/configs/cifar10_fedprox.yaml
```

Results are written to `--output-dir`: `results.json`, `plots.png`, and `experiment.log`.

---

## 3. Fast iteration (to improve results)

Use **small** runs while you tune hyperparameters or change code:

```bash
# Very fast (~1–2 min on CPU): 3 rounds, 10 clients, 3 per round
python -m experiments.mnist.run_experiment \
  --num-rounds 3 \
  --num-clients 10 \
  --clients-per-round 3 \
  --local-epochs 2 \
  --eval-frequency 1 \
  --output-dir ./results/iter
```

Then scale up once you’re happy:

```bash
python -m experiments.mnist.run_experiment \
  --num-rounds 50 \
  --num-clients 50 \
  --clients-per-round 10 \
  --output-dir ./results/full
```

---

## 4. Tips to improve results

### Algorithm and data

| What to try | Why |
|-------------|-----|
| **FedProx** for non-IID data | `--algorithm fedprox --fedprox-mu 0.01` (or 0.1 for very heterogeneous data) improves stability when client data is skewed. |
| **Tune `--non-iid-alpha`** | Lower (e.g. 0.3) = more non-IID; higher (e.g. 1.0) = closer to IID. Match to your target scenario. |
| **More clients per round** | e.g. `--clients-per-round 20` often improves convergence (more updates per round). |
| **More rounds** | Increase `--num-rounds` (e.g. 150–200 for CIFAR-10) until validation accuracy plateaus. |

### Learning and training

| What to try | Why |
|-------------|-----|
| **Learning rate** | Try `--learning-rate 0.001` or `0.02`; 0.01 is default. Lower can help stability; higher can speed up early rounds. |
| **Local epochs** | `--local-epochs 3` or `5`; more epochs per client can improve accuracy but increase communication cost and overfitting risk. |
| **Batch size** | `--batch-size 64` can speed up and sometimes stabilize training. |

### Reproducibility and logging

| What to try | Why |
|-------------|-----|
| **Fixed seed** | `--seed 42` (default) for reproducible runs when comparing changes. |
| **Carbon / green** | `--carbon-tracking --green-aware` to log CO₂ and use carbon-aware scheduling in the server. |
| **Privacy** | `--privacy differential --epsilon 1.0` to enable differential privacy (may slightly lower accuracy). |

### Example: stronger MNIST run

```bash
python -m experiments.mnist.run_experiment \
  --algorithm fedprox \
  --fedprox-mu 0.01 \
  --num-rounds 80 \
  --num-clients 50 \
  --clients-per-round 15 \
  --learning-rate 0.01 \
  --local-epochs 5 \
  --non-iid-alpha 0.5 \
  --carbon-tracking \
  --output-dir ./results/mnist_improved
```

### Example: stronger CIFAR-10 run

```bash
python -m experiments.cifar10.run_experiment \
  --algorithm fedprox \
  --fedprox-mu 0.01 \
  --num-rounds 150 \
  --num-clients 50 \
  --clients-per-round 15 \
  --learning-rate 0.01 \
  --batch-size 64 \
  --non-iid-alpha 0.3 \
  --output-dir ./results/cifar10_improved
```

---

## 5. Where results and code live

- **Output**: `./results/<output-dir>/`  
  - `results.json` – final accuracy, loss, carbon, timing, training history.  
  - `plots.png` – accuracy, loss, carbon, time per round.  
  - `experiment.log` – detailed logs.

- **Change behavior**:  
  - Algorithms: `src/metafed/core/client.py`, `src/metafed/core/aggregation.py`, `src/metafed/algorithms/`.  
  - Server/orchestration: `src/metafed/core/server.py`, `src/metafed/orchestration/`.  
  - Data: `src/metafed/data/loaders.py`.  
  - Models: `src/metafed/models/simple_cnn.py`.

Because you installed with `pip install -e .`, edits in `src/` are used on the next run without reinstalling.

---

## 6. Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'metafed'` | Run from **project root** and use `python -m experiments.mnist.run_experiment` (not from another directory). |
| `ModuleNotFoundError: No module named 'experiments'` | Same: run from project root. |
| Out of memory | Use `--batch-size 16`, `--clients-per-round 5`, or `--device cpu`. |
| Runs too slow | Use fewer rounds/clients for iteration; use `--device cuda` if you have a GPU. |
| Config not applied | Config file path is relative to current dir; use `--config experiments/configs/mnist_fedavg.yaml` from project root. |

---

## 7. Quick reference: main CLI options

```
--algorithm       fedavg | fedprox | scaffold
--num-rounds      total FL rounds
--num-clients     total clients in the federation
--clients-per-round   clients selected each round
--learning-rate   e.g. 0.01
--local-epochs    local training epochs per round
--batch-size      e.g. 32 or 64
--non-iid-alpha   data heterogeneity (lower = more non-IID)
--fedprox-mu      only for FedProx (e.g. 0.01)
--output-dir      where results.json and plots go
--config          path to YAML config
--device          auto | cpu | cuda
--carbon-tracking --green-aware   enable carbon tracking
--privacy differential --epsilon 1.0   enable DP
```

Use this file as the single place to “get the project to run” and to improve results over time.
