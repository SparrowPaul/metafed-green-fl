"""
MNIST Federated Learning Experiment Runner.

This script runs federated learning experiments on the MNIST dataset
with various algorithms, orchestration strategies, and privacy settings.
"""

import sys
import os
import argparse
import logging
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Dict, List, Any, Optional
import time
import json

# Add src to path to import metafed modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from metafed.core.client import Client, FedProxClient, SCAFFOLDClient
from metafed.core.server import FederatedServer
from metafed.core.aggregation import FedAvgAggregator, FedProxAggregator, SCAFFOLDAggregator
from metafed.orchestration.random_orchestrator import RandomOrchestrator
from metafed.orchestration.green_registry_orchestrator import GreenRegistryOrchestrator
from metafed.blockchain.federated_registry import FederatedRegistry
from metafed.green.carbon_tracking import CarbonTracker
from metafed.utils.logging_config import setup_logging
from metafed.utils.metrics import compute_accuracy, plot_results, compute_participation_fairness
from metafed.data.loaders import create_federated_datasets
from metafed.models.simple_cnn import ResNet18


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MNIST Federated Learning Experiment")
    
    # General settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    
    # Federated learning settings
    parser.add_argument("--algorithm", type=str, default="fedavg", 
                       choices=["fedavg", "fedprox", "scaffold"],
                       help="Federated learning algorithm")
    parser.add_argument("--num-clients", type=int, default=50, help="Total number of clients")
    parser.add_argument("--clients-per-round", type=int, default=10, help="Clients per round")
    parser.add_argument("--num-rounds", type=int, default=100, help="Number of FL rounds")
    parser.add_argument("--local-epochs", type=int, default=5, help="Local training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    
    # Data settings
    parser.add_argument("--non-iid-alpha", type=float, default=0.5, help="Non-IID alpha parameter")
    
    # FedProx specific
    parser.add_argument("--fedprox-mu", type=float, default=0.01, help="FedProx mu parameter")
    
    # Orchestration
    parser.add_argument("--orchestrator", type=str, default="random", 
                       choices=["random", "rl", "registry", "osmd"],
                       help="Client orchestration strategy (registry = blockchain + carbon-aware)")
    
    # Green computing
    parser.add_argument("--green-aware", action="store_true", help="Enable carbon-aware scheduling")
    parser.add_argument("--carbon-tracking", action="store_true", help="Enable carbon tracking")
    # Phase 2: incentives (reputation in registry orchestrator)
    parser.add_argument("--use-incentives", action="store_true", help="Use reputation/incentives in registry orchestrator")
    
    # Baseline B: OSMD bandit sampling (Zhao et al., 2025)
    parser.add_argument("--osmd-alpha", type=float, default=0.4,
                        help="OSMD lower bound on sampling prob (alpha/M). "
                             "Zhao et al. (2025) use 0.4.")
    parser.add_argument("--osmd-eta", type=float, default=None,
                        help="OSMD learning rate. If None, uses adaptive "
                             "schedule eta_t = alpha/sqrt(t+1).")

    # Baseline A: HybridBN aggregation (Chen et al., 2025)
    parser.add_argument("--bn-aggregation", type=str, default="metafed",
                        choices=["metafed", "hybridbn"],
                        help="BatchNorm buffer aggregation method. "
                             "'metafed' = original simple mean (default). "
                             "'hybridbn' = Chen et al. (2025) unbiased estimator.")
    # Phase 2: identity-style non-IID (Metaverse-like, more skewed)
    parser.add_argument("--identity-non-iid", action="store_true", help="Use identity-style non-IID (smaller alpha=0.2)")
    
    # Privacy
    parser.add_argument("--privacy", type=str, choices=["none", "differential"], 
                       default="none", help="Privacy mechanism")
    parser.add_argument("--epsilon", type=float, default=1.0, help="DP epsilon parameter")
    
    # Evaluation
    parser.add_argument("--eval-frequency", type=int, default=5, help="Evaluation frequency")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using default settings")
        return {}


def setup_device(device_arg: str) -> str:
    """Setup computation device."""
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    
    logging.info(f"Using device: {device}")
    return device


def create_model() -> nn.Module:
    """Create model for MNIST."""
    return ResNet18(num_classes=10, input_channels=1)


def create_clients(
    train_datasets: List[Subset],
    model_template: nn.Module,
    algorithm: str,
    lr: float,
    device: str,
    local_epochs: int,
    fedprox_mu: float = 0.01
) -> List[Client]:
    """Create federated learning clients."""
    clients = []
    
    for i, dataset in enumerate(train_datasets):
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        if algorithm == "fedavg":
            client = Client(
                client_id=i,
                train_loader=train_loader,
                model_template=model_template,
                lr=lr,
                device=device,
                local_epochs=local_epochs
            )
        elif algorithm == "fedprox":
            client = FedProxClient(
                client_id=i,
                train_loader=train_loader,
                model_template=model_template,
                lr=lr,
                device=device,
                local_epochs=local_epochs,
                mu=fedprox_mu
            )
        elif algorithm == "scaffold":
            client = SCAFFOLDClient(
                client_id=i,
                train_loader=train_loader,
                model_template=model_template,
                lr=lr,
                device=device,
                local_epochs=local_epochs
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        clients.append(client)
    
    logging.info(f"Created {len(clients)} {algorithm} clients")
    return clients


def create_and_populate_registry(
    clients: List[Client],
    train_datasets: List[Subset],
    seed: int = 42,
) -> FederatedRegistry:
    """
    Create blockchain-based Federated Registry and register all clients.
    
    Assigns each client synthetic capability (based on data size) and
    carbon_region (gCO2/kWh) for green-aware selection (Metaverse FL Phase 1).
    """
    rng = np.random.default_rng(seed)
    registry = FederatedRegistry()
    n_ds = len(train_datasets) if train_datasets else 0
    max_size = max(len(d) for d in train_datasets) if train_datasets else 1
    # Carbon regions in gCO2/kWh: low ~80, medium ~150, high ~250 (paper-style)
    region_options = np.array([80.0, 120.0, 150.0, 200.0, 250.0])
    # Phase 2: Metaverse-like latency (ms) and bandwidth (Mbps)
    latency_range = (20.0, 150.0)
    bandwidth_range = (50.0, 200.0)
    for i, client in enumerate(clients):
        # Capability: higher for clients with more data, plus small noise
        size_ratio = len(train_datasets[i]) / max_size if i < n_ds else 0.5
        capability = float(np.clip(size_ratio + rng.uniform(-0.1, 0.1), 0.2, 1.0))
        # Carbon region: random assignment to simulate geographic distribution
        carbon_region = float(rng.choice(region_options))
        # Phase 2: latency and bandwidth for Metaverse-aware selection
        latency_ms = float(rng.uniform(*latency_range))
        bandwidth_mbps = float(rng.uniform(*bandwidth_range))
        registry.register(
            client_id=client.id,
            capability=capability,
            carbon_region=carbon_region,
            latency_ms=latency_ms,
            bandwidth_mbps=bandwidth_mbps,
        )
    logging.info(
        f"Federated Registry: registered {len(clients)} clients (chain length={registry.get_chain_length()}), "
        f"chain verified={registry.verify_chain()}"
    )
    return registry


def create_aggregator(algorithm: str):
    """Create aggregator based on algorithm."""
    if algorithm == "fedavg":
        return FedAvgAggregator()
    elif algorithm == "fedprox":
        return FedProxAggregator()
    elif algorithm == "scaffold":
        return SCAFFOLDAggregator()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def create_orchestrator(
    orchestrator_type: str,
    registry: Optional[FederatedRegistry] = None,
    seed: Optional[int] = None,
    use_incentives: bool = False,
    num_clients=50,
    clients_per_round=10,
    osmd_alpha=0.4,
    osmd_eta=None,
):
    """Create client orchestrator."""
    if orchestrator_type == "random":
        return RandomOrchestrator(seed=seed)
    elif orchestrator_type == "rl":
        logging.warning("RL orchestrator not implemented, using random")
        return RandomOrchestrator(seed=seed)
    elif orchestrator_type == "registry":
        if registry is None:
            raise ValueError("Registry orchestrator requires a FederatedRegistry.")
        return GreenRegistryOrchestrator(registry=registry, seed=seed, use_incentives=use_incentives)
    elif orchestrator_type == "osmd":
        from src.metafed.orchestration.osmd_orchestrator import OSMDOrchestrator
        return OSMDOrchestrator(
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            alpha=osmd_alpha,
            eta=osmd_eta,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown orchestrator: {orchestrator_type}")


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Run federated learning experiment."""
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = setup_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(os.path.join(args.output_dir, "experiment.log"))
    
    logging.info("Starting MNIST federated learning experiment")
    logging.info(f"Configuration: {vars(args)}")
    
    # Load data (Phase 2: identity-non-iid uses smaller alpha for Metaverse-like skew)
    non_iid_alpha = 0.2 if args.identity_non_iid else args.non_iid_alpha
    if args.identity_non_iid:
        logging.info("Using identity-style non-IID (alpha=0.2) for Metaverse-like heterogeneity")
    logging.info("Loading and partitioning MNIST dataset")
    train_datasets, test_loader = create_federated_datasets(
        dataset_name="mnist",
        num_clients=args.num_clients,
        non_iid_alpha=non_iid_alpha,
        batch_size=args.batch_size
    )
    
    # Create model
    model_template = create_model()
    logging.info(f"Created model: {model_template.__class__.__name__}")
    
    # Create clients
    clients = create_clients(
        train_datasets=train_datasets,
        model_template=model_template,
        algorithm=args.algorithm,
        lr=args.learning_rate,
        device=device,
        local_epochs=args.local_epochs,
        fedprox_mu=args.fedprox_mu
    )
    
    # Create aggregator
    aggregator = create_aggregator(args.algorithm)
    
    # Create Federated Registry and register clients when using registry orchestrator
    registry = None
    if args.orchestrator == "registry":
        registry = create_and_populate_registry(clients, train_datasets, seed=args.seed)
    
    # Create orchestrator (Phase 2: use_incentives for reputation-based selection)
    orchestrator = create_orchestrator(
        args.orchestrator,
        registry=registry,
        seed=args.seed,
        use_incentives=args.use_incentives,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        osmd_alpha=args.osmd_alpha,
        osmd_eta=args.osmd_eta,
    )
    
    # Carbon-aware: enable when green flag set or when using registry (so intensity is passed)
    carbon_aware = args.green_aware or (args.orchestrator == "registry")
    
    # Create server (Phase 3: pass registry for reputation updates and round commits)
    server = FederatedServer(
        model_template=model_template,
        orchestrator=orchestrator,
        num_rounds=args.num_rounds,
        clients_per_round=args.clients_per_round,
        device=device,
        carbon_aware=carbon_aware,
        privacy_budget=args.epsilon if args.privacy == "differential" else None,
        registry=registry,
    )
    # Apply BN aggregation method (Baseline A: HybridBN or default MetaFed)
    server.bn_aggregation = args.bn_aggregation
    
    # Run federated learning
    logging.info("Starting federated learning training")
    start_time = time.time()
    
    results = server.run_federated_learning(
        clients=clients,
        aggregator=aggregator,
        test_loader=test_loader,
        eval_frequency=args.eval_frequency
    )
    
    training_time = time.time() - start_time
    results["total_training_time"] = training_time

    # Phase 3: Participation fairness (Jain index, min rate)
    results["participation_fairness"] = compute_participation_fairness(
        results["training_history"]["selected_clients"],
        len(clients),
    )
    logging.info(
        f"Participation fairness: Jain index={results['participation_fairness']['jain_fairness_index']:.4f}, "
        f"min_rate={results['participation_fairness']['min_participation_rate']:.4f}"
    )
    
    logging.info(f"Experiment completed in {training_time:.2f} seconds")
    
    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        # Convert tensors to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                json_results[key] = value.tolist()
            elif key == "final_model_state":
                # Skip model state dict for JSON
                continue
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    logging.info(f"Results saved to {results_path}")
    
    # Plot results if matplotlib is available
    try:
        plot_results(results, save_path=os.path.join(args.output_dir, "plots.png"))
        logging.info("Plots saved")
    except ImportError:
        logging.warning("Matplotlib not available, skipping plots")
    
    return results


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Update args with config values (command line takes precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    try:
        results = run_experiment(args)
        
        # Print summary
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Algorithm: {args.algorithm}")
        print(f"Rounds: {args.num_rounds}")
        print(f"Clients: {args.num_clients} (per round: {args.clients_per_round})")
        
        if "final_accuracy" in results:
            print(f"Final Accuracy: {results['final_accuracy']:.2f}%")
        
        if "total_carbon_emission" in results:
            print(f"Carbon Emission: {results['total_carbon_emission']:.6f} kg CO2")
        if "participation_fairness" in results:
            pf = results["participation_fairness"]
            print(f"Participation Jain Fairness: {pf.get('jain_fairness_index', 0):.4f}")
            print(f"Min Participation Rate: {pf.get('min_participation_rate', 0):.4f}")
        print(f"Training Time: {results['total_training_time']:.2f} seconds")
        print("="*50)
        
    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()