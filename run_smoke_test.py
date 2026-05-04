#!/usr/bin/env python3
"""
Smoke test for MetaFed-FL updated code (Phase 1-3).
Run before pushing to Colab to verify imports and new modules.
Usage: from project root, run:  python run_smoke_test.py
"""

import sys
import os

# Project root and src (same as experiments)
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

def test_blockchain_and_orchestrator():
    """Test blockchain registry and green registry orchestrator (no torch)."""
    # Load modules under proper package names so .base and ..blockchain resolve
    import importlib.util
    _src = os.path.join(_root, "src")
    # 1) metafed package stub
    if "metafed" not in sys.modules:
        import types
        metafed = types.ModuleType("metafed")
        metafed.__path__ = [os.path.join(_src, "metafed")]
        sys.modules["metafed"] = metafed
    # 2) metafed.blockchain
    if "metafed.blockchain" not in sys.modules:
        import types
        m = types.ModuleType("metafed.blockchain")
        m.__path__ = [os.path.join(_src, "metafed", "blockchain")]
        sys.modules["metafed.blockchain"] = m
    # 3) metafed.orchestration
    if "metafed.orchestration" not in sys.modules:
        import types
        m = types.ModuleType("metafed.orchestration")
        m.__path__ = [os.path.join(_src, "metafed", "orchestration")]
        sys.modules["metafed.orchestration"] = m
    # 4) Load base orchestrator first (orchestrator imports .base)
    spec_base = importlib.util.spec_from_file_location(
        "metafed.orchestration.base",
        os.path.join(_src, "metafed", "orchestration", "base.py"),
        submodule_search_locations=[_src],
    )
    mod_base = importlib.util.module_from_spec(spec_base)
    sys.modules["metafed.orchestration.base"] = mod_base
    spec_base.loader.exec_module(mod_base)
    # 5) Load federated registry
    spec_r = importlib.util.spec_from_file_location(
        "metafed.blockchain.federated_registry",
        os.path.join(_src, "metafed", "blockchain", "federated_registry.py"),
        submodule_search_locations=[_src],
    )
    mod_r = importlib.util.module_from_spec(spec_r)
    sys.modules["metafed.blockchain.federated_registry"] = mod_r
    spec_r.loader.exec_module(mod_r)
    FederatedRegistry = mod_r.FederatedRegistry
    ClientMetadata = mod_r.ClientMetadata
    # 6) Load green registry orchestrator (imports .base and ..blockchain)
    spec_o = importlib.util.spec_from_file_location(
        "metafed.orchestration.green_registry_orchestrator",
        os.path.join(_src, "metafed", "orchestration", "green_registry_orchestrator.py"),
        submodule_search_locations=[_src],
    )
    mod_o = importlib.util.module_from_spec(spec_o)
    mod_o.__package__ = "metafed.orchestration"  # so "from .base" resolves to metafed.orchestration.base
    sys.modules["metafed.orchestration.green_registry_orchestrator"] = mod_o
    spec_o.loader.exec_module(mod_o)
    GreenRegistryOrchestrator = mod_o.GreenRegistryOrchestrator

    # Registry
    r = FederatedRegistry()
    r.register(0, capability=0.8, carbon_region=100.0, latency_ms=30.0, bandwidth_mbps=150.0)
    r.register(1, capability=0.5, carbon_region=200.0)
    assert r.get_chain_length() >= 3, "Chain should have genesis + 2 registrations"
    assert r.verify_chain(), "Chain verification should pass"
    assert r.get_client(0) is not None and r.get_client(0).capability == 0.8
    assert r.get_client(0).get_latency_ms() == 30.0
    assert r.get_client(0).get_bandwidth_mbps() == 150.0

    # Reputation (Phase 2)
    r.update_reputation(0, 0.1)
    r.update_reputation(1, 0.05)
    assert r.get_reputation(0) == 0.1 and r.get_reputation(1) == 0.05

    # Round commit (Phase 3)
    h = r.commit_round(round_id=1, selected_clients=[0, 1], carbon_emission_g=300.0, avg_loss=0.5, test_accuracy=95.0)
    assert len(h) == 64, "Block hash should be 64-char hex"
    assert r.verify_chain()

    # Orchestrator
    orch = GreenRegistryOrchestrator(registry=r, seed=42)
    class MockClient:
        id = 0
    clients = [MockClient()]
    MockClient.id = 0
    selected = orch.select_clients(clients, num_select=1, round_num=1, carbon_intensity=0.15)
    assert len(selected) == 1

    orch_inv = GreenRegistryOrchestrator(registry=r, seed=42, use_incentives=True)
    selected2 = orch_inv.select_clients(clients, num_select=1, round_num=2, carbon_intensity=0.2)
    assert len(selected2) == 1
    print("  [PASS] blockchain + GreenRegistryOrchestrator + reputation + commit_round")
    return True


def test_participation_fairness():
    """Test participation fairness metric (requires torch because metrics.py imports it at top level)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  [SKIP] participation_fairness (torch not installed; metrics module needs torch)")
        return True
    from metafed.utils.metrics import compute_participation_fairness

    selected_per_round = [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]]
    out = compute_participation_fairness(selected_per_round, num_clients=5)
    assert "jain_fairness_index" in out
    assert "min_participation_rate" in out
    assert 0 <= out["jain_fairness_index"] <= 1
    assert out["total_rounds"] == 4
    print("  [PASS] compute_participation_fairness")
    return True


def test_carbon_tracker():
    """Test carbon tracker time-of-day and g/kWh (no torch)."""
    import importlib.util
    _src = os.path.join(_root, "src")
    spec = importlib.util.spec_from_file_location(
        "carbon_tracking",
        os.path.join(_src, "metafed", "green", "carbon_tracking.py"),
        submodule_search_locations=[_src],
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    CarbonTracker = mod.CarbonTracker

    ct = CarbonTracker(region="US", use_time_of_day=True, i_base_g=150.0, amplitude_g=70.0)
    g = ct.get_current_intensity_g_per_kwh()
    assert 20 <= g <= 350
    kg = ct.get_current_intensity()
    assert 0.02 <= kg <= 0.35
    print("  [PASS] CarbonTracker (time-of-day, g/kWh)")
    return True


def test_full_experiment_minimal():
    """Run minimal MNIST experiment (2 rounds, 5 clients, 2 per round) if torch available."""
    try:
        import torch
    except ImportError:
        print("  [SKIP] full experiment (torch not installed; run on Colab to verify)")
        return True

    from experiments.mnist.run_experiment import run_experiment
    from argparse import Namespace

    args = Namespace(
        seed=42,
        device="cpu",
        config=None,
        output_dir=os.path.join(_root, "results", "smoke_test"),
        algorithm="fedavg",
        num_clients=5,
        clients_per_round=2,
        num_rounds=2,
        local_epochs=1,
        batch_size=32,
        learning_rate=0.01,
        non_iid_alpha=0.5,
        fedprox_mu=0.01,
        orchestrator="random",
        green_aware=False,
        carbon_tracking=False,
        privacy="none",
        epsilon=1.0,
        eval_frequency=1,
        use_incentives=False,
        identity_non_iid=False,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    results = run_experiment(args)
    assert "final_accuracy" in results or "training_history" in results
    assert "participation_fairness" in results
    assert results["participation_fairness"]["total_rounds"] == 2
    print("  [PASS] minimal MNIST experiment (random, 2 rounds)")
    return True


def test_registry_orchestrator_experiment():
    """Run minimal experiment with registry orchestrator if torch available."""
    try:
        import torch
    except ImportError:
        print("  [SKIP] registry experiment (torch not installed)")
        return True

    from experiments.mnist.run_experiment import run_experiment
    from argparse import Namespace

    args = Namespace(
        seed=42,
        device="cpu",
        config=None,
        output_dir=os.path.join(_root, "results", "smoke_test_registry"),
        algorithm="fedavg",
        num_clients=5,
        clients_per_round=2,
        num_rounds=2,
        local_epochs=1,
        batch_size=32,
        learning_rate=0.01,
        non_iid_alpha=0.5,
        fedprox_mu=0.01,
        orchestrator="registry",
        green_aware=False,
        carbon_tracking=False,
        privacy="none",
        epsilon=1.0,
        eval_frequency=1,
        use_incentives=False,
        identity_non_iid=False,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    results = run_experiment(args)
    assert "final_accuracy" in results or "training_history" in results
    assert "participation_fairness" in results
    assert "total_carbon_emission_g" in results
    print("  [PASS] minimal MNIST experiment (registry orchestrator, 2 rounds)")
    return True


def main():
    print("MetaFed-FL smoke test (updated code)")
    print("-" * 50)
    ok = True
    try:
        test_blockchain_and_orchestrator()
    except Exception as e:
        print(f"  [FAIL] blockchain/orchestrator: {e}")
        ok = False
    try:
        test_participation_fairness()
    except Exception as e:
        print(f"  [FAIL] participation_fairness: {e}")
        ok = False
    try:
        test_carbon_tracker()
    except Exception as e:
        print(f"  [FAIL] carbon_tracker: {e}")
        ok = False
    try:
        test_full_experiment_minimal()
    except Exception as e:
        print(f"  [FAIL] minimal experiment: {e}")
        ok = False
    try:
        test_registry_orchestrator_experiment()
    except Exception as e:
        print(f"  [FAIL] registry experiment: {e}")
        ok = False
    print("-" * 50)
    if ok:
        print("All smoke tests passed. Safe to push to Colab.")
        return 0
    print("Some tests failed. Fix before pushing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
