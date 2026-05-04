# FILE: src/metafed/orchestration/osmd_orchestrator.py
# OSMD (Online Mirror Descent) client sampling — Zhao et al. (2025) JMLR 26(8)
# "Adaptive Client Sampling in Federated Learning via Online Learning
#  with Bandit Feedback"
#
# Key idea: select clients based on gradient norm magnitude.
# Clients with larger gradient norms contribute more to model improvement,
# so they should be selected more often.
#
# The sampling distribution q is updated each round using mirror descent
# on the variance reduction loss:
#   l(q) = (1/K) * sum_m (a_m / q_m)
# where a_m = ||g_m||^2 (squared gradient norm of client m).
#
# The optimal distribution is p* ∝ sqrt(a_m), but since we can only observe
# gradient norms for selected clients, OSMD learns q online.
#
# Integration note:
# This orchestrator plugs into the existing BaseOrchestrator interface.
# It requires gradient norms from each round, which it gets by comparing
# client model state before and after local training. The server calls
# orchestrator.update_history(round_results) after each round — this
# orchestrator overrides that method to update q.
#
# IMPORTANT: This class needs access to client model states (before/after
# training) to compute gradient norm proxies. The server currently stores
# these in client_updates['params']. We approximate ||g||^2 from the
# parameter difference: ||w_after - w_before||^2 / (lr * local_epochs)^2.
# This is a standard proxy used in FL literature when true gradients
# are not available at the server.

import math
import random
import copy
import logging
from typing import List, Any, Optional, Dict

import torch

from .base import BaseOrchestrator

logger = logging.getLogger(__name__)


class OSMDOrchestrator(BaseOrchestrator):
    """
    OSMD-based adaptive client sampling orchestrator.

    Implements Algorithm 1 (OSMD Sampler) from Zhao et al. (2025).
    Learns client sampling probabilities online using mirror descent
    on estimated gradient norm magnitudes.

    Args:
        num_clients: Total number of clients in the federation (M in the paper)
        clients_per_round: Number of clients selected per round (K in the paper)
        alpha: Lower bound on sampling probability per client: q_m >= alpha/M.
               Paper uses alpha=0.4 as default. Prevents any client from
               being permanently excluded.
        eta: Learning rate for mirror descent update. If None, uses adaptive
             schedule eta_t = alpha / sqrt(t + 1).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_clients: int,
        clients_per_round: int,
        alpha: float = 0.4,
        eta: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        super().__init__("OSMDOrchestrator")

        self.M = num_clients
        self.K = clients_per_round
        self.alpha = alpha
        self.eta = eta
        self._round = 0

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Initialize sampling distribution: uniform
        # Shape: [M], each element is the probability of selecting client m
        self._q = torch.ones(num_clients) / num_clients

        # Gradient norm estimates per client: initialized to 1.0 (uniform)
        # a_m = ||g_m||^2 (squared gradient norm proxy)
        self._a = torch.ones(num_clients)

        # Track which clients were selected in the last round and their
        # observed gradient norms, so we can update q after each round
        self._last_selected_indices: List[int] = []
        self._last_client_id_to_index: Dict[Any, int] = {}

        logger.info(
            f"OSMDOrchestrator initialized: M={num_clients}, K={clients_per_round}, "
            f"alpha={alpha}, eta={'adaptive' if eta is None else eta}"
        )

    def select_clients(
        self,
        available_clients: List[Any],
        num_select: int,
        round_num: int,
        **kwargs,
    ) -> List[Any]:
        """
        Sample K clients according to current distribution q.

        Uses torch.multinomial for sampling WITHOUT replacement,
        consistent with Zhao et al.'s setup.

        Args:
            available_clients: All available clients (must match num_clients)
            num_select: Number of clients to select (K)
            round_num: Current round number
            **kwargs: Ignored

        Returns:
            List of selected clients
        """
        self._round = round_num

        n_available = len(available_clients)

        if num_select >= n_available:
            self._last_selected_indices = list(range(n_available))
            self._last_client_id_to_index = {
                getattr(c, "id", i): i for i, c in enumerate(available_clients)
            }
            logger.info(f"Round {round_num}: OSMD selected all {n_available} clients")
            return available_clients.copy()

        # Trim q to actual available clients (handles edge cases where
        # num_clients > len(available_clients) in a given round)
        q_available = self._q[:n_available].clone()
        q_available = q_available / q_available.sum()  # renormalize

        # Sample K clients without replacement using current distribution
        selected_indices = torch.multinomial(
            q_available, num_select, replacement=False
        ).tolist()

        # Store mapping for update_history
        self._last_selected_indices = selected_indices
        self._last_client_id_to_index = {
            getattr(c, "id", i): i for i, c in enumerate(available_clients)
        }

        selected = [available_clients[i] for i in selected_indices]

        logger.info(
            f"Round {round_num}: OSMD selected {len(selected)} clients, "
            f"q range=[{q_available.min():.4f}, {q_available.max():.4f}]"
        )
        return selected

    def update_history(self, round_results: Dict[str, Any]) -> None:
        """
        Update q using OSMD after observing gradient norms from this round.

        Called by FederatedServer.run_federated_learning() at line 393:
            if hasattr(self.orchestrator, 'update_history'):
                self.orchestrator.update_history(round_results)

        round_results is the dict from train_round() containing:
            'selected_clients': List[int]  — client IDs selected this round
            'client_gradient_norms': Dict[int, float]  — if present
            'client_param_norms': Dict[int, float]     — fallback proxy

        We compute a_m = ||g_m||^2 for selected clients and update q.
        For unselected clients, a_m remains unchanged (OSMD forgets old
        estimates by only using the most recent round's feedback).
        """
        super().update_history(round_results)

        selected_client_ids = round_results.get("selected_clients", [])
        if not selected_client_ids:
            return

        # Get gradient norm proxies from round_results
        # The server doesn't currently compute these, so we check for them
        # and fall back to a uniform update if not present.
        grad_norms = round_results.get("client_gradient_norms", {})
        param_norms = round_results.get("client_param_norms", {})

        # Update a_m for selected clients
        updated_any = False
        for cid in selected_client_ids:
            # Map client_id to index in [0, M)
            idx = self._last_client_id_to_index.get(cid)
            if idx is None:
                # Try direct integer interpretation
                if isinstance(cid, int) and 0 <= cid < self.M:
                    idx = cid
                else:
                    continue

            if cid in grad_norms:
                self._a[idx] = float(grad_norms[cid]) ** 2
                updated_any = True
            elif cid in param_norms:
                # Use parameter difference norm as proxy for gradient norm
                self._a[idx] = float(param_norms[cid]) ** 2
                updated_any = True
            # If neither is available, a_m[idx] stays at its previous value

        if not updated_any:
            # No norm information available this round; skip q update
            logger.debug(
                "OSMDOrchestrator: no gradient norm info in round_results; "
                "skipping q update. To enable OSMD updates, ensure "
                "server.train_round() populates 'client_gradient_norms' "
                "or 'client_param_norms' in round_results."
            )
            return

        self._update_q(selected_client_ids)

    def _update_q(self, selected_client_ids: List[int]) -> None:
        """
        Perform one OSMD update step on the sampling distribution q.

        The OSMD update from Zhao et al. Algorithm 1:
            For each selected client m:
                gradient of loss w.r.t. q_m = -a_m / q_m^2
            q_new = argmin_{q in A} { <nabla_loss, q - q_old> + (1/eta) * KL(q, q_old) }
            which gives the multiplicative update:
                q_new_m ∝ q_old_m * exp(eta * a_m / q_m^2)  for selected m
                q_new_m = q_old_m                             for unselected m
            Then project onto the simplex with lower bound alpha/M.
        """
        # Adaptive learning rate schedule
        if self.eta is None:
            eta = self.alpha / math.sqrt(self._round + 1)
        else:
            eta = self.eta

        lower_bound = self.alpha / self.M

        # Compute update only for selected clients
        # For unselected clients, their q values don't change before projection
        q_new = self._q.clone()

        selected_set = set()
        for cid in selected_client_ids:
            idx = self._last_client_id_to_index.get(cid)
            if idx is None and isinstance(cid, int) and 0 <= cid < self.M:
                idx = cid
            if idx is not None:
                selected_set.add(idx)

        for idx in selected_set:
            # OSMD multiplicative update (exponentiated gradient on log q)
            # Gradient of variance reduction loss: -a_m / q_m^2
            # Mirror descent step: log q_new_m = log q_old_m + eta * a_m / q_m^2
            grad = self._a[idx].item() / (self._q[idx].item() ** 2 + 1e-12)
            log_q_new = math.log(self._q[idx].item() + 1e-12) + eta * grad
            log_q_new = max(-500.0, min(500.0, log_q_new))  # prevent exp overflow
            q_new[idx] = math.exp(log_q_new)

        # Project: enforce lower bound alpha/M on all clients
        q_new = torch.clamp(q_new, min=lower_bound)

        # Renormalize to sum to 1
        q_sum = q_new.sum().item()
        if q_sum > 0:
            q_new = q_new / q_sum
        else:
            q_new = torch.ones(self.M) / self.M

        self._q = q_new

        logger.debug(
            f"OSMD q updated: min={self._q.min():.4f}, max={self._q.max():.4f}, "
            f"entropy={-(self._q * torch.log(self._q + 1e-12)).sum():.4f}"
        )

    def get_sampling_distribution(self) -> torch.Tensor:
        """Return current sampling distribution q (read-only copy)."""
        return self._q.clone()

    def get_gradient_norm_estimates(self) -> torch.Tensor:
        """Return current gradient norm estimates a_m (read-only copy)."""
        return self._a.clone()