"""
Carbon-aware client orchestrator using the blockchain Federated Registry.

Selects clients based on registry metadata (capability, carbon_region,
latency, bandwidth) and optional reputation (Phase 2). Favors green and
capable participants (Metaverse FL).
"""

import random
from typing import List, Any, Optional
import logging

from .base import BaseOrchestrator
from ..blockchain.federated_registry import FederatedRegistry

logger = logging.getLogger(__name__)

# Paper: I_threshold = 100 gCO2/kWh
DEFAULT_I_THRESHOLD_G = 100.0


class GreenRegistryOrchestrator(BaseOrchestrator):
    """
    Carbon-aware client selection using the Federated Registry.

    Uses registry metadata (capability, carbon_region, latency, bandwidth)
    and optional reputation (Phase 2) to compute priority.
    """

    def __init__(
        self,
        registry: FederatedRegistry,
        seed: Optional[int] = None,
        i_threshold_g: float = DEFAULT_I_THRESHOLD_G,
        use_incentives: bool = False,
        latency_weight: float = 0.1,
        bandwidth_weight: float = 0.1,
    ):
        """
        Initialize green registry orchestrator.

        Args:
            registry: Blockchain-based federated registry with client metadata
            seed: Random seed for tie-breaking and sampling
            i_threshold_g: Carbon intensity threshold in gCO2/kWh (paper: 100)
            use_incentives: If True, factor reputation into priority (Phase 2)
            latency_weight: Weight for latency factor (lower latency = higher priority)
            bandwidth_weight: Weight for bandwidth factor (higher bandwidth = higher priority)
        """
        super().__init__("GreenRegistryOrchestrator")
        self.registry = registry
        self.i_threshold_g = i_threshold_g
        self.use_incentives = use_incentives
        self.latency_weight = latency_weight
        self.bandwidth_weight = bandwidth_weight
        if seed is not None:
            random.seed(seed)
        self.seed = seed
        logger.info(
            f"Initialized GreenRegistryOrchestrator with registry ({registry.get_chain_length()} blocks), "
            f"I_threshold={i_threshold_g} gCO2/kWh, use_incentives={use_incentives}"
        )

    def _carbon_intensity_to_g_per_kwh(self, carbon_intensity: Optional[float]) -> float:
        """
        Convert carbon intensity to gCO2/kWh for priority computation.

        Server/CarbonTracker may provide intensity in kg CO2/kWh; paper uses g.
        """
        if carbon_intensity is None:
            return 150.0  # default mid value in g
        # If value is small (e.g. 0.15) assume kg -> g
        if carbon_intensity < 10.0:
            return carbon_intensity * 1000.0
        return carbon_intensity

    def _priority(
        self,
        capability: float,
        carbon_region: float,
        i_current_g: float,
        reputation: float = 0.0,
        latency_ms: float = 50.0,
        bandwidth_mbps: float = 100.0,
    ) -> float:
        """
        Compute selection priority (higher = prefer more).

        Base: capability * green_factor * grid_factor.
        Phase 2: * (1 + latency_factor + bandwidth_factor), * (0.7 + 0.3 * norm(reputation)).
        """
        green_factor = self.i_threshold_g / max(self.i_threshold_g, carbon_region)
        grid_factor = self.i_threshold_g / max(self.i_threshold_g, i_current_g)
        base = capability * green_factor * (0.5 + 0.5 * grid_factor)
        # Metaverse: prefer lower latency (Phase 2)
        latency_factor = 1.0 / (1.0 + latency_ms / 200.0)  # 0–1, higher for low latency
        bandwidth_factor = min(1.0, bandwidth_mbps / 150.0)  # 0–1, higher for high bandwidth
        base *= 1.0 + self.latency_weight * latency_factor + self.bandwidth_weight * bandwidth_factor
        # Reputation incentive (Phase 2): normalize rep to [0,1] and blend
        if self.use_incentives and reputation > 0:
            # Simple norm: rep / (1 + rep) in [0,1]
            rep_norm = reputation / (1.0 + reputation)
            base *= 0.7 + 0.3 * rep_norm
        return base

    def select_clients(
        self,
        available_clients: List[Any],
        num_select: int,
        round_num: int,
        carbon_intensity: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Select clients using registry metadata and carbon-aware priority.

        Args:
            available_clients: List of all available clients (must have .id)
            num_select: Number of clients to select
            round_num: Current round number
            carbon_intensity: Current grid carbon intensity (kg or g CO2/kWh)
            **kwargs: Ignored

        Returns:
            List of selected clients (top by priority, with random tie-breaking)
        """
        i_current_g = self._carbon_intensity_to_g_per_kwh(carbon_intensity)

        # Build (client, priority) for clients that are in the registry
        scored: List[tuple] = []
        for client in available_clients:
            client_id = getattr(client, "id", None)
            if client_id is None:
                client_id = id(client)  # fallback
            meta = self.registry.get_client(client_id)
            rep = self.registry.get_reputation(client_id) if self.use_incentives else 0.0
            if meta is None:
                priority = self._priority(0.5, 150.0, i_current_g, reputation=rep, latency_ms=50.0, bandwidth_mbps=100.0)
                logger.debug(f"Client {client_id} not in registry, using default priority {priority:.3f}")
            else:
                priority = self._priority(
                    meta.capability,
                    meta.carbon_region,
                    i_current_g,
                    reputation=rep,
                    latency_ms=meta.get_latency_ms(),
                    bandwidth_mbps=meta.get_bandwidth_mbps(),
                )
            scored.append((client, priority))

        # Sort by priority descending; break ties randomly
        scored.sort(key=lambda x: (x[1], random.random()), reverse=True)

        if num_select >= len(available_clients):
            selected = available_clients.copy()
        else:
            selected = [c for c, _ in scored[:num_select]]

        logger.info(
            f"Round {round_num}: Selected {len(selected)} clients (carbon-aware registry), "
            f"carbon_intensity={i_current_g:.0f} gCO2/kWh"
        )
        return selected
