"""
Blockchain-based Federated Registry (F).

Implements a lightweight mock blockchain / DHT for Metaverse FL:
- Client registration with identity, capability, carbon_region, latency, bandwidth
- Reputation/token for incentive-based selection (Phase 2)
- Round commits on-chain for audit (Phase 3)
- Tamper-evident chain for verification
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClientMetadata:
    """Metadata for a registered client (resource provider)."""

    client_id: int
    capability: float  # Normalized compute capability in [0, 1] or similar
    carbon_region: float  # Carbon intensity in gCO2/kWh (lower = greener)
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)  # e.g. latency_ms, bandwidth_mbps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "capability": self.capability,
            "carbon_region": self.carbon_region,
            "timestamp": self.timestamp,
            **self.extra,
        }

    def get_latency_ms(self) -> float:
        """Metaverse: latency in ms (lower = better). Default 50."""
        return float(self.extra.get("latency_ms", 50.0))

    def get_bandwidth_mbps(self) -> float:
        """Metaverse: bandwidth in Mbps (higher = better). Default 100."""
        return float(self.extra.get("bandwidth_mbps", 100.0))


@dataclass
class Block:
    """Single block in the registry chain (immutable once created)."""

    index: int
    prev_hash: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    nonce: int = 0

    def hash(self) -> str:
        payload = json.dumps(
            {
                "index": self.index,
                "prev_hash": self.prev_hash,
                "data": self.data,
                "timestamp": self.timestamp,
                "nonce": self.nonce,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class FederatedRegistry:
    """
    Blockchain-based Federated Registry for resource discovery and identity.

    Stores client metadata (capability, carbon_region, latency, bandwidth) in a
    tamper-evident chain. Supports reputation/token (Phase 2) and round commits
    on-chain (Phase 3) for audit and incentives.
    """

    def __init__(self):
        self._chain: List[Block] = []
        self._index: Dict[int, ClientMetadata] = {}  # client_id -> latest metadata
        self._reputation: Dict[int, float] = {}  # client_id -> reputation score (Phase 2)
        self._genesis()

    def _genesis(self) -> None:
        """Create genesis block."""
        genesis = Block(
            index=0,
            prev_hash="0" * 64,
            data={"action": "genesis", "message": "Federated Registry initialized"},
        )
        self._chain.append(genesis)
        logger.info("Federated Registry: genesis block created")

    def register(
        self,
        client_id: int,
        capability: float,
        carbon_region: float,
        **extra: Any,
    ) -> str:
        """
        Register a client (resource provider) on the registry chain.

        Args:
            client_id: Unique client identifier
            capability: Normalized compute capability (e.g. in [0.2, 1.0])
            carbon_region: Carbon intensity in gCO2/kWh (lower = greener region)
            **extra: Optional metadata (e.g. latency_ms, bandwidth_mbps)

        Returns:
            Hash of the new block
        """
        metadata = ClientMetadata(
            client_id=client_id,
            capability=capability,
            carbon_region=carbon_region,
            extra=extra,
        )
        self._index[client_id] = metadata

        prev = self._chain[-1]
        block = Block(
            index=len(self._chain),
            prev_hash=prev.hash(),
            data={
                "action": "register",
                "client_id": client_id,
                "capability": capability,
                "carbon_region": carbon_region,
                "timestamp": metadata.timestamp,
                **extra,
            },
        )
        self._chain.append(block)
        self._reputation[client_id] = self._reputation.get(client_id, 0.0)
        block_hash = block.hash()
        logger.debug(
            f"Registered client {client_id} on registry (capability={capability:.3f}, "
            f"carbon_region={carbon_region:.0f} gCO2/kWh), block hash={block_hash[:16]}..."
        )
        return block_hash

    def get_reputation(self, client_id: int) -> float:
        """Return current reputation/token score for a client (Phase 2)."""
        return self._reputation.get(client_id, 0.0)

    def update_reputation(self, client_id: int, delta: float) -> None:
        """Update reputation by delta (can be negative)."""
        self._reputation[client_id] = self._reputation.get(client_id, 0.0) + delta
        logger.debug(f"Registry: client {client_id} reputation delta={delta:.4f}, now={self._reputation[client_id]:.4f}")

    def update_reputation_from_round(self, round_results: Dict[str, Any]) -> None:
        """
        Update reputation for selected clients based on round outcomes (Phase 2).

        Rewards: green (lower carbon region), quality (lower loss / higher accuracy).
        """
        selected = round_results.get("selected_clients", [])
        if not selected:
            return
        carbon_g = round_results.get("carbon_emission_g") or (round_results.get("carbon_emission", 0) * 1000)
        avg_loss = round_results.get("avg_loss", 0.0)
        test_acc = round_results.get("test_accuracy")
        n = len(selected)
        # Green reward: lower carbon this round -> small positive for all selected
        green_bonus = max(0, 0.1 * (400 - carbon_g) / 400) if carbon_g else 0.05
        # Quality: lower loss -> bonus; higher accuracy -> bonus
        loss_bonus = max(0, 0.15 * (2.0 - avg_loss) / 2.0) if avg_loss is not None else 0.05
        acc_bonus = max(0, 0.1 * (test_acc / 100.0)) if test_acc is not None else 0.05
        delta = (green_bonus + loss_bonus + acc_bonus) / max(1, n)
        for cid in selected:
            self.update_reputation(cid, delta)

    def commit_round(
        self,
        round_id: int,
        selected_clients: List[int],
        carbon_emission_g: float,
        avg_loss: float,
        test_accuracy: Optional[float] = None,
    ) -> str:
        """
        Append a round summary block to the chain for audit (Phase 3).
        """
        prev = self._chain[-1]
        block = Block(
            index=len(self._chain),
            prev_hash=prev.hash(),
            data={
                "action": "round_commit",
                "round_id": round_id,
                "selected_clients": selected_clients,
                "carbon_emission_g": carbon_emission_g,
                "avg_loss": avg_loss,
                "test_accuracy": test_accuracy,
                "timestamp": time.time(),
            },
        )
        self._chain.append(block)
        h = block.hash()
        logger.debug(f"Registry: round {round_id} committed, block hash={h[:16]}...")
        return h

    def get_client(self, client_id: int) -> Optional[ClientMetadata]:
        """Return latest metadata for a client, or None if not registered."""
        return self._index.get(client_id)

    def get_all_registered(self) -> List[ClientMetadata]:
        """Return list of all currently registered clients (latest metadata)."""
        return list(self._index.values())

    def get_registered_ids(self) -> List[int]:
        """Return list of all registered client IDs."""
        return list(self._index.keys())

    def get_chain_length(self) -> int:
        """Return number of blocks (including genesis)."""
        return len(self._chain)

    def get_chain(self) -> List[Block]:
        """Return the full chain for verification/audit (read-only)."""
        return self._chain.copy()

    def verify_chain(self) -> bool:
        """Verify integrity of the chain (prev_hash links)."""
        for i in range(1, len(self._chain)):
            if self._chain[i].prev_hash != self._chain[i - 1].hash():
                return False
        return True
