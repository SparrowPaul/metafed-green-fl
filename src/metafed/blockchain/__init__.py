"""
Blockchain-based Federated Registry for Metaverse FL.

This module provides a blockchain-backed registry for resource discovery
and identity verification in federated Metaverse systems.
"""

from .federated_registry import FederatedRegistry, Block, ClientMetadata

__all__ = ["FederatedRegistry", "Block", "ClientMetadata"]
