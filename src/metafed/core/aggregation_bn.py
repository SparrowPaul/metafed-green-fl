# FILE: src/metafed/core/aggregation_bn.py
# HybridBN buffer aggregation — Chen et al. (2025) ICML
# Replaces MetaFed's simple mean aggregation for BN running stats.
#
# Key difference from MetaFed's current method (server.py lines 231-249):
#   MetaFed:  global_var = mean(local_vars)
#   HybridBN: global_var = mean(local_var + (local_mean - global_mean)^2)
#             i.e. uses the law of total variance to get an unbiased estimate.
#
# Integer buffers (num_batches_tracked) are handled identically to MetaFed:
# first client value is used.

import torch
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def aggregate_buffers_hybridbn(
    client_updates: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate BatchNorm buffers using the HybridBN unbiased estimator.

    For running_mean buffers:
        global_mean = weighted average of local means
        weight_i = num_samples_i / total_samples

    For running_var buffers:
        global_var = weighted average of (local_var + (local_mean - global_mean)^2)
        This is the law of total variance — it corrects for the bias introduced
        when clients have different local data distributions.

    For all other float buffers:
        Simple weighted mean (same as MetaFed).

    For integer buffers (e.g. num_batches_tracked):
        First client value (same as MetaFed).

    Args:
        client_updates: List of dicts, each with keys:
            'buffers': Dict[str, torch.Tensor]  — from client.get_model_buffers()
            'num_samples': int                  — number of training samples
        device: torch.device to move tensors to

    Returns:
        Dict[str, torch.Tensor] of aggregated buffers.
        Returns empty dict if no buffers are present.
    """
    if not client_updates or "buffers" not in client_updates[0]:
        return {}

    buf_names = list(client_updates[0]["buffers"].keys())
    if not buf_names:
        return {}

    # Compute per-client weights by num_samples (same as FedAvg parameter agg)
    total_samples = sum(u["num_samples"] for u in client_updates)
    if total_samples == 0:
        weights = [1.0 / len(client_updates)] * len(client_updates)
    else:
        weights = [u["num_samples"] / total_samples for u in client_updates]

    aggregated: Dict[str, torch.Tensor] = {}

    # Pre-compute global running_mean for all BN layers so we can use it
    # when computing the corrected variance. We identify running_mean buffers
    # by name suffix.
    global_means: Dict[str, torch.Tensor] = {}
    for name in buf_names:
        tensors = [
            u["buffers"][name].to(device)
            for u in client_updates
            if name in u.get("buffers", {})
        ]
        if not tensors:
            continue
        if tensors[0].is_floating_point() and name.endswith("running_mean"):
            # Weighted mean of local means
            stacked = torch.stack(tensors)          # [K, features]
            w = torch.tensor(weights, dtype=stacked.dtype, device=device)
            global_means[name] = (stacked * w.view(-1, 1)).sum(dim=0)

    # Now aggregate all buffers
    for name in buf_names:
        tensors = [
            u["buffers"][name].to(device)
            for u in client_updates
            if name in u.get("buffers", {})
        ]
        if not tensors:
            continue

        t0 = tensors[0]

        if not (t0.is_floating_point() or t0.is_complex()):
            # Integer buffers: use first client's value (same as MetaFed)
            aggregated[name] = t0.clone()
            continue

        # Float buffers
        w = torch.tensor(weights, dtype=t0.dtype, device=device)
        stacked = torch.stack(tensors)              # [K, features]

        if name.endswith("running_mean"):
            # Already computed in global_means pass
            aggregated[name] = global_means[name]

        elif name.endswith("running_var"):
            # HybridBN variance correction:
            # global_var = E[local_var] + E[(local_mean - global_mean)^2]
            #            = sum_i w_i * local_var_i
            #            + sum_i w_i * (local_mean_i - global_mean)^2
            mean_name = name.replace("running_var", "running_mean")
            if mean_name in global_means:
                global_mean = global_means[mean_name]       # [features]
                local_means_tensors = [
                    u["buffers"][mean_name].to(device)
                    for u in client_updates
                    if mean_name in u.get("buffers", {})
                ]
                if local_means_tensors and len(local_means_tensors) == len(tensors):
                    local_means_stacked = torch.stack(local_means_tensors)  # [K, features]
                    # E[local_var]: weighted mean of local variances
                    term1 = (stacked * w.view(-1, 1)).sum(dim=0)
                    # E[(local_mean - global_mean)^2]: weighted mean of squared deviations
                    deviations = local_means_stacked - global_mean.unsqueeze(0)  # [K, features]
                    term2 = (deviations.pow(2) * w.view(-1, 1)).sum(dim=0)
                    aggregated[name] = term1 + term2
                    logger.debug(
                        f"HybridBN variance correction applied to {name}: "
                        f"term1_mean={term1.mean().item():.4f}, "
                        f"term2_mean={term2.mean().item():.4f}"
                    )
                else:
                    # Fallback: simple weighted mean (can't apply correction)
                    aggregated[name] = (stacked * w.view(-1, 1)).sum(dim=0)
            else:
                # No corresponding running_mean found: simple weighted mean
                aggregated[name] = (stacked * w.view(-1, 1)).sum(dim=0)

        else:
            # Other float buffers: simple weighted mean
            aggregated[name] = (stacked * w.view(-1, 1)).sum(dim=0)

    logger.debug(
        f"HybridBN aggregated {len(aggregated)} buffers from "
        f"{len(client_updates)} clients"
    )
    return aggregated


def aggregate_buffers_metafed(
    client_updates: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    MetaFed's original BN buffer aggregation (simple mean for float,
    first-client for integer). Kept here for easy A/B switching in server.py.

    This is identical to the inline code in server.py lines 231-249.
    """
    if not client_updates or "buffers" not in client_updates[0]:
        return {}

    buf_names = list(client_updates[0]["buffers"].keys())
    aggregated: Dict[str, torch.Tensor] = {}

    for name in buf_names:
        tensors = [
            u["buffers"][name].to(device)
            for u in client_updates
            if name in u.get("buffers", {})
        ]
        if not tensors:
            continue
        t0 = tensors[0]
        if t0.is_floating_point() or t0.is_complex():
            aggregated[name] = torch.stack(tensors).mean(dim=0)
        else:
            aggregated[name] = t0.clone()

    return aggregated