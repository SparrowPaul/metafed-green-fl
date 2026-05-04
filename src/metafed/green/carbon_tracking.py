"""
Carbon Tracking Module.

This module provides functionality for tracking carbon emissions
and carbon intensity for sustainable federated learning.
Supports time-of-day model I(t) from the MetaFed paper for green scheduling.
"""

import math
import time
import logging
from typing import Optional, Dict, Any
import random

logger = logging.getLogger(__name__)


class CarbonTracker:
    """
    Carbon emission tracker for federated learning operations.
    
    Tracks carbon emissions during training rounds and provides
    carbon intensity information for green scheduling. Optional
    time-of-day model: I(t) = I_base + A*sin(2*pi*t/T + phi) + noise (paper Eq. 8).
    """
    
    def __init__(
        self,
        region: str = "US",
        mock_data: bool = True,
        use_time_of_day: bool = True,
        i_base_g: float = 150.0,
        amplitude_g: float = 70.0,
        period_hours: float = 24.0,
    ):
        """
        Initialize carbon tracker.
        
        Args:
            region: Geographic region for carbon intensity data
            mock_data: Whether to use mock data (for development/testing)
            use_time_of_day: If True, use sinusoidal I(t) model (paper); else use region mock
            i_base_g: Base carbon intensity in gCO2/kWh (paper: 150)
            amplitude_g: Amplitude A in gCO2/kWh (paper: 70)
            period_hours: Period T in hours (paper: 24)
        """
        self.region = region
        self.mock_data = mock_data
        self.use_time_of_day = use_time_of_day
        self.i_base_g = i_base_g
        self.amplitude_g = amplitude_g
        self.period_hours = period_hours
        self._start_wall_time = time.time()
        self.tracking_start_time = None
        self.current_session_emission = 0.0
        self.total_emission = 0.0
        
        # Mock carbon intensity data (kg CO2/kWh) for non-time-of-day mode
        self.mock_intensities = {
            'US': {'avg': 0.4, 'range': (0.2, 0.8)},
            'EU': {'avg': 0.3, 'range': (0.1, 0.6)},
            'ASIA': {'avg': 0.5, 'range': (0.3, 0.9)}
        }
        
        logger.info(
            f"Initialized carbon tracker for region {region}, "
            f"use_time_of_day={use_time_of_day}, I_base={i_base_g}g, A={amplitude_g}g, T={period_hours}h"
        )
    
    def get_current_intensity(self) -> float:
        """
        Get current carbon intensity.
        
        Returns:
            Carbon intensity in kg CO2/kWh (for backward compatibility)
        """
        return self.get_current_intensity_g_per_kwh() / 1000.0
    
    def get_current_intensity_g_per_kwh(self) -> float:
        """
        Get current carbon intensity in gCO2/kWh (paper units).
        
        Uses time-of-day model I(t) = I_base + A*sin(2*pi*t/T + phi) + noise
        when use_time_of_day is True.
        
        Returns:
            Carbon intensity in gCO2/kWh
        """
        if self.mock_data and self.use_time_of_day:
            # Paper Eq. (8): I(t) = I_base + A*sin(2*pi*t/T + phi) + epsilon(t)
            t_hours = (time.time() - self._start_wall_time) / 3600.0
            phi = 0.0  # phase
            sinusoidal = self.i_base_g + self.amplitude_g * math.sin(
                2 * math.pi * t_hours / self.period_hours + phi
            )
            noise = random.gauss(0, 10.0)  # small noise
            intensity_g = max(20.0, min(350.0, sinusoidal + noise))
            return intensity_g
        if self.mock_data:
            # Legacy: region-based mock (convert to g)
            base = self.mock_intensities.get(self.region, {'avg': 0.4})['avg']
            rng = self.mock_intensities.get(self.region, {'range': (0.2, 0.8)})['range']
            variation = random.uniform(-0.1, 0.1)
            intensity_kg = max(rng[0], min(rng[1], base + variation))
            return intensity_kg * 1000.0  # kg -> g
        # Real API path
        return self._fetch_real_carbon_intensity_g() or 150.0
    
    def _fetch_real_carbon_intensity_g(self) -> Optional[float]:
        """Fetch real carbon intensity in gCO2/kWh (placeholder)."""
        logger.warning("Real carbon intensity API not implemented, using mock")
        return None
    
    def _fetch_real_carbon_intensity(self) -> float:
        """
        Fetch real carbon intensity data from external API.
        
        This is a placeholder for real API integration.
        
        Returns:
            Current carbon intensity in kg CO2/kWh
        """
        g = self._fetch_real_carbon_intensity_g()
        return (g or 150.0) / 1000.0
    
    def start_tracking(self) -> None:
        """Start tracking carbon emissions for the current session."""
        self.tracking_start_time = time.time()
        self.current_session_emission = 0.0
        logger.debug("Started carbon emission tracking")
    
    def stop_tracking(self) -> float:
        """
        Stop tracking and calculate emissions for the current session.
        
        Returns:
            Carbon emissions in kg CO2 for the session
        """
        if self.tracking_start_time is None:
            logger.warning("Carbon tracking was not started")
            return 0.0
        
        # Calculate energy consumption and emissions
        duration_hours = (time.time() - self.tracking_start_time) / 3600
        
        # Estimate power consumption (this is a simplified model)
        # In practice, this would use more sophisticated power monitoring
        estimated_power_kw = self._estimate_power_consumption()
        energy_consumption_kwh = estimated_power_kw * duration_hours
        
        # Calculate carbon emissions
        carbon_intensity = self.get_current_intensity()
        session_emission = energy_consumption_kwh * carbon_intensity
        
        self.current_session_emission = session_emission
        self.total_emission += session_emission
        
        logger.info(f"Session carbon emission: {session_emission:.6f} kg CO2 "
                   f"(duration: {duration_hours:.4f}h, intensity: {carbon_intensity:.3f} kg CO2/kWh)")
        
        self.tracking_start_time = None
        return session_emission
    
    def _estimate_power_consumption(self) -> float:
        """
        Estimate power consumption during the tracking session.
        
        This is a simplified model. In practice, you would use:
        - Hardware-specific power monitoring
        - GPU utilization metrics
        - CPU utilization metrics
        - System-level power monitoring tools
        
        Returns:
            Estimated power consumption in kW
        """
        # Mock power consumption based on typical ML workload
        # Assumes: CPU + GPU training
        base_power = 0.1  # Base system power (kW)
        compute_power = 0.3  # Additional power for ML computation (kW)
        
        # Add some randomness to simulate varying workloads
        variation = random.uniform(0.8, 1.2)
        total_power = (base_power + compute_power) * variation
        
        return total_power
    
    def get_total_emissions(self) -> float:
        """
        Get total carbon emissions tracked so far.
        
        Returns:
            Total carbon emissions in kg CO2
        """
        return self.total_emission
    
    def get_carbon_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive carbon tracking metrics.
        
        Returns:
            Dictionary with carbon metrics
        """
        return {
            'total_emission_kg_co2': self.total_emission,
            'current_session_emission_kg_co2': self.current_session_emission,
            'current_carbon_intensity_kg_per_kwh': self.get_current_intensity(),
            'region': self.region
        }
    
    def is_green_time(self, threshold: float = 0.3) -> bool:
        """
        Check if current time is considered "green" for scheduling.
        
        Args:
            threshold: Carbon intensity threshold (kg CO2/kWh)
        
        Returns:
            True if current carbon intensity is below threshold
        """
        current_intensity = self.get_current_intensity()
        is_green = current_intensity < threshold
        
        logger.debug(f"Carbon intensity: {current_intensity:.3f}, threshold: {threshold}, "
                    f"is_green: {is_green}")
        
        return is_green
    
    def get_green_score(self) -> float:
        """
        Get a green score (0-1) based on current carbon intensity.
        
        Lower carbon intensity results in higher green score.
        
        Returns:
            Green score between 0 and 1
        """
        current_intensity = self.get_current_intensity()
        max_intensity = self.mock_intensities.get(self.region, {'range': (0.2, 0.8)})['range'][1]
        
        # Normalize to 0-1 scale (higher score = greener)
        green_score = max(0.0, 1.0 - (current_intensity / max_intensity))
        
        return green_score
    
    def reset(self) -> None:
        """Reset all tracking metrics."""
        self.tracking_start_time = None
        self.current_session_emission = 0.0
        self.total_emission = 0.0
        logger.info("Carbon tracker reset")