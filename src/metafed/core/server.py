"""
Federated Learning Server Implementation.

This module contains the central server that coordinates federated learning
across multiple clients, including model aggregation and orchestration.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import copy
import logging
import time
from ..orchestration.base import BaseOrchestrator
from ..green.carbon_tracking import CarbonTracker
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..blockchain.federated_registry import FederatedRegistry

logger = logging.getLogger(__name__)


class FederatedServer:
    """
    Central federated learning server.
    
    Coordinates federated learning rounds, manages client selection,
    and performs model aggregation.
    """
    
    def __init__(
        self,
        model_template: nn.Module,
        orchestrator: BaseOrchestrator,
        num_rounds: int = 100,
        clients_per_round: int = 10,
        device: str = "cpu",
        carbon_aware: bool = False,
        privacy_budget: Optional[float] = None,
        registry: Optional["FederatedRegistry"] = None,
    ):
        """
        Initialize federated server.

        Args:
            model_template: Global model template
            orchestrator: Client orchestration strategy
            num_rounds: Number of federated learning rounds
            clients_per_round: Number of clients to select per round
            device: Device for computation
            carbon_aware: Enable carbon-aware scheduling
            privacy_budget: Differential privacy budget
            registry: Optional blockchain registry for reputation + round commits (Phase 2/3)
        """
        self.global_model = copy.deepcopy(model_template)
        self.global_model.to(device)
        self.orchestrator = orchestrator
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.device = device
        self.carbon_aware = carbon_aware
        self.privacy_budget = privacy_budget
        self.registry = registry
        self.bn_aggregation = 'metafed'  # default; override externally for HybridBN

        # Initialize carbon tracker if needed
        self.carbon_tracker = CarbonTracker() if carbon_aware else None
        
        # Training history (carbon_emissions in kg; carbon_emissions_g in g/round for paper reporting)
        self.training_history = {
            'rounds': [],
            'accuracies': [],
            'losses': [],
            'carbon_emissions': [],
            'carbon_emissions_g': [],
            'selected_clients': [],
            'training_times': []
        }
        
        # Privacy accounting
        self.privacy_spent = 0.0
        
        logger.info(f"Initialized federated server with {num_rounds} rounds, "
                   f"{clients_per_round} clients per round")
    
    def get_global_params(self) -> Dict[str, torch.Tensor]:
        """
        Get global model parameters.
        
        Returns:
            Dictionary of global model parameters
        """
        return {name: param.clone() for name, param in self.global_model.named_parameters()}
    
    def update_global_model(
        self,
        aggregated_params: Dict[str, torch.Tensor],
        aggregated_buffers: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """
        Update global model with aggregated parameters and optional buffers.
        Buffers (e.g. BatchNorm running_mean, running_var) must be aggregated
        and set so evaluation reflects the trained model; otherwise test
        accuracy stays at random.
        
        Args:
            aggregated_params: Aggregated model parameters
            aggregated_buffers: Optional aggregated buffers (e.g. BN stats)
        """
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_params:
                    src = aggregated_params[name]
                    if src.device != param.device:
                        src = src.to(param.device)
                    param.copy_(src)
            if aggregated_buffers:
                for name, buf in self.global_model.named_buffers():
                    if name in aggregated_buffers:
                        src = aggregated_buffers[name]
                        if src.device != buf.device:
                            src = src.to(buf.device)
                        buf.copy_(src)
                logger.info("Global model updated with aggregated parameters and BatchNorm buffers")
    
    def select_clients(self, available_clients: List[Any], round_num: int) -> List[Any]:
        """
        Select clients for current round.
        
        Args:
            available_clients: List of available clients
            round_num: Current round number
            
        Returns:
            List of selected clients
        """
        # Get carbon intensity if carbon-aware
        carbon_intensity = None
        if self.carbon_aware and self.carbon_tracker:
            carbon_intensity = self.carbon_tracker.get_current_intensity()
        
        # Use orchestrator to select clients
        selected_clients = self.orchestrator.select_clients(
            available_clients=available_clients,
            num_select=min(self.clients_per_round, len(available_clients)),
            round_num=round_num,
            carbon_intensity=carbon_intensity
        )
        
        logger.info(f"Round {round_num}: Selected {len(selected_clients)} clients")
        return selected_clients
    
    def evaluate_model(self, test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Evaluate global model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (accuracy, loss)
        """
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                loss = criterion(outputs, target)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        
        return accuracy, avg_loss
    
    def train_round(
        self,
        selected_clients: List[Any],
        aggregator: Any,
        round_num: int
    ) -> Dict[str, Any]:
        """
        Execute one round of federated training.
        
        Args:
            selected_clients: Clients participating in this round
            aggregator: Model aggregation strategy
            round_num: Current round number
            
        Returns:
            Dictionary with round results
        """
        round_start_time = time.time()
        
        # Track carbon emissions for this round
        if self.carbon_tracker:
            self.carbon_tracker.start_tracking()
        
        # Send global model to selected clients
        global_params = self.get_global_params()
        client_updates = []
        
        # Collect updates from clients (params and buffers for models with BatchNorm)
        for client in selected_clients:
            # Send global model to client
            client.update_model(global_params)
            
            # Get client update (params, num_samples, loss) and buffers
            if hasattr(client, 'train'):
                update_result = client.train()
                client_updates.append({
                    'client_id': client.id,
                    'params': update_result[0],
                    'num_samples': update_result[1] if len(update_result) > 1 else 1,
                    'loss': update_result[2] if len(update_result) > 2 else 0.0
                })
                if hasattr(client, 'get_model_buffers'):
                    client_updates[-1]['buffers'] = client.get_model_buffers()
        
        # Aggregate client updates (parameters)
        aggregated_params = aggregator.aggregate(client_updates)
        
        # Aggregate buffers (BatchNorm running_mean, running_var, etc.)
        # bn_aggregation controls the method:
        #   'metafed'  — original simple mean (default, always safe)
        #   'hybridbn' — Chen et al. (2025) unbiased variance estimator
        aggregated_buffers = None
        if client_updates and 'buffers' in client_updates[0]:
            from ..core.aggregation_bn import aggregate_buffers_hybridbn, aggregate_buffers_metafed
            bn_method = getattr(self, 'bn_aggregation', 'metafed')
            if bn_method == 'hybridbn':
                aggregated_buffers = aggregate_buffers_hybridbn(client_updates, self.device)
            else:
                aggregated_buffers = aggregate_buffers_metafed(client_updates, self.device)
            if aggregated_buffers:
                logger.debug(
                    f"Aggregated {len(aggregated_buffers)} buffers from "
                    f"{len(client_updates)} clients using bn_aggregation='{bn_method}'"
                )
        
        # Apply differential privacy if enabled (params only; buffers left as-is)
        if self.privacy_budget and self.privacy_budget > self.privacy_spent:
            aggregated_params = self._apply_differential_privacy(aggregated_params)
        
        # Update global model with aggregated params and buffers
        self.update_global_model(aggregated_params, aggregated_buffers)
        
        # Calculate round metrics
        round_time = time.time() - round_start_time
        carbon_emission = 0.0
        
        if self.carbon_tracker:
            carbon_emission = self.carbon_tracker.stop_tracking()
        
        # Average loss across clients
        avg_loss = sum(update['loss'] for update in client_updates) / len(client_updates)

        # Compute per-client parameter norm proxies for OSMD orchestrator.
        # ||w_after - w_before||_F is a proxy for the gradient signal magnitude.
        # The global params before training are in global_params (set at line 210).
        client_param_norms: Dict[int, float] = {}
        if hasattr(self.orchestrator, 'update_history'):
            for update in client_updates:
                cid = update['client_id']
                after_params = update['params']
                norm_sq = 0.0
                for pname, p_after in after_params.items():
                    if pname in global_params:
                        diff = p_after.to(self.device) - global_params[pname].to(self.device)
                        norm_sq += diff.norm().item() ** 2
                client_param_norms[cid] = norm_sq ** 0.5

        carbon_emission_g = carbon_emission * 1000.0  # g CO2 for paper-style reporting
        round_results = {
            'round': round_num,
            'selected_clients': [client.id for client in selected_clients],
            'avg_loss': avg_loss,
            'training_time': round_time,
            'carbon_emission': carbon_emission,
            'carbon_emission_g': carbon_emission_g,
            'num_participants': len(selected_clients),
            'client_param_norms': client_param_norms,
        }
        
        logger.info(f"Round {round_num} completed in {round_time:.2f}s, "
                   f"loss: {avg_loss:.4f}, carbon: {carbon_emission:.4f}kg CO2 ({carbon_emission_g:.2f}g/round)")
        
        return round_results
    
    def _apply_differential_privacy(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply differential privacy to aggregated parameters.
        
        Args:
            params: Aggregated parameters
            
        Returns:
            Privacy-preserving parameters
        """
        # Simple Gaussian mechanism for differential privacy
        epsilon = 0.1  # Privacy parameter
        sensitivity = 1.0  # L2 sensitivity
        noise_scale = sensitivity / epsilon
        
        private_params = {}
        for name, param in params.items():
            noise = torch.normal(0, noise_scale, size=param.shape, device=param.device)
            private_params[name] = param + noise
        
        # Update privacy spent
        self.privacy_spent += epsilon
        
        logger.info(f"Applied differential privacy, epsilon spent: {self.privacy_spent:.3f}")
        
        return private_params
    
    def run_federated_learning(
        self,
        clients: List[Any],
        aggregator: Any,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        eval_frequency: int = 5
    ) -> Dict[str, Any]:
        """
        Run complete federated learning process.
        
        Args:
            clients: List of all available clients
            aggregator: Model aggregation strategy
            test_loader: Test data for evaluation
            eval_frequency: Frequency of model evaluation
            
        Returns:
            Complete training results
        """
        logger.info(f"Starting federated learning with {len(clients)} clients")
        
        for round_num in range(1, self.num_rounds + 1):
            # Select clients for this round
            selected_clients = self.select_clients(clients, round_num)
            
            # Execute training round
            round_results = self.train_round(selected_clients, aggregator, round_num)
            
            # Evaluate model periodically
            if test_loader and (round_num % eval_frequency == 0 or round_num == self.num_rounds):
                accuracy, loss = self.evaluate_model(test_loader)
                round_results['test_accuracy'] = accuracy
                round_results['test_loss'] = loss
                
                logger.info(f"Round {round_num} evaluation - Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
            
            # Store results
            self.training_history['rounds'].append(round_num)
            self.training_history['losses'].append(round_results['avg_loss'])
            self.training_history['carbon_emissions'].append(round_results['carbon_emission'])
            self.training_history['carbon_emissions_g'].append(round_results.get('carbon_emission_g', round_results['carbon_emission'] * 1000.0))
            self.training_history['selected_clients'].append(round_results['selected_clients'])
            self.training_history['training_times'].append(round_results['training_time'])
            
            if 'test_accuracy' in round_results:
                self.training_history['accuracies'].append(round_results['test_accuracy'])

            # Phase 2/3: Update registry reputation and commit round on-chain
            if self.registry is not None:
                self.registry.update_reputation_from_round(round_results)
                self.registry.commit_round(
                    round_id=round_num,
                    selected_clients=round_results['selected_clients'],
                    carbon_emission_g=round_results.get('carbon_emission_g', round_results['carbon_emission'] * 1000.0),
                    avg_loss=round_results['avg_loss'],
                    test_accuracy=round_results.get('test_accuracy'),
                )
            
            # Update orchestrator with round results
            if hasattr(self.orchestrator, 'update_history'):
                self.orchestrator.update_history(round_results)
        
        total_carbon_kg = sum(self.training_history['carbon_emissions'])
        final_results = {
            'training_history': self.training_history,
            'final_model_state': self.global_model.state_dict(),
            'total_carbon_emission': total_carbon_kg,
            'total_carbon_emission_g': total_carbon_kg * 1000.0,
            'total_training_time': sum(self.training_history['training_times']),
            'privacy_spent': self.privacy_spent
        }
        
        if test_loader:
            final_accuracy, final_loss = self.evaluate_model(test_loader)
            final_results['final_accuracy'] = final_accuracy
            final_results['final_loss'] = final_loss
        
        logger.info("Federated learning completed!")
        logger.info(f"Total carbon emission: {final_results['total_carbon_emission']:.4f}kg CO2 ({final_results.get('total_carbon_emission_g', 0):.2f}g cumulative)")
        logger.info(f"Total training time: {final_results['total_training_time']:.2f}s")
        
        if 'final_accuracy' in final_results:
            logger.info(f"Final accuracy: {final_results['final_accuracy']:.2f}%")
        
        return final_results