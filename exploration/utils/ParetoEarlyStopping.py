import numpy as np
from pytorch_lightning.callbacks import Callback
from typing import List, Dict

class ParetoEarlyStopping(Callback):
    """
    Early stopping callback that monitors multiple metrics for Pareto improvements.
    Stops when no Pareto improvements are made for patience epochs.
    
    Args:
        monitor_metrics (List[str]): List of metric names to monitor
        mode (str): Whether metrics should be minimized ('min') or maximized ('max')
        patience (int): Number of epochs to wait for Pareto improvement
        min_delta (float): Minimum change to qualify as an improvement
        verbose (bool): If True, prints messages
    """
    
    def __init__(
        self,
        monitor: List[str],
        mode: str = 'min',
        patience: int = 3,
        min_delta: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__()
        self.monitor_metrics = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        # Initialize state
        self.wait_count = 0
        self.stopped_epoch = 0
        self.pareto_front = []
        
    def _get_metric_values(self, trainer) -> List[float]:
        """Get current values of monitored metrics."""
        return [trainer.callback_metrics[metric].item() for metric in self.monitor_metrics]
    
    def _dominates(self, a: List[float], b: List[float]) -> bool:
        """Check if point a dominates point b in the Pareto sense."""
        if self.mode == 'min':
            return (all(a_i <= b_i + self.min_delta for a_i, b_i in zip(a, b)) and 
                   any(a_i < b_i - self.min_delta for a_i, b_i in zip(a, b)))
        else:
            return (all(a_i >= b_i - self.min_delta for a_i, b_i in zip(a, b)) and 
                   any(a_i > b_i + self.min_delta for a_i, b_i in zip(a, b)))
    
    def _is_pareto_improvement(self, point: List[float]) -> bool:
        """Check if a point is a Pareto improvement over the current front."""
        if not self.pareto_front:
            return True
            
        # Check if the new point dominates any point in the current front
        dominates_existing = any(self._dominates(point, p) for p in self.pareto_front)
        # Check if the new point is not dominated by any point in the current front
        not_dominated = not any(self._dominates(p, point) for p in self.pareto_front)
        
        return dominates_existing or not_dominated
    
    def _update_pareto_front(self, point: List[float]):
        """Update the Pareto front with a new point."""
        # Remove points that are dominated by the new point
        self.pareto_front = [p for p in self.pareto_front if not self._dominates(point, p)]
        # Add the new point if it's not dominated
        if not any(self._dominates(p, point) for p in self.pareto_front):
            self.pareto_front.append(point)
    
    def on_validation_end(self, trainer, pl_module):
        """Check for Pareto improvements after validation."""
        current_metrics = self._get_metric_values(trainer)
        
        if self._is_pareto_improvement(current_metrics):
            self.wait_count = 0
            self._update_pareto_front(current_metrics)
            if self.verbose:
                print(f'Epoch {trainer.current_epoch}: Pareto improvement found')
        else:
            self.wait_count += 1
            if self.verbose:
                print(f'Epoch {trainer.current_epoch}: No Pareto improvement. '
                      f'Waiting {self.wait_count}/{self.patience}')
        
        # Stop training if no improvements for patience epochs
        if self.wait_count >= self.patience:
            self.stopped_epoch = trainer.current_epoch
            trainer.should_stop = True