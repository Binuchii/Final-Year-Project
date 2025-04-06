import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import math
import logging
import pandas as pd
from dataclasses import dataclass
from collections import OrderedDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCTSConfig:
    """Configuration parameters for Monte Carlo Tree Search"""
    num_simulations: int = 50
    c_puct: float = 1.0
    max_cache_size: int = 10000

class MCTSNode:
    """Represents a node in the MCTS search tree"""
    def __init__(self, prior: float, state: Optional[np.ndarray] = None):
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.children: OrderedDict[int, 'MCTSNode'] = OrderedDict()
        self.state = state
    
    def expanded(self) -> bool:
        """Check if node has been expanded"""
        return len(self.children) > 0
    
    def value(self) -> float:
        """Calculate node value (average of all visits)"""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

class MCTS:
    """Monte Carlo Tree Search algorithm implementation"""
    def __init__(self, model: 'SimplifiedF1Net', config: MCTSConfig):
        self.model = model
        self.config = config
        self.root: Optional[MCTSNode] = None
        self.node_count = 0
    
    def search(self, state: np.ndarray) -> np.ndarray:
        """Conduct MCTS search and return action probabilities"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        self.root = MCTSNode(prior=1.0, state=state)
        self.node_count = 1
        
        self.model.eval()
        
        for _ in range(self.config.num_simulations):
            node = self.root
            search_path = [node]
            
            # Select leaf node
            while node.expanded():
                action, node = self._select_child(node)
                search_path.append(node)
            
            # Evaluate position with neural network
            with torch.no_grad():
                policy, value = self.model(state_tensor)
            
            policy = policy.detach().numpy()[0]
            value = value.item()
            
            # Update node values
            self._backpropagate(search_path, value, policy)
        
        self.model.train()
        
        return self._get_action_probs(self.root)
    
    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Select child with highest UCB score"""
        return max(
            node.children.items(), 
            key=lambda x: self._ucb_score(node, x[1], self.config.c_puct)
        )
    
    def _ucb_score(self, parent: MCTSNode, child: MCTSNode, c_puct: float) -> float:
        """Calculate UCB score with exploration bonus"""
        prior_score = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        value_score = -child.value()  # Negative for minimizing position
        exploration_bonus = 0.5 * math.sqrt(2 * math.log(parent.visit_count + 1) / (child.visit_count + 1))
        
        return value_score + prior_score + exploration_bonus
    
    def _backpropagate(self, search_path: List[MCTSNode], value: float, policy: np.ndarray):
        """Update values in all nodes of the search path"""
        for node in search_path:
            node.value_sum += value
            node.visit_count += 1
            
            if not node.expanded():
                for i in range(20):  # 20 possible positions
                    new_node = MCTSNode(prior=policy[i])
                    node.children[i] = new_node
                    self.node_count += 1
    
    def _get_action_probs(self, root: MCTSNode) -> np.ndarray:
        """Convert visit counts to action probability distribution"""
        visits = np.array([child.visit_count for child in root.children.values()])
        
        return visits / visits.sum() if visits.sum() > 0 else np.ones(20) / 20

class ModelConfig:
    """Neural network architecture configuration"""
    def __init__(self):
        # Input feature dimensions
        self.driver_features = 20
        self.constructor_features = 30
        self.qualifying_features = 60
        self.weather_features = 1
        self.circuit_features = 3
        
        # Combined state dimension
        self.state_size = (
            self.driver_features + 
            self.constructor_features + 
            self.qualifying_features + 
            self.weather_features + 
            self.circuit_features
        )  # Total: 114
        
        # Network parameters
        self.hidden_size = 64
        self.dropout_rate = 0.2
        self.attention_heads = 4

class SimplifiedF1Net(nn.Module):
    def __init__(self, config: ModelConfig):
        super(SimplifiedF1Net, self).__init__()
        self.config = config
        
        # Feature encoding layers
        self.driver_encoder = self._create_encoder(
            config.driver_features, 
            config.hidden_size
        )
        self.constructor_encoder = self._create_encoder(
            config.constructor_features, 
            config.hidden_size
        )
        self.qualifying_encoder = self._create_encoder(
            config.qualifying_features, 
            config.hidden_size
        )
        
        # Circuit feature encoder
        self.circuit_encoder = nn.Sequential(
            nn.Linear(config.circuit_features, config.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size)
        )
        
        # Attention mechanism for circuit-specific adaptations
        self.circuit_attention = nn.Sequential(
            nn.Linear(config.hidden_size, 3),
            nn.Softmax(dim=-1)
        )
        
        # Main processing network
        combined_size = config.hidden_size * 4 + config.weather_features
        
        self.main_network = nn.Sequential(
            nn.Linear(combined_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size * 2),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size)
        )
        
        # Output heads
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size // 2),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, 20),
            nn.Softmax(dim=-1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size // 2),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Tanh()
        )
    
    def _create_encoder(self, input_size: int, output_size: int) -> nn.Sequential:
        """Create a standardized feature encoder block"""
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(self.config.dropout_rate)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Split input features
        driver_feats = state[:, :self.config.driver_features]
        constructor_feats = state[:, self.config.driver_features:self.config.driver_features + self.config.constructor_features]
        qualifying_feats = state[:, self.config.driver_features + self.config.constructor_features:self.config.driver_features + self.config.constructor_features + self.config.qualifying_features]
        weather_feats = state[:, -self.config.circuit_features-1:-self.config.circuit_features]
        circuit_feats = state[:, -self.config.circuit_features:]
        
        # Encode individual feature sets
        encoded_driver = self.driver_encoder(driver_feats)
        encoded_constructor = self.constructor_encoder(constructor_feats)
        encoded_qualifying = self.qualifying_encoder(qualifying_feats)
        encoded_circuit = self.circuit_encoder(circuit_feats)
        
        # Apply circuit-specific attention weights
        attention_weights = self.circuit_attention(encoded_circuit)
        
        # Weight features by circuit characteristics
        attended_driver = encoded_driver * attention_weights[:, 0:1]
        attended_constructor = encoded_constructor * attention_weights[:, 1:2]
        attended_qualifying = encoded_qualifying * attention_weights[:, 2:3]
        
        # Combine all features
        combined = torch.cat([
            attended_driver,
            attended_constructor,
            attended_qualifying,
            encoded_circuit,
            weather_feats
        ], dim=1)
        
        # Process through main network
        features = self.main_network(combined)
        
        # Generate policy and value outputs
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value

def convert_time_to_seconds(time_str: str) -> Optional[float]:
    """Convert qualifying time string to seconds with error handling"""
    if not isinstance(time_str, (str, int, float)) or pd.isna(time_str):
        return None
        
    try:
        # Clean input
        if isinstance(time_str, str):
            time_str = time_str.strip()
        
        # Parse time format
        if isinstance(time_str, str) and ':' in time_str:
            # Format: "1:23.456"
            minutes, seconds = time_str.split(':')
            total_seconds = float(minutes) * 60 + float(seconds)
        else:
            # Direct numeric input
            total_seconds = float(time_str)
        
        # Validate result is within reasonable range
        return total_seconds if 30 <= total_seconds <= 180 else None
            
    except (ValueError, TypeError):
        return None
