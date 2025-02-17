import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import math
import logging
from dataclasses import dataclass
from collections import OrderedDict
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCTSConfig:
    """Configuration for MCTS parameters"""
    num_simulations: int = 50
    c_puct: float = 1.0
    max_cache_size: int = 10000
    cache_cleanup_factor: float = 0.5

class MCTSNode:
    """Enhanced Node in the MCTS tree with memory management"""
    def __init__(self, prior: float, state: Optional[np.ndarray] = None):
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.children: OrderedDict[int, 'MCTSNode'] = OrderedDict()
        self.state = state
        self.last_access_time = time.time()
        self._cached_ucb_scores: Dict[int, float] = {}
    
    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def update_access_time(self):
        """Update the last access time of the node"""
        self.last_access_time = time.time()
        self._cached_ucb_scores.clear()  # Clear cached scores on access

class MCTS:
    """Improved Monte Carlo Tree Search implementation with memory management"""
    def __init__(self, model: 'SimplifiedF1Net', config: MCTSConfig):
        self.model = model
        self.config = config
        self.root: Optional[MCTSNode] = None
        self.node_count = 0
    
    def cleanup_cache(self):
        """Clean up old nodes to manage memory"""
        if self.root is None or self.node_count < self.config.max_cache_size:
            return
            
        cleanup_size = int(self.config.max_cache_size * self.config.cache_cleanup_factor)
        current_time = time.time()
        
        def cleanup_recursive(node: MCTSNode) -> int:
            cleaned = 0
            children_to_remove = []
            
            for action, child in node.children.items():
                if (current_time - child.last_access_time > 3600 and  # 1 hour old
                    child.visit_count < node.visit_count * 0.1):  # Rarely visited
                    children_to_remove.append(action)
                    cleaned += 1
                else:
                    cleaned += cleanup_recursive(child)
                    
            for action in children_to_remove:
                del node.children[action]
                
            return cleaned
            
        nodes_cleaned = cleanup_recursive(self.root)
        self.node_count -= nodes_cleaned
        logger.info(f"Cleaned up {nodes_cleaned} nodes from MCTS tree")
    
    def search(self, state: np.ndarray) -> np.ndarray:
        """Perform MCTS search with proper batch handling"""
        try:
            if not isinstance(state, np.ndarray):
                raise ValueError("State must be a numpy array")
            
            # Convert to tensor and add batch dimension if needed
            state_tensor = torch.FloatTensor(state)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            # Create root node
            self.root = MCTSNode(prior=1.0, state=state)
            self.node_count = 1
            
            # Temporarily set model to eval mode
            self.model.eval()
            
            for _ in range(self.config.num_simulations):
                node = self.root
                search_path = [node]
                
                # Selection
                while node.expanded():
                    action, node = self._select_child(node)
                    search_path.append(node)
                
                # Expansion and Evaluation
                with torch.no_grad():
                    # Ensure we have a batch dimension but don't use training mode
                    policy, value = self.model(state_tensor)
                
                policy = policy.detach().numpy()[0]
                value = value.item()
                
                # Backpropagate
                self._backpropagate(search_path, value, policy)
                
                # Check memory usage
                if self.node_count >= self.config.max_cache_size:
                    self.cleanup_cache()
            
            # Restore model mode
            self.model.train()
            
            return self._get_action_probs(self.root)
            
        except Exception as e:
            logger.error(f"Error in MCTS search: {str(e)}")
            raise
    
    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Select child node with highest UCB score using caching"""
        node.update_access_time()
        
        # Calculate UCB scores for all children if cache is empty
        if not node._cached_ucb_scores:
            node._cached_ucb_scores = {
                action: self._ucb_score(node, child, self.config.c_puct)
                for action, child in node.children.items()
            }
        
        best_action = max(node._cached_ucb_scores.items(), key=lambda x: x[1])[0]
        return best_action, node.children[best_action]
    
    def _ucb_score(self, parent: MCTSNode, child: MCTSNode, c_puct: float) -> float:
        """Calculate UCB score with validation"""
        if parent.visit_count == 0:
            raise ValueError("Parent node must have at least one visit")
            
        prior_score = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        value_score = -child.value()  # Negative because we want to minimize position
        return value_score + prior_score
    
    def _backpropagate(self, search_path: List[MCTSNode], value: float, policy: np.ndarray):
        """Backpropagate value through search path with validation"""
        if not search_path:
            raise ValueError("Search path cannot be empty")
            
        if not isinstance(policy, np.ndarray) or policy.shape[0] != 20:
            raise ValueError("Invalid policy shape")
            
        for node in search_path:
            node.value_sum += value
            node.visit_count += 1
            
            if not node.expanded():
                for i in range(20):  # 20 possible positions
                    new_node = MCTSNode(prior=policy[i])
                    node.children[i] = new_node
                    self.node_count += 1
    
    def _get_action_probs(self, root: MCTSNode) -> np.ndarray:
        """Get action probabilities based on visit counts"""
        visits = np.zeros(20)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        
        if visits.sum() > 0:
            probs = visits / visits.sum()
        else:
            probs = np.ones(20) / 20
        return probs

class ModelConfig:
    """Configuration for neural network architecture"""
    def __init__(self):
        # Update sizes to match actual feature dimensions
        self.driver_features = 20
        self.constructor_features = 30
        self.qualifying_features = 60  # 20 drivers * 3 qualifying sessions
        self.weather_features = 1
        self.circuit_features = 3
        self.state_size = (
            self.driver_features + 
            self.constructor_features + 
            self.qualifying_features + 
            self.weather_features + 
            self.circuit_features
        )  # Total: 114
        self.hidden_size = 64
        self.dropout_rate = 0.2
        self.attention_heads = 4

class SimplifiedF1Net(nn.Module):
    def __init__(self, config: ModelConfig):
        super(SimplifiedF1Net, self).__init__()
        self.config = config
        
        # Feature encoders with circuit-specific attention
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
        
        # Special circuit encoder with increased capacity
        self.circuit_encoder = nn.Sequential(
            nn.Linear(config.circuit_features, config.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size)
        )
        
        # Circuit-aware attention mechanism
        self.circuit_attention = nn.Sequential(
            nn.Linear(config.hidden_size, 3),  # 3 attention weights for each feature type
            nn.Softmax(dim=-1)
        )
        
        # Main network layers with circuit modulation
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
        
        # Policy head with reduced temperature
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size // 2),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, 20),  # 20 possible positions
            nn.Softmax(dim=-1)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size // 2),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Tanh()
        )
    
    def _create_encoder(self, input_size: int, output_size: int) -> nn.Sequential:
        """Create an encoder block with proper regularization"""
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(self.config.dropout_rate)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure we have a batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Split state into components
        driver_feats = state[:, :self.config.driver_features]
        constructor_feats = state[:, self.config.driver_features:self.config.driver_features + self.config.constructor_features]
        qualifying_feats = state[:, self.config.driver_features + self.config.constructor_features:self.config.driver_features + self.config.constructor_features + self.config.qualifying_features]
        weather_feats = state[:, -self.config.circuit_features-1:-self.config.circuit_features]
        circuit_feats = state[:, -self.config.circuit_features:]
        
        # Encode features
        encoded_driver = self.driver_encoder(driver_feats)
        encoded_constructor = self.constructor_encoder(constructor_feats)
        encoded_qualifying = self.qualifying_encoder(qualifying_feats)
        encoded_circuit = self.circuit_encoder(circuit_feats)
        
        # Apply circuit-based attention
        attention_weights = self.circuit_attention(encoded_circuit)
        
        # Modulate features based on circuit characteristics
        attended_driver = encoded_driver * attention_weights[:, 0:1]
        attended_constructor = encoded_constructor * attention_weights[:, 1:2]
        attended_qualifying = encoded_qualifying * attention_weights[:, 2:3]
        
        # Combine features
        combined = torch.cat([
            attended_driver,
            attended_constructor,
            attended_qualifying,
            encoded_circuit,
            weather_feats
        ], dim=1)
        
        # Process through main network
        features = self.main_network(combined)
        
        # Generate outputs with raw logits
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value

def convert_time_to_seconds(time_str: str) -> Optional[float]:
    """
    Convert a qualifying time string to seconds.
    Handles formats like "1:23.456" or "83.456"
    
    Args:
        time_str: String representing a qualifying time
        
    Returns:
        Float representing time in seconds, or None if invalid
    """
    if pd.isna(time_str) or time_str == 'N/A' or time_str == '\\N':
        return None
        
    try:
        if isinstance(time_str, str):
            # Remove any whitespace
            time_str = time_str.strip()
            
            if ':' in time_str:
                # Format: "1:23.456"
                minutes, seconds = time_str.split(':')
                total_seconds = float(minutes) * 60 + float(seconds)
            else:
                # Format: "83.456"
                total_seconds = float(time_str)
            
            # Validate the time is reasonable (between 30 seconds and 3 minutes)
            if 30 <= total_seconds <= 180:
                return total_seconds
            else:
                print(f"Warning: Qualifying time outside reasonable range: {total_seconds} seconds")
                return None
                
        elif isinstance(time_str, (int, float)):
            # Direct numeric input
            total_seconds = float(time_str)
            if 30 <= total_seconds <= 180:
                return total_seconds
                
        return None
            
    except (ValueError, TypeError) as e:
        print(f"Error parsing time string '{time_str}': {str(e)}")
        return None
    
# Updated driver feature processing
def process_driver_features(self, circuit_data: pd.DataFrame, circuit_name: str) -> np.ndarray:
    """Process driver features correctly with proper DataFrame handling"""
    try:
        # Get current driver IDs
        current_driver_ids = self.driver_mapping.get_current_driver_ids()
        
        # Initialize feature arrays for all drivers
        features = np.zeros((len(current_driver_ids), 3))  # 3 features per driver
        
        for idx, driver_id in enumerate(current_driver_ids):
            # Get qualifying data for this driver
            driver_quals = self.kaggle_data['qualifying'][
                self.kaggle_data['qualifying']['driverId'] == driver_id
            ].copy()
            
            # Get circuit-specific qualifying data
            if circuit_name:
                circuit_quals = driver_quals[
                    driver_quals['raceId'].isin(
                        self.race_data[
                            self.race_data['name'].str.contains(circuit_name, case=False)
                        ]['raceId']
                    )
                ]
            else:
                circuit_quals = driver_quals

            # Calculate features
            features[idx] = self._calculate_driver_features(circuit_quals)
            
        return features.flatten()
        
    except Exception as e:
        logger.error(f"Error in process_driver_features: {str(e)}")
        return np.zeros(len(current_driver_ids) * 3)

def _calculate_driver_features(self, quals: pd.DataFrame) -> np.ndarray:
    """Calculate specific features for a driver from qualifying data"""
    if quals.empty:
        return np.array([0.0, 0.0, 0.0])
        
    # Q3 appearances ratio
    q3_ratio = len(quals[quals['q3'].notna()]) / len(quals)
    
    # Recent performance
    recent_times = []
    for _, qual in quals.iterrows():
        if pd.notna(qual['q3']) and qual['q3'] != 'N/A':
            time_secs = convert_time_to_seconds(qual['q3'])
            if time_secs:
                recent_times.append(time_secs)
    
    if recent_times:
        recent_times = recent_times[-5:]  # Last 5 races
        min_time = min(recent_times)
        recent_perf = 1 - (sum(recent_times) / len(recent_times) - min_time) / min_time
        consistency = 1 / (1 + np.std(recent_times) / np.mean(recent_times))
    else:
        recent_perf = 0.0
        consistency = 0.0
    
    return np.array([q3_ratio, recent_perf, consistency])