import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import math

class SimplifiedF1Net(nn.Module):
    """
    Simplified AlphaZero network focusing on circuit and driver features
    for qualifying prediction
    """
    def __init__(self, state_size: int = 23):  # 20 drivers + 3 sector times
        super(SimplifiedF1Net, self).__init__()
        
        # Input sizes
        self.driver_features = 20  # One feature per driver
        self.sector_features = 3   # Three sector times
        
        # Encoder layers
        self.driver_encoder = nn.Sequential(
            nn.Linear(self.driver_features, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.2)
        )
        
        self.sector_encoder = nn.Sequential(
            nn.Linear(self.sector_features, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Dropout(0.2)
        )
        
        # Attention mechanism for combining features
        self.attention = nn.MultiheadAttention(embed_dim=48, num_heads=4)  # 48 = 32 + 16
        
        # Policy head (predicting position probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 20),  # 20 possible positions
            nn.Softmax(dim=-1)
        )
        
        # Value head (predicting qualifying performance)
        self.value_head = nn.Sequential(
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Tanh()  # Output between -1 and 1
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Split state into driver and sector features
        driver_state = state[:, :self.driver_features]
        sector_state = state[:, -self.sector_features:]
        
        # Encode features
        driver_encoded = self.driver_encoder(driver_state)
        sector_encoded = self.sector_encoder(sector_state)
        
        # Combine encoded features
        combined_features = torch.cat([driver_encoded, sector_encoded], dim=-1)
        
        # Apply self-attention
        combined_features = combined_features.unsqueeze(0)  # Add sequence dimension
        attended_features, _ = self.attention(
            combined_features, 
            combined_features, 
            combined_features
        )
        attended_features = attended_features.squeeze(0)  # Remove sequence dimension
        
        # Get policy and value predictions
        policy = self.policy_head(attended_features)
        value = self.value_head(attended_features)
        
        return policy, value

class MCTS:
    """Monte Carlo Tree Search implementation for F1 qualifying prediction"""
    def __init__(self, model: SimplifiedF1Net, num_simulations: int = 50):
        self.model = model
        self.num_simulations = num_simulations
        
    def search(self, state: np.ndarray) -> np.ndarray:
        root = MCTSNode(prior=1.0)
        root.state = state
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.expanded():
                action, node = self._select_child(node)
                search_path.append(node)
            
            # Expansion and Evaluation
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy, value = self.model(state_tensor)
            
            policy = policy.detach().numpy()[0]
            value = value.item()
            
            # Backpropagate
            self._backpropagate(search_path, value, policy)
        
        # Get action probabilities
        return self._get_action_probs(root)
    
    def _select_child(self, node: 'MCTSNode') -> Tuple[int, 'MCTSNode']:
        """Select child node with highest UCB score"""
        c_puct = 1.0  # Exploration constant
        
        ucb_scores = {
            action: self._ucb_score(node, child, c_puct)
            for action, child in node.children.items()
        }
        
        best_action = max(ucb_scores.items(), key=lambda x: x[1])[0]
        return best_action, node.children[best_action]
    
    def _ucb_score(self, parent: 'MCTSNode', child: 'MCTSNode', c_puct: float) -> float:
        """Calculate UCB score for a child node"""
        prior_score = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        value_score = -child.value()  # Negative because we want to minimize position
        return value_score + prior_score
    
    def _backpropagate(self, search_path: List['MCTSNode'], value: float, policy: np.ndarray):
        """Backpropagate value through search path"""
        for node in search_path:
            node.value_sum += value
            node.visit_count += 1
            
            if not node.expanded():
                for i in range(20):  # 20 possible positions
                    node.children[i] = MCTSNode(prior=policy[i])
    
    def _get_action_probs(self, root: 'MCTSNode') -> np.ndarray:
        """Get action probabilities based on visit counts"""
        visits = np.zeros(20)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        
        if visits.sum() > 0:
            probs = visits / visits.sum()
        else:
            probs = np.ones(20) / 20
        return probs

class MCTSNode:
    """Node in the MCTS tree"""
    def __init__(self, prior: float):
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.children = {}
        self.state = None
    
    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class QualifyingPredictor:
    """Main class for making qualifying predictions"""
    def __init__(self, data_processor, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.data_processor = data_processor
        self.device = device
        self.model = SimplifiedF1Net().to(device)
        self.mcts = MCTS(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train_step(self, state_batch: torch.Tensor, target_batch: torch.Tensor) -> Dict:
        """Perform a single training step"""
        self.optimizer.zero_grad()
        
        # Get predictions
        policy_pred, value_pred = self.model(state_batch)
        
        # Calculate losses
        # Convert target_batch to float for policy loss
        target_batch_float = target_batch.float()
        
        # Use KL divergence loss for policy
        policy_loss = F.kl_div(
            F.log_softmax(policy_pred, dim=1),
            target_batch_float,
            reduction='batchmean'
        )
        
        # Value loss using the mean position as target
        value_target = (torch.argmax(target_batch_float, dim=1).float() + 1) / 20.0  # Normalize to [0,1]
        value_loss = F.mse_loss(value_pred.squeeze(), value_target)
        
        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss  # Weigh value loss less
        
        # Backpropagate
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def predict_qualifying(self, circuit_name: str) -> Dict:
        """Predict qualifying results for a specific circuit"""
        # Get circuit data
        circuit_data = self.data_processor.circuits_data.get(circuit_name.lower())
        if circuit_data is None:
            return {"error": f"No data found for circuit: {circuit_name}"}
        
        # Create state representation
        state = self._create_state(circuit_data)
        
        # Get predictions using MCTS
        action_probs = self.mcts.search(state)
        
        # Get confidence score from value network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, value = self.model(state_tensor)
            confidence = (value.item() + 1) / 2
        
        # Get current driver IDs from mapping
        driver_ids = self.data_processor.driver_mapping.get_current_driver_ids()

        # Get top 5 predictions
        top5_indices = np.argsort(action_probs)[-5:][::-1]
        
        # Create predictions with proper driver mapping
        predictions = {
            'top5': [],
            'confidence_score': float(confidence)
        }
        
        for i, idx in enumerate(top5_indices):
            driver_id = driver_ids[idx] if idx < len(driver_ids) else None
            driver_code = self.data_processor.driver_mapping.get_driver_code(driver_id) or f"DRV{idx+1}"
            
            predictions['top5'].append({
                'position': i + 1,
                'probability': float(action_probs[idx]),
                'driver_code': driver_code
            })
        
        return predictions
    
    def _create_state(self, circuit_data: pd.DataFrame) -> np.ndarray:
        """Create state representation from circuit data"""
        # Get normalized sector times
        sector_times = np.zeros(3)
        for i, col in enumerate(['sector1_normalized', 'sector2_normalized', 'sector3_normalized']):
            if col in circuit_data.columns:
                sector_times[i] = circuit_data[col].mean()
        
        # Create driver performance vector (placeholder - you'll want to add actual driver performance data)
        driver_perf = np.zeros(20)
        
        # Combine features
        state = np.concatenate([driver_perf, sector_times])
        return state

def train_step(self, state_batch: torch.Tensor, target_batch: torch.Tensor) -> Dict:
    """Single training step"""
    policy_pred, value_pred = self.model(state_batch)
    
    # Calculate losses
    policy_loss = F.cross_entropy(policy_pred, target_batch)
    value_loss = F.mse_loss(value_pred.squeeze(), target_batch.float().mean(dim=1))
    
    total_loss = policy_loss + value_loss
    
    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'total_loss': total_loss.item()
    }