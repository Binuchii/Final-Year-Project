import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

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

class MCTS:
    """Monte Carlo Tree Search implementation for F1 qualifying prediction"""
    def __init__(self, model: 'SimplifiedF1Net', num_simulations: int = 50):
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
        
        return self._get_action_probs(root)
    
    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Select child node with highest UCB score"""
        c_puct = 1.0  # Exploration constant
        
        ucb_scores = {
            action: self._ucb_score(node, child, c_puct)
            for action, child in node.children.items()
        }
        
        best_action = max(ucb_scores.items(), key=lambda x: x[1])[0]
        return best_action, node.children[best_action]
    
    def _ucb_score(self, parent: MCTSNode, child: MCTSNode, c_puct: float) -> float:
        """Calculate UCB score for a child node"""
        prior_score = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        value_score = -child.value()  # Negative because we want to minimize position
        return value_score + prior_score
    
    def _backpropagate(self, search_path: List[MCTSNode], value: float, policy: np.ndarray):
        """Backpropagate value through search path"""
        for node in search_path:
            node.value_sum += value
            node.visit_count += 1
            
            if not node.expanded():
                for i in range(20):  # 20 possible positions
                    node.children[i] = MCTSNode(prior=policy[i])
    
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

class SimplifiedF1Net(nn.Module):
    def __init__(self, state_size: int = 103):
        super(SimplifiedF1Net, self).__init__()
        
        # Input sizes
        self.driver_features = 20
        self.qual_features = 20
        self.recent_qual_features = 20
        self.track_perf_features = 20
        self.experience_features = 20
        self.sector_features = 3
        
        # Network architecture
        self.driver_encoder = nn.Sequential(
            nn.Linear(20, 48),
            nn.ReLU(),
            nn.LayerNorm(48),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.LayerNorm(48)
        )
        
        self.track_encoder = nn.Sequential(
            nn.Linear(20, 48),
            nn.ReLU(),
            nn.LayerNorm(48),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.LayerNorm(48)
        )
        
        self.sector_encoder = nn.Sequential(
            nn.Linear(self.sector_features, 16),
            nn.ReLU(),
            nn.LayerNorm(16)
        )
        
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        
        self.policy_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 20),
            nn.Softmax(dim=-1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = state.size(0)
        
        # Split state into components
        driver_base = state[:, :self.driver_features]
        qual_perf = state[:, self.driver_features:2*self.driver_features]
        recent_qual = state[:, 2*self.driver_features:3*self.driver_features]
        track_perf = state[:, 3*self.driver_features:4*self.driver_features]
        experience = state[:, 4*self.driver_features:5*self.driver_features]
        sector_state = state[:, -self.sector_features:]
        
        # Process features
        base_encoded = self.driver_encoder(driver_base)
        qual_encoded = self.driver_encoder(qual_perf)
        recent_encoded = self.driver_encoder(recent_qual)
        track_encoded = self.track_encoder(track_perf)
        exp_encoded = self.driver_encoder(experience)
        
        # Combine features with AlphaZero-style weighting
        driver_encoded = (0.15 * base_encoded + 
                         0.25 * qual_encoded + 
                         0.30 * recent_encoded +
                         0.20 * track_encoded +
                         0.10 * exp_encoded)
        
        sector_encoded = self.sector_encoder(sector_state)
        combined_features = torch.cat([driver_encoded, sector_encoded], dim=1)
        
        # Apply attention mechanism
        combined_features = combined_features.unsqueeze(0)
        attended_features, _ = self.attention(
            combined_features,
            combined_features,
            combined_features
        )
        attended_features = attended_features.squeeze(0)
        
        return self.policy_head(attended_features), self.value_head(attended_features)

class QualifyingPredictor:
    def __init__(self, data_processor, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.data_processor = data_processor
        self.device = device
        self.model = SimplifiedF1Net().to(device)
        self.mcts = MCTS(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def _calculate_track_performance(self, driver_id: int, circuit_name: str, qualifying_data: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate driver's performance metrics at specific track
        Returns: (track_performance, consistency, improvement_rate)
        """
        if 'q3' not in qualifying_data.columns:
            return 0.0, 0.0, 0.0
            
        # Get qualifying data for this driver at this circuit
        track_quals = qualifying_data[
            (qualifying_data['driverId'] == driver_id) & 
            (qualifying_data['circuit_name'].str.lower() == circuit_name.lower())
        ] if 'circuit_name' in qualifying_data.columns else qualifying_data[
            qualifying_data['driverId'] == driver_id
        ]
        
        if track_quals.empty:
            return 0.0, 0.0, 0.0
            
        # Calculate performance based on Q3 times
        q3_times = []
        dates = []
        for _, qual in track_quals.iterrows():
            if pd.notna(qual['q3']) and qual['q3'] != 'N/A':
                try:
                    time_str = qual['q3']
                    if isinstance(time_str, str) and ':' in time_str:
                        mins, secs = time_str.split(':')
                        time_secs = float(mins) * 60 + float(secs)
                    else:
                        time_secs = float(time_str)
                    q3_times.append(time_secs)
                    if 'date' in qual:
                        dates.append(pd.to_datetime(qual['date']))
                except (ValueError, TypeError):
                    continue
        
        if not q3_times:
            return 0.0, 0.0, 0.0
            
        # Calculate base performance (normalized)
        min_time = min(q3_times)
        normalized_times = [1 - (t - min_time) / min_time for t in q3_times]
        
        # Calculate consistency (inverse of standard deviation)
        consistency = 1.0 / (1.0 + np.std(normalized_times))
        
        # Calculate improvement rate if dates are available
        improvement_rate = 0.0
        if dates and len(dates) > 1:
            # Calculate time differences between consecutive races
            time_diffs = np.diff(q3_times)
            # Negative diff means improvement
            improvements = [1 if diff < 0 else 0 for diff in time_diffs]
            improvement_rate = np.mean(improvements)
        
        # Weight recent results more heavily
        weights = np.exp(-0.3 * np.arange(len(normalized_times)))
        weights = weights / weights.sum()
        track_performance = np.sum(normalized_times * weights)
        
        return track_performance, consistency, improvement_rate

    def _create_state(self, circuit_data: pd.DataFrame, circuit_name: Optional[str] = None) -> np.ndarray:
        """Create enhanced state representation for AlphaZero model"""
        driver_ids = self.data_processor.driver_mapping.get_current_driver_ids()
        
        # Initialize feature vectors
        driver_perf = np.zeros(20)
        qual_perf = np.zeros(20)
        recent_qual = np.zeros(20)
        track_perf = np.zeros(20)
        track_consistency = np.zeros(20)
        track_improvement = np.zeros(20)
        experience = np.zeros(20)
        
        qual_data = self.data_processor.kaggle_data['qualifying']
        
        for idx, driver_id in enumerate(driver_ids):
            if driver_id is None:
                continue
                
            driver_quals = qual_data[qual_data['driverId'] == driver_id]
            
            if not driver_quals.empty:
                # Calculate experience weight
                num_races = len(driver_quals)
                max_races = len(qual_data['raceId'].unique())
                experience[idx] = min(1.0, num_races / max_races)
                
                # Process qualifying times
                q_times = []
                recent_times = []
                all_qual_positions = []
                
                for _, qual in driver_quals.iterrows():
                    if pd.notna(qual['q3']) and qual['q3'] != 'N/A':
                        try:
                            time_str = qual['q3']
                            if isinstance(time_str, str) and ':' in time_str:
                                mins, secs = time_str.split(':')
                                time_secs = float(mins) * 60 + float(secs)
                            else:
                                time_secs = float(time_str)
                            q_times.append(time_secs)
                            recent_times.append(time_secs)
                        except (ValueError, TypeError):
                            continue
                
                if q_times:
                    # Calculate overall qualifying performance
                    min_time = min(q_times)
                    qual_perf[idx] = 1 - np.mean([(t - min_time) / min_time for t in q_times])
                
                if recent_times:
                    # Calculate recent qualifying performance (last 5 races)
                    recent_times = recent_times[-5:]
                    recent_min = min(recent_times)
                    recent_qual[idx] = 1 - np.mean([(t - recent_min) / recent_min for t in recent_times])
                
                # Calculate track-specific performance metrics
                if circuit_name:
                    perf, cons, imp = self._calculate_track_performance(
                        driver_id, circuit_name, qual_data
                    )
                    track_perf[idx] = perf
                    track_consistency[idx] = cons
                    track_improvement[idx] = imp
        
        # Get normalized sector times and circuit characteristics
        sector_times = np.zeros(3)
        for i, col in enumerate(['sector1_normalized', 'sector2_normalized', 'sector3_normalized']):
            if col in circuit_data.columns:
                sector_times[i] = circuit_data[col].mean()
        
        # Calculate base driver performance with enhanced track metrics
        driver_perf = (0.25 * qual_perf + 
                    0.30 * recent_qual + 
                    0.25 * track_perf + 
                    0.10 * track_consistency +
                    0.10 * track_improvement)
        
        # Add circuit characteristics effect
        if 'avg_speed' in circuit_data.columns:
            circuit_speed = circuit_data['avg_speed'].mean()
            # Adjust driver performance based on circuit characteristics
            speed_factor = circuit_speed / circuit_data['avg_speed'].max()
            driver_perf *= (0.8 + 0.4 * speed_factor)  # Scale effect
        
        # Combine all features
        state = np.concatenate([
            driver_perf,
            qual_perf,
            recent_qual,
            track_perf,
            experience,
            sector_times
        ])
        
        return state

    def predict_qualifying(self, circuit_name: str) -> Dict:
        """Predict qualifying results for a specific circuit"""
        circuit_data = self.data_processor.circuits_data.get(circuit_name.lower())
        if circuit_data is None:
            return {"error": f"No data found for circuit: {circuit_name}"}
        
        state = self._create_state(circuit_data, circuit_name)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy, value = self.model(state_tensor)
            
            action_probs = policy.cpu().numpy()[0]
            confidence = (value.item() + 1) / 2
        
        driver_ids = self.data_processor.driver_mapping.get_current_driver_ids()
        top5_indices = np.argsort(action_probs)[-5:][::-1]
        
        predictions = {
            'top5': [],
            'confidence_score': float(confidence)
        }
        
        qual_data = self.data_processor.kaggle_data['qualifying']
        
        for i, idx in enumerate(top5_indices):
            driver_id = driver_ids[idx] if idx < len(driver_ids) else None
            driver_code = self.data_processor.driver_mapping.get_driver_code(driver_id) or f"DRV{idx+1}"
            
            if driver_id:
                driver_quals = qual_data[qual_data['driverId'] == driver_id]
                q3_apps = len(driver_quals[driver_quals['q3'].notna() & (driver_quals['q3'] != 'N/A')])
                
                # Calculate consistency
                q3_times = []
                for _, qual in driver_quals.iterrows():
                    if pd.notna(qual['q3']) and qual['q3'] != 'N/A':
                        try:
                            time_str = qual['q3']
                            if isinstance(time_str, str) and ':' in time_str:
                                mins, secs = time_str.split(':')
                                time_secs = float(mins) * 60 + float(secs)
                            else:
                                time_secs = float(time_str)
                            q3_times.append(time_secs)
                        except (ValueError, TypeError):
                            continue
                
                consistency = np.std(q3_times) if q3_times else None
            else:
                q3_apps = 0
                consistency = None
            
            predictions['top5'].append({
                'position': i + 1,
                'driver_code': driver_code,
                'probability': float(action_probs[idx]),
                'q3_appearances': q3_apps,
                'consistency': consistency
            })
        
        return predictions

    def train_step(self, state_batch: torch.Tensor, target_batch: torch.Tensor) -> Dict:
        """Perform a single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get predictions
        policy_pred, value_pred = self.model(state_batch)
        
        # Calculate policy loss using KL divergence
        policy_loss = F.kl_div(
            F.log_softmax(policy_pred, dim=1),
            target_batch.float(),
            reduction='batchmean'
        )
        
        # Calculate value loss - use mean position as target
        value_target = (torch.argmax(target_batch, dim=1).float() + 1) / 20.0  # Normalize to [0,1]
        value_loss = F.mse_loss(value_pred.squeeze(), value_target)
        
        # Combined loss with weighting
        total_loss = policy_loss + 0.5 * value_loss
        
        # Backpropagate
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }