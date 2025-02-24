from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Change this from relative to absolute import
from mcts_and_nn import (
    MCTSConfig,
    ModelConfig,
    SimplifiedF1Net,
    MCTS,
    convert_time_to_seconds
)

logger = logging.getLogger(__name__)

@dataclass
class PredictorConfig:
    """Configuration for QualifyingPredictor"""
    learning_rate: float = 0.001
    batch_size: int = 32
    model_save_dir: str = "models"
    cache_dir: str = "cache"
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    min_delta: float = 0.001

class QualifyingPredictor:
    """Improved QualifyingPredictor with better error handling and performance"""
    def __init__(
        self,
        data_processor: Any,
        config: PredictorConfig,
        model_config: ModelConfig,
        mcts_config: MCTSConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.data_processor = data_processor
        self.config = config
        self.device = device
        self.model = SimplifiedF1Net(model_config).to(device)
        self.mcts = MCTS(self.model, mcts_config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Create directories
        Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'epochs_without_improvement': 0
        }
        
        # Feature calculation cache
        self._feature_cache = {}

    def _create_state(self, circuit_data: pd.DataFrame, circuit_name: str) -> np.ndarray:
        """Create complete state representation with all required features."""
        try:
            logger.info(f"Creating state for circuit: {circuit_name}")
            
            if circuit_data is None or circuit_data.empty:
                raise ValueError("Invalid circuit data provided")

            # Get race ID for this circuit
            race_ids = self.data_processor.race_data[
                self.data_processor.race_data['name'].str.contains(circuit_name, case=False)
            ]['raceId'].tolist()

            if not race_ids:
                raise ValueError(f"No race data found for circuit: {circuit_name}")

            # Use the most recent race ID for this circuit
            race_id = max(race_ids)

            # Get race and constructor info
            race_info = self.data_processor.race_data[self.data_processor.race_data['raceId'] == race_id]
            constructor_info = self.data_processor.constructor_data[
                self.data_processor.constructor_data['raceId'] == race_id
            ]

            # 1. Driver features (20)
            driver_features = self._process_driver_features(circuit_data, circuit_name)
            
            # 2. Constructor features (30)
            constructor_features = self._process_constructor_features(constructor_info)
            
            # 3. Qualifying features (60)
            qualifying_features = self._process_qualifying_features(circuit_data, circuit_name)
            
            # 4. Weather features (1)
            weather_features = self._process_weather_features(race_info)
            
            # 5. Circuit features (3)
            circuit_features = self._process_circuit_features(circuit_data, circuit_name)

            # Combine all features
            state = np.concatenate([
                driver_features,      # 20
                constructor_features, # 30
                qualifying_features,  # 60
                weather_features,     # 1
                circuit_features      # 3
            ])

            logger.info(f"State created with shape: {state.shape}")
            return state

        except Exception as e:
            logger.error(f"Error in state creation: {e}", exc_info=True)
            raise
    
    def predict_qualifying(self, circuit_name: str) -> Dict:
        """
        Predict qualifying with enhanced circuit-specific processing.
        """
        try:
            logger.info(f"Predicting qualifying for circuit: {circuit_name}")
            
            # Get circuit data
            circuit_data = self.data_processor.circuits_data.get(circuit_name.lower())
            if circuit_data is None:
                raise ValueError(f"No data found for circuit: {circuit_name}")
            
            # Create state representation
            state = self._create_state(circuit_data, circuit_name)
            
            # Log state statistics by feature group
            feature_sizes = {
                'driver': 20,
                'constructor': 30,
                'qualifying': 60,
                'weather': 1,
                'circuit': 3
            }
            
            start_idx = 0
            for feature_name, size in feature_sizes.items():
                feature_slice = state[start_idx:start_idx + size]
                stats = {
                    'mean': np.mean(feature_slice),
                    'std': np.std(feature_slice),
                    'min': np.min(feature_slice),
                    'max': np.max(feature_slice)
                }
                logger.info(f"{feature_name.capitalize()} features stats: {stats}")
                start_idx += size
            
            # Convert to tensor and predict
            self.model.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                policy, value = self.model(state_tensor)
                
                # APPROACH #1 & #4: Apply adjusted temperature based on circuit characteristics
                # Make base temperature more aggressive (lower)
                base_temperature = 0.1  # Changed from 0.3 to 0.1 for more peaked distribution
                circuit_scaling = self._get_circuit_scaling(circuit_name)
                temperature = base_temperature / circuit_scaling
                
                # Apply temperature scaling directly to logits before softmax
                scaled_policy = policy / temperature
                action_probs = F.softmax(scaled_policy, dim=1).cpu().numpy()[0]
                
                # Instead of small noise, add position-biased noise to favor better positions
                position_bias = np.array([(20-i)/20 for i in range(20)]) * 0.05
                action_probs = action_probs + position_bias
                
                circuit_type = self._get_circuit_type(circuit_name)
                driver_ids = self.data_processor.driver_mapping.get_current_driver_ids()

                # Adjust probability distribution based on circuit type
                for idx, driver_id in enumerate(driver_ids):
                    if idx < len(action_probs):
                        driver_code = self.data_processor.driver_mapping.get_driver_code(driver_id)
                        
                        # Circuit type preferences - comprehensive for all drivers
                        driver_circuit_preferences = {
                            # Red Bull
                            'VER': {'high_speed': 1.4, 'technical': 1.1, 'street': 0.9},
                            'PER': {'high_speed': 0.9, 'technical': 0.9, 'street': 1.5},
                            
                            # Ferrari
                            'LEC': {'high_speed': 1.1, 'technical': 1.2, 'street': 1.5},
                            'SAI': {'high_speed': 1.2, 'technical': 1.3, 'street': 1.1},
                            
                            # Mercedes
                            'HAM': {'high_speed': 1.3, 'technical': 1.4, 'street': 1.0},
                            'RUS': {'high_speed': 1.2, 'technical': 1.3, 'street': 1.2},
                            
                            # McLaren
                            'NOR': {'high_speed': 1.3, 'technical': 1.2, 'street': 1.2},
                            'PIA': {'high_speed': 1.2, 'technical': 1.1, 'street': 1.2},
                            
                            # Aston Martin
                            'ALO': {'high_speed': 1.2, 'technical': 1.3, 'street': 1.4},
                            'STR': {'high_speed': 1.0, 'technical': 0.9, 'street': 1.2},
                            
                            # Alpine
                            'GAS': {'high_speed': 1.1, 'technical': 1.2, 'street': 1.0},
                            'OCO': {'high_speed': 1.0, 'technical': 1.3, 'street': 1.1},
                            
                            # Williams
                            'ALB': {'high_speed': 1.3, 'technical': 1.0, 'street': 1.2},
                            'SAR': {'high_speed': 1.1, 'technical': 0.8, 'street': 0.9},
                            
                            # RB (AlphaTauri)
                            'TSU': {'high_speed': 1.0, 'technical': 1.0, 'street': 1.2},
                            'RIC': {'high_speed': 1.2, 'technical': 1.1, 'street': 1.0},
                            
                            # Sauber (Alfa Romeo)
                            'BOT': {'high_speed': 1.3, 'technical': 1.1, 'street': 0.9},
                            'ZHO': {'high_speed': 1.0, 'technical': 0.9, 'street': 1.1},
                            
                            # Haas
                            'MAG': {'high_speed': 1.2, 'technical': 1.0, 'street': 1.1},
                            'HUL': {'high_speed': 1.2, 'technical': 1.1, 'street': 1.0}
                        }
                        
                        if driver_code in driver_circuit_preferences and circuit_type in driver_circuit_preferences[driver_code]:
                            type_factor = driver_circuit_preferences[driver_code][circuit_type]
                            action_probs[idx] *= type_factor

                # Re-normalize
                action_probs = action_probs / np.sum(action_probs)
                
                confidence = (value.item() + 1) / 2
            
            # APPROACH #5: Apply Constructor Weighting
            constructor_strengths = self._get_constructor_strengths()
            for idx, driver_id in enumerate(self.data_processor.driver_mapping.get_current_driver_ids()):
                if idx < len(action_probs):
                    constructor_id = self._get_driver_constructor(driver_id)
                    if constructor_id is not None:
                        constructor_weight = constructor_strengths[constructor_id]
                        action_probs[idx] *= constructor_weight
            
            # Re-normalize
            action_probs = action_probs / np.sum(action_probs)
            
            # APPROACH #6: Incorporate Recent Form
            recent_form_factors = self._calculate_recent_form()
            for idx, driver_id in enumerate(self.data_processor.driver_mapping.get_current_driver_ids()):
                if idx < len(action_probs):
                    form_factor = recent_form_factors[driver_id]
                    action_probs[idx] *= form_factor
            
            # Re-normalize
            action_probs = action_probs / np.sum(action_probs)
            
            # APPROACH #7: Add Confidence Amplifier
            confidence_factor = confidence * 4  # Increased from 2 to 4 for more pronounced effect
            confidence_adjusted_probs = np.power(action_probs, confidence_factor)
            
            # Re-normalize
            action_probs = confidence_adjusted_probs / np.sum(confidence_adjusted_probs)
            
            # Apply extreme amplification to create greater spread
            skewed_probs = np.power(action_probs, 3)  # Apply cubic function to increase spread
            action_probs = skewed_probs / np.sum(skewed_probs)
            
            # Format predictions with detailed driver info
            predictions = self._format_predictions(action_probs, confidence, circuit_name)
            
            return predictions
                
        except Exception as e:
            logger.error(f"Error in prediction: {e}", exc_info=True)
            return {"error": str(e)}

    def _get_circuit_type(self, circuit_name: str) -> str:
        """Categorize circuit by type"""
        circuit_types = {
            # Street circuits with walls close to the track, high emphasis on qualifying
            'street': [
                'monaco', 'singapore', 'azerbaijan', 'saudi', 'las vegas',
                'baku', 'jeddah', 'marina bay'
            ],
            
            # High-speed circuits with long straights and fast corners
            'high_speed': [
                'monza', 'spa', 'austria', 'silverstone', 'british', 'miami',
                'belgian', 'bahrain', 'canadian', 'italian', 'australian',
                'red bull ring', 'spielberg', 'montreal', 'interlagos', 'brazilian'
            ],
            
            # Technical circuits with complex sections and emphasis on downforce
            'technical': [
                'hungary', 'barcelona', 'spanish', 'zandvoort', 'dutch',
                'japan', 'japanese', 'suzuka', 'abu', 'yas marina', 'mexican',
                'mexico city', 'imola', 'emilia', 'chinese', 'shanghai',
                'portuguese', 'portimao', 'french', 'paul ricard', 'qatar',
                'hungarian', 'sochi', 'russian', 'catalonia'
            ]
        }
        
        normalized_name = circuit_name.lower()
        
        for circuit_type, circuits in circuit_types.items():
            if any(c in normalized_name for c in circuits):
                return circuit_type
        
        # Default to technical if unknown
        return 'technical'
    def _get_driver_constructor(self, driver_id: int) -> int:
        """Get constructor ID for a driver"""
        try:
            # Check if we have driver_standings data which typically has the constructor mapping
            if 'driver_standings' in self.data_processor.kaggle_data:
                driver_standings = self.data_processor.kaggle_data['driver_standings']
                # Get the most recent entry for this driver
                driver_data = driver_standings[driver_standings['driverId'] == driver_id].sort_values('raceId', ascending=False)
                
                if not driver_data.empty and 'constructorId' in driver_data.columns:
                    return driver_data.iloc[0]['constructorId']
            
            # Alternative approach if the first method doesn't work
            if 'results' in self.data_processor.kaggle_data:
                results = self.data_processor.kaggle_data['results']
                # Get the most recent result for this driver
                driver_results = results[results['driverId'] == driver_id].sort_values('raceId', ascending=False)
                
                if not driver_results.empty and 'constructorId' in driver_results.columns:
                    return driver_results.iloc[0]['constructorId']
            
            # If we still don't have a result, use a fallback mapping
            return self._get_fallback_constructor_mapping().get(driver_id, None)
                
        except Exception as e:
            logger.error(f"Error getting driver constructor: {str(e)}")
            return None
    
    def _get_fallback_constructor_mapping(self) -> Dict[int, int]:
        """Provide a fallback mapping of drivers to constructors for 2024 F1 season"""
        # This mapping uses IDs from your database
        # If your ID system is different, adjust accordingly
        return {
            # Red Bull Racing
            1: 1,    # Max Verstappen
            2: 1,    # Sergio Perez
            
            # Ferrari
            3: 2,    # Charles Leclerc
            4: 2,    # Carlos Sainz
            
            # Mercedes
            5: 3,    # Lewis Hamilton
            6: 3,    # George Russell
            
            # McLaren
            7: 4,    # Lando Norris
            8: 4,    # Oscar Piastri
            
            # Aston Martin
            9: 5,    # Fernando Alonso
            10: 5,   # Lance Stroll
            
            # Alpine
            11: 6,   # Pierre Gasly
            12: 6,   # Esteban Ocon
            
            # Williams
            13: 7,   # Alex Albon
            14: 7,   # Logan Sargeant
            
            # RB (AlphaTauri/Toro Rosso)
            15: 8,   # Yuki Tsunoda
            16: 8,   # Daniel Ricciardo
            
            # Sauber (Alfa Romeo)
            17: 9,   # Valtteri Bottas
            18: 9,   # Zhou Guanyu
            
            # Haas
            19: 10,  # Kevin Magnussen
            20: 10,  # Nico Hulkenberg
        }

    def _get_constructor_strengths(self) -> Dict[int, float]:
        """Calculate strength factors for each constructor based on performance"""
        try:
            strengths = {}
            constructor_data = self.data_processor.kaggle_data.get('constructor_standings', pd.DataFrame())
            
            if constructor_data.empty:
                return {k: 1.0 for k in range(1, 11)}  # Default value if no data
                
            # Get the most recent constructor standings
            recent_standings = constructor_data.sort_values('raceId', ascending=False)
            constructor_groups = recent_standings.groupby('constructorId')
            
            # Get the most recent entry for each constructor
            latest_standings = []
            for constructor_id, group in constructor_groups:
                latest_standings.append(group.iloc[0])
                
            latest_standings_df = pd.DataFrame(latest_standings)
            
            if not latest_standings_df.empty and 'points' in latest_standings_df:
                max_points = latest_standings_df['points'].max()
                if max_points > 0:
                    for _, row in latest_standings_df.iterrows():
                        constructor_id = row['constructorId']
                        points = row['points']
                        # Scale from 0.8 to 1.5 based on points (better teams get higher factor)
                        strengths[constructor_id] = 0.8 + 0.7 * (points / max_points)
                else:
                    # If no points data, use default values
                    for constructor_id in latest_standings_df['constructorId'].unique():
                        strengths[constructor_id] = 1.0
            
            # Ensure we have a default for any constructor not found
            return defaultdict(lambda: 1.0, strengths)
        except Exception as e:
            logger.error(f"Error calculating constructor strengths: {e}")
            return defaultdict(lambda: 1.0)
    
    def _get_recent_races(self, n_races: int = 5) -> List[int]:
        """Get IDs of the n most recent races"""
        try:
            race_data = self.data_processor.race_data
            if race_data is not None and not race_data.empty:
                # Sort races by date (or race ID if date not available)
                if 'date' in race_data.columns:
                    sorted_races = race_data.sort_values('date', ascending=False)
                else:
                    sorted_races = race_data.sort_values('raceId', ascending=False)
                    
                # Get the n most recent race IDs
                return sorted_races['raceId'].head(n_races).tolist()
            return []
        except Exception as e:
            logger.error(f"Error getting recent races: {e}")
            return []

    def _get_driver_recent_results(self, driver_id: int, race_ids: List[int]) -> List[Dict]:
        """Get a driver's qualifying results from recent races"""
        try:
            qualifying_data = self.data_processor.kaggle_data.get('qualifying', pd.DataFrame())
            
            if qualifying_data.empty:
                return []
                
            # Filter for this driver and these races
            driver_quals = qualifying_data[
                (qualifying_data['driverId'] == driver_id) & 
                (qualifying_data['raceId'].isin(race_ids))
            ]
            
            # Sort by race ID descending (most recent first)
            driver_quals = driver_quals.sort_values('raceId', ascending=False)
            
            results = []
            for _, qual in driver_quals.iterrows():
                position = qual.get('position', None)
                if position is not None:
                    try:
                        position = int(position)
                    except (ValueError, TypeError):
                        position = 20  # Default to back of grid if invalid
                        
                    results.append({
                        'raceId': qual['raceId'],
                        'position': position
                    })
                    
            return results
        except Exception as e:
            logger.error(f"Error getting driver recent results: {e}")
            return []

    def _calculate_recent_form(self) -> Dict[int, float]:
        """Calculate form factors for each driver based on recent results"""
        try:
            form_factors = {}
            
            # Get recent races (e.g. last 5 races)
            recent_races = self._get_recent_races(5)
            
            for driver_id in self.data_processor.driver_mapping.get_current_driver_ids():
                # Calculate recent performance metrics
                recent_results = self._get_driver_recent_results(driver_id, recent_races)
                
                # Default form factor
                form_factor = 1.0
                
                if recent_results:
                    # Example formula: weight recent qualifying positions
                    # Lower positions (better performance) = higher factor
                    weighted_sum = 0
                    weights_sum = 0
                    
                    # More recent races have higher weights
                    for i, result in enumerate(recent_results):
                        position = result.get('position', 10)
                        weight = len(recent_results) - i  # More recent = higher weight
                        weighted_sum += position * weight
                        weights_sum += weight
                    
                    if weights_sum > 0:
                        avg_position = weighted_sum / weights_sum
                        
                        # Convert to a factor (1.5 for P1, 0.7 for P20)
                        form_factor = 1.5 - (avg_position - 1) * (0.8 / 19)
                
                form_factors[driver_id] = form_factor
                
            # Ensure default values for any missing drivers
            return defaultdict(lambda: 1.0, form_factors)
        except Exception as e:
            logger.error(f"Error calculating recent form: {e}")
            return defaultdict(lambda: 1.0)
        
    def _get_circuit_scaling(self, circuit_name: str) -> float:
        """Returns a scaling factor based on circuit characteristics
        - Higher values (>1.0): More predictable circuits where qualifying position strongly correlates with results
        - Lower values (<1.0): Less predictable circuits with more variables affecting qualifying
        """
        circuit_predictability = {
            # Street circuits - typically more predictable qualifying due to limited overtaking
            'monaco': 1.7,           # Most predictable, qualifying is crucial, very difficult to overtake
            'singapore': 1.5,        # Street circuit, difficult to overtake
            'azerbaijan': 1.4,       # Baku street circuit with challenging corners
            
            # Technical circuits - reward driver skill and setup precision
            'hungarian': 1.4,        # Tight, technical circuit similar to Monaco but not a street circuit
            'dutch': 1.3,            # Narrow with technical sections
            'spanish': 1.3,          # Technical circuit with high importance on aerodynamics
            'japanese': 1.3,         # Technical track that rewards precision
            
            # Balanced circuits
            'british': 1.1,          # Technical sections but also fast corners
            'australian': 1.1,       # Balanced circuit with technical sections
            'austrian': 1.0,         # Short circuit with some overtaking opportunities
            'bahrain': 1.0,          # Balanced track with multiple overtaking spots
            'canadian': 1.0,         # Balanced with some technical sections
            
            # Circuits with more overtaking possibilities
            'emilia romagna': 0.9,   # Imola has some overtaking opportunities
            'belgian': 0.9,          # Long circuit with varied sections
            'chinese': 0.9,          # Good overtaking opportunities
            'saudi arabian': 0.9,    # Fast street circuit with some passing zones
            
            # High-speed circuits with good overtaking - qualifying less crucial
            'italian': 0.8,          # Monza - high speed with good slipstream opportunities
            'united states': 0.8,    # COTA has multiple overtaking spots
            'mexican': 0.8,          # Good overtaking opportunities
            'miami': 0.8,            # Newer circuit with multiple passing zones
        }
        
        # Handle different naming conventions
        normalized_name = circuit_name.lower().strip()
        if 'mexico' in normalized_name:
            return circuit_predictability.get('mexican', 1.0)
        elif 'hungary' in normalized_name:
            return circuit_predictability.get('hungarian', 1.0)
        elif 'italy' in normalized_name or 'monza' in normalized_name:
            return circuit_predictability.get('italian', 1.0)
        elif 'us' in normalized_name or 'cota' in normalized_name:
            return circuit_predictability.get('united states', 1.0)
        elif 'imola' in normalized_name:
            return circuit_predictability.get('emilia romagna', 1.0)
        
        # For any other circuits or if name doesn't match exactly
        for circuit_key in circuit_predictability:
            if circuit_key in normalized_name:
                return circuit_predictability[circuit_key]
        
        # Default value if no match is found
        return 1.0
    def train_step(self, state_batch: torch.Tensor, target_batch: torch.Tensor) -> Dict[str, float]:
        """Perform a single training step with proper mode handling."""
        try:
            # Ensure model is in training mode
            self.model.train()
            self.optimizer.zero_grad()

            # Get predictions
            policy_pred, value_pred = self.model(state_batch)

            # Calculate losses
            policy_loss = F.kl_div(
                F.log_softmax(policy_pred, dim=1),
                target_batch.float(),
                reduction='batchmean'
            )

            value_target = (torch.argmax(target_batch, dim=1).float() + 1) / 20.0
            value_loss = F.mse_loss(value_pred.squeeze(), value_target)

            # Combined loss
            total_loss = policy_loss + 0.5 * value_loss

            # Backpropagate
            total_loss.backward()
            self.optimizer.step()

            return {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'total_loss': total_loss.item()
            }

        except Exception as e:
            logger.error(f"Error in training step: {e}", exc_info=True)
            raise

    def save_model(self, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }
        
        path = Path(self.config.model_save_dir) / f'model_epoch_{epoch}_loss_{loss:.4f}.pt'
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint.get('history', self.history)
            logger.info(f"Model loaded from {path}")
            return checkpoint.get('epoch', 0), checkpoint.get('loss', float('inf'))
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def _format_predictions(self, action_probs: np.ndarray, confidence: float, circuit_name: str) -> Dict:
        """Format predictions with enhanced driver stats and circuit-specific performance."""
        try:
            driver_ids = self.data_processor.driver_mapping.get_current_driver_ids()
            
            # Get qualifying data for this circuit
            qual_data = self.data_processor.kaggle_data['qualifying']
            circuit_quals = qual_data[
                qual_data['raceId'].isin(
                    self.data_processor.race_data[
                        self.data_processor.race_data['name'].str.contains(circuit_name, case=False)
                    ]['raceId']
                )
            ]
            
            # Apply driver circuit factors with stronger effect
            for idx, driver_id in enumerate(self.data_processor.driver_mapping.get_current_driver_ids()):
                if idx < len(action_probs):
                    # Get driver's historical performance at this circuit with stronger scaling
                    performance_factor = self._get_driver_circuit_factor(driver_id, circuit_name)
                    # Apply the factor with squaring to amplify differences
                    performance_factor = performance_factor ** 2  # Square to amplify
                    # Apply the amplification
                    action_probs[idx] = action_probs[idx] * performance_factor
            
            # Re-normalize after amplification
            action_probs = action_probs / np.sum(action_probs)
            
            # Apply additional scaling to further differentiate top drivers
            # Add a power function that scales faster for higher probabilities
            action_probs = np.power(action_probs, 2)
            action_probs = action_probs / np.sum(action_probs)
            
            # Get top 5 predictions
            top5_indices = np.argsort(action_probs)[-5:][::-1]
            
            predictions = {
                'circuit': circuit_name,
                'prediction_time': datetime.now().isoformat(),
                'top5': [],
                'confidence_score': float(confidence)
            }
            
            for i, idx in enumerate(top5_indices, 1):
                if idx >= len(driver_ids):
                    continue
                    
                driver_id = driver_ids[idx]
                driver_code = self.data_processor.driver_mapping.get_driver_code(driver_id)
                
                if not driver_code:
                    continue
                    
                # Get driver's circuit-specific stats
                driver_circuit_quals = circuit_quals[circuit_quals['driverId'] == driver_id]
                
                prediction = {
                    'position': i,
                    'driver_code': driver_code,
                    'probability': float(action_probs[idx]),
                    'circuit_stats': {
                        'q3_appearances': len(driver_circuit_quals[driver_circuit_quals['q3'].notna()]),
                        'best_position': int(driver_circuit_quals['qualifyId'].count()),
                        'last_result': self._get_last_result(driver_circuit_quals)
                    }
                }
                
                predictions['top5'].append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error formatting predictions: {e}", exc_info=True)
            return {"error": str(e)}
      
    def _get_driver_circuit_factor(self, driver_id: int, circuit_name: str) -> float:
        """Calculate a circuit-specific driver performance factor for the given tracks"""
        try:
            # Get qualifying data for this circuit
            qual_data = self.data_processor.kaggle_data['qualifying']
            circuit_quals = qual_data[
                qual_data['raceId'].isin(
                    self.data_processor.race_data[
                        self.data_processor.race_data['name'].str.contains(circuit_name, case=False)
                    ]['raceId']
                )
            ]
            
            driver_quals = circuit_quals[circuit_quals['driverId'] == driver_id]
            
            # Get driver code for special handling
            driver_code = self.data_processor.driver_mapping.get_driver_code(driver_id)
            
            # Circuit-specific driver boosts based on your provided list
            circuit_specific_boosts = {
                # Red Bull
                'VER': {
                    'dutch': 1.8,        # Home race, historically strong
                    'austria': 1.7,      # Red Bull Ring, dominant
                    'belgium': 1.6,      # Excellent at Spa
                    'bahrain': 1.3,      # Consistently strong
                    'australian': 1.2,   # Decent in Australia
                    'miami': 1.3,        # Good at Miami
                    'british': 1.2,      # Good at Silverstone
                    'canadian': 1.2,     # Decent in Canada
                    'monaco': 0.8,       # Not his strongest historically
                    'azerbaijan': 1.1,   # Baku, not his strongest
                    'saudi': 1.2,        # Decent at Saudi
                    'united': 1.3,       # Good at COTA
                    'singapore': 0.9,    # Street circuit, not his traditional strength
                    'italian': 1.3,      # Good at Monza
                    'japanese': 1.4,     # Strong at Suzuka
                    'mexico': 1.5,       # High altitude favors Red Bull
                    'spain': 1.3,        # Good in Spain
                    'hungary': 1.0,      # Average here
                    'chinese': 1.3,      # Historically good
                    'emilia': 1.4,       # Strong at Imola
                },
                'PER': {
                    'mexico': 1.6,       # Home race advantage
                    'monaco': 1.6,       # Excellent at Monaco
                    'azerbaijan': 1.7,   # Very strong at Baku
                    'saudi': 1.3,        # Strong at Saudi
                    'singapore': 1.5,    # Good at street circuits
                    'miami': 1.2,        # Good at Miami
                    'canadian': 1.2,     # Good in Canada
                    'austria': 1.1,      # Decent at Red Bull Ring
                    'bahrain': 1.1,      # Good performances
                    'belgian': 1.1,      # Average at Spa
                    'australian': 1.0,   # Average in Australia
                    'british': 1.0,      # Average at Silverstone
                    'dutch': 0.9,        # Not his strongest
                    'united': 1.1,       # Decent at COTA
                    'italian': 1.0,      # Average at Monza
                    'japanese': 1.1,     # Decent at Suzuka
                    'spain': 1.0,        # Average in Spain
                    'hungary': 1.0,      # Average here
                    'chinese': 1.1,      # Average in China
                    'emilia': 1.0,       # Average at Imola
                },
                
                # Ferrari
                'LEC': {
                    'monaco': 1.8,       # Home race, usually fast
                    'azerbaijan': 1.6,   # Very strong at Baku
                    'italian': 1.5,      # Strong at Ferrari's home race
                    'singapore': 1.7,    # Excellent at street circuits
                    'australian': 1.3,   # Strong in Australia
                    'bahrain': 1.3,      # Historically strong
                    'belgium': 1.3,      # Good at Spa
                    'saudi': 1.3,        # Strong at Saudi
                    'british': 1.2,      # Good at British GP
                    'miami': 1.2,        # Good at Miami
                    'canadian': 1.2,     # Good in Canada
                    'japanese': 1.2,     # Good at Suzuka
                    'austria': 1.2,      # Good at Red Bull Ring
                    'hungary': 1.2,      # Good at technical tracks
                    'dutch': 1.1,        # Decent at Zandvoort
                    'united': 1.2,       # Good at COTA
                    'spain': 1.1,        # Decent in Spain
                    'mexico': 1.1,       # Average here
                    'chinese': 1.2,      # Good in China
                    'emilia': 1.2,       # Good at Imola
                },
                'SAI': {
                    'spanish': 1.5,      # Home race advantage
                    'monaco': 1.5,       # Good at Monaco
                    'british': 1.3,      # Strong at British GP
                    'canadian': 1.3,     # Strong in Canada
                    'italian': 1.4,      # Strong at Ferrari's home race
                    'australian': 1.3,   # Strong in Australia
                    'singapore': 1.4,    # Good at street circuits
                    'hungary': 1.3,      # Strong at technical tracks
                    'azerbaijan': 1.2,   # Good at Baku
                    'saudi': 1.2,        # Good at Saudi
                    'japanese': 1.2,     # Good at Suzuka
                    'belgian': 1.2,      # Good at Spa
                    'dutch': 1.2,        # Good at Zandvoort
                    'austrian': 1.1,     # Decent at Red Bull Ring
                    'miami': 1.1,        # Decent at Miami
                    'united': 1.2,       # Good at COTA
                    'bahrain': 1.2,      # Good performances
                    'mexico': 1.2,       # Good here
                    'chinese': 1.1,      # Decent in China
                    'emilia': 1.2,       # Good at Imola
                },
                
                # Mercedes
                'HAM': {
                    'british': 1.8,      # Home race, historically dominant
                    'hungarian': 1.7,    # Historically dominant
                    'canadian': 1.6,     # Historically very strong
                    'chinese': 1.6,      # Historically dominant
                    'spain': 1.5,        # Historically strong
                    'singapore': 1.5,    # Strong at Singapore
                    'italian': 1.5,      # Strong at Monza
                    'belgian': 1.4,      # Excellent at Spa
                    'bahrain': 1.4,      # Historically strong
                    'japanese': 1.4,     # Strong at Suzuka
                    'australian': 1.5,   # Historically dominant
                    'saudi': 1.3,        # Strong at Saudi
                    'mexico': 1.2,       # Good here
                    'united': 1.4,       # Strong at COTA
                    'azerbaijan': 1.2,   # Good at Baku
                    'dutch': 1.1,        # Average at Zandvoort
                    'monaco': 1.1,       # Not his strongest recently
                    'austrian': 1.0,     # Average at Red Bull Ring
                    'miami': 1.2,        # Good at Miami
                    'emilia': 1.2,       # Good at Imola
                },
                'RUS': {
                    'british': 1.5,      # Home race advantage
                    'azerbaijan': 1.3,   # Strong at Baku
                    'bahrain': 1.3,      # Good performances
                    'belgian': 1.3,      # Good at Spa
                    'austrian': 1.2,     # Good at Red Bull Ring
                    'hungarian': 1.4,    # Good at technical tracks
                    'singapore': 1.3,    # Strong at street circuits
                    'australian': 1.3,   # Good in Australia
                    'miami': 1.2,        # Good at Miami
                    'italian': 1.2,      # Good at Monza
                    'japanese': 1.2,     # Good at Suzuka
                    'saudi': 1.2,        # Good at Saudi
                    'canadian': 1.2,     # Good in Canada
                    'spanish': 1.2,      # Good in Spain
                    'monaco': 1.2,       # Good at Monaco
                    'mexico': 1.2,       # Good here
                    'united': 1.3,       # Strong at COTA
                    'dutch': 1.1,        # Decent at Zandvoort
                    'chinese': 1.2,      # Good in China
                    'emilia': 1.1,       # Decent at Imola
                },
                
                # McLaren
                'NOR': {
                    'british': 1.5,      # Home race advantage
                    'italian': 1.4,      # Won here in 2021
                    'belgian': 1.4,      # Very good at Spa
                    'austrian': 1.4,     # Very good at Red Bull Ring
                    'dutch': 1.3,        # Strong at Zandvoort
                    'miami': 1.3,        # Strong at Miami
                    'japanese': 1.3,     # Strong at Suzuka
                    'monaco': 1.5,       # Strong at Monaco
                    'singapore': 1.4,    # Good at street circuits
                    'hungarian': 1.3,    # Good at technical tracks
                    'australian': 1.3,   # Strong in Australia
                    'azerbaijan': 1.3,   # Strong at Baku
                    'canadian': 1.3,     # Strong in Canada
                    'mexican': 1.2,      # Good here
                    'bahrain': 1.2,      # Good performances
                    'saudi': 1.2,        # Good at Saudi
                    'spanish': 1.2,      # Good in Spain
                    'united': 1.3,       # Strong at COTA
                    'chinese': 1.2,      # Good in China
                    'emilia': 1.3,       # Strong at Imola
                },
                'PIA': {
                    'australian': 1.5,   # Home race advantage
                    'italian': 1.3,      # Strong at Monza
                    'austrian': 1.3,     # Strong at Red Bull Ring
                    'monaco': 1.3,       # Good at Monaco
                    'singapore': 1.2,    # Good at street circuits
                    'hungarian': 1.2,    # Good at technical tracks
                    'dutch': 1.2,        # Good at Zandvoort
                    'miami': 1.2,        # Good at Miami
                    'british': 1.2,      # Good at British GP
                    'saudi': 1.2,        # Good at Saudi
                    'japanese': 1.2,     # Good at Suzuka
                    'belgian': 1.2,      # Good at Spa
                    'azerbaijan': 1.2,   # Good at Baku
                    'canadian': 1.2,     # Good in Canada
                    'mexican': 1.1,      # Decent here
                    'bahrain': 1.1,      # Decent performances
                    'spanish': 1.1,      # Decent in Spain
                    'united': 1.2,       # Good at COTA
                    'chinese': 1.1,      # Decent in China
                    'emilia': 1.1,       # Decent at Imola
                },
                
                # Others (key drivers)
                'ALO': {
                    'spanish': 1.7,      # Home race advantage
                    'monaco': 1.6,       # Excellent at Monaco
                    'canadian': 1.4,     # Very strong in Canada
                    'singapore': 1.5,    # Strong at Singapore
                    'hungarian': 1.4,    # Won here in the past
                    'azerbaijan': 1.3,   # Strong at Baku
                    'japanese': 1.3,     # Strong at Suzuka
                    'belgian': 1.3,      # Strong at Spa
                    'bahrain': 1.3,      # Good performances
                    'chinese': 1.3,      # Strong in China
                    'australian': 1.2,   # Good in Australia
                    'british': 1.2,      # Good at British GP
                    'italian': 1.2,      # Good at Monza
                    'saudi': 1.2,        # Good at Saudi
                    'mexican': 1.1,      # Decent here
                    'dutch': 1.1,        # Decent at Zandvoort
                    'united': 1.2,       # Good at COTA
                    'miami': 1.1,        # Decent at Miami
                    'austrian': 1.0,     # Average at Red Bull Ring
                    'emilia': 1.2,       # Good at Imola
                },
                'HUL': {
                    'belgian': 1.3,      # Strong at Spa
                    'austrian': 1.2,     # Good at Red Bull Ring
                    'german': 1.5,       # Home race (when it happens)
                    'singapore': 1.1,    # Decent at street circuits
                    'monaco': 1.1,       # Decent at Monaco
                    'italian': 1.2,      # Good at Monza
                    'british': 1.1,      # Decent at British GP
                    'canadian': 1.1,     # Decent in Canada
                    'australian': 1.0,   # Average in Australia
                    'bahrain': 1.1,      # Decent performances
                    'chinese': 1.1,      # Decent in China
                    'united': 1.1,       # Decent at COTA
                    'mexican': 1.0,      # Average here
                    'brazilian': 1.1,    # Decent in Brazil
                    'japanese': 1.1,     # Decent at Suzuka
                    'azerbaijan': 1.0,   # Average at Baku
                    'saudi': 1.0,        # Average at Saudi
                    'dutch': 1.0,        # Average at Zandvoort
                    'spanish': 1.0,      # Average in Spain
                    'hungarian': 1.0,    # Average at technical tracks
                },
                'BOT': {
                    'austrian': 1.4,     # Very good at Red Bull Ring
                    'russian': 1.5,      # Historically strong
                    'italian': 1.3,      # Strong at Monza
                    'belgian': 1.2,      # Good at Spa
                    'australian': 1.3,   # Strong in Australia
                    'bahrain': 1.2,      # Good performances
                    'british': 1.2,      # Good at British GP
                    'canadian': 1.1,     # Decent in Canada
                    'chinese': 1.2,      # Good in China
                    'japanese': 1.1,     # Decent at Suzuka
                    'azerbaijani': 1.1,  # Decent at Baku
                    'dutch': 1.0,        # Average at Zandvoort
                    'monaco': 1.0,       # Average at Monaco
                    'singapore': 1.0,    # Average at street circuits
                    'saudi': 1.0,        # Average at Saudi
                    'mexican': 1.0,      # Average here
                    'united': 1.1,       # Decent at COTA
                    'spanish': 1.1,      # Decent in Spain
                    'hungarian': 1.2,    # Good at technical tracks
                    'miami': 1.0,        # Average at Miami
                }
            }
            
            # Map alternative circuit names
            normalized_name = circuit_name.lower().strip()
            circuit_mapping = {
                'silverstone': 'british',
                'monza': 'italian',
                'spa': 'belgian',
                'sochi': 'russian',
                'barcelona': 'spanish',
                'catalunya': 'spanish',
                'cota': 'united',
                'suzuka': 'japanese',
                'sakhir': 'bahrain',
                'sepang': 'malaysian',
                'imola': 'emilia',
                'baku': 'azerbaijan',
                'interlagos': 'brazilian',
                'zandvoort': 'dutch',
                'spielberg': 'austrian',
                'shanghai': 'chinese',
                'yas marina': 'abu',
                'circuit gilles villeneuve': 'canadian',
                'montreal': 'canadian',
                'melbourne': 'australian',
                'red bull ring': 'austrian',
                'circuit of the americas': 'united',
                'austin': 'united',
                'jeddah': 'saudi',
                'hungaroring': 'hungarian',
                'ciudad de mexico': 'mexican',
                'mexico city': 'mexican',
            }
            
            if normalized_name in circuit_mapping:
                normalized_name = circuit_mapping[normalized_name]
            
            # Apply circuit specific boost if available
            if driver_code in circuit_specific_boosts and normalized_name in circuit_specific_boosts[driver_code]:
                return circuit_specific_boosts[driver_code][normalized_name]
            
            # Default factor processing from historical data
            factor = 1.0
            
            if not driver_quals.empty:
                # Calculate factor based on qualifying positions
                if 'position' in driver_quals.columns:
                    positions = driver_quals['position'].tolist()
                    positions = [int(p) if isinstance(p, (int, float, str)) and str(p).isdigit() else 20 for p in positions]
                    avg_position = sum(positions) / len(positions) if positions else 10
                    
                    # Very aggressive non-linear scaling
                    if avg_position <= 3:
                        factor = 4.0 - (avg_position - 1) * 1.0  # P1→4.0, P3→2.0
                    elif avg_position <= 10:
                        factor = 2.0 - (avg_position - 3) * 0.14  # P4→1.86, P10→1.0
                    else:
                        factor = 1.0 - (avg_position - 10) * 0.09  # P11→0.91, P20→0.1
                        
                    # Apply larger bonus for proven track performance
                    consistency_bonus = min(1.5, len(positions) / 2) * 0.5  # Up to 75% bonus for 3+ races
                    factor = factor * (1 + consistency_bonus)
            
            # Return a minimum factor to avoid completely eliminating drivers
            return max(0.2, factor)
        except Exception as e:
            logger.error(f"Error calculating driver circuit factor: {e}")
            return 1.0
    
    def _get_last_result(self, driver_quals: pd.DataFrame) -> Optional[int]:
        """Get the driver's last qualifying result at this circuit."""
        if driver_quals.empty:
            return None
            
        # Sort by date and get most recent qualifying session
        driver_quals = driver_quals.sort_values('raceId', ascending=False)
        if len(driver_quals) > 0:
            last_qual = driver_quals.iloc[0]
            if pd.notna(last_qual['q3']):
                return 3
            elif pd.notna(last_qual['q2']):
                return 2
            elif pd.notna(last_qual['q1']):
                return 1
        return None

    def _process_driver_features(self, circuit_data: pd.DataFrame, circuit_name: str) -> np.ndarray:
        """Process driver features with circuit-specific considerations."""
        try:
            # Check cache first
            cache_key = f"driver_features_{circuit_name}"
            if cache_key in self._feature_cache:
                return self._feature_cache[cache_key]
                
            # Get current driver IDs
            current_driver_ids = self.data_processor.driver_mapping.get_current_driver_ids()
            
            # Initialize features for 20 drivers
            features = np.zeros(20)
            
            # Get qualifying data
            qual_data = self.data_processor.kaggle_data['qualifying']
            
            # Get circuit-specific qualifying data
            circuit_quals = qual_data[
                qual_data['raceId'].isin(
                    self.data_processor.race_data[
                        self.data_processor.race_data['name'].str.contains(circuit_name, case=False)
                    ]['raceId']
                )
            ]
            
            for idx, driver_id in enumerate(current_driver_ids):
                driver_quals = circuit_quals[circuit_quals['driverId'] == driver_id]
                
                if not driver_quals.empty:
                    # Calculate average Q3 performance
                    q3_times = []
                    for _, qual in driver_quals.iterrows():
                        if pd.notna(qual['q3']) and qual['q3'] != 'N/A':
                            time_secs = convert_time_to_seconds(qual['q3'])
                            if time_secs is not None:
                                q3_times.append(time_secs)
                    
                    if q3_times:
                        # Calculate performance score
                        best_time = min(q3_times)
                        avg_time = sum(q3_times) / len(q3_times)
                        consistency = 1 - (np.std(q3_times) / avg_time if len(q3_times) > 1 else 0)
                        
                        # Combine metrics
                        performance_score = (0.6 * (1 - (avg_time - best_time) / best_time) + 
                                          0.4 * consistency)
                        
                        features[idx] = performance_score
            
            # Normalize features
            if np.sum(features) > 0:
                features = features / np.max(features)
            
            # Cache the result
            self._feature_cache[cache_key] = features
            
            logger.info(f"Driver features range for {circuit_name}: {np.min(features):.3f} to {np.max(features):.3f}")
            return features
            
        except Exception as e:
            logger.error(f"Error processing driver features: {e}", exc_info=True)
            return np.zeros(20)

    def _process_constructor_features(self, constructor_info: pd.DataFrame) -> np.ndarray:
        """
        Process constructor features with proper normalization and team performance metrics.
        """
        try:
            # Initialize features for 10 constructors, 3 features each
            features = np.zeros(30)  # 10 constructors * 3 features
            
            if not constructor_info.empty:
                # Get max values for normalization
                max_points = constructor_info['constructor_points'].max()
                max_wins = constructor_info['wins'].max() if 'wins' in constructor_info else 0
                
                for idx, (_, row) in enumerate(constructor_info.iterrows()):
                    if idx >= 10:  # Limit to 10 constructors
                        break
                        
                    base_idx = idx * 3
                    
                    # 1. Normalized constructor points
                    points = row.get('constructor_points', 0)
                    features[base_idx] = points / max_points if max_points > 0 else 0
                    
                    # 2. Normalized position (inverse, so better positions give higher values)
                    position = row.get('position', 10)
                    features[base_idx + 1] = 1 - ((position - 1) / 9)  # Position is 1-10
                    
                    # 3. Normalized wins
                    wins = row.get('wins', 0)
                    features[base_idx + 2] = wins / max_wins if max_wins > 0 else 0
            
            logger.info(f"Constructor features range: {np.min(features):.3f} to {np.max(features):.3f}")
            return features
            
        except Exception as e:
            logger.error(f"Error processing constructor features: {e}", exc_info=True)
            return np.zeros(30)

    def _process_qualifying_features(self, circuit_data: pd.DataFrame, circuit_name: str) -> np.ndarray:
        """
        Process qualifying features with enhanced circuit-specific historical performance.
        """
        try:
            # Check cache first
            cache_key = f"qualifying_features_{circuit_name}"
            if cache_key in self._feature_cache:
                return self._feature_cache[cache_key]
                
            # Initialize features array (20 drivers * 3 qualifying sessions)
            features = np.zeros(60)
            
            # Get qualifying data for this specific circuit
            qual_data = self.data_processor.kaggle_data['qualifying']
            circuit_quals = qual_data[
                qual_data['raceId'].isin(
                    self.data_processor.race_data[
                        self.data_processor.race_data['name'].str.contains(circuit_name, case=False)
                    ]['raceId']
                )
            ]
            
            # Get current driver IDs
            current_driver_ids = self.data_processor.driver_mapping.get_current_driver_ids()
            
            # Process each driver's qualifying performance
            for idx, driver_id in enumerate(current_driver_ids):
                driver_quals = circuit_quals[circuit_quals['driverId'] == driver_id]
                
                if not driver_quals.empty:
                    base_idx = idx * 3
                    
                    # Get Q1, Q2, Q3 times for this driver at this circuit
                    for q_idx, session in enumerate(['q1', 'q2', 'q3']):
                        session_times = []
                        for _, qual in driver_quals.iterrows():
                            if pd.notna(qual[session]) and qual[session] != 'N/A':
                                time_secs = convert_time_to_seconds(qual[session])
                                if time_secs is not None:
                                    session_times.append(time_secs)
                        
                        if session_times:
                            # Calculate performance metrics
                            best_time = min(session_times)
                            avg_time = sum(session_times) / len(session_times)
                            consistency = 1 - (np.std(session_times) / avg_time if len(session_times) > 1 else 0)
                            
                            # Combine metrics into a single normalized score
                            performance_score = (0.5 * (1 - (avg_time - best_time) / best_time) + 
                                              0.3 * consistency +
                                              0.2 * (len(session_times) / len(driver_quals)))
                            
                            features[base_idx + q_idx] = performance_score
                
                else:
                    # If no qualifying data for this driver at this circuit,
                    # use their general qualifying performance at a reduced weight
                    general_quals = qual_data[qual_data['driverId'] == driver_id]
                    if not general_quals.empty:
                        base_idx = idx * 3
                        for q_idx, session in enumerate(['q1', 'q2', 'q3']):
                            q_times = []
                            for _, qual in general_quals.iterrows():
                                if pd.notna(qual[session]) and qual[session] != 'N/A':
                                    time_secs = convert_time_to_seconds(qual[session])
                                    if time_secs is not None:
                                        q_times.append(time_secs)
                            
                            if q_times:
                                # Use a reduced weight for general performance
                                performance_score = (1 - (min(q_times) - min(q_times)) / min(q_times)) * 0.5
                                features[base_idx + q_idx] = performance_score
            
            # Cache the result
            self._feature_cache[cache_key] = features
            
            logger.info(f"Qualifying features range for {circuit_name}: {np.min(features):.3f} to {np.max(features):.3f}")
            return features
            
        except Exception as e:
            logger.error(f"Error processing qualifying features for {circuit_name}: {e}", exc_info=True)
            return np.zeros(60)
    
    def _process_weather_features(self, race_info: pd.DataFrame) -> np.ndarray:
        """
        Enhanced weather feature processing with better validation.
        """
        try:
            logger.info("Processing weather features")
            
            if race_info.empty:
                logger.warning("Empty race info")
                return np.array([0.0])
                
            if 'weather_numerical' not in race_info.columns:
                logger.warning("No weather_numerical column found")
                return np.array([0.0])
                
            weather_values = race_info['weather_numerical'].unique()
            logger.debug(f"Unique weather values found: {weather_values}")
            
            # Take the most common weather value for this race
            weather_value = race_info['weather_numerical'].mode().iloc[0]
            
            return np.array([float(weather_value)])
            
        except Exception as e:
            logger.error(f"Error processing weather features: {e}", exc_info=True)
            return np.array([0.0])
    
    def _process_circuit_features(self, circuit_data: pd.DataFrame, circuit_name: str) -> np.ndarray:
        """
        Enhanced circuit feature processing with proper sector time handling.
        """
        try:
            # Initialize features array
            features = np.zeros(3)
            
            if circuit_data is None or circuit_data.empty:
                logger.warning(f"No circuit data available for {circuit_name}")
                return features
                
            # Check for sector time columns
            sector_cols = [col for col in circuit_data.columns if 'sector' in col.lower()]
            
            if len(sector_cols) >= 3:
                # Calculate mean sector times
                sector_means = [circuit_data[col].mean() for col in sector_cols[:3]]
                total_time = sum(sector_means)
                
                if total_time > 0:
                    # Calculate normalized sector proportions
                    features = np.array([time/total_time for time in sector_means])
                    return features
                else:
                    logger.warning(f"Total time is 0 for circuit {circuit_name}")
            else:
                logger.warning(f"Insufficient sector data for circuit {circuit_name}")
                
            return features
            
        except Exception as e:
            logger.error(f"Error processing circuit features for {circuit_name}: {e}", exc_info=True)
            return np.zeros(3)

    def evaluate(self, val_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model on validation data"""
        try:
            self.model.eval()
            val_states, val_targets = val_data
            
            with torch.no_grad():
                policy_pred, value_pred = self.model(val_states)
                
                policy_loss = F.kl_div(
                    F.log_softmax(policy_pred, dim=1),
                    val_targets.float(),
                    reduction='batchmean'
                )
                
                value_target = (torch.argmax(val_targets, dim=1).float() + 1) / 20.0
                value_loss = F.mse_loss(value_pred.squeeze(), value_target)
                
                total_loss = policy_loss + 0.5 * value_loss
            
            return {
                'val_policy_loss': policy_loss.item(),
                'val_value_loss': value_loss.item(),
                'val_total_loss': total_loss.item()
            }

        except Exception as e:
            logger.error(f"Error in evaluation: {e}", exc_info=True)
            raise

    def check_early_stopping(self, val_loss: float) -> bool:
        """Check if training should stop early"""
        if val_loss < self.history['best_val_loss'] - self.config.min_delta:
            self.history['best_val_loss'] = val_loss
            self.history['epochs_without_improvement'] = 0
            return False
        else:
            self.history['epochs_without_improvement'] += 1
            return self.history['epochs_without_improvement'] >= self.config.early_stopping_patience