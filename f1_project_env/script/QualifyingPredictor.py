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
        """Create complete state representation with enhanced error handling for Monaco."""
        try:
            logger.info(f"Creating state for circuit: {circuit_name}")
            
            if circuit_data is None or circuit_data.empty:
                raise ValueError("Invalid circuit data provided")

            # Get race ID for this circuit with better error handling for Monaco
            race_ids = self.data_processor.race_data[
                self.data_processor.race_data['name'].str.contains(circuit_name, case=False) |
                (circuit_name.lower() == 'monaco' and self.data_processor.race_data['name'].str.contains('monte carlo', case=False))
            ]['raceId'].tolist()

            if not race_ids:
                # Special case for Monaco
                if circuit_name.lower() == 'monaco':
                    # Try a generic approach with Monte Carlo
                    race_ids = self.data_processor.race_data[
                        self.data_processor.race_data['name'].str.contains('monte', case=False) |
                        self.data_processor.race_data['name'].str.contains('monaco', case=False)
                    ]['raceId'].tolist()
                    
                    if not race_ids:
                        # If still no results, use a fallback race ID
                        most_recent_race_id = self.data_processor.race_data['raceId'].max()
                        race_ids = [most_recent_race_id]
                        logger.warning(f"No specific race data found for Monaco, using fallback race ID: {most_recent_race_id}")
                else:
                    raise ValueError(f"No race data found for circuit: {circuit_name}")

            # Use the most recent race ID for this circuit
            race_id = max(race_ids)
            logger.info(f"Using race ID {race_id} for circuit: {circuit_name}")

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
        Predict qualifying with more balanced distributions.
        Modified to better reflect 2023-2024 F1 performance with more realistic spread.
        """
        try:
            logger.info(f"Predicting qualifying for circuit: {circuit_name}")
            
            # Get all available circuit keys
            available_circuits = sorted(self.data_processor.circuits_data.keys())
            
            # Convert to standardized name
            standardized_name = self.get_standardized_circuit_name(circuit_name)
            logger.info(f"Standardized circuit name: '{circuit_name}' -> '{standardized_name}'")
            
            # Get circuit data with standardized name
            circuit_data = self.data_processor.circuits_data.get(standardized_name.lower())
            
            # If not found directly, try to find partial matches
            if circuit_data is None:
                # First try more advanced partial matching
                best_match = None
                best_score = 0
                
                for key in available_circuits:
                    # Simple matching score based on shared characters
                    # This could be improved with more sophisticated string matching algorithms
                    circuit_lower = circuit_name.lower()
                    key_lower = key.lower()
                    
                    # Check for direct substring matches first
                    if key_lower in circuit_lower or circuit_lower in key_lower:
                        score = min(len(key_lower), len(circuit_lower))
                        if score > best_score:
                            best_score = score
                            best_match = key
                            
                    # Otherwise calculate character overlap
                    else:
                        # Count common characters (simple approximation)
                        common_chars = set(circuit_lower) & set(key_lower)
                        score = len(common_chars)
                        if score > best_score:
                            best_score = score
                            best_match = key
                
                if best_match and best_score > 3:  # Threshold to avoid spurious matches
                    logger.info(f"Found alternative match: '{best_match}' (score: {best_score})")
                    standardized_name = best_match
                    circuit_data = self.data_processor.circuits_data[best_match]
            
            if circuit_data is None:
                # Log available circuits for diagnostics
                logger.error(f"No data found for circuit: {circuit_name} (standardized: {standardized_name})")
                logger.error(f"Available circuits: {available_circuits}")
                return {"error": f"No data found for circuit: {circuit_name}. Available circuits: {', '.join(available_circuits[:5])}..."}
            
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
                
                # REBALANCED: More moderate temperature scaling
                base_temperature = 0.2  # Increased from 0.08 to 0.2 for less peaked distribution
                circuit_scaling = self._get_circuit_scaling(circuit_name)
                temperature = base_temperature / circuit_scaling
                
                # Apply temperature scaling directly to logits before softmax
                scaled_policy = policy / temperature
                action_probs = F.softmax(scaled_policy, dim=1).cpu().numpy()[0]
                
                # Add position-biased noise to favor better positions
                position_bias = np.array([(20-i)/20 for i in range(20)]) * 0.03  # Reduced from 0.05
                action_probs = action_probs + position_bias

                # Apply updated team recency bias
                action_probs = self._apply_team_recency_bias(action_probs)
                
                circuit_type = self._get_circuit_type(circuit_name)
                driver_ids = self.data_processor.driver_mapping.get_current_driver_ids()

                # REBALANCED: Circuit type preferences with more balanced distribution
                for idx, driver_id in enumerate(driver_ids):
                    if idx < len(action_probs):
                        driver_code = self.data_processor.driver_mapping.get_driver_code(driver_id)
                        
                        # More balanced circuit type preferences
                        driver_circuit_preferences = {
                            # Red Bull
                            'VER': {'high_speed': 1.3, 'technical': 1.2, 'street': 1.1},
                            'PER': {'high_speed': 0.9, 'technical': 0.9, 'street': 1.3},
                            
                            # Ferrari
                            'LEC': {'high_speed': 1.1, 'technical': 1.2, 'street': 1.3},
                            'SAI': {'high_speed': 1.2, 'technical': 1.2, 'street': 1.1},
                            
                            # Mercedes
                            'HAM': {'high_speed': 1.1, 'technical': 1.2, 'street': 1.0},
                            'RUS': {'high_speed': 1.1, 'technical': 1.2, 'street': 1.1},
                            
                            # McLaren
                            'NOR': {'high_speed': 1.3, 'technical': 1.2, 'street': 1.1},
                            'PIA': {'high_speed': 1.2, 'technical': 1.1, 'street': 1.1},
                        }
                        
                        if driver_code in driver_circuit_preferences and circuit_type in driver_circuit_preferences[driver_code]:
                            type_factor = driver_circuit_preferences[driver_code][circuit_type]
                            action_probs[idx] *= type_factor

                # Re-normalize
                action_probs = action_probs / np.sum(action_probs)
                
                confidence = (value.item() + 1) / 2
            
            # Apply Constructor Weighting
            constructor_strengths = self._get_constructor_strengths()
            for idx, driver_id in enumerate(self.data_processor.driver_mapping.get_current_driver_ids()):
                if idx < len(action_probs):
                    constructor_id = self._get_driver_constructor(driver_id)
                    if constructor_id is not None:
                        constructor_weight = constructor_strengths[constructor_id]
                        action_probs[idx] *= constructor_weight
            
            # Re-normalize
            action_probs = action_probs / np.sum(action_probs)
            
            # Apply Recent Form factors
            recent_form_factors = self._calculate_recent_form()
            for idx, driver_id in enumerate(self.data_processor.driver_mapping.get_current_driver_ids()):
                if idx < len(action_probs):
                    form_factor = recent_form_factors[driver_id]
                    action_probs[idx] *= form_factor
            
            # Re-normalize
            action_probs = action_probs / np.sum(action_probs)
            
            # REBALANCED: Reduced confidence amplifier
            confidence_factor = confidence * 2  # Reduced from 4 to 2
            confidence_adjusted_probs = np.power(action_probs, confidence_factor)
            
            # Re-normalize
            action_probs = confidence_adjusted_probs / np.sum(confidence_adjusted_probs)
            
            # REBALANCED: Less extreme amplification
            skewed_probs = np.power(action_probs, 1.5)  # Reduced from 3 to 1.5
            action_probs = skewed_probs / np.sum(skewed_probs)
            
            # REBALANCED: Driver-specific 2023-2024 performance boost
            driver_2024_boost = {
                'VER': 1.3,  # Max Verstappen
                'NOR': 1.25, # Lando Norris
                'PIA': 1.15, # Oscar Piastri
                'LEC': 1.2,  # Charles Leclerc
                'SAI': 1.15, # Carlos Sainz
                'HAM': 1.05, # Lewis Hamilton
                'RUS': 1.1,  # George Russell
                'PER': 0.95, # Sergio Perez
                'ALO': 1.05, # Fernando Alonso
                'ALB': 1.05, # Alex Albon (Williams performing better)
            }
            
            for idx, driver_id in enumerate(driver_ids):
                if idx < len(action_probs):
                    driver_code = self.data_processor.driver_mapping.get_driver_code(driver_id)
                    if driver_code in driver_2024_boost:
                        action_probs[idx] *= driver_2024_boost[driver_code]
            
            # Re-normalize
            action_probs = action_probs / np.sum(action_probs)
            
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
        """Calculate balanced strength factors for each constructor based on 2023-2024 performance"""
        # REBALANCED: More realistic constructor strength distribution
        constructor_strengths_2024 = {
            1: 1.3,  # Red Bull Racing
            2: 1.2,  # Ferrari
            3: 1.1,  # Mercedes
            4: 1.2, # McLaren
            5: 1.0,  # Aston Martin
            6: 0.9,  # Alpine
            7: 0.95, # Williams
            8: 0.9,  # RB (AlphaTauri)
            9: 0.85, # Sauber (Alfa Romeo)
            10: 0.9, # Haas
        }
        
        # Create defaultdict with these values
        return defaultdict(lambda: 1.0, constructor_strengths_2024)
    
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
            
            # NEW: Apply time-weighted Q3 performance boost
            q3_performance_boost = 1.2  # Base configurable parameter
            max_years_relevance = 5     # How many years back to consider Q3 performances

            for driver_idx, driver_id in enumerate(driver_ids):
                if driver_idx < len(action_probs):
                    # Get driver's Q3 record at this circuit
                    driver_quals = circuit_quals[circuit_quals['driverId'] == driver_id]
                    
                    if not driver_quals.empty and 'raceId' in driver_quals.columns:
                        # Sort by race ID (assuming higher race IDs are more recent)
                        driver_quals = driver_quals.sort_values('raceId', ascending=False)
                        
                        # Get year information if available, otherwise use race ID as proxy
                        if 'year' in driver_quals.columns:
                            driver_quals['years_ago'] = datetime.now().year - driver_quals['year']
                        else:
                            # Estimate years based on race ID differences
                            most_recent_race_id = self.data_processor.race_data['raceId'].max()
                            avg_races_per_year = 22  # Average F1 races per year
                            driver_quals['years_ago'] = (most_recent_race_id - driver_quals['raceId']) / avg_races_per_year
                            
                        # Filter to only races within the relevance window
                        relevant_quals = driver_quals[driver_quals['years_ago'] <= max_years_relevance]
                        
                        # Calculate time-weighted Q3 factor
                        q3_factor = 1.0
                        
                        if not relevant_quals.empty:
                            total_weight = 0
                            weighted_q3_score = 0
                            
                            for _, qual_row in relevant_quals.iterrows():
                                # Calculate time decay factor (1.0 for current year, decreasing for older years)
                                years_ago = qual_row['years_ago']
                                time_weight = self._calculate_time_weight(years_ago, max_years_relevance)
                                
                                # Check if driver made Q3
                                if pd.notna(qual_row['q3']):
                                    # If position data is available, factor in their Q3 performance
                                    q3_performance = 1.0
                                    if 'position' in qual_row and pd.notna(qual_row['position']):
                                        try:
                                            position = int(qual_row['position'])
                                            # Higher boost for better positions (P1-P5)
                                            if position <= 5:
                                                q3_performance = 1.4 - ((position - 1) * 0.08)  # P1: 1.4, P5: 1.08
                                            elif position <= 10:
                                                q3_performance = 1.0
                                        except (ValueError, TypeError):
                                            pass
                                            
                                    # Add this Q3 appearance to the weighted score
                                    weighted_q3_score += time_weight * q3_performance
                                    
                                total_weight += time_weight
                            
                            # Calculate final Q3 factor, normalizing by total weight
                            if total_weight > 0:
                                normalized_q3_score = weighted_q3_score / total_weight
                                q3_factor = 1.0 + (normalized_q3_score * q3_performance_boost * 0.1)
                        
                        # Apply the calculated factor
                        action_probs[driver_idx] *= q3_factor
                        
                        # Log significant Q3 boosts for debugging
                        if q3_factor > 1.2:
                            driver_code = self.data_processor.driver_mapping.get_driver_code(driver_id)
                            recent_q3s = len(relevant_quals[relevant_quals['q3'].notna()])
                            logger.info(f"Applied time-weighted Q3 boost of {q3_factor:.2f} to {driver_code} at {circuit_name} (recent Q3s: {recent_q3s})")
            
            # Re-normalize after Q3 boost
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
                'confidence_score': float(confidence),
                'q3_influence': q3_performance_boost  # Add this to track Q3 influence level
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
                q3_appearances = len(driver_circuit_quals[driver_circuit_quals['q3'].notna()])
                
                # Get Q3 stats for more detailed output
                q3_stats = {}
                if q3_appearances > 0:
                    # Get Q3 times if available
                    q3_times = []
                    for _, qual in driver_circuit_quals.iterrows():
                        if pd.notna(qual['q3']) and qual['q3'] != 'N/A':
                            time_secs = convert_time_to_seconds(qual['q3'])
                            if time_secs is not None:
                                q3_times.append(time_secs)
                    
                    if q3_times:
                        q3_stats['avg_time'] = sum(q3_times) / len(q3_times)
                        q3_stats['best_time'] = min(q3_times)
                        q3_stats['consistency'] = 1 - (np.std(q3_times) / q3_stats['avg_time'] if len(q3_times) > 1 else 0)
                
                prediction = {
                    'position': i,
                    'driver_code': driver_code,
                    'probability': float(action_probs[idx]),
                    'circuit_stats': {
                        'q3_appearances': q3_appearances,
                        'best_position': self._get_best_position(driver_circuit_quals),
                        'last_result': self._get_last_result(driver_circuit_quals),
                        'q3_stats': q3_stats
                    }
                }
                
                predictions['top5'].append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error formatting predictions: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _calculate_time_weight(self, years_ago: float, max_years_relevance: float) -> float:
        """
        Calculate time weight with exponential decay rather than linear.
        This gives much higher weight to very recent results.
        
        Args:
            years_ago: How many years ago the result occurred
            max_years_relevance: Maximum years to consider
            
        Returns:
            Weight factor between 0.05 and 1.0
        """
        # Exponential decay with half-life of ~1.5 years
        # This means results from 1.5 years ago are worth half as much as current results
        decay_factor = 0.5  # Controls how quickly the influence drops off
        
        # Exponential decay formula: weight = base_weight * e^(-decay_factor * years_ago)
        # Normalized so most recent result = 1.0
        weight = np.exp(-decay_factor * years_ago)
        
        # Set minimum weight to 0.05 (instead of 0.2 in original)
        return max(0.05, weight)
    
    def _apply_team_recency_bias(self, action_probs: np.ndarray) -> np.ndarray:
        """
        Apply a more balanced recency bias factor at the team level based on 2023-2024 form.
        Creates a more realistic spread among teams without any single team dominating.
        
        Args:
            action_probs: Current probability array
            
        Returns:
            Updated probability array with team recency bias applied
        """
        # Map constructor IDs to their drivers
        constructor_to_drivers = defaultdict(list)
        driver_to_position = {}  # For accessing positions by driver ID
        
        for idx, driver_id in enumerate(self.data_processor.driver_mapping.get_current_driver_ids()):
            if idx < len(action_probs):
                constructor_id = self._get_driver_constructor(driver_id)
                if constructor_id:
                    constructor_to_drivers[constructor_id].append(driver_id)
                    driver_to_position[driver_id] = idx
        
        # REBALANCED: 2023-2024 Team Performance Factors - More balanced distribution
        # The top 4 teams are now closer together
        team_performance_2024 = {
            1: 1.3,  # Red Bull Racing
            2: 1.2,  # Ferrari
            3: 1.1,  # Mercedes
            4: 1.2, # McLaren 
            5: 1.0,  # Aston Martin
            6: 0.9,  # Alpine
            7: 0.95, # Williams (slight improvement)
            8: 0.9,  # RB (AlphaTauri)
            9: 0.85, # Sauber (Alfa Romeo)
            10: 0.9, # Haas
        }
        
        # Apply team recency bias to each driver
        for constructor_id, driver_ids in constructor_to_drivers.items():
            bias_factor = team_performance_2024.get(constructor_id, 1.0)
            for driver_id in driver_ids:
                idx = driver_to_position.get(driver_id)
                if idx is not None:
                    action_probs[idx] *= bias_factor
        
        # Log strong team performances
        strong_teams = {c_id: factor for c_id, factor in team_performance_2024.items() if factor > 1.1}
        if strong_teams:
            logger.info(f"Applied team recency bias to {len(strong_teams)} teams: {strong_teams}")
        
        # Re-normalize
        return action_probs / np.sum(action_probs)

    def _get_best_position(self, driver_quals: pd.DataFrame) -> int:
        """Get the driver's best qualifying position at this circuit."""
        if driver_quals.empty:
            return 20
            
        if 'position' in driver_quals.columns:
            positions = driver_quals['position'].tolist()
            positions = [int(p) if isinstance(p, (int, float, str)) and str(p).isdigit() else 20 for p in positions]
            return min(positions) if positions else 20
        
        return 20
      
    def _get_driver_circuit_factor(self, driver_id: int, circuit_name: str) -> float:
        """Calculate a circuit-specific driver performance factor with balanced weighting"""
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
            
            # REBALANCED: Circuit-specific driver boosts with more realistic distribution
            circuit_specific_boosts = {
                # Red Bull - Strong but more balanced
                'VER': {
                    'dutch': 1.5,        # Home race, historically strong
                    'austria': 1.5,      # Red Bull Ring, strong
                    'belgium': 1.4,      # Excellent at Spa
                    'bahrain': 1.4,      # Strong opener
                    'australian': 1.3,   # Good in Australia
                    'miami': 1.3,        # Good at Miami
                    'british': 1.3,      # Good at Silverstone
                    'canadian': 1.3,     # Good in Canada
                    'monaco': 1.2,       # Improved in recent years
                    'azerbaijan': 1.2,   # Good at Baku
                    'saudi': 1.3,        # Good at Saudi
                    'united': 1.4,       # Strong at COTA
                    'singapore': 1.2,    # Improved at Singapore
                    'italian': 1.4,      # Strong at Monza
                    'japanese': 1.4,     # Strong at Suzuka
                    'mexico': 1.4,       # Strong at high altitude
                    'spain': 1.3,        # Good in Spain
                    'hungary': 1.3,      # Good at Hungaroring
                    'chinese': 1.3,      # Good in China
                    'emilia': 1.3,       # Good at Imola
                },
                'PER': {
                    'mexico': 1.3,       # Home race advantage
                    'monaco': 1.2,       # Good but inconsistent at Monaco
                    'azerbaijan': 1.3,   # Strong at Baku
                    'saudi': 1.3,        # Strong at Saudi
                    'singapore': 1.1,    # Not as strong in recent years
                    'miami': 1.0,        # Average at Miami
                    'canadian': 1.0,     # Average in Canada
                    'austria': 0.9,      # Below average at Red Bull Ring
                    'bahrain': 1.0,      # Average performances
                    'belgian': 0.9,      # Average at Spa
                    'australian': 0.9,   # Struggles in Australia
                    'british': 0.9,      # Struggles at Silverstone
                    'dutch': 0.9,        # Consistently underperforms
                    'united': 0.9,       # Average at COTA
                    'italian': 0.9,      # Below average at Monza
                    'japanese': 0.9,     # Average at Suzuka
                    'spain': 0.9,        # Below average in Spain
                    'hungary': 0.9,      # Struggles at Hungaroring
                    'chinese': 0.9,      # Average in China
                    'emilia': 0.9,       # Below average at Imola
                },
                
                # Ferrari - Consistent challenger
                'LEC': {
                    'monaco': 1.5,       # Home race, usually fast (pole positions)
                    'azerbaijan': 1.4,   # Very good at Baku
                    'italian': 1.3,      # Good at Ferrari's home race
                    'singapore': 1.4,    # Strong at street circuits
                    'australian': 1.3,   # Good in Australia
                    'bahrain': 1.3,      # Historically good
                    'belgium': 1.3,      # Good at Spa
                    'saudi': 1.3,        # Good at Saudi
                    'british': 1.2,      # Good at British GP
                    'miami': 1.2,        # Good at Miami
                    'canadian': 1.2,     # Good in Canada
                    'japanese': 1.2,     # Good at Suzuka
                    'austria': 1.2,      # Good at Red Bull Ring
                    'hungary': 1.2,      # Good at technical tracks
                    'dutch': 1.1,        # Decent at Zandvoort
                    'united': 1.2,       # Good at COTA
                    'spain': 1.2,        # Good in Spain
                    'mexico': 1.1,       # Average here
                    'chinese': 1.2,      # Good in China
                    'emilia': 1.2,       # Good at Imola
                },
                'SAI': {
                    'spanish': 1.3,      # Home race advantage
                    'monaco': 1.3,       # Good at Monaco
                    'british': 1.2,      # Good at British GP
                    'canadian': 1.2,     # Good in Canada
                    'italian': 1.3,      # Strong at Ferrari's home race
                    'australian': 1.3,   # Strong in Australia (won in 2024)
                    'singapore': 1.3,    # Good at street circuits
                    'hungary': 1.2,      # Good at technical tracks
                    'azerbaijan': 1.2,   # Good at Baku
                    'saudi': 1.3,        # Strong at Saudi
                    'japanese': 1.1,     # Good at Suzuka
                    'belgian': 1.2,      # Good at Spa
                    'dutch': 1.2,        # Good at Zandvoort
                    'austrian': 1.1,     # Decent at Red Bull Ring
                    'miami': 1.2,        # Good at Miami
                    'united': 1.2,       # Good at COTA
                    'bahrain': 1.2,      # Good performances
                    'mexico': 1.1,       # Good here
                    'chinese': 1.2,      # Good in China
                    'emilia': 1.2,       # Good at Imola
                },
                
                # Mercedes - Competitive but not dominant
                'HAM': {
                    'british': 1.4,      # Home race, still strong
                    'hungarian': 1.3,    # Historically strong
                    'canadian': 1.2,     # Historically strong
                    'chinese': 1.2,      # Good performance
                    'spain': 1.2,        # Solid performances
                    'singapore': 1.1,    # Average at Singapore recently
                    'italian': 1.1,      # Average at Monza
                    'belgian': 1.1,      # Average at Spa
                    'bahrain': 1.1,      # Average performances recently
                    'japanese': 1.1,     # Above average at Suzuka
                    'australian': 1.1,   # Above average in Australia
                    'saudi': 1.1,        # Average at Saudi
                    'mexico': 1.1,       # Average here
                    'united': 1.2,       # Above average at COTA
                    'azerbaijan': 1.1,   # Average at Baku
                    'dutch': 1.0,        # Average at Zandvoort
                    'monaco': 1.0,       # Not his strongest recently
                    'austrian': 1.0,     # Average at Red Bull Ring
                    'miami': 1.0,        # Average at Miami
                    'emilia': 1.0,       # Average at Imola
                },
                'RUS': {
                    'british': 1.3,      # Home race advantage
                    'azerbaijan': 1.3,   # Strong at Baku
                    'bahrain': 1.2,      # Good performances
                    'belgian': 1.2,      # Good at Spa
                    'austrian': 1.2,     # Good at Red Bull Ring
                    'hungarian': 1.3,    # Strong at technical tracks
                    'singapore': 1.2,    # Strong at street circuits
                    'australian': 1.2,   # Good in Australia
                    'miami': 1.2,        # Good at Miami
                    'italian': 1.2,      # Good at Monza
                    'japanese': 1.2,     # Good at Suzuka
                    'saudi': 1.2,        # Good at Saudi
                    'canadian': 1.2,     # Good in Canada
                    'spanish': 1.2,      # Good in Spain
                    'monaco': 1.1,      # Good at Monaco
                    'mexico': 1.1,      # Good here
                    'united': 1.2,      # Good at COTA
                    'dutch': 1.2,       # Good at Zandvoort
                    'chinese': 1.2,     # Good in China
                    'emilia': 1.1,      # Good at Imola
                },
                
                # McLaren - Strong contender but not overwhelmingly dominant
                'NOR': {
                    'british': 1.3,      # Home race advantage
                    'italian': 1.3,      # Strong at Monza
                    'belgian': 1.3,      # Strong at Spa
                    'austrian': 1.3,     # Strong at Red Bull Ring
                    'dutch': 1.2,        # Good at Zandvoort
                    'miami': 1.2,        # Good at Miami
                    'japanese': 1.3,     # Strong at Suzuka
                    'monaco': 1.3,       # Strong at Monaco
                    'singapore': 1.3,    # Good at street circuits
                    'hungarian': 1.3,    # Good at technical tracks
                    'australian': 1.3,   # Good in Australia
                    'azerbaijan': 1.2,   # Good at Baku
                    'canadian': 1.3,     # Good in Canada
                    'mexican': 1.2,      # Good in Mexico
                    'bahrain': 1.3,      # Good performances
                    'saudi': 1.3,        # Good at Saudi
                    'spanish': 1.3,      # Good in Spain
                    'united': 1.4,       # Strong at COTA
                    'chinese': 1.3,      # Good in China
                    'emilia': 1.3,       # Good at Imola
                },
                'PIA': {
                    'australian': 1.3,   # Home race advantage
                    'italian': 1.2,      # Good at Monza
                    'austrian': 1.3,     # Good at Red Bull Ring
                    'monaco': 1.2,       # Good at Monaco
                    'singapore': 1.2,    # Good at street circuits
                    'hungarian': 1.2,    # Good at technical tracks
                    'dutch': 1.2,        # Good at Zandvoort
                    'miami': 1.2,        # Good at Miami
                    'british': 1.2,      # Good at British GP
                    'saudi': 1.2,        # Good at Saudi
                    'japanese': 1.2,     # Good at Suzuka
                    'belgian': 1.2,      # Good at Spa
                    'azerbaijan': 1.1,   # Good at Baku
                    'canadian': 1.1,     # Good in Canada
                    'mexican': 1.1,      # Good here
                    'bahrain': 1.1,      # Good performances
                    'spanish': 1.1,      # Good in Spain
                    'united': 1.1,       # Good at COTA
                    'chinese': 1.1,      # Good in China
                    'emilia': 1.1,       # Good at Imola
                }
            }
            
            # Handle different naming conventions
            normalized_name = circuit_name.lower().strip()
            if 'mexico' in normalized_name:
                return circuit_specific_boosts.get(driver_code, {}).get('mexican', 1.0)
            elif 'hungary' in normalized_name:
                return circuit_specific_boosts.get(driver_code, {}).get('hungarian', 1.0)
            elif 'italy' in normalized_name or 'monza' in normalized_name:
                return circuit_specific_boosts.get(driver_code, {}).get('italian', 1.0)
            elif 'us' in normalized_name or 'cota' in normalized_name:
                return circuit_specific_boosts.get(driver_code, {}).get('united', 1.0)
            elif 'imola' in normalized_name:
                return circuit_specific_boosts.get(driver_code, {}).get('emilia', 1.0)
            
            # For any other circuits or if name doesn't match exactly
            for circuit_key in circuit_specific_boosts.get(driver_code, {}):
                if circuit_key in normalized_name:
                    return circuit_specific_boosts[driver_code][circuit_key]
            
            # Default value if no match is found
            return 1.0
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
        
    def get_standardized_circuit_name(self, circuit_name: str) -> str:
        """Map file names to standardized circuit names in the database"""
        
        # Create a mapping dictionary for circuit names
        circuit_mapping = {
            # Original mappings
            'emilia romagna': 'emilia',
            'mexico city': 'mexico',
            'monaco': 'monaco',      # Changed from 'monaco gp' to 'monte' to match the circuits_data
            'saudi arabian': 'saudi',
            'united states': 'us',  # Changed from 'us' to match the circuits_data keys
            
            # Add additional mappings for possible variations
            'bahrain': 'bahrain',
            'jeddah': 'saudi',
            'baku': 'azerbaijan', 
            'melbourne': 'australian',
            'albert park': 'australian',
            'catalunya': 'spanish',
            'barcelona': 'spanish',
            'monte carlo': 'monaco',
            'silverstone': 'british',
            'spielberg': 'austrian',
            'red bull ring': 'austrian',
            'hungaroring': 'hungarian',
            'spa': 'belgian',
            'monza': 'italian',
            'marina bay': 'singapore',
            'suzuka': 'japanese',
            'losail': 'qatar',
            'cota': 'us',
            'circuit of the americas': 'us',
            'autodromo hermanos rodriguez': 'mexican',
            'interlagos': 'brazilian',
            'jose carlos pace': 'brazilian',
            'yas marina': 'abu',
            'miami international': 'miami',
            'zandvoort': 'dutch',
            'las vegas': 'vegas'
        }
        
        # Convert to lowercase for case-insensitive matching
        circuit_name_lower = circuit_name.lower()
        
        # First try exact match
        if circuit_name_lower in circuit_mapping:
            return circuit_mapping[circuit_name_lower]
            
        # Then try partial match
        for key, value in circuit_mapping.items():
            if key in circuit_name_lower or circuit_name_lower in key:
                return value
                
        # If no match found, check if it's already a standard name by looking in circuits_data keys
        if circuit_name_lower in self.data_processor.circuits_data:
            return circuit_name_lower
                
        # Return the original if no mapping found
        return circuit_name_lower
    
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