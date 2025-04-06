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
    learning_rate: float = 0.001
    batch_size: int = 32
    model_save_dir: str = "models"
    cache_dir: str = "cache"
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    min_delta: float = 0.001

class QualifyingPredictor:
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
        
        Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'epochs_without_improvement': 0
        }
        
        self._feature_cache = {}

    def _create_state(self, circuit_data: pd.DataFrame, circuit_name: str) -> np.ndarray:
        try:
            logger.info(f"Creating state for circuit: {circuit_name}")
            
            if circuit_data is None or circuit_data.empty:
                raise ValueError("Invalid circuit data provided")

            race_ids = self.data_processor.race_data[
                self.data_processor.race_data['name'].str.contains(circuit_name, case=False) |
                (circuit_name.lower() == 'monaco' and self.data_processor.race_data['name'].str.contains('monte carlo', case=False))
            ]['raceId'].tolist()

            if not race_ids:
                if circuit_name.lower() == 'monaco':
                    race_ids = self.data_processor.race_data[
                        self.data_processor.race_data['name'].str.contains('monte', case=False) |
                        self.data_processor.race_data['name'].str.contains('monaco', case=False)
                    ]['raceId'].tolist()
                    
                    if not race_ids:
                        most_recent_race_id = self.data_processor.race_data['raceId'].max()
                        race_ids = [most_recent_race_id]
                        logger.warning(f"No specific race data found for Monaco, using fallback race ID: {most_recent_race_id}")
                else:
                    raise ValueError(f"No race data found for circuit: {circuit_name}")

            race_id = max(race_ids)
            logger.info(f"Using race ID {race_id} for circuit: {circuit_name}")

            race_info = self.data_processor.race_data[self.data_processor.race_data['raceId'] == race_id]
            constructor_info = self.data_processor.constructor_data[
                self.data_processor.constructor_data['raceId'] == race_id
            ]

            driver_features = self._process_driver_features(circuit_data, circuit_name)
            constructor_features = self._process_constructor_features(constructor_info)
            qualifying_features = self._process_qualifying_features(circuit_data, circuit_name)
            weather_features = self._process_weather_features(race_info)
            circuit_features = self._process_circuit_features(circuit_data, circuit_name)

            state = np.concatenate([
                driver_features,
                constructor_features,
                qualifying_features,
                weather_features,
                circuit_features
            ])

            logger.info(f"State created with shape: {state.shape}")
            return state

        except Exception as e:
            logger.error(f"Error in state creation: {e}", exc_info=True)
            raise
    
    def predict_qualifying(self, circuit_name: str) -> Dict:
        try:
            logger.info(f"Predicting qualifying for circuit: {circuit_name}")
            
            available_circuits = sorted(self.data_processor.circuits_data.keys())
            
            standardized_name = self.get_standardized_circuit_name(circuit_name)
            logger.info(f"Standardized circuit name: '{circuit_name}' -> '{standardized_name}'")
            
            circuit_data = self.data_processor.circuits_data.get(standardized_name.lower())
            
            if circuit_data is None:
                best_match = None
                best_score = 0
                
                for key in available_circuits:
                    circuit_lower = circuit_name.lower()
                    key_lower = key.lower()
                    
                    if key_lower in circuit_lower or circuit_lower in key_lower:
                        score = min(len(key_lower), len(circuit_lower))
                        if score > best_score:
                            best_score = score
                            best_match = key
                            
                    else:
                        common_chars = set(circuit_lower) & set(key_lower)
                        score = len(common_chars)
                        if score > best_score:
                            best_score = score
                            best_match = key
                
                if best_match and best_score > 3:
                    logger.info(f"Found alternative match: '{best_match}' (score: {best_score})")
                    standardized_name = best_match
                    circuit_data = self.data_processor.circuits_data[best_match]
            
            if circuit_data is None:
                logger.error(f"No data found for circuit: {circuit_name} (standardized: {standardized_name})")
                logger.error(f"Available circuits: {available_circuits}")
                return {"error": f"No data found for circuit: {circuit_name}. Available circuits: {', '.join(available_circuits[:5])}..."}
            
            state = self._create_state(circuit_data, circuit_name)
            
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
            
            self.model.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                policy, value = self.model(state_tensor)
                
                base_temperature = 0.2
                circuit_scaling = self._get_circuit_scaling(circuit_name)
                temperature = base_temperature / circuit_scaling
                
                scaled_policy = policy / temperature
                action_probs = F.softmax(scaled_policy, dim=1).cpu().numpy()[0]
                
                position_bias = np.array([(20-i)/20 for i in range(20)]) * 0.03
                action_probs = action_probs + position_bias

                action_probs = self._apply_team_recency_bias(action_probs)
                
                circuit_type = self._get_circuit_type(circuit_name)
                driver_ids = self.data_processor.driver_mapping.get_current_driver_ids()

                for idx, driver_id in enumerate(driver_ids):
                    if idx < len(action_probs):
                        driver_code = self.data_processor.driver_mapping.get_driver_code(driver_id)
                        
                        driver_circuit_preferences = {
                            'VER': {'high_speed': 1.3, 'technical': 1.2, 'street': 1.1},
                            'PER': {'high_speed': 0.9, 'technical': 0.9, 'street': 1.3},
                            
                            'LEC': {'high_speed': 1.1, 'technical': 1.2, 'street': 1.3},
                            'SAI': {'high_speed': 1.2, 'technical': 1.2, 'street': 1.1},
                            
                            'HAM': {'high_speed': 1.1, 'technical': 1.2, 'street': 1.0},
                            'RUS': {'high_speed': 1.1, 'technical': 1.2, 'street': 1.1},
                            
                            'NOR': {'high_speed': 1.3, 'technical': 1.2, 'street': 1.1},
                            'PIA': {'high_speed': 1.2, 'technical': 1.1, 'street': 1.1},
                        }
                        
                        if driver_code in driver_circuit_preferences and circuit_type in driver_circuit_preferences[driver_code]:
                            type_factor = driver_circuit_preferences[driver_code][circuit_type]
                            action_probs[idx] *= type_factor

                action_probs = action_probs / np.sum(action_probs)
                
                confidence = (value.item() + 1) / 2
            
            constructor_strengths = self._get_constructor_strengths()
            for idx, driver_id in enumerate(self.data_processor.driver_mapping.get_current_driver_ids()):
                if idx < len(action_probs):
                    constructor_id = self._get_driver_constructor(driver_id)
                    if constructor_id is not None:
                        constructor_weight = constructor_strengths[constructor_id]
                        action_probs[idx] *= constructor_weight
            
            action_probs = action_probs / np.sum(action_probs)
            
            recent_form_factors = self._calculate_recent_form()
            for idx, driver_id in enumerate(self.data_processor.driver_mapping.get_current_driver_ids()):
                if idx < len(action_probs):
                    form_factor = recent_form_factors[driver_id]
                    action_probs[idx] *= form_factor
            
            action_probs = action_probs / np.sum(action_probs)
            
            confidence_factor = confidence * 2
            confidence_adjusted_probs = np.power(action_probs, confidence_factor)
            
            action_probs = confidence_adjusted_probs / np.sum(confidence_adjusted_probs)
            
            skewed_probs = np.power(action_probs, 1.5)
            action_probs = skewed_probs / np.sum(skewed_probs)
            
            driver_2024_boost = {
                'VER': 1.3,
                'NOR': 1.2,
                'PIA': 1.15,
                'LEC': 1.2,
                'SAI': 1.15,
                'HAM': 1.05,
                'RUS': 1.1,
                'PER': 0.95,
                'ALO': 1.05,
                'ALB': 1.05,
            }
            
            for idx, driver_id in enumerate(driver_ids):
                if idx < len(action_probs):
                    driver_code = self.data_processor.driver_mapping.get_driver_code(driver_id)
                    if driver_code in driver_2024_boost:
                        action_probs[idx] *= driver_2024_boost[driver_code]
            
            action_probs = action_probs / np.sum(action_probs)
            
            predictions = self._format_predictions(action_probs, confidence, circuit_name)
            
            return predictions
                
        except Exception as e:
            logger.error(f"Error in prediction: {e}", exc_info=True)
            return {"error": str(e)}

    def _get_circuit_type(self, circuit_name: str) -> str:
        circuit_types = {
            'street': [
                'monaco', 'singapore', 'azerbaijan', 'saudi', 'las vegas',
                'baku', 'jeddah', 'marina bay'
            ],
            
            'high_speed': [
                'monza', 'spa', 'austria', 'silverstone', 'british', 'miami',
                'belgian', 'bahrain', 'canadian', 'italian', 'australian',
                'red bull ring', 'spielberg', 'montreal', 'interlagos', 'brazilian'
            ],
            
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
        
        return 'technical'
    def _get_driver_constructor(self, driver_id: int) -> int:
        try:
            if 'driver_standings' in self.data_processor.kaggle_data:
                driver_standings = self.data_processor.kaggle_data['driver_standings']
                driver_data = driver_standings[driver_standings['driverId'] == driver_id].sort_values('raceId', ascending=False)
                
                if not driver_data.empty and 'constructorId' in driver_data.columns:
                    return driver_data.iloc[0]['constructorId']
            
            if 'results' in self.data_processor.kaggle_data:
                results = self.data_processor.kaggle_data['results']
                driver_results = results[results['driverId'] == driver_id].sort_values('raceId', ascending=False)
                
                if not driver_results.empty and 'constructorId' in driver_results.columns:
                    return driver_results.iloc[0]['constructorId']
            
            return self._get_fallback_constructor_mapping().get(driver_id, None)
                
        except Exception as e:
            logger.error(f"Error getting driver constructor: {str(e)}")
            return None
    
    def _get_fallback_constructor_mapping(self) -> Dict[int, int]:
        return {
            1: 1,
            2: 1,
            
            3: 2,
            4: 2,
            
            5: 3,
            6: 3,
            
            7: 4,
            8: 4,
            
            9: 5,
            10: 5,
            
            11: 6,
            12: 6,
            
            13: 7,
            14: 7,
            
            15: 8,
            16: 8,
            
            17: 9,
            18: 9,
            
            19: 10,
            20: 10,
        }

    def _get_constructor_strengths(self) -> Dict[int, float]:
        constructor_strengths_2024 = {
            1: 1.3,
            2: 1.2,
            3: 1.1,
            4: 1.15,
            5: 1.0,
            6: 0.9,
            7: 0.95,
            8: 0.9,
            9: 0.85,
            10: 0.9,
        }
        
        return defaultdict(lambda: 1.0, constructor_strengths_2024)
    
    def _get_recent_races(self, n_races: int = 5) -> List[int]:
        try:
            race_data = self.data_processor.race_data
            if race_data is not None and not race_data.empty:
                if 'date' in race_data.columns:
                    sorted_races = race_data.sort_values('date', ascending=False)
                else:
                    sorted_races = race_data.sort_values('raceId', ascending=False)
                    
                return sorted_races['raceId'].head(n_races).tolist()
            return []
        except Exception as e:
            logger.error(f"Error getting recent races: {e}")
            return []

    def _get_driver_recent_results(self, driver_id: int, race_ids: List[int]) -> List[Dict]:
        try:
            qualifying_data = self.data_processor.kaggle_data.get('qualifying', pd.DataFrame())
            
            if qualifying_data.empty:
                return []
                
            driver_quals = qualifying_data[
                (qualifying_data['driverId'] == driver_id) & 
                (qualifying_data['raceId'].isin(race_ids))
            ]
            
            driver_quals = driver_quals.sort_values('raceId', ascending=False)
            
            results = []
            for _, qual in driver_quals.iterrows():
                position = qual.get('position', None)
                if position is not None:
                    try:
                        position = int(position)
                    except (ValueError, TypeError):
                        position = 20
                        
                    results.append({
                        'raceId': qual['raceId'],
                        'position': position
                    })
                    
            return results
        except Exception as e:
            logger.error(f"Error getting driver recent results: {e}")
            return []

    def _calculate_recent_form(self) -> Dict[int, float]:
        try:
            form_factors = {}
            
            recent_races = self._get_recent_races(5)
            
            for driver_id in self.data_processor.driver_mapping.get_current_driver_ids():
                recent_results = self._get_driver_recent_results(driver_id, recent_races)
                
                form_factor = 1.0
                
                if recent_results:
                    weighted_sum = 0
                    weights_sum = 0
                    
                    for i, result in enumerate(recent_results):
                        position = result.get('position', 10)
                        weight = len(recent_results) - i
                        weighted_sum += position * weight
                        weights_sum += weight
                    
                    if weights_sum > 0:
                        avg_position = weighted_sum / weights_sum
                        
                        form_factor = 1.5 - (avg_position - 1) * (0.8 / 19)
                
                form_factors[driver_id] = form_factor
                
            return defaultdict(lambda: 1.0, form_factors)
        except Exception as e:
            logger.error(f"Error calculating recent form: {e}")
            return defaultdict(lambda: 1.0)
        
    def _get_circuit_scaling(self, circuit_name: str) -> float:
        circuit_predictability = {
            'monaco': 1.7,
            'singapore': 1.5,
            'azerbaijan': 1.4,
            
            'hungarian': 1.4,
            'dutch': 1.3,
            'spanish': 1.3,
            'japanese': 1.3,
            
            'british': 1.1,
            'australian': 1.1,
            'austrian': 1.0,
            'bahrain': 1.0,
            'canadian': 1.0,
            
            'emilia romagna': 0.9,
            'belgian': 0.9,
            'chinese': 0.9,
            'saudi arabian': 0.9,
            
            'italian': 0.8,
            'united states': 0.8,
            'mexican': 0.8,
            'miami': 0.8,
        }
        
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
        
        for circuit_key in circuit_predictability:
            if circuit_key in normalized_name:
                return circuit_predictability[circuit_key]
        
        return 1.0
    def train_step(self, state_batch: torch.Tensor, target_batch: torch.Tensor) -> Dict[str, float]:
        try:
            self.model.train()
            self.optimizer.zero_grad()

            policy_pred, value_pred = self.model(state_batch)

            policy_loss = F.kl_div(
                F.log_softmax(policy_pred, dim=1),
                target_batch.float(),
                reduction='batchmean'
            )

            value_target = (torch.argmax(target_batch, dim=1).float() + 1) / 20.0
            value_loss = F.mse_loss(value_pred.squeeze(), value_target)

            total_loss = policy_loss + 0.5 * value_loss

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
    
    def calibrate_confidence(self, original_confidence: float, 
                       evaluation_metrics: dict = None, 
                       calibration_method: str = 'rescale') -> float:
        if calibration_method == 'rescale':
            min_input = 0.3
            max_input = 0.6
            min_output = 0.65
            max_output = 0.95
            
            clamped = max(min_input, min(max_input, original_confidence))
            
            calibrated = min_output + (clamped - min_input) * (max_output - min_output) / (max_input - min_input)
            return calibrated
        
        elif calibration_method == 'metrics_based' and evaluation_metrics is not None:
            weights = {
                'top5_f1': 0.5,
                'position_within_1': 0.3,
                'exact_accuracy': 0.2
            }
            
            calibrated = 0.0
            for metric, weight in weights.items():
                if metric in evaluation_metrics:
                    calibrated += evaluation_metrics[metric] * weight
            
            return min(1.0, max(0.0, calibrated))
        
        elif calibration_method == 'sigmoid':
            import math
            alpha = 10.0
            shifted = original_confidence - 0.5
            calibrated = 1 / (1 + math.exp(-alpha * shifted))
            return calibrated
        
        else:
            return original_confidence
        

    def _format_predictions(self, action_probs: np.ndarray, confidence: float, circuit_name: str) -> Dict:
        try:
            driver_ids = self.data_processor.driver_mapping.get_current_driver_ids()
            
            qual_data = self.data_processor.kaggle_data['qualifying']
            circuit_quals = qual_data[
                qual_data['raceId'].isin(
                    self.data_processor.race_data[
                        self.data_processor.race_data['name'].str.contains(circuit_name, case=False)
                    ]['raceId']
                )
            ]
            
            for idx, driver_id in enumerate(self.data_processor.driver_mapping.get_current_driver_ids()):
                if idx < len(action_probs):
                    performance_factor = self._get_driver_circuit_factor(driver_id, circuit_name)
                    performance_factor = performance_factor ** 2
                    action_probs[idx] = action_probs[idx] * performance_factor
            
            action_probs = action_probs / np.sum(action_probs)
            
            q3_performance_boost = 1.2
            max_years_relevance = 5

            for driver_idx, driver_id in enumerate(driver_ids):
                if driver_idx < len(action_probs):
                    driver_quals = circuit_quals[circuit_quals['driverId'] == driver_id]
                    
                    if not driver_quals.empty and 'raceId' in driver_quals.columns:
                        driver_quals = driver_quals.sort_values('raceId', ascending=False)
                        
                        if 'year' in driver_quals.columns:
                            driver_quals['years_ago'] = datetime.now().year - driver_quals['year']
                        else:
                            most_recent_race_id = self.data_processor.race_data['raceId'].max()
                            avg_races_per_year = 22
                            driver_quals['years_ago'] = (most_recent_race_id - driver_quals['raceId']) / avg_races_per_year
                            
                        relevant_quals = driver_quals[driver_quals['years_ago'] <= max_years_relevance]
                        
                        q3_factor = 1.0
                        
                        if not relevant_quals.empty:
                            total_weight = 0
                            weighted_q3_score = 0
                            
                            for _, qual_row in relevant_quals.iterrows():
                                years_ago = qual_row['years_ago']
                                time_weight = self._calculate_time_weight(years_ago, max_years_relevance)
                                
                                if pd.notna(qual_row['q3']):
                                    q3_performance = 1.0
                                    if 'position' in qual_row and pd.notna(qual_row['position']):
                                        try:
                                            position = int(qual_row['position'])
                                            if position <= 5:
                                                q3_performance = 1.4 - ((position - 1) * 0.08)
                                            elif position <= 10:
                                                q3_performance = 1.0
                                        except (ValueError, TypeError):
                                            pass
                                            
                                    weighted_q3_score += time_weight * q3_performance
                                    
                                total_weight += time_weight
                            
                            if total_weight > 0:
                                normalized_q3_score = weighted_q3_score / total_weight
                                q3_factor = 1.0 + (normalized_q3_score * q3_performance_boost * 0.1)
                        
                        action_probs[driver_idx] *= q3_factor
                        
                        if q3_factor > 1.2:
                            driver_code = self.data_processor.driver_mapping.get_driver_code(driver_id)
                            recent_q3s = len(relevant_quals[relevant_quals['q3'].notna()])
                            logger.info(f"Applied time-weighted Q3 boost of {q3_factor:.2f} to {driver_code} at {circuit_name} (recent Q3s: {recent_q3s})")
            
            action_probs = action_probs / np.sum(action_probs)
            
            action_probs = np.power(action_probs, 2)
            action_probs = action_probs / np.sum(action_probs)
            
            top5_indices = np.argsort(action_probs)[-5:][::-1]
            
            top5_probs = np.array([action_probs[idx] for idx in top5_indices])
            
            top5_probs_normalized = top5_probs / np.sum(top5_probs)

            evaluation_metrics = getattr(self, 'evaluation_metrics', None)
            calibrated_confidence = self.calibrate_confidence(confidence, evaluation_metrics, 'rescale')
            
            predictions = {
                'circuit': circuit_name,
                'prediction_time': datetime.now().isoformat(),
                'top5': [],
                'confidence_score': float(calibrated_confidence),
                'q3_influence': q3_performance_boost
            }
            
            for i, (idx, normalized_prob) in enumerate(zip(top5_indices, top5_probs_normalized), 1):
                if idx >= len(driver_ids):
                    continue
                    
                driver_id = driver_ids[idx]
                driver_code = self.data_processor.driver_mapping.get_driver_code(driver_id)
                
                if not driver_code:
                    continue
                    
                driver_circuit_quals = circuit_quals[circuit_quals['driverId'] == driver_id]
                q3_appearances = len(driver_circuit_quals[driver_circuit_quals['q3'].notna()])
                
                q3_stats = {}
                if q3_appearances > 0:
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
                    'probability': float(normalized_prob),
                    'original_probability': float(action_probs[idx] * 100),
                    'circuit_stats': {
                        'q3_appearances': q3_appearances,
                        'best_position': self._get_best_position(driver_circuit_quals),
                        'last_result': self._get_last_result(driver_circuit_quals),
                        'q3_stats': q3_stats
                    }
                }
                
                predictions['top5'].append(prediction)
            
            predictions['probability_note'] = "Top 5 probabilities normalized to sum to 100%"
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error formatting predictions: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _calculate_time_weight(self, years_ago: float, max_years_relevance: float) -> float:
        decay_factor = 0.5
        
        weight = np.exp(-decay_factor * years_ago)
        
        return max(0.05, weight)
    
    def _apply_team_recency_bias(self, action_probs: np.ndarray) -> np.ndarray:
        constructor_to_drivers = defaultdict(list)
        driver_to_position = {}
        
        for idx, driver_id in enumerate(self.data_processor.driver_mapping.get_current_driver_ids()):
            if idx < len(action_probs):
                constructor_id = self._get_driver_constructor(driver_id)
                if constructor_id:
                    constructor_to_drivers[constructor_id].append(driver_id)
                    driver_to_position[driver_id] = idx
        
        team_performance_2024 = {
            1: 1.3,
            2: 1.2,
            3: 1.1,
            4: 1.15,
            5: 1.0,
            6: 0.9,
            7: 0.95,
            8: 0.9,
            9: 0.85,
            10: 0.9,
        }
        
        for constructor_id, driver_ids in constructor_to_drivers.items():
            bias_factor = team_performance_2024.get(constructor_id, 1.0)
            for driver_id in driver_ids:
                idx = driver_to_position.get(driver_id)
                if idx is not None:
                    action_probs[idx] *= bias_factor
        
        strong_teams = {c_id: factor for c_id, factor in team_performance_2024.items() if factor > 1.1}
        if strong_teams:
            logger.info(f"Applied team recency bias to {len(strong_teams)} teams: {strong_teams}")
        
        return action_probs / np.sum(action_probs)

    def _get_best_position(self, driver_quals: pd.DataFrame) -> int:
        if not driver_quals.empty and 'position' in driver_quals.columns:
            positions = []
            for p in driver_quals['position'].tolist():
                try:
                    if isinstance(p, (int, float)) or (isinstance(p, str) and p.isdigit()):
                        positions.append(int(p))
                except (ValueError, TypeError):
                    pass
                    
            if positions:
                return min(positions)
        
        return None
      
    def _get_driver_circuit_factor(self, driver_id: int, circuit_name: str) -> float:
        try:
            qual_data = self.data_processor.kaggle_data['qualifying']
            circuit_quals = qual_data[
                qual_data['raceId'].isin(
                    self.data_processor.race_data[
                        self.data_processor.race_data['name'].str.contains(circuit_name, case=False)
                    ]['raceId']
                )
            ]
            
            driver_quals = circuit_quals[circuit_quals['driverId'] == driver_id]
            
            driver_code = self.data_processor.driver_mapping.get_driver_code(driver_id)
            
            circuit_specific_boosts = {
                'VER': {
                    'dutch': 1.5,
                    'austria': 1.5,
                    'belgium': 1.4,
                    'bahrain': 1.4,
                    'australian': 1.3,
                    'miami': 1.3,
                    'british': 1.3,
                    'canadian': 1.3,
                    'monaco': 1.2,
                    'azerbaijan': 1.2,
                    'saudi': 1.3,
                    'united': 1.4,
                    'singapore': 1.2,
                    'italian': 1.4,
                    'japanese': 1.4,
                    'mexico': 1.4,
                    'spain': 1.3,
                    'hungary': 1.3,
                    'chinese': 1.3,
                    'emilia': 1.3,
                },
                'PER': {
                    'mexico': 1.3,
                    'monaco': 1.2,
                    'azerbaijan': 1.3,
                    'saudi': 1.3,
                    'singapore': 1.1,
                    'miami': 1.0,
                    'canadian': 1.0,
                    'austria': 0.9,
                    'bahrain': 1.0,
                    'belgian': 0.9,
                    'australian': 0.9,
                    'british': 0.9,
                    'dutch': 0.9,
                    'united': 0.9,
                    'italian': 0.9,
                    'japanese': 0.9,
                    'spain': 0.9,
                    'hungary': 0.9,
                    'chinese': 0.9,
                    'emilia': 0.9,
                },
                
                'LEC': {
                    'monaco': 1.5,
                    'azerbaijan': 1.4,
                    'italian': 1.3,
                    'singapore': 1.4,
                    'australian': 1.3,
                    'bahrain': 1.3,
                    'belgium': 1.3,
                    'saudi': 1.3,
                    'british': 1.2,
                    'miami': 1.2,
                    'canadian': 1.2,
                    'japanese': 1.2,
                    'austria': 1.2,
                    'hungary': 1.2,
                    'dutch': 1.1,
                    'united': 1.2,
                    'spain': 1.2,
                    'mexico': 1.1,
                    'chinese': 1.2,
                    'emilia': 1.2,
                },
                'SAI': {
                    'spanish': 1.3,
                    'monaco': 1.3,
                    'british': 1.2,
                    'canadian': 1.2,
                    'italian': 1.3,
                    'australian': 1.3,
                    'singapore': 1.3,
                    'hungary': 1.2,
                    'azerbaijan': 1.2,
                    'saudi': 1.3,
                    'japanese': 1.1,
                    'belgian': 1.2,
                    'dutch': 1.2,
                    'austrian': 1.1,
                    'miami': 1.2,
                    'united': 1.2,
                    'bahrain': 1.2,
                    'mexico': 1.1,
                    'chinese': 1.2,
                    'emilia': 1.2,
                },
                
                'HAM': {
                    'british': 1.4,
                    'hungarian': 1.3,
                    'canadian': 1.2,
                    'chinese': 1.2,
                    'spain': 1.2,
                    'singapore': 1.1,
                    'italian': 1.1,
                    'belgian': 1.1,
                    'bahrain': 1.1,
                    'japanese': 1.1,
                    'australian': 1.1,
                    'saudi': 1.1,
                    'mexico': 1.1,
                    'united': 1.2,
                    'azerbaijan': 1.1,
                    'dutch': 1.0,
                    'monaco': 1.0,
                    'austrian': 1.0,
                    'miami': 1.0,
                    'emilia': 1.0,
                },
                'RUS': {
                    'british': 1.3,
                    'azerbaijan': 1.3,
                    'bahrain': 1.2,
                    'belgian': 1.2,
                    'austrian': 1.2,
                    'hungarian': 1.3,
                    'singapore': 1.2,
                    'australian': 1.2,
                    'miami': 1.2,
                    'italian': 1.2,
                    'japanese': 1.2,
                    'saudi': 1.2,
                    'canadian': 1.2,
                    'spanish': 1.2,
                    'monaco': 1.1,
                    'mexico': 1.1,
                    'united': 1.2,
                    'dutch': 1.2,
                    'chinese': 1.2,
                    'emilia': 1.1,
                },
                
                'NOR': {
                    'british': 1.3,
                    'italian': 1.2,
                    'belgian': 1.2,
                    'austrian': 1.2,
                    'dutch': 1.2,
                    'miami': 1.2,
                    'japanese': 1.2,
                    'monaco': 1.2,
                    'singapore': 1.2,
                    'hungarian': 1.2,
                    'australian': 1.2,
                    'azerbaijan': 1.2,
                    'canadian': 1.2,
                    'mexican': 1.2,
                    'bahrain': 1.2,
                    'saudi': 1.2,
                    'spanish': 1.2,
                    'united': 1.3,
                    'chinese': 1.2,
                    'emilia': 1.2,
                },
                'PIA': {
                    'australian': 1.3,
                    'italian': 1.1,
                    'austrian': 1.2,
                    'monaco': 1.1,
                    'singapore': 1.1,
                    'hungarian': 1.1,
                    'dutch': 1.1,
                    'miami': 1.1,
                    'british': 1.1,
                    'saudi': 1.1,
                    'japanese': 1.1,
                    'belgian': 1.1,
                    'azerbaijan': 1.0,
                    'canadian': 1.0,
                    'mexican': 1.0,
                    'bahrain': 1.0,
                    'spanish': 1.0,
                    'united': 1.0,
                    'chinese': 1.0,
                    'emilia': 1.0,
                }
            }
            
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
            
            for circuit_key in circuit_specific_boosts.get(driver_code, {}):
                if circuit_key in normalized_name:
                    return circuit_specific_boosts[driver_code][circuit_key]
            
            return 1.0
        except Exception as e:
            logger.error(f"Error calculating driver circuit factor: {e}")
            return 1.0
    
    def _get_last_result(self, driver_quals: pd.DataFrame) -> Optional[int]:
        if driver_quals.empty:
            return None
            
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
        try:
            cache_key = f"driver_features_{circuit_name}"
            if cache_key in self._feature_cache:
                return self._feature_cache[cache_key]
                
            current_driver_ids = self.data_processor.driver_mapping.get_current_driver_ids()
            
            features = np.zeros(20)
            
            qual_data = self.data_processor.kaggle_data['qualifying']
            
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
                    q3_times = []
                    for _, qual in driver_quals.iterrows():
                        if pd.notna(qual['q3']) and qual['q3'] != 'N/A':
                            time_secs = convert_time_to_seconds(qual['q3'])
                            if time_secs is not None:
                                q3_times.append(time_secs)
                    
                    if q3_times:
                        best_time = min(q3_times)
                        avg_time = sum(q3_times) / len(q3_times)
                        consistency = 1 - (np.std(q3_times) / avg_time if len(q3_times) > 1 else 0)
                        
                        performance_score = (0.6 * (1 - (avg_time - best_time) / best_time) + 
                                          0.4 * consistency)
                        
                        features[idx] = performance_score
            
            if np.sum(features) > 0:
                features = features / np.max(features)
            
            self._feature_cache[cache_key] = features
            
            logger.info(f"Driver features range for {circuit_name}: {np.min(features):.3f} to {np.max(features):.3f}")
            return features
            
        except Exception as e:
            logger.error(f"Error processing driver features: {e}", exc_info=True)
            return np.zeros(20)

    def _process_constructor_features(self, constructor_info: pd.DataFrame) -> np.ndarray:
        try:
            features = np.zeros(30)
            
            if not constructor_info.empty:
                max_points = constructor_info['constructor_points'].max()
                max_wins = constructor_info['wins'].max() if 'wins' in constructor_info else 0
                
                for idx, (_, row) in enumerate(constructor_info.iterrows()):
                    if idx >= 10:
                        break
                        
                    base_idx = idx * 3
                    
                    points = row.get('constructor_points', 0)
                    features[base_idx] = points / max_points if max_points > 0 else 0
                    
                    position = row.get('position', 10)
                    features[base_idx + 1] = 1 - ((position - 1) / 9)
                    
                    wins = row.get('wins', 0)
                    features[base_idx + 2] = wins / max_wins if max_wins > 0 else 0
            
            logger.info(f"Constructor features range: {np.min(features):.3f} to {np.max(features):.3f}")
            return features
            
        except Exception as e:
            logger.error(f"Error processing constructor features: {e}", exc_info=True)
            return np.zeros(30)

    def _process_qualifying_features(self, circuit_data: pd.DataFrame, circuit_name: str) -> np.ndarray:
        try:
            cache_key = f"qualifying_features_{circuit_name}"
            if cache_key in self._feature_cache:
                return self._feature_cache[cache_key]
                
            features = np.zeros(60)
            
            qual_data = self.data_processor.kaggle_data['qualifying']
            circuit_quals = qual_data[
                qual_data['raceId'].isin(
                    self.data_processor.race_data[
                        self.data_processor.race_data['name'].str.contains(circuit_name, case=False)
                    ]['raceId']
                )
            ]
            
            current_driver_ids = self.data_processor.driver_mapping.get_current_driver_ids()
            
            for idx, driver_id in enumerate(current_driver_ids):
                driver_quals = circuit_quals[circuit_quals['driverId'] == driver_id]
                
                if not driver_quals.empty:
                    base_idx = idx * 3
                    
                    for q_idx, session in enumerate(['q1', 'q2', 'q3']):
                        session_times = []
                        for _, qual in driver_quals.iterrows():
                            if pd.notna(qual[session]) and qual[session] != 'N/A':
                                time_secs = convert_time_to_seconds(qual[session])
                                if time_secs is not None:
                                    session_times.append(time_secs)
                        
                        if session_times:
                            best_time = min(session_times)
                            avg_time = sum(session_times) / len(session_times)
                            consistency = 1 - (np.std(session_times) / avg_time if len(session_times) > 1 else 0)
                            
                            performance_score = (0.5 * (1 - (avg_time - best_time) / best_time) + 
                                              0.3 * consistency +
                                              0.2 * (len(session_times) / len(driver_quals)))
                            
                            features[base_idx + q_idx] = performance_score
                
                else:
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
                                performance_score = (1 - (min(q_times) - min(q_times)) / min(q_times)) * 0.5
                                features[base_idx + q_idx] = performance_score
            
            self._feature_cache[cache_key] = features
            
            logger.info(f"Qualifying features range for {circuit_name}: {np.min(features):.3f} to {np.max(features):.3f}")
            return features
            
        except Exception as e:
            logger.error(f"Error processing qualifying features for {circuit_name}: {e}", exc_info=True)
            return np.zeros(60)
        
    def get_standardized_circuit_name(self, circuit_name: str) -> str:
        circuit_name_lower = circuit_name.lower().strip()
        
        circuit_mapping = {
            'emilia romagna': 'emilia',
            'mexico city': 'mexico',
            'monaco': 'monte',
            'saudi arabian': 'saudi',
            'united states': 'us',
            
            'bahrain': 'bahrain',
            'jeddah': 'saudi',
            'baku': 'azerbaijan', 
            'melbourne': 'australian',
            'albert park': 'australian',
            'catalunya': 'spanish',
            'barcelona': 'spanish',
            'monte carlo': 'monte',
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
        
        if circuit_name_lower in circuit_mapping:
            return circuit_mapping[circuit_name_lower]
            
        for key, value in sorted(circuit_mapping.items(), key=lambda x: len(x[0]), reverse=True):
            if key in circuit_name_lower or circuit_name_lower in key:
                return value
        
        if ' ' in circuit_name_lower:
            words = circuit_name_lower.split()
            for word in words:
                if word in circuit_mapping:
                    return circuit_mapping[word]
                
                for key, value in circuit_mapping.items():
                    if word in key or key in word:
                        return value
        
        if circuit_name_lower in self.data_processor.circuits_data:
            return circuit_name_lower
            
        first_word = circuit_name_lower.split()[0] if ' ' in circuit_name_lower else circuit_name_lower
        return first_word
    
    def _process_weather_features(self, race_info: pd.DataFrame) -> np.ndarray:
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
            
            weather_value = race_info['weather_numerical'].mode().iloc[0]
            
            return np.array([float(weather_value)])
            
        except Exception as e:
            logger.error(f"Error processing weather features: {e}", exc_info=True)
            return np.array([0.0])
    
    def _process_circuit_features(self, circuit_data: pd.DataFrame, circuit_name: str) -> np.ndarray:
        try:
            features = np.zeros(3)
            
            if circuit_data is None or circuit_data.empty:
                logger.warning(f"No circuit data available for {circuit_name}")
                return features
                
            sector_cols = [col for col in circuit_data.columns if 'sector' in col.lower()]
            
            if len(sector_cols) >= 3:
                sector_means = [circuit_data[col].mean() for col in sector_cols[:3]]
                total_time = sum(sector_means)
                
                if total_time > 0:
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
        if val_loss < self.history['best_val_loss'] - self.config.min_delta:
            self.history['best_val_loss'] = val_loss
            self.history['epochs_without_improvement'] = 0
            return False
        else:
            self.history['epochs_without_improvement'] += 1
            return self.history['epochs_without_improvement'] >= self.config.early_stopping_patience
