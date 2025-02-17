from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import logging
from datetime import datetime
from pathlib import Path
import json
from functools import lru_cache

# Change this from relative to absolute import
from mcts_and_nn import (
    MCTSConfig,
    ModelConfig,
    SimplifiedF1Net,
    MCTS
)

logger = logging.getLogger(__name__)

@dataclass
class PredictorConfig:
    """Configuration for QualifyingPredictor"""
    learning_rate: float = 0.001
    batch_size: int = 32
    model_save_dir: str = "models"
    cache_dir: str = "cache"
    max_cache_size: int = 1000
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    min_delta: float = 0.001

class DataCache:
    """Cache for processed data to improve performance"""
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with timestamp update"""
        if key in self.cache:
            self.access_times[key] = datetime.now().timestamp()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set item in cache with LRU cleanup"""
        if len(self.cache) >= self.max_size:
            # Remove least recently used items
            sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
            remove_count = int(self.max_size * 0.2)  # Remove 20% of items
            for k, _ in sorted_items[:remove_count]:
                del self.cache[k]
                del self.access_times[k]
        
        self.cache[key] = value
        self.access_times[key] = datetime.now().timestamp()

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
        self.data_cache = DataCache(max_size=config.max_cache_size)
        
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

    @lru_cache(maxsize=128)
    def _calculate_track_performance(
        self, 
        driver_id: int, 
        circuit_name: str, 
        qualifying_data: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """Calculate driver's performance metrics at specific track with caching"""
        try:
            # Create cache key
            cache_key = f"track_perf_{driver_id}_{circuit_name.lower()}"
            cached_result = self.data_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            if 'q3' not in qualifying_data.columns:
                raise ValueError("Required column 'q3' not found in qualifying data")

            # Normalize circuit name for comparison
            normalized_circuit_name = circuit_name.lower()
            # Handle special cases
            if normalized_circuit_name == 'são':
                normalized_circuit_name = 'sao'
            elif normalized_circuit_name == 'méxico':
                normalized_circuit_name = 'mexico'
            elif normalized_circuit_name == 'monaco':
                normalized_circuit_name = 'monte'

            # Get qualifying data for this driver at this circuit
            if 'circuit_name' in qualifying_data.columns:
                track_quals = qualifying_data[
                    (qualifying_data['driverId'] == driver_id) & 
                    (qualifying_data['circuit_name'].str.lower().str.contains(normalized_circuit_name))
                ]
            else:
                track_quals = qualifying_data[qualifying_data['driverId'] == driver_id]

            if track_quals.empty:
                logger.warning(f"No qualifying data found for driver {driver_id} at circuit {circuit_name}")
                return 0.0, 0.0, 0.0

            # Process qualifying times
            q3_times = []
            dates = []
            
            for _, qual in track_quals.iterrows():
                if pd.notna(qual['q3']) and qual['q3'] != 'N/A':
                    try:
                        time_str = qual['q3']
                        time_secs = self._parse_time_string(time_str)
                        if time_secs is not None:
                            if time_secs > 0:  # Validate time is positive
                                q3_times.append(time_secs)
                                if 'date' in qual:
                                    dates.append(pd.to_datetime(qual['date']))
                            else:
                                logger.warning(f"Invalid negative time found: {time_secs}")
                    except Exception as e:
                        logger.warning(f"Error parsing time '{time_str}': {str(e)}")
                        continue

            if not q3_times:
                logger.warning(f"No valid Q3 times found for driver {driver_id} at circuit {circuit_name}")
                return 0.0, 0.0, 0.0

            # Calculate performance metrics
            result = self._compute_performance_metrics(q3_times, dates)
            
            # Validate results before caching
            if all(isinstance(x, float) for x in result):
                self.data_cache.set(cache_key, result)
            else:
                logger.error(f"Invalid performance metrics calculated: {result}")
                return 0.0, 0.0, 0.0

            logger.debug(f"Calculated performance metrics for driver {driver_id} at {circuit_name}: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in track performance calculation: {str(e)}")
            return 0.0, 0.0, 0.0

    def _parse_time_string(self, time_str: str) -> Optional[float]:
        """Parse qualifying time string to seconds with improved validation"""
        try:
            if pd.isna(time_str) or time_str == 'N/A' or time_str == '\\N':
                return None
                
            if isinstance(time_str, str):
                # Remove any whitespace
                time_str = time_str.strip()
                
                if ':' in time_str:
                    mins, secs = time_str.split(':')
                    total_secs = float(mins) * 60 + float(secs)
                else:
                    total_secs = float(time_str)
                
                # Validate the time is reasonable (between 30 seconds and 3 minutes)
                if 30 <= total_secs <= 180:
                    return total_secs
                else:
                    logger.warning(f"Qualifying time outside reasonable range: {total_secs} seconds")
                    return None
            elif isinstance(time_str, (int, float)):
                # Direct numeric input
                if 30 <= float(time_str) <= 180:
                    return float(time_str)
                    
            return None
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing time string '{time_str}': {str(e)}")
            return None

    def _compute_performance_metrics(
        self, 
        times: List[float], 
        dates: Optional[List[datetime]] = None
    ) -> Tuple[float, float, float]:
        """Compute performance metrics from qualifying times"""
        min_time = min(times)
        normalized_times = [1 - (t - min_time) / min_time for t in times]
        
        # Calculate consistency
        consistency = 1.0 / (1.0 + np.std(normalized_times))
        
        # Calculate improvement rate
        improvement_rate = 0.0
        if dates and len(dates) > 1:
            time_diffs = np.diff(times)
            improvements = [1 if diff < 0 else 0 for diff in time_diffs]
            improvement_rate = np.mean(improvements)
        
        # Calculate weighted performance
        weights = np.exp(-0.3 * np.arange(len(normalized_times)))
        weights = weights / weights.sum()
        track_performance = np.sum(normalized_times * weights)
        
        return track_performance, consistency, improvement_rate

    def _create_state(self, circuit_data: pd.DataFrame, circuit_name: str) -> np.ndarray:
      """Create complete state representation with all required features."""
      try:
          print("\nCreating state in qualifying predictor...")
          
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
          print(f"Driver features shape: {driver_features.shape}")

          # 2. Constructor features (30)
          constructor_features = self._process_constructor_features(constructor_info)
          print(f"Constructor features shape: {constructor_features.shape}")

          # 3. Qualifying features (60)
          qualifying_features = self._process_qualifying_features(circuit_data, circuit_name)
          print(f"Qualifying features shape: {qualifying_features.shape}")

          # 4. Weather features (1)
          weather_features = self._process_weather_features(race_info)
          print(f"Weather features shape: {weather_features.shape}")

          # 5. Circuit features (3)
          circuit_features = self._process_circuit_features(circuit_data, circuit_name)
          print(f"Circuit features shape: {circuit_features.shape}")

          # Combine all features
          state = np.concatenate([
              driver_features,      # 20
              constructor_features, # 30
              qualifying_features,  # 60
              weather_features,     # 1
              circuit_features      # 3
          ])

          print(f"\nFinal state composition:")
          print(f"- Driver features: {len(driver_features)}")
          print(f"- Constructor features: {len(constructor_features)}")
          print(f"- Qualifying features: {len(qualifying_features)}")
          print(f"- Weather features: {len(weather_features)}")
          print(f"- Circuit features: {len(circuit_features)}")
          print(f"Total state size: {len(state)}")

          return state

      except Exception as e:
          print(f"\nError in state creation:")
          print(f"Exception type: {type(e)}")
          print(f"Exception message: {str(e)}")
          print("\nStack trace:")
          import traceback
          traceback.print_exc()
          raise

    def predict_qualifying(self, circuit_name: str) -> Dict:
        """
        Predict qualifying with enhanced circuit-specific processing.
        """
        try:
            print(f"\nPredicting qualifying for circuit: {circuit_name}")
            
            # Get circuit data
            circuit_data = self.data_processor.circuits_data.get(circuit_name.lower())
            if circuit_data is None:
                raise ValueError(f"No data found for circuit: {circuit_name}")
            
            # Create state representation
            state = self._create_state(circuit_data, circuit_name)
            
            # Validate state values and print detailed stats
            print("\nState feature statistics:")
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
                print(f"\n{feature_name.capitalize()} Features:")
                print(f"Mean: {np.mean(feature_slice):.3f}")
                print(f"Std: {np.std(feature_slice):.3f}")
                print(f"Min: {np.min(feature_slice):.3f}")
                print(f"Max: {np.max(feature_slice):.3f}")
                start_idx += size
            
            # Convert to tensor and predict
            self.model.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                policy, value = self.model(state_tensor)
                
                # Use a lower temperature for more diverse predictions
                temperature = 1.0  # Reduced from 1.5
                scaled_policy = policy / temperature
                action_probs = F.softmax(scaled_policy, dim=1).cpu().numpy()[0]
                
                # Add noise to break ties and increase diversity
                noise = np.random.normal(0, 0.01, action_probs.shape)
                action_probs = F.softmax(torch.FloatTensor(action_probs + noise), dim=0).numpy()
                
                confidence = (value.item() + 1) / 2
            
            # Format predictions with detailed driver info
            predictions = self._format_predictions(action_probs, confidence, circuit_name)
            
            # Print prediction diversity stats
            unique_probs = len(set([round(p, 3) for p in action_probs]))
            prob_stats = {
                'min': np.min(action_probs),
                'max': np.max(action_probs),
                'std': np.std(action_probs)
            }
            print(f"\nPrediction statistics:")
            print(f"Unique probabilities: {unique_probs}")
            print(f"Probability range: {prob_stats['min']:.3f} to {prob_stats['max']:.3f}")
            print(f"Probability std: {prob_stats['std']:.3f}")
            
            return predictions
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {"error": str(e)}

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
            print(f"Error in training step: {str(e)}")
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
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _format_predictions(self, action_probs: np.ndarray, confidence: float, circuit_name: str) -> Dict:
      """
      Format predictions with enhanced driver stats and circuit-specific performance.
      """
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
          print(f"Error formatting predictions: {str(e)}")
          return {"error": str(e)}
      
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

    def _get_driver_info(self, driver_id: int, qual_data: pd.DataFrame) -> Dict:
        """Get detailed driver information"""
        try:
            driver_quals = qual_data[qual_data['driverId'] == driver_id]
            
            q3_times = []
            for _, qual in driver_quals.iterrows():
                if pd.notna(qual['q3']) and qual['q3'] != 'N/A':
                    time_secs = self._parse_time_string(qual['q3'])
                    if time_secs is not None:
                        q3_times.append(time_secs)

            return {
                'q3_appearances': len(driver_quals[driver_quals['q3'].notna()]),
                'total_races': len(driver_quals),
                'consistency': float(np.std(q3_times)) if q3_times else None,
                'best_time': float(min(q3_times)) if q3_times else None,
                'average_time': float(np.mean(q3_times)) if q3_times else None,
                'recent_form': self._calculate_recent_form(q3_times),
                'experience_level': len(driver_quals) / 100.0  # Normalized experience
            }

        except Exception as e:
            logger.error(f"Error getting driver info: {str(e)}")
            return {
                'q3_appearances': 0,
                'total_races': 0,
                'consistency': None,
                'best_time': None,
                'average_time': None,
                'recent_form': None,
                'experience_level': 0.0
            }

    def _calculate_recent_form(self, times: List[float], window: int = 5) -> Optional[float]:
        """Calculate driver's recent form based on latest performances"""
        if not times or len(times) < window:
            return None
            
        recent_times = times[-window:]
        if not recent_times:
            return None
            
        # Calculate trend in recent performances
        # Negative value means improving (times decreasing)
        # Positive value means declining (times increasing)
        trends = np.diff(recent_times)
        return float(np.mean(trends))

    def _process_driver_features(self, circuit_data: pd.DataFrame, circuit_name: str) -> np.ndarray:
      """Process driver features with circuit-specific considerations."""
      try:
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
                          time_secs = self._parse_qualifying_time(qual['q3'])
                          if time_secs > 0:
                              q3_times.append(time_secs)
                  
                  if q3_times:
                      # Calculate performance score
                      best_time = min(q3_times)
                      avg_time = sum(q3_times) / len(q3_times)
                      consistency = 1 - (np.std(q3_times) / avg_time if len(q3_times) > 1 else 0)
                      
                      # Combine metrics
                      performance_score = (0.6 * (1 - (best_time - min(q3_times)) / best_time) + 
                                        0.4 * consistency)
                      
                      features[idx] = performance_score
          
          # Normalize features
          if np.sum(features) > 0:
              features = features / np.max(features)
          
          print(f"Driver features range for {circuit_name}: {np.min(features):.3f} to {np.max(features):.3f}")
          return features
          
      except Exception as e:
          print(f"Error processing driver features: {str(e)}")
          return np.zeros(20)
            
    def convert_time_to_seconds(time_str: str) -> Optional[float]:
        """Convert qualifying time string to seconds"""
        if pd.isna(time_str) or time_str == 'N/A' or time_str == '\\N':
            return None
          
        try:
            if isinstance(time_str, str):
                time_str = time_str.strip()
                if ':' in time_str:
                    minutes, seconds = time_str.split(':')
                    return float(minutes) * 60 + float(seconds)
                return float(time_str)
            elif isinstance(time_str, (int, float)):
                return float(time_str)
            return None
        except (ValueError, TypeError):
            return None
    
    def _get_driver_features(self, race_info: pd.DataFrame) -> np.ndarray:
      """Extract and normalize driver-related features for current F1 drivers only."""
      # Get current driver IDs from mapping
      current_driver_ids = self.driver_mapping.get_current_driver_ids()
      features = np.zeros((len(current_driver_ids), 3))  # 3 features per driver
      
      for idx, driver_id in enumerate(current_driver_ids):
          driver_data = race_info[race_info['driverId'] == driver_id]
          
          if not driver_data.empty:
              # Extract features
              features[idx, 0] = self._normalize_position(driver_data['position'].iloc[0])
              
              # Q3 appearances
              q3_data = driver_data['q3'].notna().sum() / len(driver_data)
              features[idx, 1] = q3_data
              
              # Recent qualifying performance
              qual_times = []
              for _, row in driver_data.iterrows():
                  if pd.notna(row['q3']) and row['q3'] != 'N/A':
                      time_secs = self._parse_time_string(row['q3'])
                      if time_secs is not None:
                          qual_times.append(time_secs)
              
              if qual_times:
                  min_time = min(qual_times)
                  avg_time = sum(qual_times) / len(qual_times)
                  features[idx, 2] = 1 - (avg_time - min_time) / min_time
              else:
                  features[idx, 2] = 0.0
      
      return features.flatten()

    def _normalize_position(self, position: float) -> float:
        """Normalize position to [0,1] range"""
        if pd.isna(position):
            return 0.0
        return 1 - (position - 1) / 19  # 20 positions, normalized to [0,1]

    def _process_qualifying_features(self, circuit_data: pd.DataFrame, circuit_name: str) -> np.ndarray:
      """
      Process qualifying features with enhanced circuit-specific historical performance.
      """
      try:
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
                              time_secs = self._parse_qualifying_time(qual[session])
                              if time_secs > 0:
                                  session_times.append(time_secs)
                      
                      if session_times:
                          # Calculate performance metrics
                          best_time = min(session_times)
                          avg_time = sum(session_times) / len(session_times)
                          consistency = 1 - (np.std(session_times) / avg_time if len(session_times) > 1 else 0)
                          
                          # Combine metrics into a single normalized score
                          performance_score = (0.5 * (1 - (best_time - min(session_times)) / best_time) + 
                                            0.3 * consistency +
                                            0.2 * (len(session_times) / len(driver_quals)))
                          
                          features[base_idx + q_idx] = performance_score
              
              else:
                  # If no qualifying data for this driver at this circuit,
                  # use their general qualifying performance
                  general_quals = qual_data[qual_data['driverId'] == driver_id]
                  if not general_quals.empty:
                      base_idx = idx * 3
                      for q_idx, session in enumerate(['q1', 'q2', 'q3']):
                          q_times = []
                          for _, qual in general_quals.iterrows():
                              if pd.notna(qual[session]) and qual[session] != 'N/A':
                                  time_secs = self._parse_qualifying_time(qual[session])
                                  if time_secs > 0:
                                      q_times.append(time_secs)
                          
                          if q_times:
                              # Use a reduced weight for general performance
                              performance_score = (1 - (min(q_times) - min(q_times)) / min(q_times)) * 0.5
                              features[base_idx + q_idx] = performance_score
          
          print(f"Qualifying features range for {circuit_name}: {np.min(features):.3f} to {np.max(features):.3f}")
          return features
          
      except Exception as e:
          print(f"Error processing qualifying features for {circuit_name}: {str(e)}")
          return np.zeros(60)
    
    def _process_weather_features(self, race_info: pd.DataFrame) -> np.ndarray:
        """
        Enhanced weather feature processing with better validation.
        """
        try:
            print("\nProcessing weather features:")
            print(f"Race info columns: {race_info.columns.tolist()}")
            
            if race_info.empty:
                print("Warning: Empty race info")
                return np.array([0.0])
                
            if 'weather_numerical' not in race_info.columns:
                print("Warning: No weather_numerical column found")
                return np.array([0.0])
                
            weather_values = race_info['weather_numerical'].unique()
            print(f"Unique weather values found: {weather_values}")
            
            # Take the most common weather value for this race
            weather_value = race_info['weather_numerical'].mode().iloc[0]
            print(f"Selected weather value: {weather_value}")
            
            return np.array([float(weather_value)])
            
        except Exception as e:
            print(f"Error processing weather features: {str(e)}")
            return np.array([0.0])
    
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
              max_wins = constructor_info['wins'].max()
              
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
          
          print(f"Constructor features range: {np.min(features):.3f} to {np.max(features):.3f}")
          return features
          
      except Exception as e:
          print(f"Error processing constructor features: {str(e)}")
          return np.zeros(30)

    def _process_circuit_features(self, circuit_data: pd.DataFrame, circuit_name: str) -> np.ndarray:
        """
        Enhanced circuit feature processing with proper sector time handling.
        """
        try:
            # Initialize features array
            features = np.zeros(3)
            
            if circuit_data is None or circuit_data.empty:
                print(f"No circuit data available for {circuit_name}")
                return features
                
            print(f"\nProcessing circuit data for {circuit_name}:")
            print(f"Available columns: {circuit_data.columns.tolist()}")
            print(f"Number of rows: {len(circuit_data)}")
            
            # Check for sector time columns
            sector_cols = [col for col in circuit_data.columns if 'sector' in col.lower()]
            print(f"Found sector columns: {sector_cols}")
            
            if len(sector_cols) >= 3:
                # Calculate mean sector times
                sector_means = [circuit_data[col].mean() for col in sector_cols[:3]]
                total_time = sum(sector_means)
                
                if total_time > 0:
                    # Calculate normalized sector proportions
                    features = np.array([time/total_time for time in sector_means])
                    
                    # Add some validation prints
                    print(f"\nCircuit {circuit_name} analysis:")
                    print(f"Raw sector means: {sector_means}")
                    print(f"Total lap time: {total_time:.3f}")
                    print(f"Normalized features: {features}")
                    
                    return features
                else:
                    print(f"Warning: Total time is 0 for circuit {circuit_name}")
            else:
                print(f"Warning: Insufficient sector data for circuit {circuit_name}")
                
            return features
            
        except Exception as e:
            print(f"Error processing circuit features for {circuit_name}: {str(e)}")
            print(f"Circuit data head:\n{circuit_data.head()}")
            return np.zeros(3)

    def _parse_qualifying_time(self, time_str: str) -> float:
      """Parse qualifying time string to seconds."""
      if pd.isna(time_str) or time_str == 'N/A' or time_str == '\\N':
          return 0.0
          
      try:
          if isinstance(time_str, str):
              if ':' in time_str:
                  minutes, seconds = time_str.split(':')
                  return float(minutes) * 60 + float(seconds)
              return float(time_str)
          elif isinstance(time_str, (int, float)):
              return float(time_str)
      except (ValueError, TypeError):
          return 0.0
      
      return 0.0
    
    def _combine_features(
        self,
        feature_arrays: Dict[str, np.ndarray],
        sector_times: np.ndarray
    ) -> np.ndarray:
        """Combine all features into final state representation"""
        try:
            # Calculate base driver performance with enhanced track metrics
            driver_perf = (
                0.25 * feature_arrays['qual_perf'] +
                0.30 * feature_arrays['recent_qual'] +
                0.25 * feature_arrays['track_perf'] +
                0.10 * feature_arrays['track_consistency'] +
                0.10 * feature_arrays['track_improvement']
            )
            
            # Make sure all arrays are the correct shape
            for key, arr in feature_arrays.items():
                if not isinstance(arr, np.ndarray):
                    feature_arrays[key] = np.array(arr)
                if len(arr.shape) == 1:
                    feature_arrays[key] = arr.reshape(-1)
            
            # Combine all features in the required order to match ModelConfig
            state = np.concatenate([
                driver_perf,                    # 20 features
                feature_arrays['qual_perf'],    # 20 features
                feature_arrays['recent_qual'],  # 20 features
                feature_arrays['track_perf'],   # 20 features
                feature_arrays['experience'],   # 20 features
                sector_times                    # 3 features
            ])
            
            return state

        except Exception as e:
            logger.error(f"Error combining features: {str(e)}")
            raise

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
            logger.error(f"Error in evaluation: {str(e)}")
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