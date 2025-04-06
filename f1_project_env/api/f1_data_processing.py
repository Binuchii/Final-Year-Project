import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

class F1DriverMapping:
    """A class to handle F1 driver mappings between different identification systems."""
    def __init__(self):
        # Initialize current drivers mapping
        self._current_drivers = {
            'VER': 830, 'PER': 815, 'HAM': 1, 'RUS': 847,
            'LEC': 844, 'SAI': 832, 'NOR': 846, 'PIA': 857,
            'ALO': 4, 'STR': 840, 'GAS': 842, 'OCO': 839,
            'ALB': 848, 'SAR': 858, 'BOT': 822, 'ZHO': 855,
            'TSU': 852, 'RIC': 817, 'MAG': 825, 'HUL': 807
        }
        
        # Initialize mappings
        self.driver_mappings = {
            'id_to_code': {id_: code for code, id_ in self._current_drivers.items()},
            'number_to_code': {
                '1': 'VER', '11': 'PER', '44': 'HAM', '63': 'RUS',
                '16': 'LEC', '55': 'SAI', '4': 'NOR', '81': 'PIA',
                '14': 'ALO', '18': 'STR', '10': 'GAS', '31': 'OCO',
                '23': 'ALB', '2': 'SAR', '77': 'BOT', '24': 'ZHO',
                '22': 'TSU', '3': 'RIC', '20': 'MAG', '27': 'HUL',
            },
            'code_to_name': {
                'VER': 'Max Verstappen', 'PER': 'Sergio Perez',
                'HAM': 'Lewis Hamilton', 'RUS': 'George Russell',
                'LEC': 'Charles Leclerc', 'SAI': 'Carlos Sainz',
                'NOR': 'Lando Norris', 'PIA': 'Oscar Piastri',
                'ALO': 'Fernando Alonso', 'STR': 'Lance Stroll',
                'GAS': 'Pierre Gasly', 'OCO': 'Esteban Ocon',
                'ALB': 'Alexander Albon', 'SAR': 'Logan Sargeant',
                'BOT': 'Valtteri Bottas', 'ZHO': 'Zhou Guanyu',
                'TSU': 'Yuki Tsunoda', 'RIC': 'Daniel Ricciardo',
                'MAG': 'Kevin Magnussen', 'HUL': 'Nico Hulkenberg'
            }
        }
        
        # Create reverse mappings
        self.driver_mappings['code_to_id'] = {v: k for k, v in self.driver_mappings['id_to_code'].items()}
        self.driver_mappings['code_to_number'] = {v: k for k, v in self.driver_mappings['number_to_code'].items()}

    def get_current_driver_ids(self) -> list:
        """Get list of current driver IDs."""
        return list(self._current_drivers.values())

    def get_driver_code(self, identifier) -> str:
        """Get driver code from either driver ID or number."""
        if isinstance(identifier, (int, float)):
            return self.driver_mappings['id_to_code'].get(int(identifier))
        elif isinstance(identifier, str):
            if identifier.isdigit():
                return self.driver_mappings['number_to_code'].get(identifier)
            return identifier if identifier in self.driver_mappings['code_to_name'] else None
        return None

    def get_driver_id(self, identifier) -> int:
        """Get driver ID from either driver code or number."""
        if isinstance(identifier, str):
            if identifier in self._current_drivers:
                return self._current_drivers[identifier]
            elif identifier in self.driver_mappings['number_to_code']:
                code = self.driver_mappings['number_to_code'][identifier]
                return self._current_drivers.get(code)
        return None

    def get_driver_number(self, identifier) -> str:
        """Get driver number from either driver code or ID."""
        code = None
        if isinstance(identifier, str):
            code = identifier if identifier in self.driver_mappings['code_to_number'] else None
        elif isinstance(identifier, (int, float)):
            code = self.driver_mappings['id_to_code'].get(int(identifier))
        return self.driver_mappings['code_to_number'].get(code) if code else None

    def get_driver_name(self, identifier) -> str:
        """Get driver full name from code, ID, or number."""
        code = self.get_driver_code(identifier)
        return self.driver_mappings['code_to_name'].get(code) if code else None
    
class F1DataProcessor:
    """Data processor for F1 AlphaZero model."""
    def __init__(self, data_dir: str, circuits_folder: Optional[str] = None):
        self.data_dir = data_dir
        self.circuits_folder = circuits_folder or os.path.join(data_dir, "calculated_variables")
        self.kaggle_data = {}
        self.circuits_data = {}
        self.race_data = None
        self.constructor_data = None
        self.state_dim = None
        self.action_dim = None
        self.driver_mapping = F1DriverMapping()
        self._load_and_clean_data()
        
    def _load_kaggle_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load historical F1 data from a CSV file."""
        try:
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                print(f"Loaded {len(data)} rows from {file_path}")
                return data
            else:
                print(f"File not found: {file_path}")
                return None
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _load_and_clean_data(self):
        """Load and clean all required F1 datasets."""
        self.kaggle_data = {
            "drivers": self._clean_drivers_data(self._load_kaggle_data(os.path.join(self.data_dir, "drivers.csv"))),
            "races": self._clean_races_data(self._load_kaggle_data(os.path.join(self.data_dir, "races.csv"))),
            "results": self._clean_results_data(self._load_kaggle_data(os.path.join(self.data_dir, "results.csv"))),
            "constructor_results": self._clean_constructorResults_data(self._load_kaggle_data(os.path.join(self.data_dir, "constructor_results.csv"))),
            "constructor_standings": self._clean_Standings_data(self._load_kaggle_data(os.path.join(self.data_dir, "constructor_standings.csv"))),
            "teams": self._clean_teams_data(self._load_kaggle_data(os.path.join(self.data_dir, "constructors.csv"))),
            "qualifying": self._clean_qualifying_data(self._load_kaggle_data(os.path.join(self.data_dir, "qualifying.csv"))),
            "weather": self._clean_weather_data(self._load_kaggle_data(os.path.join(self.data_dir, "cleaned_weather_with_raceId.csv")))
        }
        
        # Load circuit data
        self.circuits_data = {}
        circuits_folder = "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/api/data/calculated_variables"
        for filename in os.listdir(circuits_folder):
            if filename.endswith(".csv"):
                circuit_name = filename.split(" ")[0].lower()
                file_path = os.path.join(circuits_folder, filename)
                df = pd.read_csv(file_path)
                self.circuits_data[circuit_name] = self._clean_circuit_data(df)
        
        # Create merged datasets for state representation
        self._create_merged_datasets()
        
    def _clean_drivers_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the drivers dataset and add driver codes."""
        if data is not None:
            data = data.copy()
            columns_to_keep = ["driverId", "number", "forename", "surname"]
            data = data[columns_to_keep]
            data['driverCode'] = data['driverId'].apply(self.driver_mapping.get_driver_code)
            data = data[data['driverCode'].notna()]
        return data

    def _clean_races_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the races dataset."""
        if data is not None:
            data = data.copy()
            data = data[data["year"].isin([2021, 2022, 2023, 2024])]
            columns_to_keep = ["raceId", "year", "circuitId", "name"]
            data = data[columns_to_keep]
        return data

    def _clean_results_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the results dataset."""
        if data is not None:
            data = data.copy()
            columns_to_keep = ["resultId", "raceId", "driverId", "constructorId", "position"]
            data = data[columns_to_keep]
            data.loc[:, 'position'] = pd.to_numeric(data['position'], errors='coerce')
            data.loc[data['position'].isna(), 'position'] = 20
        return data

    def _clean_constructorResults_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the constructor results dataset."""
        if data is not None:
            columns_to_keep = ["raceId", "constructorId", "points"]
            data = data[columns_to_keep]
            data = data[data['raceId'] >= 1053]
            data = data.rename(columns={'points': 'points_AtRace'})
        return data

    def _clean_Standings_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the standings dataset."""
        if data is not None:
            columns_to_keep = ["constructorStandingsId", "raceId", "constructorId", 
                             "points", "position", "wins"]
            data = data[columns_to_keep]
            data = data[data['raceId'] >= 1053]
            data = data.rename(columns={'points': 'constructor_points'})
        return data

    def _clean_teams_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the teams dataset."""
        if data is not None:
            columns_to_keep = ["constructorId", "name"]
            data = data[columns_to_keep]
        return data

    def _clean_qualifying_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the qualifying dataset."""
        if data is not None:
            columns_to_keep = ["qualifyId", "raceId", "driverId", "q1", "q2", "q3"]
            data = data[columns_to_keep]
            data.fillna({"q1": "N/A", "q2": "N/A", "q3": "N/A"}, inplace=True)
        return data

    def _clean_weather_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the weather dataset."""
        if data is not None:
            columns_to_keep = ["meeting_key", "circuit_short_name", "location", 
                             "country_name", "date", "weather_type", "raceId"]
            data = data[columns_to_keep]
            
            weather_mapping = {'No Rain': 0, 'Rain': 1}
            data['weather_numerical'] = data['weather_type'].map(weather_mapping)
        return data
    
    def _clean_circuit_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean circuit-specific data."""
        if df is not None:
            df = df.copy()
            df = df.dropna()
            
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'sector1' in df.columns:
                max_sector_time = max(
                    df['sector1'].max(),
                    df['sector2'].max() if 'sector2' in df.columns else 0,
                    df['sector3'].max() if 'sector3' in df.columns else 0
                )
                for sector in ['sector1', 'sector2', 'sector3']:
                    if sector in df.columns:
                        df.loc[:, f'{sector}_normalized'] = df[sector] / max_sector_time
        
        return df

    def _get_circuit_features(self, race_info: pd.DataFrame) -> np.ndarray:
        """Extract circuit-specific features."""
        features = np.zeros(3)
        
        try:
            if race_info.empty:
                return features
                
            circuit_name = race_info['name'].iloc[0].split(' ')[0].lower()
            
            if circuit_name == 'são':
                circuit_name = 'sao'
            elif circuit_name == 'méxico':
                circuit_name = 'mexico'
            elif circuit_name == 'monaco':
                circuit_name = 'monte'
            
            if circuit_name in self.circuits_data:
                circuit_df = self.circuits_data[circuit_name]
                
                for i, col in enumerate(['sector1_normalized', 'sector2_normalized', 'sector3_normalized']):
                    if col in circuit_df.columns:
                        features[i] = circuit_df[col].mean()
        
        except Exception as e:
            print(f"Error processing circuit features: {str(e)}")
        
        return features
        
    def _create_merged_datasets(self):
        """Create merged datasets with proper DataFrame operations."""
        # Merge race data
        if all(k in self.kaggle_data for k in ["races", "results", "qualifying", "drivers"]):
            merged_race = pd.merge(self.kaggle_data["races"], self.kaggle_data["results"], 
                                on="raceId", how="inner")
            merged_race = pd.merge(merged_race, self.kaggle_data["qualifying"], 
                                on=["raceId", "driverId"], how="inner")
            self.race_data = pd.merge(merged_race, self.kaggle_data["drivers"], 
                                    on="driverId", how="inner")
            
            # Validate numeric columns
            numeric_columns = ['position', 'points_AtRace', 'constructor_points', 'wins']
            self.race_data = self.race_data.copy()
            for col in numeric_columns:
                if col in self.race_data.columns:
                    self.race_data[col] = pd.to_numeric(self.race_data[col], errors='coerce')
                    col_mean = self.race_data[col].mean()
                    self.race_data[col] = self.race_data[col].fillna(col_mean if pd.notnull(col_mean) else 0)

        # Merge constructor data
        if all(k in self.kaggle_data for k in ["constructor_standings", "teams", "constructor_results"]):
            merged_constructor = pd.merge(self.kaggle_data["constructor_standings"], 
                                      self.kaggle_data["teams"], 
                                      on="constructorId", how="inner")
            self.constructor_data = pd.merge(merged_constructor, 
                                          self.kaggle_data["constructor_results"],
                                          on=["constructorId", "raceId"], how="inner")

        # Add weather data to race_data
        if "weather" in self.kaggle_data and self.race_data is not None:
            self.race_data = self.race_data.copy()
            
            unique_weather = self.kaggle_data["weather"][["raceId", "weather_numerical"]].drop_duplicates("raceId")
            
            self.race_data = pd.merge(
                self.race_data,
                unique_weather,
                on="raceId",
                how="left"
            )
            
            self.race_data["weather_numerical"] = self.race_data["weather_numerical"].fillna(0)
        
        self.filter_races_with_circuit_data()

    def get_state_representation(self, race_id: int) -> np.ndarray:
        """Create a state representation with detailed dimension checking."""
        race_info = self.race_data[self.race_data['raceId'] == race_id]
        constructor_info = self.constructor_data[self.constructor_data['raceId'] == race_id]
        
        driver_features = self._get_driver_features(race_info)
        print(f"Driver features shape: {driver_features.shape}, Expected: (20,)")
        
        constructor_features = self._get_constructor_features(constructor_info)
        print(f"Constructor features shape: {constructor_features.shape}, Expected: (30,)")
        
        qualifying_features = self._get_qualifying_features(race_info)
        print(f"Qualifying features shape: {qualifying_features.shape}, Expected: (60,)")
        
        weather_features = self._get_weather_features(race_info)
        print(f"Weather features shape: {weather_features.shape}, Expected: (1,)")
        
        circuit_features = self._get_circuit_features(race_info)
        print(f"Circuit features shape: {circuit_features.shape}, Expected: (3,)")
        
        expected_sizes = {
            'driver': 20,
            'constructor': 30,
            'qualifying': 60,
            'weather': 1,
            'circuit': 3
        }
        
        if len(driver_features) != expected_sizes['driver']:
            print(f"WARNING: Padding driver features from {len(driver_features)} to {expected_sizes['driver']}")
            driver_features = np.pad(driver_features, 
                                (0, expected_sizes['driver'] - len(driver_features)), 
                                'constant')
        
        if len(constructor_features) != expected_sizes['constructor']:
            print(f"WARNING: Padding constructor features from {len(constructor_features)} to {expected_sizes['constructor']}")
            constructor_features = np.pad(constructor_features, 
                                        (0, expected_sizes['constructor'] - len(constructor_features)), 
                                        'constant')
        
        if len(qualifying_features) != expected_sizes['qualifying']:
            print(f"WARNING: Padding qualifying features from {len(qualifying_features)} to {expected_sizes['qualifying']}")
            qualifying_features = np.pad(qualifying_features, 
                                    (0, expected_sizes['qualifying'] - len(qualifying_features)), 
                                    'constant')
        
        if len(weather_features) != expected_sizes['weather']:
            print(f"WARNING: Adjusting weather features from {len(weather_features)} to {expected_sizes['weather']}")
            weather_features = weather_features[:expected_sizes['weather']]
        
        if len(circuit_features) != expected_sizes['circuit']:
            print(f"WARNING: Padding circuit features from {len(circuit_features)} to {expected_sizes['circuit']}")
            circuit_features = np.pad(circuit_features, 
                                    (0, expected_sizes['circuit'] - len(circuit_features)), 
                                    'constant')
        
        state = np.concatenate([
            driver_features,
            constructor_features,
            qualifying_features,
            weather_features,
            circuit_features
        ])
        
        print(f"\nFinal state shape: {state.shape}, Expected: (114,)")
        
        if state.shape[0] != 114:
            print(f"ERROR: Incorrect state size. Got {state.shape[0]}, expected 114")
            print("Individual feature sizes:")
            print(f"- Driver features: {len(driver_features)}")
            print(f"- Constructor features: {len(constructor_features)}")
            print(f"- Qualifying features: {len(qualifying_features)}")
            print(f"- Weather features: {len(weather_features)}")
            print(f"- Circuit features: {len(circuit_features)}")
        
        return state

    def _get_driver_features(self, race_info: pd.DataFrame) -> np.ndarray:
        """Extract and normalize driver-related features."""
        current_driver_ids = self.driver_mapping.get_current_driver_ids()
        features = np.zeros((len(current_driver_ids)))
        
        for idx, driver_id in enumerate(current_driver_ids):
            driver_data = race_info[race_info['driverId'] == driver_id]
            if not driver_data.empty:
                position = driver_data['position'].iloc[0]
                features[idx] = 1 - (position - 1) / 20
        
        return features

    def _get_constructor_features(self, constructor_info: pd.DataFrame) -> np.ndarray:
        """Extract and normalize constructor-related features."""
        features = np.zeros(30)
        
        if not constructor_info.empty:
            for idx, (_, row) in enumerate(constructor_info.iterrows()):
                if idx >= 10:
                    break
                    
                base_idx = idx * 3
                features[base_idx] = row['constructor_points'] / max(row['constructor_points'], 1)
                features[base_idx + 1] = 1 - (row['position'] - 1) / 10
                features[base_idx + 2] = row['wins'] / max(row['wins'], 1)
        
        return features

    def _get_qualifying_features(self, race_info: pd.DataFrame) -> np.ndarray:
        """Extract and normalize qualifying features."""
        features = np.zeros(60)
        
        current_driver_ids = self.driver_mapping.get_current_driver_ids()
        
        for idx, driver_id in enumerate(current_driver_ids):
            driver_data = race_info[race_info['driverId'] == driver_id]
            if not driver_data.empty:
                base_idx = idx * 3
                
                for q_idx, session in enumerate(['q1', 'q2', 'q3']):
                    time_str = driver_data[session].iloc[0]
                    time_secs = self._parse_qualifying_time(time_str)
                    if time_secs > 0:
                        features[base_idx + q_idx] = time_secs
        
        non_zero_mask = features > 0
        if non_zero_mask.any():
            max_time = np.max(features[non_zero_mask])
            features[non_zero_mask] = features[non_zero_mask] / max_time
        
        return features
    
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

    def _get_weather_features(self, race_info: pd.DataFrame) -> np.ndarray:
        """Extract weather features."""
        if race_info.empty or 'weather_numerical' not in race_info.columns:
            return np.array([0.0])
        
        weather_value = race_info['weather_numerical'].iloc[0]
        return np.array([float(weather_value)])

    def get_action_space(self) -> Dict[str, List[float]]:
        """Define the action space for the F1 AlphaZero model."""
        return {
            'tire_strategy': list(range(4)),
            'pit_stop_timing': np.linspace(0, 1, 100).tolist(),
            'fuel_mode': list(range(8)),
            'ers_deployment': np.linspace(0, 1, 20).tolist(),
            'wet_weather_setup': [0, 1]
        }

    def get_reward(self, race_id: int, position: int) -> float:
        """Calculate the reward for a given race position."""
        points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 
                        6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        return points_system.get(position, 0)

    def filter_races_with_circuit_data(self):
        """Filter out races that don't have matching circuit data."""
        if self.race_data is None:
            print("No race data available to filter")
            return 0, 0

        initial_race_count = len(self.race_data['raceId'].unique())
        
        valid_races = []
        
        for race_id in self.race_data['raceId'].unique():
            race_info = self.race_data[self.race_data['raceId'] == race_id].iloc[0]
            circuit_name = race_info['name'].split(' ')[0].lower()
            
            if circuit_name == 'são':
                circuit_name = 'sao'
            elif circuit_name == 'méxico':
                circuit_name = 'mexico'
            elif circuit_name == 'monaco':
                circuit_name = 'monte'
                
            if circuit_name in self.circuits_data:
                valid_races.append(race_id)
        
        self.race_data = self.race_data[self.race_data['raceId'].isin(valid_races)].copy()
        
        if self.constructor_data is not None:
            self.constructor_data = self.constructor_data[
                self.constructor_data['raceId'].isin(valid_races)
            ].copy()
        
        final_race_count = len(valid_races)
        
        print(f"\nFiltering Summary:")
        print(f"Initial number of races: {initial_race_count}")
        print(f"Races with valid circuit data: {final_race_count}")
        print(f"Removed races: {initial_race_count - final_race_count}")
        
        kept_circuits = self.race_data['name'].unique()
        print("\nKept Circuits:")
        for circuit in sorted(kept_circuits):
            print(f"✓ {circuit}")
        
        return initial_race_count, final_race_count

class F1Environment:
    """F1 racing environment for AlphaZero training."""
    def __init__(self, data_processor: F1DataProcessor):
        self.data_processor = data_processor
        self.current_race_id = None
        self.current_state = None
        self.current_position = None
        self.lap_number = 0
        self.max_laps = 50
        
        self.points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        
    def reset(self, race_id: Optional[int] = None) -> np.ndarray:
        """Reset the environment to start a new episode."""
        if race_id is None:
            race_ids = self.data_processor.race_data['raceId'].unique()
            race_id = np.random.choice(race_ids)
            
        self.current_race_id = race_id
        self.current_state = self.data_processor.get_state_representation(race_id)
        self.current_position = np.random.randint(1, 21)
        self.lap_number = 0
        
        print(f"Starting new race (ID: {race_id}) from position {self.current_position}")
        return self.current_state
        
    def step(self, action: Dict[str, float]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one time step within the environment."""
        try:
            if not isinstance(action, dict):
                raise ValueError("Action must be a dictionary")
            
            self.lap_number += 1
            
            position_change = self._simulate_position_change(action)
            new_position = max(1, min(20, self.current_position + position_change))
            
            reward = self._calculate_reward(new_position)
            
            self.current_position = new_position
            
            next_state = self._simulate_race_step(action)
            
            done = self._is_race_finished()
            
            info = {
                'position': self.current_position,
                'race_id': self.current_race_id,
                'lap': self.lap_number,
                'points': reward,
                'weather_condition': self._get_weather_condition()
            }
            
            self.current_state = next_state
            return next_state, reward, done, info
            
        except Exception as e:
            print(f"Warning: Error in environment step: {e}")
            return self.current_state, 0.0, True, {
                'position': self.current_position,
                'race_id': self.current_race_id,
                'weather_condition': 'No Rain',
                'error': str(e)
            }
    
    def _simulate_position_change(self, action: Dict[str, float]) -> int:
        """Simulate how actions affect position changes."""
        tire_strategy = action.get('tire_strategy', 0)
        fuel_mode = action.get('fuel_mode', 0)
        ers_deployment = action.get('ers_deployment', 0)
        wet_setup = action.get('wet_weather_setup', 0)
        
        position_change = 0
        
        weather = self._get_weather_condition()
        
        if tire_strategy >= 3:
            if weather == 'Rain' and wet_setup == 0:
                position_change += np.random.choice([1, 2])
            else:
                position_change += np.random.choice([-2, -1, 1])
        else:
            position_change += np.random.choice([-1, 0, 1])
            
        if fuel_mode >= 6:
            position_change += np.random.choice([-1, 0])
        elif fuel_mode <= 2:
            position_change += np.random.choice([0, 1])
            
        if ers_deployment > 0.8:
            position_change += np.random.choice([-1, 0])
        
        max_possible_gain = self.current_position - 1
        max_possible_loss = 20 - self.current_position
        
        return np.clip(position_change, -max_possible_gain, max_possible_loss)
    
    def _calculate_reward(self, position: int) -> float:
        """Calculate reward based on the F1 points system."""
        return self.points_system.get(position, 0)
    
    def _simulate_race_step(self, action: Dict[str, float]) -> np.ndarray:
        """Simulate the effect of actions on the race state."""
        return self.current_state
        
    def _is_race_finished(self) -> bool:
        """Check if the race is finished based on lap number."""
        return self.lap_number >= self.max_laps
        
    def _get_weather_condition(self) -> str:
        """Get the current weather condition for the race."""
        try:
            race_info = self.data_processor.race_data[
                self.data_processor.race_data['raceId'] == self.current_race_id
            ]
            if not race_info.empty and 'weather_numerical' in race_info.columns:
                weather_value = race_info['weather_numerical'].iloc[0]
                return 'Rain' if weather_value == 1 else 'No Rain'
            return 'No Rain'
        except Exception as e:
            print(f"Warning: Could not get weather condition: {e}")
            return 'No Rain'
