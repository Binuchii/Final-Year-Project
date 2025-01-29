import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

class F1DataProcessor:
    """
    Data processor for F1 AlphaZero model that handles data loading, cleaning,
    and state representation for reinforcement learning.
    """
    def __init__(self, data_dir: str, circuits_folder: Optional[str] = None):
        self.data_dir = data_dir
        self.circuits_folder = circuits_folder or os.path.join(data_dir, "calculated_variables")
        self.kaggle_data = {}
        self.circuits_data = {}
        self.race_data = None
        self.constructor_data = None
        self.state_dim = None
        self.action_dim = None
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
            "drivers": self._clean_drivers_data(self._load_kaggle_data(os.path.join(self.data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/drivers.csv"))),
            "races": self._clean_races_data(self._load_kaggle_data(os.path.join(self.data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/races.csv"))),
            "results": self._clean_results_data(self._load_kaggle_data(os.path.join(self.data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/results.csv"))),
            "constructor_results": self._clean_constructorResults_data(self._load_kaggle_data(os.path.join(self.data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/constructor_results.csv"))),
            "constructor_standings": self._clean_Standings_data(self._load_kaggle_data(os.path.join(self.data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/constructor_standings.csv"))),
            "teams": self._clean_teams_data(self._load_kaggle_data(os.path.join(self.data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/constructors.csv"))),
            "qualifying": self._clean_qualifying_data(self._load_kaggle_data(os.path.join(self.data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/qualifying.csv"))),
            "weather": self._clean_weather_data(self._load_kaggle_data(os.path.join(self.data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/cleaned_weather_with_raceId.csv")))
        }
        
        # Load circuit data
        self.circuits_data = {}
        circuits_folder = "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/calculated_variables"
        for filename in os.listdir(circuits_folder):
            if filename.endswith(".csv"):
                circuit_name = filename.split(" ")[0].lower()
                file_path = os.path.join(circuits_folder, filename)
                df = pd.read_csv(file_path)
                self.circuits_data[circuit_name] = self._clean_circuit_data(df)
        
        # Create merged datasets for state representation
        self._create_merged_datasets()
        
    def _clean_drivers_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the drivers dataset."""
        if data is not None:
            columns_to_keep = ["driverId", "number", "forename", "surname"]
            data = data[columns_to_keep]
            if "number" in data.columns:
                data = data[data["number"] != "\\N"]
                data.fillna({"number": "N/A"}, inplace=True)
        return data

    def _clean_races_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the races dataset."""
        if data is not None:
            data = data[data["year"].isin([2021, 2022, 2023, 2024])]
            columns_to_keep = ["raceId", "year", "circuitId", "name"]
            data = data[columns_to_keep]
        return data

    def _clean_results_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the results dataset."""
        if data is not None:
            # Create a copy to avoid SettingWithCopyWarning
            data = data.copy()
            columns_to_keep = ["resultId", "raceId", "driverId", "constructorId", "position"]
            data = data[columns_to_keep]
            # Convert position to numeric, replacing any non-numeric values with NaN
            data.loc[:, 'position'] = pd.to_numeric(data['position'], errors='coerce')
            # Fill NaN values with a default position (e.g., last place)
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
            
            # Convert weather_type to numerical
            weather_mapping = {'No Rain': 0, 'Rain': 1}
            data['weather_numerical'] = data['weather_type'].map(weather_mapping)
        return data

    def _validate_numeric_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Validate and convert columns to numeric type."""
        # Create a copy of the DataFrame to avoid chained assignment warnings
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Calculate mean for non-null values
                col_mean = df[col].mean()
                
                # Fill NA values with mean or 0
                df[col] = df[col].fillna(col_mean if pd.notnull(col_mean) else 0)
        
        return df

    def _load_and_clean_circuit_data(self):
        """Load and clean circuit-specific sector times data."""
        try:
            for filename in os.listdir(self.circuits_folder):
                if filename.endswith(".csv"):
                    # Extract circuit name
                    circuit_name = filename.split(" ")[0].lower()
                    
                    # Load the CSV
                    file_path = os.path.join(self.circuits_folder, filename)
                    df = pd.read_csv(file_path)
                    
                    # Clean the data
                    df = self._clean_circuit_data(df)
                    
                    # Store in dictionary
                    self.circuits_data[circuit_name] = df
                    
            print(f"Loaded {len(self.circuits_data)} circuit datasets")
        except Exception as e:
            print(f"Error loading circuit data: {e}")

    def _clean_circuit_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean circuit-specific data."""
        if df is not None:
            # Create a copy to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Remove NaN values
            df = df.dropna()
            
            # Convert sector times to numeric, handling any string or invalid values
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
            
            # Normalize sector times
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
        """Extract circuit-specific features with proper error handling."""
        try:
            if race_info.empty:
                print("Warning: Empty race info provided")
                return np.zeros(3)  # Return default features for empty race info
                
            # Get circuit name from race info
            if 'name' not in race_info.columns:
                print("Warning: 'name' column not found in race info")
                return np.zeros(3)
                
            circuit_name = race_info['name'].iloc[0].split(' ')[0].lower()
            
            if circuit_name in self.circuits_data:
                circuit_df = self.circuits_data[circuit_name]
                
                # Initialize features list with default zeros
                sector_features = []
                
                # Check if we have the normalized columns
                normalized_cols = ['sector1_normalized', 'sector2_normalized', 'sector3_normalized']
                if any(col in circuit_df.columns for col in normalized_cols):
                    # Use normalized columns if they exist
                    for col in normalized_cols:
                        if col in circuit_df.columns:
                            sector_features.append(circuit_df[col].mean())
                        else:
                            sector_features.append(0.0)
                else:
                    # Fall back to regular sector columns if normalized ones don't exist
                    regular_cols = ['sector1', 'sector2', 'sector3']
                    max_sector_time = 0
                    
                    # First find max sector time for normalization
                    for col in regular_cols:
                        if col in circuit_df.columns:
                            max_sector_time = max(max_sector_time, circuit_df[col].max())
                    
                    # Then normalize and add features
                    if max_sector_time > 0:
                        for col in regular_cols:
                            if col in circuit_df.columns:
                                sector_features.append(circuit_df[col].mean() / max_sector_time)
                            else:
                                sector_features.append(0.0)
                    else:
                        sector_features = [0.0, 0.0, 0.0]
                
                # Convert to numpy array and ensure we have 3 features
                features = np.array(sector_features[:3])
                if len(features) < 3:
                    features = np.pad(features, (0, 3 - len(features)), 'constant')
                    
                return features
                
            else:
                print(f"Warning: Circuit '{circuit_name}' not found in circuits data")
                return np.zeros(3)
                
        except Exception as e:
            print(f"Error getting circuit features: {str(e)}")
            return np.zeros(3)  # Return default features on error
        
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
            self.race_data = self._validate_numeric_columns(self.race_data, numeric_columns)

        # Merge constructor data
        if all(k in self.kaggle_data for k in ["constructor_standings", "teams", "constructor_results"]):
            merged_constructor = pd.merge(self.kaggle_data["constructor_standings"], 
                                      self.kaggle_data["teams"], 
                                      on="constructorId", how="inner")
            self.constructor_data = pd.merge(merged_constructor, 
                                          self.kaggle_data["constructor_results"],
                                          on=["constructorId", "raceId"], how="inner")

        # Add weather data to race data
        if "weather" in self.kaggle_data and self.race_data is not None:
            # Create a copy of race_data before modifications
            self.race_data = self.race_data.copy()
            
            # Merge weather data
            self.race_data = pd.merge(self.race_data, 
                                    self.kaggle_data["weather"][["raceId", "weather_numerical"]], 
                                    on="raceId", how="left")
            
            # Fill NA values without using inplace
            self.race_data["weather_numerical"] = self.race_data["weather_numerical"].fillna(0)  # Default to no rain
        
        self.filter_races_with_circuit_data()

    def get_state_representation(self, race_id: int) -> np.ndarray:
        """
        Create a state representation for the AlphaZero model for a given race.
        Returns a numpy array representing the current state.
        """
        race_info = self.race_data[self.race_data['raceId'] == race_id]
        constructor_info = self.constructor_data[self.constructor_data['raceId'] == race_id]
        
        # Create feature vectors
        driver_features = self._get_driver_features(race_info)
        constructor_features = self._get_constructor_features(constructor_info)
        qualifying_features = self._get_qualifying_features(race_info)
        weather_features = self._get_weather_features(race_info)
        circuit_features = self._get_circuit_features(race_info)
        
        # Combine all features into state representation
        state = np.concatenate([
            driver_features,
            constructor_features,
            qualifying_features,
            weather_features,
            circuit_features
        ])
        
        # Print feature dimensions for debugging
        print("\nState vector composition:")
        print(f"Driver features: {len(driver_features)}")
        print(f"Constructor features: {len(constructor_features)}")
        print(f"Qualifying features: {len(qualifying_features)}")
        print(f"Weather features: {len(weather_features)}")
        print(f"Circuit features: {len(circuit_features)}")
        print(f"Total state size: {len(state)}")
        
        return state

    def _get_driver_features(self, race_info: pd.DataFrame) -> np.ndarray:
        """Extract and normalize driver-related features."""
        # Example features: driver position, points, wins
        features = race_info[['position']].values
        # Normalize positions to [0,1] range
        features = features / 20.0  # Assuming max 20 positions
        return features.flatten()

    def _get_constructor_features(self, constructor_info: pd.DataFrame) -> np.ndarray:
        """Extract and normalize constructor-related features."""
        # Example features: constructor points, position, wins
        features = constructor_info[['constructor_points', 'position', 'wins']].values
        # Normalize features
        features[:, 0] = features[:, 0] / features[:, 0].max()  # points
        features[:, 1] = features[:, 1] / 10.0  # position (assuming 10 teams)
        features[:, 2] = features[:, 2] / features[:, 2].max()  # wins
        return features.flatten()

    def _get_qualifying_features(self, race_info: pd.DataFrame) -> np.ndarray:
        """Extract and normalize qualifying-related features."""
        def convert_time_to_seconds(time_str):
            if pd.isna(time_str) or time_str == 'N/A' or time_str == '\\N':
                return 0
            try:
                if isinstance(time_str, str):
                    # Handle minute:second.millisecond format
                    if ':' in time_str:
                        minutes, rest = time_str.split(':')
                        seconds = float(minutes) * 60 + float(rest)
                    else:
                        seconds = float(time_str)
                    return seconds
                return float(time_str)
            except (ValueError, TypeError):
                return 0

        # Convert qualifying times to seconds
        q_times = pd.DataFrame()
        for col in ['q1', 'q2', 'q3']:
            q_times[col] = race_info[col].apply(convert_time_to_seconds)

        # Convert to numpy array and handle normalization
        q_times_array = q_times.values.astype(float)
        max_time = np.max(q_times_array[q_times_array > 0])
        if max_time > 0:
            q_times_array = np.where(q_times_array > 0, q_times_array / max_time, 0)

        return q_times_array.flatten()

    def _get_weather_features(self, race_info: pd.DataFrame) -> np.ndarray:
        """Extract weather features."""
        # Get weather condition (0 for no rain, 1 for rain)
        weather_features = race_info[['weather_numerical']].values
        return weather_features.flatten()

    def get_action_space(self) -> Dict[str, List[float]]:
        """
        Define the action space for the F1 AlphaZero model.
        Returns a dictionary of possible actions and their ranges.
        """
        return {
            'tire_strategy': list(range(4)),  # 4 different tire compounds
            'pit_stop_timing': np.linspace(0, 1, 100).tolist(),  # Normalized lap timing
            'fuel_mode': list(range(8)),  # Different engine modes
            'ers_deployment': np.linspace(0, 1, 20).tolist(),  # Energy deployment strategy
            'wet_weather_setup': [0, 1]  # Dry or wet weather setup
        }

    def get_reward(self, race_id: int, position: int) -> float:
        """
        Calculate the reward for a given race position.
        Returns a float value representing the reward.
        """
        # F1 points system
        points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 
                        6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        return points_system.get(position, 0)

    def filter_races_with_circuit_data(self):
        """
        Filter out races that don't have matching circuit data.
        Updates race_data and constructor_data in place.
        Returns the number of races before and after filtering.
        """
        if self.race_data is None:
            print("No race data available to filter")
            return 0, 0

        initial_race_count = len(self.race_data['raceId'].unique())
        
        # Create a mask for races with valid circuit data
        valid_races = []
        
        for race_id in self.race_data['raceId'].unique():
            race_info = self.race_data[self.race_data['raceId'] == race_id].iloc[0]
            circuit_name = race_info['name'].split(' ')[0].lower()
            
            # Apply the same fixes as in fix_circuit_name
            if circuit_name == 'são':
                circuit_name = 'sao'
            elif circuit_name == 'méxico':
                circuit_name = 'mexico'
            elif circuit_name == 'monaco':
                circuit_name = 'monte'
                
            if circuit_name in self.circuits_data:
                valid_races.append(race_id)
        
        # Filter race_data
        self.race_data = self.race_data[self.race_data['raceId'].isin(valid_races)].copy()
        
        # Filter constructor_data if it exists
        if self.constructor_data is not None:
            self.constructor_data = self.constructor_data[
                self.constructor_data['raceId'].isin(valid_races)
            ].copy()
        
        final_race_count = len(valid_races)
        
        print(f"\nFiltering Summary:")
        print(f"Initial number of races: {initial_race_count}")
        print(f"Races with valid circuit data: {final_race_count}")
        print(f"Removed races: {initial_race_count - final_race_count}")
        
        # Print which circuits were kept
        kept_circuits = self.race_data['name'].unique()
        print("\nKept Circuits:")
        for circuit in sorted(kept_circuits):
            print(f"✓ {circuit}")
        
        return initial_race_count, final_race_count

class F1Environment:
    """
    F1 racing environment for AlphaZero training.
    """
    def __init__(self, data_processor: F1DataProcessor):
        self.data_processor = data_processor
        self.current_race_id = None
        self.current_state = None
        
    def reset(self, race_id: Optional[int] = None) -> np.ndarray:
        """Reset the environment to start a new episode."""
        if race_id is None:
            # Randomly select a race from available races
            race_ids = self.data_processor.race_data['raceId'].unique()
            race_id = np.random.choice(race_ids)
            
        self.current_race_id = race_id
        self.current_state = self.data_processor.get_state_representation(race_id)
        return self.current_state
        
    def step(self, action: Dict[str, float]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step within the environment.
        Returns next_state, reward, done, info
        """
        try:
            # Validate action
            if not isinstance(action, dict):
                raise ValueError("Action must be a dictionary")
            
            # Simulate the effect of actions on the race
            next_state = self._simulate_race_step(action)
            
            # Get the current position and calculate reward
            current_position = self._get_current_position()
            reward = self.data_processor.get_reward(self.current_race_id, current_position)
            
            # Check if race is finished
            done = self._is_race_finished()
            
            # Additional info with error handling
            info = {
                'position': current_position,
                'race_id': self.current_race_id,
                'weather_condition': self._get_weather_condition()
            }
            
            self.current_state = next_state
            return next_state, reward, done, info
            
        except Exception as e:
            print(f"Warning: Error in environment step: {e}")
            # Return safe default values
            return self.current_state, 0.0, True, {
                'position': 0,
                'race_id': self.current_race_id,
                'weather_condition': 'No Rain',
                'error': str(e)
            }
    
    def _simulate_race_step(self, action: Dict[str, float]) -> np.ndarray:
        """Simulate the effect of actions on the race state."""
        # Implementation would depend on specific racing simulation logic
        # This is a placeholder that returns the current state
        return self.current_state
        
    def _get_current_position(self) -> int:
        """Get the current race position."""
        # Implementation would depend on how position is tracked
        # This is a placeholder
        return 1
        
    def _is_race_finished(self) -> bool:
        """Check if the race is finished."""
        # Implementation would depend on race completion criteria
        # This is a placeholder
        return False

    def _get_weather_condition(self) -> str:
        """Get the current weather condition for the race."""
        try:
            race_info = self.data_processor.race_data[
                self.data_processor.race_data['raceId'] == self.current_race_id
            ]
            if not race_info.empty and 'weather_numerical' in race_info.columns:
                # Convert numerical weather back to string representation
                weather_value = race_info['weather_numerical'].iloc[0]
                return 'Rain' if weather_value == 1 else 'No Rain'
            return 'No Rain'  # Default value if no weather data is found
        except Exception as e:
            print(f"Warning: Could not get weather condition: {e}")
            return 'No Rain'