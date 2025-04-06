from f1_data_processing import F1DataProcessor, F1Environment, F1DriverMapping
import pandas as pd
import numpy as np
import os

def test_driver_mapping():
    """
    Test the F1DriverMapping functionality.
    """
    print("\n=== Test 0: Driver Mapping ===")
    driver_mapping = F1DriverMapping()
    
    # Test driver code retrieval
    print("\nTesting driver code retrieval:")
    test_cases = [
        (830, 'VER'),  # Max Verstappen
        (1, 'HAM'),    # Lewis Hamilton
        ('44', 'HAM'), # Hamilton by number
        ('1', 'VER')   # Verstappen by number
    ]
    
    for input_val, expected in test_cases:
        result = driver_mapping.get_driver_code(input_val)
        print(f"Input: {input_val}, Expected: {expected}, Got: {result}")
        
    # Test driver name retrieval
    print("\nTesting driver name retrieval:")
    test_cases = [
        (830, 'Max Verstappen'),
        ('VER', 'Max Verstappen'),
        ('1', 'Max Verstappen'),
    ]
    
    for input_val, expected in test_cases:
        result = driver_mapping.get_driver_name(input_val)
        print(f"Input: {input_val}, Expected: {expected}, Got: {result}")
    
    # Test current driver IDs
    print("\nCurrent driver IDs:")
    current_ids = driver_mapping.get_current_driver_ids()
    print(f"Number of current drivers: {len(current_ids)}")
    print(f"Driver IDs: {current_ids}")

def test_data_processing():
    """
    Test the F1 data processing pipeline and print diagnostic information.
    """
    # Update these paths to your actual data directories
    data_dir = os.path.join(os.getcwd(), "data")
    circuits_folder = os.path.join(os.getcwd(), r"C:\Users\Albin Binu\Documents\College\Year 4\Final Year Project\f1_project_env\data\calculated_variables")
    
    print("\nInitializing F1DataProcessor...")
    processor = F1DataProcessor(data_dir=data_dir, circuits_folder=circuits_folder)

    # Test 1: Check if all datasets are loaded
    print("\n=== Test 1: Dataset Loading ===")
    for dataset_name, dataset in processor.kaggle_data.items():
        if dataset is not None:
            print(f"{dataset_name}: {len(dataset)} rows, {len(dataset.columns)} columns")
            print(f"Columns: {dataset.columns.tolist()}")
            
            # For drivers dataset, check driver code mapping
            if dataset_name == "drivers":
                print("\nChecking driver code mapping:")
                current_drivers = processor.driver_mapping.get_current_driver_ids()
                mapped_drivers = dataset[dataset['driverId'].isin(current_drivers)]
                print(f"Current drivers found in dataset: {len(mapped_drivers)}")
                if not mapped_drivers.empty:
                    print("Sample driver mappings:")
                    for _, driver in mapped_drivers.head().iterrows():
                        code = processor.driver_mapping.get_driver_code(driver['driverId'])
                        name = processor.driver_mapping.get_driver_name(driver['driverId'])
                        print(f"ID: {driver['driverId']}, Code: {code}, Name: {name}")
        else:
            print(f"{dataset_name}: Failed to load")

    # Test 2: Check merged datasets
    print("\n=== Test 2: Merged Datasets ===")
    if hasattr(processor, 'race_data'):
        print("\nRace Data:")
        print(f"Shape: {processor.race_data.shape}")
        print(f"Columns: {processor.race_data.columns.tolist()}")
        print("\nSample row with driver mapping:")
        sample_row = processor.race_data.iloc[0]
        print(sample_row)
        print("\nDriver details for sample row:")
        driver_id = sample_row['driverId']
        print(f"Driver Code: {processor.driver_mapping.get_driver_code(driver_id)}")
        print(f"Driver Name: {processor.driver_mapping.get_driver_name(driver_id)}")
    else:
        print("Race data merge failed")

    if hasattr(processor, 'constructor_data'):
        print("\nConstructor Data:")
        print(f"Shape: {processor.constructor_data.shape}")
        print(f"Columns: {processor.constructor_data.columns.tolist()}")
        print("\nSample row:")
        print(processor.constructor_data.iloc[0])
    else:
        print("Constructor data merge failed")

    # Test 3: Check state representation
    print("\n=== Test 3: State Representation ===")
    try:
        sample_race_id = processor.race_data['raceId'].iloc[0]
        state = processor.get_state_representation(sample_race_id)
        print(f"\nState shape: {state.shape}")
        print(f"State sample: {state[:10]}...")  # Print first 10 elements
        
        # Analyze state components
        print("\nState vector breakdown:")
        offset = 0
        
        # Driver features (20 drivers)
        driver_features = state[offset:offset+20]
        print(f"\nDriver features (positions):")
        for i, pos in enumerate(driver_features):
            if pos > 0:  # Only show non-zero positions
                driver_id = processor.driver_mapping.get_current_driver_ids()[i]
                driver_code = processor.driver_mapping.get_driver_code(driver_id)
                print(f"{driver_code}: {pos:.3f}")
        offset += 20
        
        # Constructor features
        constructor_features = state[offset:offset+30]  # Assuming 10 teams × 3 features
        print(f"\nConstructor features shape: {len(constructor_features)}")
        offset += 30
        
        # Qualifying features (20 drivers × 3 sessions)
        qualifying_features = state[offset:offset+60]
        print(f"\nQualifying features shape: {len(qualifying_features)}")
        offset += 60
        
        # Weather feature
        weather_feature = state[offset:offset+1]
        print(f"\nWeather feature: {weather_feature[0]}")
        offset += 1
        
        # Circuit features
        circuit_features = state[offset:]
        print(f"\nCircuit features: {circuit_features}")
        
    except Exception as e:
        print(f"State representation test failed: {str(e)}")

    # Test 4: Check weather integration
    print("\n=== Test 4: Weather Data Integration ===")
    if 'weather_numerical' in processor.race_data.columns:
        weather_stats = processor.race_data['weather_numerical'].value_counts()
        print("\nWeather conditions distribution:")
        print(weather_stats)
        
        missing_weather = processor.race_data['weather_numerical'].isna().sum()
        print(f"\nMissing weather data points: {missing_weather}")
    else:
        print("Weather data not properly integrated")

    # Test 5: Environment Creation
    print("\n=== Test 5: Environment Testing ===")
    try:
        env = F1Environment(processor)
        initial_state = env.reset()
        print(f"\nInitial state shape: {initial_state.shape}")
        
        action = {
            'tire_strategy': 0,
            'pit_stop_timing': 0.5,
            'fuel_mode': 3,
            'ers_deployment': 0.7,
            'wet_weather_setup': 0
        }
        next_state, reward, done, info = env.step(action)
        print("\nEnvironment step test:")
        print(f"Reward: {reward}")
        print(f"Info: {info}")
    except Exception as e:
        print(f"Environment test failed: {str(e)}")

def test_circuit_data(processor):
    """
    Test the circuit sector times data processing and integration with driver information.
    """
    print("\n=== Circuit Sector Data Analysis ===")
    
    if not hasattr(processor, 'circuits_data'):
        print("No circuit data found in the processor")
        return
        
    print(f"Total circuits loaded: {len(processor.circuits_data)}\n")
    
    # Process each circuit
    for circuit_name, circuit_df in processor.circuits_data.items():
        if circuit_df.empty:
            continue
            
        print("=" * 50)
        print(f"Circuit: {circuit_name.upper()}")
        print(f"Total Drivers: {len(circuit_df)}")
        print("-" * 50)
        
        # Display top 3 drivers' sector times
        print("\nTop 3 Drivers' Performance:")
        for idx in range(min(1, len(circuit_df))):
            driver = circuit_df.iloc[idx]
            print(f"\n{idx + 1}. {driver['Driver']} (#{driver['DriverNumber']})")
            print(f"   Position Change: P{int(driver['InitialPosition'])} → P{int(driver['FinalPosition'])}")
            print(f"   Sector Times:")
            print(f"     S1: {driver['AvgSector1']:.3f}")
            print(f"     S2: {driver['AvgSector2']:.3f}")
            print(f"     S3: {driver['AvgSector3']:.3f}")
            print(f"   Avg Lap: {driver['AvgLapTime']:.3f}")
            print(f"   Std Dev: {driver['LapTimeSTD']:.3f}")
        
        # Sector time analysis
        print("\nSector Analysis:")
        for sector in ['AvgSector1', 'AvgSector2', 'AvgSector3']:
            stats = circuit_df[sector].describe()
            fastest_driver = circuit_df.loc[circuit_df[sector].idxmin(), 'Driver']
            slowest_driver = circuit_df.loc[circuit_df[sector].idxmax(), 'Driver']
            
            print(f"\n{sector}:")
            print(f"  Mean: {stats['mean']:.3f}")
            print(f"  Fastest: {stats['min']:.3f} ({fastest_driver})")
            print(f"  Slowest: {stats['max']:.3f} ({slowest_driver})")
        
        # Overall fastest driver
        circuit_df['TotalTime'] = circuit_df['AvgSector1'] + circuit_df['AvgSector2'] + circuit_df['AvgSector3']
        fastest_idx = circuit_df['TotalTime'].idxmin()
        fastest = circuit_df.loc[fastest_idx]
        
        print("\nFastest Overall:")
        print(f"Driver: {fastest['Driver']} (#{fastest['DriverNumber']})")
        print(f"Total Time: {fastest['TotalTime']:.3f}")
        print(f"Sectors: {fastest['AvgSector1']:.3f}, {fastest['AvgSector2']:.3f}, {fastest['AvgSector3']:.3f}")
        print("\n")

def run_detailed_circuit_analysis(processor, specific_circuit=None):
    """
    Run a detailed analysis of circuit data with optional focus on a specific circuit.
    
    Args:
        processor: The F1DataProcessor instance
        specific_circuit: Optional string naming a specific circuit to analyze
    """
    if specific_circuit:
        if specific_circuit.lower() in processor.circuits_data:
            circuit_df = processor.circuits_data[specific_circuit.lower()]
            process_single_circuit(specific_circuit, circuit_df)
        else:
            print(f"Circuit '{specific_circuit}' not found in data")
    else:
        test_circuit_data(processor)

def process_single_circuit(circuit_name, circuit_df):
    """
    Process and display detailed analysis for a single circuit.
    """
    print(f"\n=== Detailed Analysis for {circuit_name.upper()} ===")
    
    # Sort drivers by total time
    circuit_df['TotalTime'] = circuit_df['AvgSector1'] + circuit_df['AvgSector2'] + circuit_df['AvgSector3']
    sorted_df = circuit_df.sort_values('TotalTime')
    
    print(f"\nDriver Rankings (by total time):")
    print("-" * 50)
    print(f"{'Pos':>3} {'Driver':10} {'#':>3} {'S1':>8} {'S2':>8} {'S3':>8} {'Total':>8}")
    print("-" * 50)
    
    for idx, driver in enumerate(sorted_df.itertuples(), 1):
        print(f"{idx:3} {driver.Driver:10} {driver.DriverNumber:3} "
              f"{driver.AvgSector1:8.3f} {driver.AvgSector2:8.3f} "
              f"{driver.AvgSector3:8.3f} {driver.TotalTime:8.3f}")
    
    # Gap analysis
    fastest_time = sorted_df['TotalTime'].min()
    print("\nGap to Fastest:")
    print("-" * 50)
    for idx, driver in enumerate(sorted_df.itertuples(), 1):
        if idx == 1:
            continue
        gap = driver.TotalTime - fastest_time
        print(f"{driver.Driver:10} +{gap:6.3f}")


if __name__ == "__main__":
    # Run tests
    test_driver_mapping()
    
    # Initialize processor
    data_dir = r"C:\Users\Albin Binu\Documents\College\Year 4\Final Year Project\f1_project_env\api\data"
    circuits_folder = r"C:\Users\Albin Binu\Documents\College\Year 4\Final Year Project\f1_project_env\api\data\calculated_variables"
    processor = F1DataProcessor(data_dir=data_dir, circuits_folder=circuits_folder)
    
    # Run all tests
    test_data_processing()
    test_circuit_data(processor)
    
    # Optional: Run detailed analysis for a specific circuit
    # run_detailed_circuit_analysis(processor, "monza")