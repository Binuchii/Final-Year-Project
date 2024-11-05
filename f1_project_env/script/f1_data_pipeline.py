import requests
import pandas as pd
import os

# BASE URL for OpenF1 API
OPENF1_BASE_URL = "https://api.openf1.org/v1/"

# Helper function to make GET requests to the OpenF1 API
def fetch_data(endpoint):
    url = f"{OPENF1_BASE_URL}{endpoint}"
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            print(f"Fetched data from {endpoint}")
            return pd.DataFrame(response.json())
        else:
            print(f"Failed to retrieve data from {endpoint}: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error occurred: {e}")
        return None

# Functions to retrieve specific data from OpenF1 API
def get_meetings():
    return fetch_data("meetings")

def get_drivers():
    return fetch_data("drivers")

def get_sessions():
    return fetch_data("sessions")

def get_pit():
    return fetch_data("pit")

def get_weather():
    return fetch_data("weather")

# Add more functions for each API method as needed

# Function to Load Kaggle CSV Data
def load_kaggle_data(file_path):
    """
    Load historical F1 data from a CSV file downloaded from Kaggle.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Function to Combine Kaggle and OpenF1 Data
def preprocess_and_merge_data(kaggle_data, api_data):
    """
    Clean and merge Kaggle historical data with OpenF1 real-time data.
    """
    # Ensure both datasets are not None
    if kaggle_data is None or api_data is None:
        print("One or both datasets are None, cannot merge.")
        return None

    # Example: Clean Kaggle data by dropping rows with missing values
    kaggle_data_clean = kaggle_data.dropna()

    # Merge Kaggle and API data on a common column (e.g., 'raceId', 'driverId')
    if 'raceId' in kaggle_data_clean.columns and 'raceId' in api_data.columns:
        merged_data = pd.merge(kaggle_data_clean, api_data, on='raceId', how='outer')
        print("Merged Kaggle and OpenF1 data on 'raceId'")
        return merged_data
    else:
        print("Common key not found in both datasets for merging.")
        return None
    
# Main Pipeline Execution
def main():
    # Step 1: Load Kaggle data files
    kaggle_circuits_data = load_kaggle_data("f1_project_env\data\circuits.csv")
    kaggle_constructor_data = load_kaggle_data("f1_project_env\data\constructor_results.csv")
    kaggle_qualifying_data = load_kaggle_data("f1_project_env\data\qualifying.csv")

    # Step 2: Fetch OpenF1 data
    meetings = get_meetings()
    drivers = get_drivers()
    pit = get_pit()
    sessions = get_sessions()
    weather = get_weather()

if __name__ == "__main__":
    main()
