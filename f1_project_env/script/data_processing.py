import os
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request
import json
import shutil


# Initialize the Flask app
app = Flask(__name__, template_folder="../templates")  # Adjust the path if templates are elsewhere

# Base URL for OpenF1 API
OPENF1_BASE_URL = "https://api.openf1.org/v1/"

# Helper function to load CSV data
def load_kaggle_data(file_path):
    """
    Load historical F1 data from a CSV file.
    """
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

def clean_circuits_data(data):
    """
    Clean the circuits dataset by removing irrelevant columns and handling missing values.
    """
    if data is not None:
        # Drop irrelevant columns
        columns_to_keep = ["circuitId", "name", "location", "country"]
        data = data[columns_to_keep]

        # Handle missing values (e.g., drop rows where location or country is missing)
        data = data.dropna(subset=["location", "country"])
    return data


def clean_constructors_data(data):
    """
    Clean the constructors dataset by removing irrelevant columns and handling missing values.
    """
    if data is not None:
        # Drop irrelevant columns
        columns_to_keep = ["constructorResultsId", "raceId", "constructorId", "points"]
        data = data[columns_to_keep]
    return data


def clean_qualifying_data(data):
    """
    Clean the qualifying dataset by removing irrelevant columns and handling missing values.
    """
    if data is not None:
        # Drop irrelevant columns
        columns_to_keep = ["qualifyId", "raceId", "driverId", "constructorId", "position", "q1", "q2", "q3"]
        data = data[columns_to_keep]

        # Handle missing qualifying times by replacing with "N/A"
        data.fillna({"q1": "N/A", "q2": "N/A", "q3": "N/A"}, inplace=True)
    return data

## TODO Clean the meetings, drivers, pit stop, and weather data fetched from the OpenF1 API
## It might not work if you run it
def clean_meetings_data(data):
    """
    Clean the meetings data fetched from the OpenF1 API.
    """
    if data is not None and not data.empty:
        # Keep relevant columns
        columns_to_keep = ["circuit_key", "meeting_key", "circuit_short_name", "year"]
        data = data[columns_to_keep]
    return data


def clean_drivers_data(data):
    """
    Clean the drivers data fetched from the OpenF1 API.
    """
    if data is not None and not data.empty:
        # Keep relevant columns
        columns_to_keep = ["driver_number", "first_name", "last_name", "meeting_key", "team_name"]
        data = data[columns_to_keep]
    return data


def clean_pit_data(data):
    """
    Clean the pit stop data fetched from the OpenF1 API.
    """
    if data is not None and not data.empty:
        # Keep relevant columns
        columns_to_keep = ["driver_number", "meeting_key", "pit_duration"]
        data = data[columns_to_keep]
    return data


def clean_weather_data(data):
    """
    Clean the weather data fetched from the OpenF1 API.
    """
    if data is not None and not data.empty:
        # Keep relevant columns
        columns_to_keep = ["humidity", "meeting_key", "pressure", "rainfall", "track_temperature"]
        data = data[columns_to_keep]
    return data


# Load datasets during app startup
data_dir = "f1_project_env/data"
kaggle_data = {
    "circuits": clean_circuits_data(load_kaggle_data(os.path.join(data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/circuits.csv"))),
    "constructors": clean_constructors_data(load_kaggle_data(os.path.join(data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/constructor_results.csv"))),
    "qualifying": clean_qualifying_data(load_kaggle_data(os.path.join(data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/qualifying.csv"))),
}

# Helper function to fetch OpenF1 API data
def fetch_openf1_data(endpoint):
    url = f"{OPENF1_BASE_URL}{endpoint}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            print(f"Failed to fetch data from {endpoint}: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

# Historic Kaggle data endpoints
@app.route("/api/circuits", methods=["GET"])
def get_circuits():
    if kaggle_data["circuits"] is not None:
        return kaggle_data["circuits"].to_json(orient="records")
    return jsonify({"error": "Circuits data not found"}), 404

@app.route("/api/constructors", methods=["GET"])
def get_constructors():
    if kaggle_data["constructors"] is not None:
        return kaggle_data["constructors"].to_json(orient="records")
    return jsonify({"error": "Constructors data not found"}), 404

@app.route("/api/qualifying", methods=["GET"])
def get_qualifying():
    if kaggle_data["qualifying"] is not None:
        return kaggle_data["qualifying"].to_json(orient="records")
    return jsonify({"error": "Qualifying data not found"}), 404

# OpenF1 API endpoints
@app.route("/api/drivers", methods=["GET"])
def get_drivers():
    # Define the directory where you want the JSON file to be saved
    data_dir = os.path.join(os.getcwd(), 'data')  # Assuming 'data' folder is in the current directory
    file_path = os.path.join(data_dir, "drivers.json")
    
    # Ensure the 'data' folder exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if file exists
    if os.path.exists(file_path):
        print(f"File already exists at: {file_path}")
        return jsonify({"message": f"File already exists at: {file_path}"}), 200
    
    # Fetch and process data
    print("Fetching drivers data from OpenF1 API...")
    data = fetch_openf1_data("drivers")
    if data is None or data.empty:
        print("Drivers data not found")
        return jsonify({"error": "Drivers data not found"}), 404
    
    cleaned_data = clean_drivers_data(data)
    print(f"Cleaned data: {cleaned_data.shape[0]} rows")
    
    if cleaned_data is not None:
        # Save to the desired path in the 'data' folder
        cleaned_data.to_json(file_path, orient="records", indent=4)
        print(f"File created successfully at: {file_path}")
        
        return jsonify({"message": f"File created successfully at: {file_path}"}), 201
    
    return jsonify({"error": "Failed to create drivers data"}), 500

@app.route("/api/meetings", methods=["GET"])
def get_meetings():
    # Define the directory where you want the JSON file to be saved
    data_dir = os.path.join(os.getcwd(), 'data')  # Assuming 'data' folder is in the current directory
    file_path = os.path.join(data_dir, "meetings.json")
    
    # Ensure the 'data' folder exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if file exists
    if os.path.exists(file_path):
        print(f"File already exists at: {file_path}")
        return jsonify({"message": f"File already exists at: {file_path}"}), 200
    
    # Fetch and process data
    print("Fetching meetings data from OpenF1 API...")
    data = fetch_openf1_data("meetings")
    if data is None or data.empty:
        print("Meetings data not found")
        return jsonify({"error": "Meetings data not found"}), 404
    
    cleaned_data = clean_meetings_data(data)
    print(f"Cleaned data: {cleaned_data.shape[0]} rows")
    
    if cleaned_data is not None:
        # Save to the desired path in the 'data' folder
        cleaned_data.to_json(file_path, orient="records", indent=4)
        print(f"File created successfully at: {file_path}")
        
        return jsonify({"message": f"File created successfully at: {file_path}"}), 201
    
    return jsonify({"error": "Failed to create meetings data"}), 500


@app.route("/api/pit", methods=["GET"])
def get_pit():
    # Define the directory where you want the JSON file to be saved
    data_dir = os.path.join(os.getcwd(), 'data')  # Assuming 'data' folder is in the current directory
    file_path = os.path.join(data_dir, "pit.json")
    
    # Ensure the 'data' folder exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if file exists
    if os.path.exists(file_path):
        print(f"File already exists at: {file_path}")
        return jsonify({"message": f"File already exists at: {file_path}"}), 200
    
    # Fetch and process data
    print("Fetching pit data from OpenF1 API...")
    data = fetch_openf1_data("pit")
    if data is None or data.empty:
        print("Pit data not found")
        return jsonify({"error": "Pit data not found"}), 404
    
    cleaned_data = clean_pit_data(data)
    print(f"Cleaned data: {cleaned_data.shape[0]} rows")
    
    if cleaned_data is not None:
        # Save to the desired path in the 'data' folder
        cleaned_data.to_json(file_path, orient="records", indent=4)
        print(f"File created successfully at: {file_path}")
        
        return jsonify({"message": f"File created successfully at: {file_path}"}), 201
    
    return jsonify({"error": "Failed to create pit data"}), 500


@app.route("/api/weather", methods=["GET"])
def get_weather():
    # Define the directory where you want the JSON file to be saved
    data_dir = os.path.join(os.getcwd(), 'data')  # Assuming 'data' folder is in the current directory
    file_path = os.path.join(data_dir, "weather.json")
    
    # Ensure the 'data' folder exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if file exists
    if os.path.exists(file_path):
        print(f"File already exists at: {file_path}")
        return jsonify({"message": f"File already exists at: {file_path}"}), 200
    
    # Fetch and process data
    print("Fetching weather data from OpenF1 API...")
    data = fetch_openf1_data("weather")
    if data is None or data.empty:
        print("Weather data not found")
        return jsonify({"error": "Weather data not found"}), 404
    
    cleaned_data = clean_weather_data(data)
    print(f"Cleaned data: {cleaned_data.shape[0]} rows")
    
    if cleaned_data is not None:
        # Save to the desired path in the 'data' folder
        cleaned_data.to_json(file_path, orient="records", indent=4)
        print(f"File created successfully at: {file_path}")
        
        return jsonify({"message": f"File created successfully at: {file_path}"}), 201
    
    return jsonify({"error": "Failed to create weather data"}), 500

# Frontend interaction to fetch data by user input
@app.route("/fetch_data", methods=["POST"])
def fetch_data():
    data_type = request.form.get("data_type")
    if data_type == "circuits":
        return get_circuits()
    elif data_type == "constructors":
        return get_constructors()
    elif data_type == "qualifying":
        return get_qualifying()
    elif data_type == "meetings":
        return get_meetings()
    elif data_type == "drivers":
        return get_drivers()
    elif data_type == "pit":
        return get_pit()
    elif data_type == "weather":
        return get_weather()
    else:
        return jsonify({"error": "Invalid data type requested"}), 400
    

# Run the app
if __name__ == "__main__":
    app.run(debug=True)