import os
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request
import json

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


def clean_constructorResults_data(data):
    """
    Clean the constructors dataset by removing irrelevant columns, handling missing values, 
    and filtering out rows with raceId lower than or equal to 1053.
    """
    if data is not None:
        # Drop irrelevant columns
        columns_to_keep = ["constructorResultsId", "raceId", "constructorId", "points"]
        data = data[columns_to_keep]
        
        # Filter out rows where raceId is less than or equal to 1053
        data = data[data['raceId'] >= 1053]
        data = data.rename(columns={'points': 'points_AtRace'})
        
    return data


def clean_qualifying_data(data):
    """
    Clean the qualifying dataset by removing irrelevant columns and handling missing values.
    """
    if data is not None:
        # Drop irrelevant columns
        columns_to_keep = ["qualifyId", "raceId", "driverId", "q1", "q2", "q3"]
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
        columns_to_keep = ["circuit_key", "meeting_key", "circuit_short_name", "location", "country_name"]
        data = data[columns_to_keep]
    return data


def clean_drivers_data(data):
    """
    Clean the drivers data fetched from the OpenF1 API.
    """
    if data is not None and not data.empty:
        # Keep relevant columns
        columns_to_keep = ["driverId","number","forename","surname"]
        data = data[columns_to_keep]

        # Remove rows where "number" is "//N"
        if "number" in data.columns:
            data = data[data["number"] != "\\N"]
            data.fillna({"number": "N/A"}, inplace=True)

    return data

def clean_races_data(data):
    """
    Clean the drivers data fetched from the OpenF1 API, keeping only the data for the years 2023-2024.
    """
    if data is not None and not data.empty:
        # Filter data to only include 2023
        data = data[data["year"].isin([2021,2022,2023, 2024])]

        # Keep relevant columns
        columns_to_keep = ["raceId", "year", "circuitId", "name"]
        data = data[columns_to_keep]

    return data

def clean_Standings_data(data):
    """
    Clean the standings data fetched from the OpenF1 API.
    """
    if data is not None and not data.empty:
        # Keep relevant columns
        columns_to_keep = ["constructorStandingsId", "raceId", "constructorId", "points", "position", "wins"]
        data = data[columns_to_keep]

        # Filter out rows where raceId is less than or equal to 1053
        data = data[data['raceId'] >= 1053]
        data = data.rename(columns={'points': 'constructor_points'})
    return data

def clean_teams_data(data):
    """
    Clean the teams data fetched from the OpenF1 API.
    """
    if data is not None and not data.empty:
        # Keep relevant columns
        columns_to_keep = ["constructorId", "name"]
        data = data[columns_to_keep]
    return data

def clean_results_data(data):
    """
    Clean the results data fetched from the OpenF1 API.
    """
    if data is not None and not data.empty:
        # Keep relevant columns
        columns_to_keep = ["resultId", "raceId", "driverId", "constructorId", "position"]
        data = data[columns_to_keep]
    return data

def clean_weather_data(data):
    if data is not None and not data.empty:
        # Keep relevant columns
        columns_to_keep = ["date", "meeting_key", "rainfall"]
        data = data[columns_to_keep]

        # Slice the first 10 characters of the 'date' column
        data['date'] = data['date'].str.slice(0, 10)

        # Group by date and calculate the average rainfall
        data = data.groupby(['date', 'meeting_key'], as_index=False).agg({'rainfall': 'mean'})

        # Add a column indicating if it rained or not
        data['weather_type'] = data['rainfall'].apply(lambda x: 'Rain' if x > 0 else 'No Rain')

        # Drop the rainfall column if no longer needed
        data = data.drop(columns=['rainfall'])

    return data


# Load datasets during app startup
data_dir = "f1_project_env/data"
kaggle_data = {
    "drivers": clean_drivers_data(load_kaggle_data(os.path.join(data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/drivers.csv"))),
    "races": clean_races_data(load_kaggle_data(os.path.join(data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/races.csv"))),
    "results": clean_results_data(load_kaggle_data(os.path.join(data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/results.csv"))),
    "constructorResults": clean_constructorResults_data(load_kaggle_data(os.path.join(data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/constructor_results.csv"))),
    "constructorStandings": clean_Standings_data(load_kaggle_data(os.path.join(data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/constructor_standings.csv"))),
    "teams": clean_teams_data(load_kaggle_data(os.path.join(data_dir, "C:/Users/Albin Binu/Documents/College/Year 4/Final Year Project/f1_project_env/data/constructors.csv"))),
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
@app.route("/api/drivers", methods=["GET"])
def get_drivers():
    if kaggle_data["drivers"] is not None:
        return kaggle_data["drivers"].to_json(orient="records")
    return jsonify({"error": "Drivers data not found"}), 404

@app.route("/api/teams", methods=["GET"])
def get_teams():
    if kaggle_data["teams"] is not None:
        return kaggle_data["teams"].to_json(orient="records")
    return jsonify({"error": "Teams data not found"}), 404

@app.route("/api/standings", methods=["GET"])
def get_Standings():
    if kaggle_data["constructorStandings"] is not None:
        return kaggle_data["constructorStandings"].to_json(orient="records")
    return jsonify({"error": "Standings data not found"}), 404

@app.route("/api/races", methods=["GET"])
def get_races():
    if kaggle_data["races"] is not None:
        return kaggle_data["races"].to_json(orient="records")
    return jsonify({"error": "Drivers data not found"}), 404

@app.route("/api/results", methods=["GET"])
def get_results():
    if kaggle_data["results"] is not None:
        return kaggle_data["results"].to_json(orient="records")
    return jsonify({"error": "Drivers data not found"}), 404

@app.route("/api/constructorResults", methods=["GET"])
def get_constructorResults():
    if kaggle_data["constructorResults"] is not None:
        return kaggle_data["constructorResults"].to_json(orient="records")
    return jsonify({"error": "Constructors Results data not found"}), 404

@app.route("/api/qualifying", methods=["GET"])
def get_qualifying():
    if kaggle_data["qualifying"] is not None:
        return kaggle_data["qualifying"].to_json(orient="records")
    return jsonify({"error": "Qualifying data not found"}), 404

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


@app.route("/api/weather", methods=["GET"])
def get_weather():
    data = fetch_openf1_data("weather")
    cleaned_data = clean_weather_data(data)
    if cleaned_data is not None:
        return cleaned_data.to_json(orient="records")
    return jsonify({"error": "Weather data not found"}), 404

# Folder where CSV files are stored
CIRCUITS_FOLDER = r"C:\Users\Albin Binu\Documents\College\Year 4\Final Year Project\f1_project_env\data\calculated_variables"

# Dictionary to store cleaned DataFrames
circuits_data = {} 

def load_and_clean_circuit_data():
    # Iterate through all CSV files in the folder
    for filename in os.listdir(CIRCUITS_FOLDER):
        if filename.endswith(".csv"):
            # Extract the circuit name (e.g., "monaco" from "monaco_circuit.csv")
            circuit_name = filename.split(" ")[0].lower()
            
            # Load the CSV into a DataFrame
            file_path = os.path.join(CIRCUITS_FOLDER, filename)
            df = pd.read_csv(file_path)
            
            # Apply your cleaning logic (example: remove NaNs)
            df = df.dropna()
            
            # Store the cleaned DataFrame in a dictionary with the circuit name as the key
            circuits_data[circuit_name] = df

# Load the data when the script runs
load_and_clean_circuit_data()

@app.route("/api/circuit/<string:circuit_name>", methods=["GET"])
def get_circuit_data(circuit_name):
    # Check if the circuit exists in the dictionary
    circuit_name = circuit_name.lower()
    if circuit_name in circuits_data:
        # Convert the DataFrame to a JSON response
        return jsonify(circuits_data[circuit_name].to_dict(orient="records"))
    else:
        return jsonify({"error": "Circuit not found"}), 404
    
# # Flask route to list all available circuits
# @app.route("/api/circuits", methods=["GET"])
# def list_all_circuits():
#     """
#     List all available circuits by name.
#     """
#     available_circuits = list(circuit_data_dict.keys())
#     return jsonify({"available_circuits": available_circuits})

@app.route("/api/race_merge", methods=["GET"])
def get_race_merge():
    """
    Join the cleaned races, results, qualifying, and drivers data on "raceId" and "driverId".
    """
    races_data = kaggle_data["races"]
    results_data = kaggle_data["results"]
    qualifying_data = kaggle_data["qualifying"]
    drivers_data = kaggle_data["drivers"]

    if races_data is not None and results_data is not None and qualifying_data is not None and drivers_data is not None:
        # Merge races and results on "raceId"
        merged_data = pd.merge(races_data, results_data, on="raceId", how="inner")
        
        # Merge the resulting dataset with qualifying data on "raceId" and "driverId"
        merged_data = pd.merge(merged_data, qualifying_data, on=["raceId", "driverId"], how="inner")
        
        # Merge the resulting dataset with drivers data on "driverId"
        merged_data = pd.merge(merged_data, drivers_data, on="driverId", how="inner")

        return merged_data.to_json(orient="records", force_ascii=False)
    else:
        return jsonify({"error": "Data not found"}), 404

    
@app.route("/api/standings_merge", methods=["GET"])
def get_standings_merge():
    """
    Join the cleaned standings, teams, and results data on "constructorId" and "raceId".
    """
    standings_data = kaggle_data["constructorStandings"]
    teams_data = kaggle_data["teams"]
    Contructor_results_data = kaggle_data["constructorResults"]

    if standings_data is not None and teams_data is not None and Contructor_results_data is not None:
        # Merge standings and teams on "constructorId"
        merged_data = pd.merge(standings_data, teams_data, on="constructorId", how="inner")
        
        # Merge the resulting dataset with results data on "constructorId" and "raceId"
        merged_data = pd.merge(merged_data, Contructor_results_data, on=["constructorId", "raceId"], how="inner")
        
        return merged_data.to_json(orient="records", force_ascii=False)
    else:
        return jsonify({"error": "Data not found"}), 404
    

# @app.route("/api/combined_merge", methods=["GET"])
# def get_full_race_data():
#     """
#     Join the cleaned races, results, qualifying, drivers, standings, teams, and constructorResults data on appropriate keys.
#     """
#     # Load data from kaggle_data (replace with actual data fetching if needed)
#     races_data = kaggle_data["races"]
#     results_data = kaggle_data["results"]
#     qualifying_data = kaggle_data["qualifying"]
#     drivers_data = kaggle_data["drivers"]
#     standings_data = kaggle_data["constructorStandings"]
#     teams_data = kaggle_data["teams"]
#     constructor_results_data = kaggle_data["constructorResults"]

#     # Check if all required datasets are available
#     if (races_data is not None and results_data is not None and qualifying_data is not None and
#         drivers_data is not None and standings_data is not None and teams_data is not None and
#         constructor_results_data is not None):
        
#         # Merge races and results on "raceId"
#         merged_data = pd.merge(races_data, results_data, on="raceId", how="inner")
        
#         # Merge the resulting dataset with qualifying data on "raceId" and "driverId"
#         merged_data = pd.merge(merged_data, qualifying_data, on=["raceId", "driverId"], how="inner")
        
#         # Merge the resulting dataset with drivers data on "driverId"
#         merged_data = pd.merge(merged_data, drivers_data, on="driverId", how="inner")
        
#         # Merge the standings data with teams data on "constructorId"
#         merged_data = pd.merge(merged_data, standings_data, on="raceId", how="inner")
#         merged_data = pd.merge(merged_data, teams_data, on="constructorId", how="inner")
        
#         # Merge the constructor results with the merged data on "constructorId" and "raceId"
#         merged_data = pd.merge(merged_data, constructor_results_data, on=["constructorId", "raceId"], how="inner")
        
#         # Return the final merged data as JSON
#         return merged_data.to_json(orient="records")
    
#     else:
#         return jsonify({"error": "Data not found"}), 404


def merge_meetings_weather(meetings_data, weather_data):
    """
    Merge the cleaned meetings data with the weather data on "meeting_key".
    """
    if meetings_data is not None and weather_data is not None:
        # Merge the datasets on "meeting_key"
        merged_data = pd.merge(meetings_data, weather_data, on="meeting_key", how="inner")
        return merged_data
    else:
        return None

@app.route("/api/meetings_weather", methods=["GET"])
def get_meetings_weather():
    """
    API endpoint to fetch and merge meetings and weather data from the OpenF1 API.
    """
    # Fetch data from the OpenF1 API
    meetings_data = fetch_openf1_data("meetings")
    weather_data = fetch_openf1_data("weather")

    # Clean the datasets
    cleaned_meetings_data = clean_meetings_data(meetings_data)
    cleaned_weather_data = clean_weather_data(weather_data)

    # Merge the cleaned datasets
    final_data = merge_meetings_weather(cleaned_meetings_data, cleaned_weather_data)

    if final_data is not None:
        return final_data.to_json(orient="records", force_ascii=False)
    else:
        return jsonify({"error": "Data not found"}), 404
    
# Frontend interaction to fetch data by user input
@app.route("/fetch_data", methods=["POST"])
def fetch_data():
    data_type = request.form.get("data_type")
    if data_type == "races":
        return get_races()
    elif data_type == "results":
        return get_results()
    elif data_type == "constructorResults":
        return get_constructorResults()
    elif data_type == "constructorStandings":
        return get_Standings()
    elif data_type == "teams":
        return get_teams()
    elif data_type == "qualifying":
        return get_qualifying()
    elif data_type == "meetings":
        return get_meetings()
    elif data_type == "drivers":
        return get_drivers()
    elif data_type == "weather":
        return get_weather()
    else:
        return jsonify({"error": "Invalid data type requested"}), 400

# Run the main function
if __name__ == "__main__":
    app.run(debug=True)