# F1 Qualifying Prediction System

This project is a machine learning-based system that predicts Formula 1 qualifying results for different circuits. It uses historical F1 data and neural networks to analyze and predict the most likely qualifying positions for drivers at any supported F1 circuit.

## 📋 Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [API Documentation](#api-documentation)
- [Frontend Features](#frontend-features)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- Predict F1 qualifying results for any supported circuit
- View probability distribution of predicted results
- Analyze circuit characteristics and their impact on qualifying
- Access detailed driver statistics for each circuit
- Web-based interface with responsive design
- RESTful API for integration with other applications

## 🏗️ System Architecture

The system follows a client-server architecture:

- **Backend**: Python Flask server that handles data processing, machine learning model training and inference
- **Frontend**: JavaScript/React-based web interface that communicates with the backend API
- **Data Processing Pipeline**: Handles historical F1 data cleaning, feature engineering, and state representation for the model
- **Machine Learning Core**: Uses a neural network model with Monte Carlo Tree Search for improved prediction accuracy

## 🔧 Technologies Used

### Backend
- Python 3.7+
- Flask (Web framework)
- PyTorch (Machine learning framework)
- Pandas & NumPy (Data processing)
- MCTS (Monte Carlo Tree Search)

### Frontend
- React
- Recharts (Data visualization)
- TailwindCSS (Styling)

### Data Sources
- Historical F1 data from Kaggle datasets
- Circuit-specific data stored in CSV format

## 📁 Project Structure

```
f1_project/
├── api/                         # Backend API
│   ├── data/                    # Data directory
│   │   └── calculated_variables/# Circuit-specific data
│   ├── models/                  # Saved model files
│   ├── templates/               # HTML templates for web interface
│   ├── app.py                   # Main Flask application
│   ├── f1_data_processing.py    # Data processor class
│   ├── mcts_and_nn.py           # Monte Carlo Tree Search and neural network model
│   └── QualifyingPredictor.py   # Main prediction class
├── static/                      # Static frontend assets
│   └── f1-prediction-app.js     # React frontend application
└── README.md                    # This file
```

## 📥 Installation

### Prerequisites
- Python 3.7+
- PyTorch
- Node.js (if you want to modify the frontend)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd f1_project
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the data directory:
   ```bash
   mkdir -p api/data/calculated_variables
   ```

5. Download and place historical F1 data in the `api/data` directory.

## 💻 Usage

### Running the Server

1. Start the Flask server:
   ```bash
   cd api
   python app.py
   ```
   The server will start on `http://localhost:5000` by default.

2. Open a web browser and navigate to `http://localhost:5000` to access the web interface.

### Using the API

- **Get Available Circuits**:
  ```
  GET /api/circuits
  ```

- **Predict Qualifying Results**:
  ```
  GET /api/predict?circuit=monaco
  ```

## 🧠 Model Details

The qualifying prediction model combines several advanced techniques:

### Neural Network
- Uses the `SimplifiedF1Net` architecture based on PyTorch
- Features encoded driver information, constructor data, historical qualifying performance, circuit characteristics, and weather conditions
- Attention mechanisms to focus on relevant features for different circuit types

### Monte Carlo Tree Search (MCTS)
- Improves prediction accuracy by exploring different potential qualifying outcomes
- Uses UCB (Upper Confidence Bound) algorithm for balancing exploration and exploitation

### Feature Engineering
- Driver features: 20 features representing driver-specific performance
- Constructor features: 30 features for team performance
- Qualifying features: 60 features capturing historical qualifying data
- Weather features: Represents weather conditions that might affect qualifying
- Circuit features: 3 features representing circuit characteristics

## 📘 API Documentation

### Endpoints

#### Get Available Circuits
- **URL**: `/api/circuits`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "circuits": [
      {"id": "monaco", "name": "Monaco"},
      {"id": "silverstone", "name": "Silverstone"},
      ...
    ]
  }
  ```

#### Predict Qualifying Results
- **URL**: `/api/predict`
- **Method**: `GET`
- **Parameters**:
  - `circuit` (required): The circuit name or ID
- **Response**:
  ```json
  {
    "circuit": "Monaco",
    "prediction_time": "2025-04-06T12:00:00.000Z",
    "top5": [
      {
        "position": 1,
        "driver_code": "VER",
        "probability": 0.32,
        "circuit_stats": {
          "q3_appearances": 5,
          "best_position": 1
        }
      },
      ...
    ],
    "confidence_score": 0.85,
    "circuit_info": {
      "name": "Monaco",
      "type": "street"
    }
  }
  ```

## 🎨 Frontend Features

The web interface provides a user-friendly way to interact with the prediction system:

- Circuit selection dropdown with all available circuits
- Prediction button to trigger the prediction process
- Detailed circuit information with characteristics visualization
- Results display showing the top 5 predicted drivers
- Probability distribution chart for visual representation
- Driver performance statistics at each circuit
- Confidence score indicating model certainty

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the project, please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
