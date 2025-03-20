import sys
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import os
from pathlib import Path
import torch

# Import your existing code - update these imports if needed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

# Import the necessary classes from your files
from mcts_and_nn import (
    MCTSConfig,
    ModelConfig,
    SimplifiedF1Net,
    MCTS,
    convert_time_to_seconds
)

# Import your F1 prediction modules
from f1_data_processing import F1DataProcessor
from QualifyingPredictor import QualifyingPredictor, PredictorConfig
from mcts_and_nn import MCTSConfig, ModelConfig
from F1PredictionEvaluator import F1PredictionEvaluator, convert_predictions_to_evaluator_format, create_actual_results_from_data

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

predictor = None
data_processor = None

# Initialize before app starts running
def initialize_app():
    global predictor, data_processor
    
    try:
        # Set paths to your data directory
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
        circuits_folder = os.path.join(data_dir, "calculated_variables")
        
        logger.info(f"Initializing data processor with data from {data_dir}")
        logger.info(f"Circuits folder: {circuits_folder}")
        
        # Check if the directories exist
        if not os.path.exists(data_dir):
            logger.error(f"Data directory does not exist: {data_dir}")
            return False
            
        if not os.path.exists(circuits_folder):
            logger.error(f"Circuits folder does not exist: {circuits_folder}")
            return False
            
        # List contents of the data directory
        logger.debug(f"Contents of data directory: {os.listdir(data_dir)}")
        if os.path.exists(circuits_folder):
            logger.debug(f"Contents of circuits folder: {os.listdir(circuits_folder)}")
        
        # Import the data processor
        from f1_data_processing import F1DataProcessor
        
        # Create the data processor
        data_processor = F1DataProcessor(data_dir, circuits_folder)
        
        # Check if circuits_data attribute exists
        if hasattr(data_processor, 'circuits_data'):
            if data_processor.circuits_data:
                available_circuits = sorted(data_processor.circuits_data.keys())
                logger.info(f"Loaded {len(available_circuits)} circuits: {', '.join(available_circuits[:5]) if available_circuits else 'None'}")
            else:
                logger.error("circuits_data exists but is empty")
                return False
        else:
            logger.error("data_processor does not have circuits_data attribute")
            return False
            
        # Initialize the predictor
        from QualifyingPredictor import QualifyingPredictor, PredictorConfig
        from mcts_and_nn import MCTSConfig, ModelConfig
        
        predictor_config = PredictorConfig()
        model_config = ModelConfig()
        mcts_config = MCTSConfig()
        
        # Set model paths
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
        Path(model_dir).mkdir(exist_ok=True, parents=True)
        predictor_config.model_save_dir = model_dir
        
        # Initialize predictor with the device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        predictor = QualifyingPredictor(
            data_processor=data_processor,
            config=predictor_config,
            model_config=model_config,
            mcts_config=mcts_config,
            device=device
        )
        
        logger.info("Initialization complete")
        return True
    except Exception as e:
        logger.error(f"ERROR during initialization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/circuits', methods=['GET'])
def get_circuits():
    """Return a list of available circuits"""
    try:
        if data_processor is None:
            logger.error("data_processor is None")
            return jsonify({"error": "Data processor not initialized. Check server logs."}), 500
            
        if not hasattr(data_processor, 'circuits_data'):
            logger.error("data_processor does not have circuits_data attribute")
            return jsonify({"error": "Data processor missing circuits_data attribute. Check server logs."}), 500
            
        if not data_processor.circuits_data:
            logger.error("circuits_data is empty")
            return jsonify({"error": "Circuit data is empty. Check server logs."}), 500
            
        available_circuits = sorted(data_processor.circuits_data.keys())
        logger.info(f"Available circuits: {available_circuits}")
        
        standardized_names = {
            circuit: circuit.title()  # Convert to title case for display
            for circuit in available_circuits
        }
        
        # Add special cases with better display names
        name_mapping = {
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
        
        for circuit in available_circuits:
            if circuit in name_mapping:
                standardized_names[circuit] = name_mapping[circuit]
        
        # Return both the internal name and display name
        result = [
            {"id": circuit, "name": standardized_names[circuit]}
            for circuit in available_circuits
        ]
        
        return jsonify({"circuits": result})
    
    except Exception as e:
        logger.error(f"Error getting circuits: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict', methods=['GET'])
def predict():
    """Predict qualifying results for a given circuit"""
    circuit = request.args.get('circuit', '')
    if not circuit:
        return jsonify({"error": "Circuit name is required"}), 400
    
    try:
        logger.info(f"Predicting qualifying for circuit: {circuit}")
        
        # Call your existing prediction method
        prediction = predictor.predict_qualifying(circuit)
        
        # Check if prediction has an error
        if isinstance(prediction, dict) and 'error' in prediction:
            return jsonify(prediction), 400
            
        logger.info(f"Prediction successful. Top driver: {prediction['top5'][0]['driver_code']} with probability {prediction['top5'][0]['probability']:.4f}")
        
        # Enhance response with additional circuit info
        prediction['circuit_info'] = {
            'name': circuit,
            'type': _get_circuit_type(circuit)
        }
        
        return jsonify(prediction)
    
    except Exception as e:
        logger.error(f"Error predicting qualifying: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _get_circuit_type(circuit_name: str) -> str:
    """Get circuit type using the method from QualifyingPredictor"""
    try:
        return predictor._get_circuit_type(circuit_name)
    except:
        # Fallback if method is not accessible
        circuit_types = {
            'street': [
                'monaco', 'singapore', 'azerbaijan', 'saudi', 'las vegas',
                'baku', 'jeddah', 'marina bay'
            ],
            'high_speed': [
                'monza', 'spa', 'austria', 'silverstone', 'british', 'miami',
                'belgian', 'bahrain', 'canadian', 'italian', 'australian',
                'red bull ring', 'spielberg', 'montreal', 'interlagos', 'brazilian'
            ]
        }
        
        normalized_name = circuit_name.lower()
        
        for circuit_type, circuits in circuit_types.items():
            if any(c in normalized_name for c in circuits):
                return circuit_type
        
        # Default to technical if unknown
        return 'technical'


if __name__ == '__main__':
    # Call initialize_app() before running the Flask app
    initialized = initialize_app()
    if not initialized:
        logger.warning("WARNING: Application initialization failed! The API will not work correctly.")
        
    # Set the port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)