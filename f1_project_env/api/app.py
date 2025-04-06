import sys
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
import torch

logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

predictor = None
data_processor = None

CIRCUIT_NAME_MAPPING = {
    'emilia romagna': 'emilia',
    'mexico city': 'mexico',
    'monaco': 'monaco',  
    'saudi arabian': 'saudi',
    'united states': 'us',
    
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

CIRCUIT_TYPES = {
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

def initialize_app():
    global predictor, data_processor
    
    try:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
        circuits_folder = os.path.join(data_dir, "calculated_variables")
        
        logger.info(f"Initializing data processor with data from {data_dir}")
        logger.info(f"Circuits folder: {circuits_folder}")
        
        if not os.path.exists(data_dir):
            logger.error(f"Data directory does not exist: {data_dir}")
            return False
            
        if not os.path.exists(circuits_folder):
            logger.error(f"Circuits folder does not exist: {circuits_folder}")
            return False
            
        logger.debug(f"Contents of data directory: {os.listdir(data_dir)}")
        if os.path.exists(circuits_folder):
            logger.debug(f"Contents of circuits folder: {os.listdir(circuits_folder)}")
        
        from f1_data_processing import F1DataProcessor
        
        data_processor = F1DataProcessor(data_dir, circuits_folder)
        
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
            
        from QualifyingPredictor import QualifyingPredictor, PredictorConfig
        from mcts_and_nn import MCTSConfig, ModelConfig
        
        predictor_config = PredictorConfig()
        model_config = ModelConfig()
        mcts_config = MCTSConfig()
        
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
        Path(model_dir).mkdir(exist_ok=True, parents=True)
        predictor_config.model_save_dir = model_dir
        
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
            circuit: circuit.title()
            for circuit in available_circuits
        }
        
        for circuit in available_circuits:
            if circuit in CIRCUIT_NAME_MAPPING:
                standardized_names[circuit] = CIRCUIT_NAME_MAPPING[circuit].title()
        
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
    circuit = request.args.get('circuit', '')
    if not circuit:
        return jsonify({"error": "Circuit name is required"}), 400
    
    try:
        logger.info(f"Predicting qualifying for circuit: {circuit}")
        
        prediction = predictor.predict_qualifying(circuit)
        
        if isinstance(prediction, dict) and 'error' in prediction:
            return jsonify(prediction), 400
            
        logger.info(f"Prediction successful. Top driver: {prediction['top5'][0]['driver_code']} with probability {prediction['top5'][0]['probability']:.4f}")
        
        prediction['circuit_info'] = {
            'name': circuit,
            'type': get_circuit_type(circuit)
        }
        
        return jsonify(prediction)
    
    except Exception as e:
        logger.error(f"Error predicting qualifying: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def get_circuit_type(circuit_name: str) -> str:
    try:
        if predictor and hasattr(predictor, '_get_circuit_type'):
            return predictor._get_circuit_type(circuit_name)
    except:
        pass
        
    normalized_name = circuit_name.lower()
    
    for circuit_type, circuits in CIRCUIT_TYPES.items():
        if any(c in normalized_name for c in circuits):
            return circuit_type
    
    return 'technical'


if __name__ == '__main__':
    initialized = initialize_app()
    if not initialized:
        logger.warning("WARNING: Application initialization failed! The API will not work correctly.")
        
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
