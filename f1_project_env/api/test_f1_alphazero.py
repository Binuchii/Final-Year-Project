import os
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from mcts_and_nn import MCTSConfig, ModelConfig, SimplifiedF1Net, MCTS
from QualifyingPredictor import QualifyingPredictor, PredictorConfig
from f1_data_processing import F1DataProcessor, F1Environment
from F1PredictionEvaluator import F1PredictionEvaluator, convert_predictions_to_evaluator_format 

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_actual_results_from_data(data_processor, year=2023):
    """Create actual qualifying results data from F1 data processor."""
    actual_results = []
    
    try:
        qualifying_data = data_processor.kaggle_data.get('qualifying', pd.DataFrame())
        races_data = data_processor.race_data
        
        if qualifying_data.empty or races_data.empty:
            logger.error("Qualifying data or races data not available")
            return actual_results
        
        # Filter races by year if available
        if 'year' in races_data.columns:
            year_races = races_data[races_data['year'] == year]
        else:
            year_races = races_data
        
        # Process each race
        for _, race in year_races.iterrows():
            race_id = race['raceId']
            circuit_name = race['name']
            
            race_qualifying = qualifying_data[qualifying_data['raceId'] == race_id]
            
            if race_qualifying.empty:
                continue
            
            race_qualifying = race_qualifying.sort_values('position')
            
            results = []
            for _, quali in race_qualifying.iterrows():
                driver_id = quali['driverId']
                driver_code = data_processor.driver_mapping.get_driver_code(driver_id)
                
                if driver_code:
                    try:
                        position = int(quali['position'])
                        results.append({
                            'driver_code': driver_code,
                            'position': position
                        })
                    except (ValueError, TypeError):
                        pass
            
            if results:
                actual_results.append({
                    'circuit': circuit_name,
                    'results': results
                })
        
        logger.info(f"Found {len(actual_results)} races with qualifying data for year {year}")
        
        # Generate mock data if no actual data found
        if not actual_results:
            logger.warning(f"No actual qualifying data found for {year}, generating mock data")
            
            circuits = list(data_processor.circuits_data.keys())[:5]
            driver_ids = data_processor.driver_mapping.get_current_driver_ids()
            
            for circuit in circuits:
                results = []
                positions = list(range(1, min(len(driver_ids) + 1, 21)))
                
                for i, driver_id in enumerate(driver_ids[:20]):
                    driver_code = data_processor.driver_mapping.get_driver_code(driver_id)
                    if driver_code:
                        results.append({
                            'driver_code': driver_code,
                            'position': positions[i]
                        })
                
                if results:
                    actual_results.append({
                        'circuit': circuit,
                        'results': results
                    })
            
            logger.info(f"Generated mock data for {len(actual_results)} circuits")
    
    except Exception as e:
        logger.error(f"Error creating actual results: {str(e)}")
    
    return actual_results

class F1AlphaZeroTester:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.results_dir = "test_results"
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_processor = F1DataProcessor(data_dir)
        self.env = F1Environment(self.data_processor)
        
        self.model_config = ModelConfig()
        self.mcts_config = MCTSConfig()
        self.predictor_config = PredictorConfig()
        
        self.predictor = QualifyingPredictor(
            data_processor=self.data_processor,
            config=self.predictor_config,
            model_config=self.model_config,
            mcts_config=self.mcts_config
        )
        
        self.evaluator = F1PredictionEvaluator()
        
    def test_data_processing(self):
        """Test data processing pipeline"""
        logger.info("Testing data processing...")
        
        driver_ids = self.data_processor.driver_mapping.get_current_driver_ids()
        logger.info(f"Found {len(driver_ids)} current drivers")
        
        circuits = list(self.data_processor.circuits_data.keys())
        logger.info(f"Loaded {len(circuits)} circuits")
        
        race_ids = self.data_processor.race_data['raceId'].unique()
        if len(race_ids) > 0:
            test_race_id = race_ids[0]
            state = self.data_processor.get_state_representation(test_race_id)
            logger.info(f"State shape: {state.shape}")
            
        return True
        
    def test_model_training(self, num_epochs: int = 10):
        """Test model training"""
        logger.info("Testing model training...")
        
        num_samples = 200
        state_size = self.model_config.state_size
        train_states = torch.randn(num_samples, state_size)
        train_targets = torch.zeros(num_samples, 20)
        train_targets[:, 0] = 1  # Dummy targets
        
        for epoch in range(num_epochs):
            loss_dict = self.predictor.train_step(train_states, train_targets)
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_dict['total_loss']:.4f}")
            
        return True
        
    def test_qualifying_prediction(self, test_circuits: List[str] = None):
        """Test qualifying predictions"""
        if test_circuits is None:
            test_circuits = [
            'australian', 'austrian', 'azerbaijan', 'bahrain', 'belgian', 
            'british', 'canadian', 'chinese', 'dutch', 'emilia romagna', 
            'hungarian', 'italian', 'japanese', 'mexico city', 'miami', 
            'monaco', 'saudi arabian', 'singapore', 'spanish', 'united states'
        ]
            
        logger.info("Testing qualifying predictions...")
        results = {}
        
        for circuit in test_circuits:
            logger.info(f"\nPredicting qualifying for {circuit.upper()}")
            predictions = self.predictor.predict_qualifying(circuit)
            self._print_predictions(predictions)
            results[circuit] = predictions
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(self.results_dir, f"test_predictions_{timestamp}.txt")
        
        with open(result_path, 'w') as f:
            for circuit, pred in results.items():
                f.write(f"\n{circuit.upper()} Predictions:\n")
                f.write("-" * 50 + "\n")
                if 'error' in pred:
                    f.write(f"Error: {pred['error']}\n")
                else:
                    for p in pred['top5']:
                        f.write(f"P{p['position']}: {p['driver_code']} ({p['probability']:.3f})\n")
                        
        return True
    
    def test_with_metrics(self, test_circuits: List[str] = None, actual_year: int = 2024):
        """Test qualifying predictions with precision, recall, and F1 metrics."""
        if test_circuits is None:
            test_circuits = [
                'australian', 'austrian', 'azerbaijan', 'bahrain', 'belgian', 
                'british', 'canadian', 'chinese', 'dutch', 'emilia romagna', 
                'hungarian', 'italian', 'japanese', 'mexico city', 'miami', 
                'monaco', 'saudi arabian', 'singapore', 'spanish', 'united states'
            ]
                
        logger.info("Testing qualifying predictions with metrics evaluation...")
        
        predictions = []
        for circuit in test_circuits:
            logger.info(f"\nPredicting qualifying for {circuit.upper()}")
            
            prediction = self.predictor.predict_qualifying(circuit)
            if 'error' not in prediction:
                prediction['circuit'] = circuit
                predictions.append(prediction)
                self._print_predictions(prediction)
        
        actual_results = create_actual_results_from_data(self.data_processor, year=actual_year)
        
        if not actual_results:
            logger.warning("No actual results available for evaluation. Using mock data for testing.")
            for pred in predictions:
                circuit = pred.get('circuit', '')
                mock_results = []
                
                for driver_pred in pred.get('top5', []):
                    driver_code = driver_pred.get('driver_code', '')
                    position = max(1, driver_pred.get('position', 0) + np.random.randint(-1, 2))
                    
                    if driver_code:
                        mock_results.append({
                            'driver_code': driver_code,
                            'position': position
                        })
                
                if len(mock_results) < 10:
                    additional_drivers = ['ALO', 'ALB', 'BOT', 'HUL', 'TSU', 'ZHO', 'GAS', 'STR', 'MAG', 'RIC']
                    for i, driver in enumerate(additional_drivers):
                        if len(mock_results) < 10:
                            mock_results.append({
                                'driver_code': driver,
                                'position': len(mock_results) + 1
                            })
                
                actual_results.append({
                    'circuit': circuit,
                    'results': mock_results
                })
            
            logger.info(f"Created mock data for {len(actual_results)} circuits")
            
        self._align_circuit_names(predictions, actual_results)
        formatted_predictions = convert_predictions_to_evaluator_format(predictions)
        metrics = self.evaluator.evaluate_qualifying_predictions(formatted_predictions, actual_results)
        comparison = self.evaluator.compare_with_baselines(formatted_predictions, actual_results)
        report = self.evaluator.generate_report(comparison)
        logger.info("\nEvaluation Report:\n" + report)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(self.results_dir, f"evaluation_report_{timestamp}.txt")
        with open(result_path, 'w') as f:
            f.write(report)
            
        try:
            import json
            metrics_path = os.path.join(self.results_dir, f"detailed_metrics_{timestamp}.json")
            
            def convert_numpy(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                return obj
            
            with open(metrics_path, 'w') as f:
                json.dump(comparison, f, default=convert_numpy, indent=4)
                
            logger.info(f"Detailed metrics saved to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving detailed metrics: {e}")
            
        return True
    
    def _align_circuit_names(self, predictions: List[Dict], actual_results: List[Dict]):
        """Align circuit names between predictions and actual results."""
        circuit_mapping = {
            'emilia romagna': ['imola', 'emilia'],
            'mexico city': ['mexican', 'mexico'],
            'monaco': ['monte carlo', 'monte'],
            'united states': ['us', 'cota', 'austin'],
            'saudi arabian': ['jeddah', 'saudi'],
            'british': ['silverstone'],
            'italian': ['monza'],
            'japanese': ['suzuka'],
            'australian': ['melbourne', 'albert park'],
            'spanish': ['barcelona', 'catalunya'],
            'hungarian': ['hungaroring'],
            'belgian': ['spa'],
            'austrian': ['red bull ring', 'spielberg'],
            'canadian': ['montreal', 'gilles villeneuve'],
            'dutch': ['zandvoort'],
            'azerbaijan': ['baku'],
            'miami': ['miami international'],
            'chinese': ['shanghai']
        }
        
        def standardize_circuit(name):
            name_lower = name.lower()
            for standard, variants in circuit_mapping.items():
                if name_lower == standard or name_lower in variants:
                    return standard
            return name_lower
            
        for pred in predictions:
            if 'circuit' in pred:
                pred['circuit'] = standardize_circuit(pred['circuit'])
                
        for actual in actual_results:
            if 'circuit' in actual:
                actual['circuit'] = standardize_circuit(actual['circuit'])
        
    def test_mcts_search(self):
        """Test MCTS search functionality"""
        logger.info("Testing MCTS search...")
        
        state = np.random.randn(self.model_config.state_size)
        action_probs = self.predictor.mcts.search(state)
        
        logger.info(f"MCTS search completed. Action probabilities shape: {action_probs.shape}")
        return True
    
    def _print_predictions(self, predictions: Dict):
        """Print prediction results"""
        if 'error' in predictions:
            logger.error(f"Prediction error: {predictions['error']}")
            return
            
        logger.info("\nTop 5 Predicted Qualifying Positions:")
        logger.info("-" * 80)
        logger.info(f"{'Position':<10} {'Driver':<10} {'Probability':<12} {'Q3 Apps':<10}")
        logger.info("-" * 80)
        
        for pred in predictions['top5']:
            logger.info(
                f"P{pred['position']:<9} "
                f"{pred['driver_code']:<10} "
                f"{pred['probability']:.3f}      "
                f"{pred.get('circuit_stats',{}).get('q3_appearances', 'N/A'):<10}"
            )
            
        logger.info("-" * 80)
    
    def check_available_circuits(self):
        """Check which circuits are available in the circuits_data dictionary."""
        print("\nAvailable circuits in dataset:")
        print("-" * 50)
        
        available_circuits = sorted(self.data_processor.circuits_data.keys())
        for circuit_key in available_circuits:
            print(f"- '{circuit_key}'")
        
        print("\nStandardization test for all test circuits:")
        print("-" * 50)
        
        test_names = [
            'emilia romagna', 'imola', 'emilia',
            'mexico city', 'mexican', 'mexico',
            'monaco', 'monte carlo', 'monte',
            'united states', 'cota', 'us', 'united',
            'saudi arabian', 'jeddah', 'saudi',
            'british', 'silverstone'
        ]
        
        for name in test_names:
            std_name = self.get_standardized_circuit_name(name)
            found = std_name in self.data_processor.circuits_data
            status = "✓" if found else "✗"
            print(f"{status} '{name}' → '{std_name}'")
        
        print("\nTest all circuits in the test suite:")
        print("-" * 50)
        
        test_circuits = [
            'australian', 'austrian', 'azerbaijan', 'bahrain', 'belgian', 
            'british', 'canadian', 'chinese', 'dutch', 'emilia romagna', 
            'hungarian', 'italian', 'japanese', 'mexico city', 'miami', 
            'monaco', 'saudi arabian', 'singapore', 'spanish', 'united states'
        ]
        
        for circuit in test_circuits:
            std_name = self.get_standardized_circuit_name(circuit)
            found = std_name in self.data_processor.circuits_data
            status = "✓" if found else "✗"
            print(f"{status} '{circuit}' → '{std_name}'")
        
        return True
    
    def get_standardized_circuit_name(self, circuit_name: str) -> str:
        """Delegate to the predictor's method to standardize circuit names."""
        return self.predictor.get_standardized_circuit_name(circuit_name)
    
    def run_all_tests(self, with_metrics: bool = True):
        """Run all tests"""
        try:
            logger.info("Starting F1 AlphaZero testing suite...")
            
            tests = [
                ('Data Processing', self.test_data_processing),
                ('Model Training', self.test_model_training),
                ('Qualifying Prediction', self.test_qualifying_prediction),
                ('MCTS Search', self.test_mcts_search)
            ]
            
            if with_metrics:
                tests.append(('Metrics Evaluation', self.test_with_metrics))
            
            results = []
            for test_name, test_func in tests:
                logger.info(f"\nRunning {test_name} test...")
                try:
                    success = test_func()
                    results.append((test_name, success))
                    logger.info(f"{test_name} test {'passed' if success else 'failed'}")
                except Exception as e:
                    logger.error(f"{test_name} test failed with error: {str(e)}")
                    results.append((test_name, False))
                    
            logger.info("\nTest Summary:")
            logger.info("-" * 50)
            for test_name, success in results:
                status = "✓" if success else "✗"
                logger.info(f"{status} {test_name}")
            logger.info("-" * 50)
            
        except Exception as e:
            logger.error(f"Testing suite failed with error: {str(e)}")


def main():
    data_dir = r"C:\Users\Albin Binu\Documents\College\Year 4\Final Year Project\f1_project_env\api\data"
    
    tester = F1AlphaZeroTester(data_dir)
    with_metrics = True
    tester.run_all_tests(with_metrics=with_metrics)

if __name__ == "__main__":
    main()
