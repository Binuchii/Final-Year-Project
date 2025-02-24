# File: test_f1_alphazero.py
import os
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Update imports to use absolute imports
from mcts_and_nn import MCTSConfig, ModelConfig, SimplifiedF1Net, MCTS
from QualifyingPredictor import QualifyingPredictor, PredictorConfig
from f1_data_processing import F1DataProcessor, F1Environment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class F1AlphaZeroTester:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.results_dir = "test_results"
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_processor = F1DataProcessor(data_dir)
        self.env = F1Environment(self.data_processor)
        
        # Initialize configurations
        self.model_config = ModelConfig()
        self.mcts_config = MCTSConfig()
        self.predictor_config = PredictorConfig()
        
        # Initialize predictor
        self.predictor = QualifyingPredictor(
            data_processor=self.data_processor,
            config=self.predictor_config,
            model_config=self.model_config,
            mcts_config=self.mcts_config
        )
        
    def test_data_processing(self):
        """Test data processing pipeline"""
        logger.info("Testing data processing...")
        
        # Test driver mapping
        driver_ids = self.data_processor.driver_mapping.get_current_driver_ids()
        logger.info(f"Found {len(driver_ids)} current drivers")
        
        # Test circuit data
        circuits = list(self.data_processor.circuits_data.keys())
        logger.info(f"Loaded {len(circuits)} circuits")
        
        # Test state representation
        race_ids = self.data_processor.race_data['raceId'].unique()
        if len(race_ids) > 0:
            test_race_id = race_ids[0]
            state = self.data_processor.get_state_representation(test_race_id)
            logger.info(f"State shape: {state.shape}")
            
        return True
        
    def test_model_training(self, num_epochs: int = 5):
        """Test model training"""
        logger.info("Testing model training...")
        
        # Create sample training data
        num_samples = 100
        state_size = self.model_config.state_size
        train_states = torch.randn(num_samples, state_size)
        train_targets = torch.zeros(num_samples, 20)
        train_targets[:, 0] = 1  # Dummy targets
        
        # Training loop
        for epoch in range(num_epochs):
            loss_dict = self.predictor.train_step(train_states, train_targets)
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_dict['total_loss']:.4f}")
            
        return True
        
    def test_qualifying_prediction(self, test_circuits: List[str] = None):
        """Test qualifying predictions"""
        if test_circuits is None:
            test_circuits = ['british', 'miami', 'bahrain', 'dutch']
            
        logger.info("Testing qualifying predictions...")
        results = {}
        
        for circuit in test_circuits:
            logger.info(f"\nPredicting qualifying for {circuit.upper()}")
            predictions = self.predictor.predict_qualifying(circuit)
            self._print_predictions(predictions)
            results[circuit] = predictions
            
        # Save results
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
        
    def test_mcts_search(self):
        """Test MCTS search functionality"""
        logger.info("Testing MCTS search...")
        
        # Create a sample state
        state = np.random.randn(self.model_config.state_size)
        
        # Perform MCTS search
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
        logger.info(f"Model Confidence: {predictions.get('confidence_score', 'N/A')}")
        
    def run_all_tests(self):
        """Run all tests"""
        try:
            logger.info("Starting F1 AlphaZero testing suite...")
            
            tests = [
                ('Data Processing', self.test_data_processing),
                ('Model Training', self.test_model_training),
                ('Qualifying Prediction', self.test_qualifying_prediction),
                ('MCTS Search', self.test_mcts_search)
            ]
            
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
                    
            # Print summary
            logger.info("\nTest Summary:")
            logger.info("-" * 50)
            for test_name, success in results:
                status = "✓" if success else "✗"
                logger.info(f"{status} {test_name}")
            logger.info("-" * 50)
            
        except Exception as e:
            logger.error(f"Testing suite failed with error: {str(e)}")

def main():
    # Update this path to your data directory
    data_dir = os.environ.get("F1_DATA_DIR", "./data")
    
    # Initialize and run tests
    tester = F1AlphaZeroTester(data_dir)
    tester.run_all_tests()

if __name__ == "__main__":
    main()