import torch
import torch.optim as optim
from f1_alphazero_model import SimplifiedF1Net, QualifyingPredictor
from f1_data_processing import F1DataProcessor
import numpy as np
from typing import Dict
import json

class F1AlphaZeroTester:
    def __init__(self, data_dir: str):
        # Initialize data processor and predictor
        self.data_processor = F1DataProcessor(data_dir)
        self.predictor = QualifyingPredictor(self.data_processor)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_model(self, num_epochs: int = 10, batch_size: int = 32):
        """Train the model using historical data"""
        print("\nStarting model training...")
        optimizer = optim.Adam(self.predictor.model.parameters(), lr=0.001)
        
        # Get all valid circuits
        circuits = list(self.data_processor.circuits_data.keys())
        print(f"Training with {len(circuits)} circuits")
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            # Create batches of circuits
            np.random.shuffle(circuits)
            for i in range(0, len(circuits), batch_size):
                batch_circuits = circuits[i:i + batch_size]
                
                # Prepare batch data
                states = []
                targets = []
                
                for circuit in batch_circuits:
                    try:
                        # Get circuit data
                        circuit_data = self.data_processor.circuits_data[circuit]
                        
                        # Create state representation
                        state = self.predictor._create_state(circuit_data)
                        
                        # Get actual qualifying results as target
                        race_info = self.data_processor.race_data[
                            self.data_processor.race_data['name'].str.contains(circuit, case=False)
                        ]
                        if not race_info.empty:
                            # Create one-hot encoded target
                            target = np.zeros(20)
                            # Get the top position for training
                            min_position = max(1, int(race_info['position'].min()))  # Ensure valid position
                            target[min_position - 1] = 1  # Convert to 0-based index
                        else:
                            # If no race data, use placeholder
                            target = np.zeros(20)
                            target[0] = 1  # Default to first position
                        
                        states.append(state)
                        targets.append(target)
                        
                    except Exception as e:
                        print(f"Error processing circuit {circuit}: {str(e)}")
                        continue
                
                if not states:
                    continue
                
                # Convert to tensors
                state_tensor = torch.FloatTensor(np.array(states)).to(self.device)
                target_tensor = torch.LongTensor(np.array(targets)).to(self.device)
                
                # Training step
                loss_dict = self.predictor.train_step(state_tensor, target_tensor)
                total_loss += loss_dict['total_loss']
                num_batches += 1
            
            # Print epoch results
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    def test_predictions(self, test_circuits: list = None):
        """Test predictions for specific circuits"""
        if test_circuits is None:
            test_circuits = ['monza', 'monaco', 'silverstone']
        
        print("\nTesting predictions...")
        results = {}
        
        for circuit in test_circuits:
            print(f"\nPredicting qualifying for {circuit.upper()}")
            predictions = self.predictor.predict_qualifying(circuit)
            self._print_predictions(predictions)
            results[circuit] = predictions
        
        return results
    
    def _print_predictions(self, predictions: Dict):
        """Print predictions in a formatted way"""
        if 'error' in predictions:
            print(f"Error: {predictions['error']}")
            return
            
        print("\nTop 5 Predicted Qualifying Positions:")
        print("-" * 50)
        print(f"{'Position':<10} {'Driver':<15} {'Probability':<12}")
        print("-" * 50)
        
        if 'top5' in predictions:
            for pred in predictions['top5']:
                position = pred.get('position', 'N/A')
                driver_code = pred.get('driver_code', 'Unknown')
                probability = pred.get('probability', 0.0)
                
                print(f"P{position:<9} {driver_code:<15} {probability:.3f}")
        
        print("-" * 50)
        if 'confidence_score' in predictions:
            print(f"Model Confidence: {predictions['confidence_score']:.3f}")
        else:
            print("Model Confidence: N/A")

def main():
    # Initialize tester
    data_dir = r"C:\Users\Albin Binu\Documents\College\Year 4\Final Year Project\f1_project_env\data"  # Update this path
    tester = F1AlphaZeroTester(data_dir)
    
    # Train the model
    tester.train_model(num_epochs=5)
    
    # Test predictions
    test_circuits = ['monaco', 'miami', 'british', 'bahrain']
    results = tester.test_predictions(test_circuits)
    
    # Save results to file
    with open('qualifying_predictions.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()