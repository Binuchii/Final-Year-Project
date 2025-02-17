import datetime as dt
import os
import pandas as pd
import torch
import torch.optim as optim
from f1_alphazero_model import SimplifiedF1Net, QualifyingPredictor
from f1_data_processing import F1DataProcessor
import numpy as np
from typing import Dict
import json

class F1AlphaZeroTester:
    def __init__(self, data_dir: str):
        self.data_processor = F1DataProcessor(data_dir)
        self.predictor = QualifyingPredictor(self.data_processor)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = 'prediction_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
    def analyze_driver_qualifying_history(self, driver_code: str) -> Dict:
        """Analyze a driver's qualifying performance history"""
        driver_id = self.data_processor.driver_mapping.get_driver_id(driver_code)
        if not driver_id:
            return {"error": f"Driver code {driver_code} not found"}
            
        qual_data = self.data_processor.kaggle_data['qualifying']
        driver_quals = qual_data[qual_data['driverId'] == driver_id]
        
        if driver_quals.empty:
            return {"error": f"No qualifying data found for {driver_code}"}
            
        # Calculate qualifying statistics
        q_times = []
        for _, qual in driver_quals.iterrows():
            times = []
            for q in ['q1', 'q2', 'q3']:
                if pd.notna(qual[q]) and qual[q] != 'N/A':
                    try:
                        if isinstance(qual[q], str):
                            if ':' in qual[q]:
                                mins, secs = qual[q].split(':')
                                time_secs = float(mins) * 60 + float(secs)
                            else:
                                time_secs = float(qual[q])
                            times.append(time_secs)
                    except (ValueError, TypeError):
                        continue
                        
            if times:
                q_times.append(min(times))
                
        if not q_times:
            return {"error": f"No valid qualifying times found for {driver_code}"}
            
        return {
            "driver_code": driver_code,
            "total_races": len(driver_quals),
            "avg_qualifying_time": np.mean(q_times),
            "best_qualifying_time": np.min(q_times),
            "qualifying_consistency": np.std(q_times),
            "q3_appearances": len(driver_quals[driver_quals['q3'].notna()])
        }
        
    def train_model(self, num_epochs: int = 10, batch_size: int = 32):
        """Train the model with enhanced features"""
        print("\nStarting enhanced model training...")
        
        # Get all valid circuits
        circuits = list(self.data_processor.circuits_data.keys())
        print(f"Training with {len(circuits)} circuits")
        
        # Track training history
        history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        }
        
        for epoch in range(num_epochs):
            total_loss = 0
            policy_loss_sum = 0
            value_loss_sum = 0
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
                        
                        # Create enhanced state representation
                        state = self.predictor._create_state(circuit_data)
                        
                        # Get qualifying results for target
                        race_info = self.data_processor.race_data[
                            self.data_processor.race_data['name'].str.contains(circuit, case=False)
                        ]
                        
                        if not race_info.empty:
                            # Create one-hot encoded target
                            target = np.zeros(20)
                            min_position = max(1, int(race_info['position'].min()))
                            target[min_position - 1] = 1
                        else:
                            target = np.zeros(20)
                            target[0] = 1
                            
                        states.append(state)
                        targets.append(target)
                        
                    except Exception as e:
                        print(f"Error processing circuit {circuit}: {str(e)}")
                        continue
                
                if not states:
                    continue
                
                # Convert to tensors
                state_tensor = torch.FloatTensor(np.array(states)).to(self.device)
                target_tensor = torch.FloatTensor(np.array(targets)).to(self.device)
                
                # Training step
                loss_dict = self.predictor.train_step(state_tensor, target_tensor)
                total_loss += loss_dict['total_loss']
                policy_loss_sum += loss_dict['policy_loss']
                value_loss_sum += loss_dict['value_loss']
                num_batches += 1
            
            # Calculate epoch averages
            if num_batches > 0:
                avg_total_loss = total_loss / num_batches
                avg_policy_loss = policy_loss_sum / num_batches
                avg_value_loss = value_loss_sum / num_batches
                
                # Update history
                history['total_loss'].append(avg_total_loss)
                history['policy_loss'].append(avg_policy_loss)
                history['value_loss'].append(avg_value_loss)
                
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"  Total Loss: {avg_total_loss:.4f}")
                print(f"  Policy Loss: {avg_policy_loss:.4f}")
                print(f"  Value Loss: {avg_value_loss:.4f}")
        
        return history
    
    def test_predictions(self, test_circuits: list = None) -> Dict:
        """Test predictions with detailed analysis"""
        if test_circuits is None:
            test_circuits = ['monza', 'monaco', 'silverstone', 'spa']
        
        print("\nTesting enhanced predictions...")
        results = {}
        
        for circuit in test_circuits:
            print(f"\nPredicting qualifying for {circuit.upper()}")
            predictions = self.predictor.predict_qualifying(circuit)
            
            # Add driver qualifying history to predictions
            if 'top5' in predictions:
                for pred in predictions['top5']:
                    driver_code = pred['driver_code']
                    driver_history = self.analyze_driver_qualifying_history(driver_code)
                    pred['driver_history'] = driver_history
            
            self._print_enhanced_predictions(predictions)
            results[circuit] = predictions
            
        # Save predictions
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qualifying_predictions_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"\nPredictions saved to {filepath}")
        return results
    
    def _print_enhanced_predictions(self, predictions: Dict):
        """Print enhanced predictions with qualifying history"""
        if 'error' in predictions:
            print(f"Error: {predictions['error']}")
            return
            
        print("\nTop 5 Predicted Qualifying Positions:")
        print("-" * 80)
        print(f"{'Position':<10} {'Driver':<10} {'Probability':<12} {'Q3 Apps':<10} {'Consistency':<12}")
        print("-" * 80)
        
        if 'top5' in predictions:
            for pred in predictions['top5']:
                position = pred.get('position', 'N/A')
                driver_code = pred.get('driver_code', 'Unknown')
                probability = pred.get('probability', 0.0)
                
                # Get driver history info
                history = pred.get('driver_history', {})
                q3_apps = history.get('q3_appearances', 'N/A')
                consistency = history.get('qualifying_consistency', 'N/A')
                if consistency != 'N/A':
                    consistency = f"{consistency:.3f}s"
                
                print(f"P{position:<9} {driver_code:<10} {probability:.3f}      {q3_apps:<10} {consistency:<12}")
        
        print("-" * 80)
        if 'confidence_score' in predictions:
            print(f"Model Confidence: {predictions['confidence_score']:.3f}")
        else:
            print("Model Confidence: N/A")

def main():
    # Initialize tester
    data_dir = r"C:\Users\Albin Binu\Documents\College\Year 4\Final Year Project\f1_project_env\data"  # Update this path
    tester = F1AlphaZeroTester(data_dir)
    
    # Train the model
    print("Starting model training...")
    history = tester.train_model(num_epochs=10)
    
    # Test predictions
    test_circuits = ['monaco', 'miami', 'british', 'bahrain']
    results = tester.test_predictions(test_circuits)
    
    # Print training history summary
    print("\nTraining History Summary:")
    print(f"Initial Total Loss: {history['total_loss'][0]:.4f}")
    print(f"Final Total Loss: {history['total_loss'][-1]:.4f}")
    print(f"Loss Improvement: {history['total_loss'][0] - history['total_loss'][-1]:.4f}")

if __name__ == "__main__":
    main()