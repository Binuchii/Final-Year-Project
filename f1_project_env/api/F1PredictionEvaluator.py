import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

class F1PredictionEvaluator:
    """
    Evaluates F1 qualifying predictions using more appropriate metrics like
    precision, recall, and F1 score instead of just exact position matching.
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_prediction(self, 
                       predicted_positions: Dict[str, int], 
                       actual_positions: Dict[str, int]) -> Dict[str, float]:
      """
      Evaluate a single prediction against actual results.
      
      Args:
          predicted_positions: Dictionary mapping driver codes to their predicted positions
          actual_positions: Dictionary mapping driver codes to their actual positions
          
      Returns:
          Dictionary containing various evaluation metrics
      """
      metrics = {}
      
      # Get common drivers between predictions and actuals
      common_drivers = set(predicted_positions.keys()) & set(actual_positions.keys())
      
      if not common_drivers:
          logger.warning("No common drivers between predictions and actual results")
          return {
              "exact_accuracy": 0.0,
              "top3_precision": 0.0,
              "top3_recall": 0.0,
              "top3_f1": 0.0,
              "top5_precision": 0.0,
              "top5_recall": 0.0,
              "top5_f1": 0.0,
              "position_mae": float('inf'),
              "position_rmse": float('inf')
          }
      
      # Create ordered lists of drivers for consistent comparison
      drivers = sorted(list(common_drivers))
      y_pred = [predicted_positions[d] for d in drivers]
      y_true = [actual_positions[d] for d in drivers]
      
      # Calculate exact position accuracy
      exact_matches = sum(1 for p, a in zip(y_pred, y_true) if p == a)
      metrics["exact_accuracy"] = exact_matches / len(drivers)
      
      # Calculate mean absolute error (MAE) of positions
      position_errors = [abs(p - a) for p, a in zip(y_pred, y_true)]
      metrics["position_mae"] = sum(position_errors) / len(position_errors)
      
      # Calculate root mean squared error (RMSE) of positions
      metrics["position_rmse"] = np.sqrt(sum(e**2 for e in position_errors) / len(position_errors))
      
      # Calculate Top-N metrics
      for n in [3, 5, 10]:
          # For classification metrics, we need binary labels
          y_true_topn = [1 if pos <= n else 0 for pos in y_true]
          y_pred_topn = [1 if pos <= n else 0 for pos in y_pred]
          
          # Calculate precision, recall, and F1 for Top-N
          try:
              # Handle case when all predictions or true values are the same class
              if all(y_true_topn) or not any(y_true_topn):
                  # All positives or all negatives in true values
                  if all(y_true_topn) == all(y_pred_topn):
                      # If predictions match the all-same pattern
                      metrics[f"top{n}_precision"] = 1.0
                      metrics[f"top{n}_recall"] = 1.0
                      metrics[f"top{n}_f1"] = 1.0
                  else:
                      # If predictions don't match the all-same pattern
                      metrics[f"top{n}_precision"] = 0.0
                      metrics[f"top{n}_recall"] = 0.0
                      metrics[f"top{n}_f1"] = 0.0
              else:
                  # Use sklearn's classification metrics for general case
                  from sklearn.metrics import precision_score, recall_score, f1_score
                  metrics[f"top{n}_precision"] = precision_score(y_true_topn, y_pred_topn, zero_division=0)
                  metrics[f"top{n}_recall"] = recall_score(y_true_topn, y_pred_topn, zero_division=0)
                  metrics[f"top{n}_f1"] = f1_score(y_true_topn, y_pred_topn, zero_division=0)
          except Exception as e:
              logger.error(f"Error calculating Top-{n} metrics: {e}")
              metrics[f"top{n}_precision"] = 0.0
              metrics[f"top{n}_recall"] = 0.0
              metrics[f"top{n}_f1"] = 0.0
      
      # Calculate position_within_n metrics (% of predictions within n positions of actual)
      for n in [1, 2, 3]:
          within_n = sum(1 for p, a in zip(y_pred, y_true) if abs(p - a) <= n)
          metrics[f"position_within_{n}"] = within_n / len(drivers)
      
      # Add confusion matrix for top 3, 5, 10 as a dictionary for easy serialization
      for n in [3, 5, 10]:
          try:
              metrics[f"top{n}_confusion_matrix"] = self._calculate_confusion_matrix(y_true, y_pred, n)
          except Exception as e:
              logger.error(f"Error calculating confusion matrix for Top-{n}: {e}")
      
      # Store metrics for later aggregation
      self.metrics_history.append(metrics)
      
      return metrics
    
    def evaluate_qualifying_predictions(self, 
                                       predictions: List[Dict], 
                                       actual_results: List[Dict]) -> Dict[str, float]:
        """
        Evaluate qualifying predictions for multiple races.
        
        Args:
            predictions: List of prediction dictionaries, each containing circuit and top5 predictions
            actual_results: List of actual result dictionaries
            
        Returns:
            Dictionary with aggregated metrics across all evaluated races
        """
        all_metrics = []
        
        for pred in predictions:
            # Extract circuit name to match predictions with actual results
            circuit = pred.get('circuit', '')
            
            # Find matching actual results
            matching_actuals = [a for a in actual_results if a.get('circuit', '') == circuit]
            
            if not matching_actuals:
                logger.warning(f"No matching actual results found for circuit: {circuit}")
                continue
                
            actual = matching_actuals[0]
            
            # Convert predictions to driver:position dictionary
            pred_positions = {}
            for p in pred.get('top5', []):
                driver_code = p.get('driver_code', '')
                position = p.get('position', 0)
                if driver_code and position > 0:
                    pred_positions[driver_code] = position
            
            # Convert actual results to driver:position dictionary
            actual_positions = {}
            for p in actual.get('results', []):
                driver_code = p.get('driver_code', '')
                position = p.get('position', 0)
                if driver_code and position > 0:
                    actual_positions[driver_code] = position
            
            # Evaluate this prediction
            if pred_positions and actual_positions:
                metrics = self.evaluate_prediction(pred_positions, actual_positions)
                metrics['circuit'] = circuit
                all_metrics.append(metrics)
        
        # Aggregate metrics across all races
        if not all_metrics:
            logger.warning("No valid prediction-actual pairs to evaluate")
            return {}
        
        aggregate_metrics = {}
        metric_keys = [k for k in all_metrics[0].keys() if k != 'circuit' and not k.endswith('confusion_matrix')]
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            aggregate_metrics[key] = sum(values) / len(values)
            aggregate_metrics[f"{key}_min"] = min(values)
            aggregate_metrics[f"{key}_max"] = max(values)
        
        # Add number of races evaluated
        aggregate_metrics["races_evaluated"] = len(all_metrics)
        
        return aggregate_metrics
    
    def compare_with_baselines(self, 
                              model_predictions: List[Dict], 
                              actual_results: List[Dict],
                              random_baseline: bool = True,
                              previous_quali_baseline: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Compare model predictions with baseline predictions.
        
        Args:
            model_predictions: List of model prediction dictionaries
            actual_results: List of actual result dictionaries
            random_baseline: Whether to include a random baseline
            previous_quali_baseline: Whether to include previous qualifying results as baseline
            
        Returns:
            Dictionary with metrics for model and baselines
        """
        comparison = {
            "model": self.evaluate_qualifying_predictions(model_predictions, actual_results)
        }
        
        # Random baseline (randomly assigns positions 1-20)
        if random_baseline:
            random_predictions = []
            for actual in actual_results:
                circuit = actual.get('circuit', '')
                drivers = [r.get('driver_code', '') for r in actual.get('results', [])]
                
                if drivers:
                    # Randomly shuffle driver positions
                    positions = list(range(1, len(drivers) + 1))
                    np.random.shuffle(positions)
                    
                    random_pred = {
                        'circuit': circuit,
                        'top5': [
                            {'driver_code': driver, 'position': pos}
                            for driver, pos in zip(drivers, positions)
                            if pos <= 5  # Only include top 5 for consistency
                        ]
                    }
                    random_predictions.append(random_pred)
            
            comparison["random_baseline"] = self.evaluate_qualifying_predictions(random_predictions, actual_results)
        
        # Previous qualifying baseline (uses previous race's qualifying results)
        if previous_quali_baseline and len(actual_results) > 1:
            prev_quali_predictions = []
            
            for i in range(1, len(actual_results)):
                prev_actual = actual_results[i-1]
                current_actual = actual_results[i]
                
                # Use previous race results as prediction for current race
                prev_quali_pred = {
                    'circuit': current_actual.get('circuit', ''),
                    'top5': [
                        {
                            'driver_code': r.get('driver_code', ''),
                            'position': r.get('position', 0)
                        }
                        for r in prev_actual.get('results', [])
                        if r.get('position', 0) <= 5  # Only include top 5
                    ]
                }
                prev_quali_predictions.append(prev_quali_pred)
            
            # Remove the first actual result since we don't have a previous result for it
            comparison["previous_quali_baseline"] = self.evaluate_qualifying_predictions(
                prev_quali_predictions, actual_results[1:])
        
        return comparison
    
    def _calculate_confusion_matrix(self, y_true: List[int], y_pred: List[int], threshold: int) -> Dict[str, int]:
      """
      Calculate confusion matrix for binary classification based on Top-N positions.
      Handles single class case properly to avoid sklearn warnings.
      
      Args:
          y_true: List of actual positions
          y_pred: List of predicted positions
          threshold: Position threshold for binary classification (e.g., top 3, top 5)
          
      Returns:
          Dictionary with confusion matrix values
      """
      # Convert to binary classification based on threshold
      y_true_bin = [1 if pos <= threshold else 0 for pos in y_true]
      y_pred_bin = [1 if pos <= threshold else 0 for pos in y_pred]
      
      # Check if we have only one class in either true or predicted
      unique_true = set(y_true_bin)
      unique_pred = set(y_pred_bin)
      
      # Initialize confusion matrix values
      cm_dict = {
          "true_negative": 0,
          "false_positive": 0,
          "false_negative": 0,
          "true_positive": 0
      }
      
      # If we have only one class, we need to handle it specially
      if len(unique_true) == 1 or len(unique_pred) == 1:
          # Count manually
          for true_val, pred_val in zip(y_true_bin, y_pred_bin):
              if true_val == 0 and pred_val == 0:
                  cm_dict["true_negative"] += 1
              elif true_val == 0 and pred_val == 1:
                  cm_dict["false_positive"] += 1
              elif true_val == 1 and pred_val == 0:
                  cm_dict["false_negative"] += 1
              elif true_val == 1 and pred_val == 1:
                  cm_dict["true_positive"] += 1
      else:
          # Use sklearn's confusion_matrix for the general case
          try:
              from sklearn.metrics import confusion_matrix
              cm = confusion_matrix(y_true_bin, y_pred_bin)
              cm_dict["true_negative"] = int(cm[0, 0])
              cm_dict["false_positive"] = int(cm[0, 1])
              cm_dict["false_negative"] = int(cm[1, 0])
              cm_dict["true_positive"] = int(cm[1, 1])
          except Exception as e:
              # Fall back to manual calculation if sklearn fails
              for true_val, pred_val in zip(y_true_bin, y_pred_bin):
                  if true_val == 0 and pred_val == 0:
                      cm_dict["true_negative"] += 1
                  elif true_val == 0 and pred_val == 1:
                      cm_dict["false_positive"] += 1
                  elif true_val == 1 and pred_val == 0:
                      cm_dict["false_negative"] += 1
                  elif true_val == 1 and pred_val == 1:
                      cm_dict["true_positive"] += 1
      
      return cm_dict    
    def generate_report(self, metrics: Dict[str, Dict[str, float]], detailed: bool = False) -> str:
        """
        Generate a human-readable report from evaluation metrics.
        
        Args:
            metrics: Dictionary with metrics for model and baselines
            detailed: Whether to include detailed metrics
            
        Returns:
            Formatted string report
        """
        report = "F1 QUALIFYING PREDICTION EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Add key metrics summary
        report += "KEY METRICS SUMMARY:\n"
        report += "-" * 30 + "\n"
        
        models = list(metrics.keys())
        
        # Main metrics to highlight in the summary
        key_metrics = [
            "top3_f1", "top5_f1", 
            "position_within_1", "position_within_2",
            "position_mae", "exact_accuracy"
        ]
        
        # Create a table header
        header = f"{'Metric':<20}"
        for model in models:
            header += f"{model:<15}"
        report += header + "\n"
        report += "-" * (20 + 15 * len(models)) + "\n"
        
        # Add each key metric
        for metric in key_metrics:
            line = f"{metric:<20}"
            for model in models:
                if model in metrics and metric in metrics[model]:
                    value = metrics[model][metric]
                    # Format based on metric type
                    if metric.startswith("position_mae") or metric.startswith("position_rmse"):
                        line += f"{value:<15.2f}"
                    else:
                        line += f"{value*100:<14.1f}%"
                else:
                    line += f"{'N/A':<15}"
            report += line + "\n"
        
        # Add interpretation
        report += "\nINTERPRETATION:\n"
        report += "-" * 30 + "\n"
        
        if "model" in metrics and "top5_f1" in metrics["model"]:
            top5_f1 = metrics["model"]["top5_f1"]
            if top5_f1 > 0.7:
                quality = "excellent"
            elif top5_f1 > 0.5:
                quality = "good"
            elif top5_f1 > 0.3:
                quality = "fair"
            else:
                quality = "needs improvement"
                
            report += f"- Your model shows {quality} performance at predicting top 5 qualifiers\n"
            
            # Position accuracy interpretation
            if "position_within_1" in metrics["model"]:
                within1 = metrics["model"]["position_within_1"] * 100
                report += f"- {within1:.1f}% of predictions are within 1 position of actual result\n"
            
            # Compare with baselines
            baseline_comparisons = []
            for baseline in [b for b in models if b != "model"]:
                if baseline in metrics and "top5_f1" in metrics[baseline]:
                    diff = metrics["model"]["top5_f1"] - metrics[baseline]["top5_f1"]
                    if diff > 0:
                        baseline_comparisons.append(f"{diff*100:.1f}% better than {baseline}")
                    elif diff < 0:
                        baseline_comparisons.append(f"{-diff*100:.1f}% worse than {baseline}")
                    else:
                        baseline_comparisons.append(f"equivalent to {baseline}")
            
            if baseline_comparisons:
                report += f"- Your model is {', '.join(baseline_comparisons)}\n"
        
        if detailed:
            # Add detailed metrics
            report += "\nDETAILED METRICS:\n"
            report += "-" * 30 + "\n"
            
            for model, model_metrics in metrics.items():
                report += f"\n{model.upper()} METRICS:\n"
                for metric, value in sorted(model_metrics.items()):
                    if not metric.endswith('confusion_matrix'):
                        if isinstance(value, float):
                            if metric.startswith("position_"):
                                report += f"  {metric:<25}: {value:.2f}\n"
                            else:
                                report += f"  {metric:<25}: {value*100:.2f}%\n"
                        else:
                            report += f"  {metric:<25}: {value}\n"
        
        report += "\nRECOMMENDATIONS:\n"
        report += "-" * 30 + "\n"
        report += "1. Use F1 score as your primary evaluation metric (especially top3_f1 or top5_f1)\n"
        report += "2. Track 'position_within_n' metrics to measure how close predictions are\n"
        report += "3. Compare against baselines to demonstrate model value\n"
        
        return report


def convert_predictions_to_evaluator_format(predictions: List[Dict]) -> List[Dict]:
    """
    Converts the model's prediction output to the format needed by F1PredictionEvaluator.
    
    Args:
        predictions: List of prediction dictionaries from QualifyingPredictor
        
    Returns:
        Reformatted predictions list
    """
    formatted_predictions = []
    
    for pred in predictions:
        if isinstance(pred, dict) and 'top5' in pred:
            formatted_pred = {
                'circuit': pred.get('circuit', ''),
                'top5': []
            }
            
            # Keep only relevant fields for evaluation
            for driver_pred in pred['top5']:
                formatted_pred['top5'].append({
                    'driver_code': driver_pred.get('driver_code', ''),
                    'position': driver_pred.get('position', 0)
                })
                
            formatted_predictions.append(formatted_pred)
    
    return formatted_predictions


def create_actual_results_from_data(data_processor, year=2023) -> List[Dict]:
    """
    Create actual qualifying results data from F1 data processor.
    
    Args:
        data_processor: F1DataProcessor instance
        year: Year to filter results by
        
    Returns:
        List of actual qualifying results in the format needed by the evaluator
    """
    actual_results = []
    
    # Get qualifying data
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
            # If year not available, use all races
            year_races = races_data
        
        # Process each race
        for _, race in year_races.iterrows():
            race_id = race['raceId']
            circuit_name = race['name']
            
            # Get qualifying results for this race
            race_qualifying = qualifying_data[qualifying_data['raceId'] == race_id]
            
            if race_qualifying.empty:
                continue
            
            # Sort by qualifying position
            race_qualifying = race_qualifying.sort_values('position')
            
            # Create results entry
            results = []
            for _, quali in race_qualifying.iterrows():
                driver_id = quali['driverId']
                
                # Get driver code from mapping
                driver_code = data_processor.driver_mapping.get_driver_code(driver_id)
                
                if driver_code:
                    try:
                        position = int(quali['position'])
                        results.append({
                            'driver_code': driver_code,
                            'position': position
                        })
                    except (ValueError, TypeError):
                        # Skip if position can't be converted to int
                        pass
            
            # Add to actual results list
            if results:
                actual_results.append({
                    'circuit': circuit_name,
                    'results': results
                })
        
        # Log number of races found
        logger.info(f"Found {len(actual_results)} races with qualifying data for year {year}")
        
        # If no results found for specific year, try to generate some mock data
        if not actual_results:
            # Generate mock data for testing
            logger.warning(f"No actual qualifying data found for {year}, generating mock data")
            
            # Get available circuits and driver IDs
            circuits = list(data_processor.circuits_data.keys())[:5]  # Use first 5 circuits
            driver_ids = data_processor.driver_mapping.get_current_driver_ids()
            
            for circuit in circuits:
                results = []
                positions = list(range(1, min(len(driver_ids) + 1, 21)))
                
                for i, driver_id in enumerate(driver_ids[:20]):  # Max 20 drivers
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