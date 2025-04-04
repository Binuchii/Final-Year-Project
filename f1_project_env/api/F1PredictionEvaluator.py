import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from collections import defaultdict
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class F1PredictionEvaluator:
    """
    Evaluates F1 qualifying predictions using appropriate metrics and generates
    visualizations for thesis presentation.
    """
    
    def __init__(self, visualization_dir: str = 'evaluation_plots'):
        self.metrics_history = []
        self.training_history = {}
        self.visualization_dir = visualization_dir
        self.comparison_results = {}
        self.learning_rate_results = {}
        
        # Create directory for visualization outputs if it doesn't exist
        os.makedirs(visualization_dir, exist_ok=True)
    
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
        
        # Store position errors for later visualization
        metrics["position_errors"] = position_errors
        metrics["raw_position_errors"] = [p - a for p, a in zip(y_pred, y_true)]
        
        # Store actual and predicted positions for confusion matrix
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
        metrics["drivers"] = drivers
        
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
                    metrics[f"top{n}_precision"] = precision_score(y_true_topn, y_pred_topn, zero_division=0)
                    metrics[f"top{n}_recall"] = recall_score(y_true_topn, y_pred_topn, zero_division=0)
                    metrics[f"top{n}_f1"] = f1_score(y_true_topn, y_pred_topn, zero_division=0)
                
                # Store probabilities for ROC curve calculation
                # For now we use binary classification output directly
                metrics[f"top{n}_true"] = y_true_topn
                metrics[f"top{n}_pred"] = y_pred_topn
                
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
                                       actual_results: List[Dict],
                                       model_name: str = "model") -> Dict[str, float]:
        """
        Evaluate qualifying predictions for multiple races.
        
        Args:
            predictions: List of prediction dictionaries, each containing circuit and top5 predictions
            actual_results: List of actual result dictionaries
            model_name: Name of the model for comparison purposes
            
        Returns:
            Dictionary with aggregated metrics across all evaluated races
        """
        all_metrics = []
        combined_position_errors = []
        combined_raw_errors = []
        combined_y_true = []
        combined_y_pred = []
        combined_top3_true = []
        combined_top3_pred = []
        combined_top5_true = []
        combined_top5_pred = []
        
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
                
                # Collect data for combined visualizations
                if 'position_errors' in metrics:
                    combined_position_errors.extend(metrics['position_errors'])
                if 'raw_position_errors' in metrics:
                    combined_raw_errors.extend(metrics['raw_position_errors'])
                if 'y_true' in metrics and 'y_pred' in metrics:
                    combined_y_true.extend(metrics['y_true'])
                    combined_y_pred.extend(metrics['y_pred'])
                if 'top3_true' in metrics and 'top3_pred' in metrics:
                    combined_top3_true.extend(metrics['top3_true'])
                    combined_top3_pred.extend(metrics['top3_pred'])
                if 'top5_true' in metrics and 'top5_pred' in metrics:
                    combined_top5_true.extend(metrics['top5_true'])
                    combined_top5_pred.extend(metrics['top5_pred'])
        
        # Aggregate metrics across all races
        if not all_metrics:
            logger.warning("No valid prediction-actual pairs to evaluate")
            return {}
        
        aggregate_metrics = {}
        metric_keys = [k for k in all_metrics[0].keys() if k != 'circuit' and not k.endswith('confusion_matrix') 
                      and k not in ['position_errors', 'raw_position_errors', 'y_true', 'y_pred', 'drivers',
                                  'top3_true', 'top3_pred', 'top5_true', 'top5_pred', 'top10_true', 'top10_pred']]
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            aggregate_metrics[key] = sum(values) / len(values)
            aggregate_metrics[f"{key}_min"] = min(values)
            aggregate_metrics[f"{key}_max"] = max(values)
        
        # Add number of races evaluated
        aggregate_metrics["races_evaluated"] = len(all_metrics)
        
        # Store combined data for visualizations
        aggregate_metrics["combined_position_errors"] = combined_position_errors
        aggregate_metrics["combined_raw_errors"] = combined_raw_errors
        aggregate_metrics["combined_y_true"] = combined_y_true
        aggregate_metrics["combined_y_pred"] = combined_y_pred
        aggregate_metrics["combined_top3_true"] = combined_top3_true
        aggregate_metrics["combined_top3_pred"] = combined_top3_pred
        aggregate_metrics["combined_top5_true"] = combined_top5_true
        aggregate_metrics["combined_top5_pred"] = combined_top5_pred
        
        # Store in comparison results for later visualization
        self.comparison_results[model_name] = aggregate_metrics
        
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
        comparison = {}
        
        # Evaluate main model
        comparison["model"] = self.evaluate_qualifying_predictions(model_predictions, actual_results, "model")
        
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
            
            comparison["random_baseline"] = self.evaluate_qualifying_predictions(
                random_predictions, actual_results, "random_baseline")
        
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
                prev_quali_predictions, actual_results[1:], "previous_quali_baseline")
        
        # Generate comparison visualizations
        self.generate_comparison_visualizations()
        
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
            "position_mae","position_rmse", "exact_accuracy"
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
                    if not metric.endswith('confusion_matrix') and not metric.startswith('combined_'):
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
        
        # Add path to generated visualizations
        report += f"\nVISUALIZATIONS:\n"
        report += "-" * 30 + "\n"
        report += f"Visualizations have been saved to: {os.path.abspath(self.visualization_dir)}\n"
        
        return report
    
    # New methods for thesis visualizations
    def store_training_history(self, history: Dict[str, List[float]]):
        """
        Store training history for later visualization
        
        Args:
            history: Dictionary containing training metrics (e.g., train_loss, val_loss)
        """
        self.training_history = history
        
        # Generate training convergence plot
        self.plot_training_convergence()
    
    def store_learning_rate_results(self, lr_values: List[float], performances: List[float], 
                                  metric_name: str = 'val_loss'):
        """
        Store learning rate analysis results for visualization
        
        Args:
            lr_values: List of learning rate values tested
            performances: List of performance metrics corresponding to each learning rate
            metric_name: Name of the performance metric (e.g., 'val_loss', 'top5_f1')
        """
        self.learning_rate_results = {
            'lr_values': lr_values,
            'performances': performances,
            'metric_name': metric_name
        }
        
        # Generate learning rate analysis plot
        self.plot_learning_rate_analysis()
    
    def plot_training_convergence(self):
        """
        Generate training convergence plot showing loss curves over epochs.
        This addresses the first visualization requirement for section 5.1.
        """
        if not self.training_history:
            logger.warning("No training history available to plot")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot training loss if available
        if 'train_loss' in self.training_history:
            plt.plot(self.training_history['train_loss'], label='Training Loss', color='#1E88E5', linewidth=2)
        
        # Plot validation loss if available
        if 'val_loss' in self.training_history:
            plt.plot(self.training_history['val_loss'], label='Validation Loss', color='#D81B60', linewidth=2)
        
        # Add other metrics if available
        for key, values in self.training_history.items():
            if key not in ['train_loss', 'val_loss'] and isinstance(values, list):
                plt.plot(values, label=key, linewidth=1.5, alpha=0.7)
        
        plt.title('Training Convergence', fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Add annotations for minimum validation loss
        if 'val_loss' in self.training_history:
            val_loss = self.training_history['val_loss']
            min_val_loss = min(val_loss)
            min_epoch = val_loss.index(min_val_loss)
            plt.annotate(f'Min Val Loss: {min_val_loss:.4f}',
                        xy=(min_epoch, min_val_loss),
                        xytext=(min_epoch + 1, min_val_loss * 1.2),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                        fontsize=12)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, 'training_convergence.png'), dpi=300)
        plt.close()
        
        logger.info(f"Training convergence plot saved to {self.visualization_dir}")
    
    def plot_learning_rate_analysis(self):
        """
        Generate plot showing model performance against different learning rates.
        This addresses the second visualization requirement for section 5.1.
        """
        if not self.learning_rate_results:
            logger.warning("No learning rate results available to plot")
            return
        
        lr_values = self.learning_rate_results['lr_values']
        performances = self.learning_rate_results['performances']
        metric_name = self.learning_rate_results['metric_name']
        
        plt.figure(figsize=(10, 6))
        
        # Use log scale for x-axis (learning rates)
        plt.semilogx(lr_values, performances, 'o-', color='#1E88E5', linewidth=2, markersize=8)
        
        # Find and highlight the best learning rate
        best_idx = np.argmin(performances) if 'loss' in metric_name.lower() else np.argmax(performances)
        best_lr = lr_values[best_idx]
        best_performance = performances[best_idx]
        
        plt.scatter([best_lr], [best_performance], s=150, c='#D81B60', zorder=10, 
                  label=f'Best LR: {best_lr:.2e}')
        
        plt.annotate(f'Best: {best_performance:.4f}',
                    xy=(best_lr, best_performance),
                    xytext=(best_lr * 1.5, best_performance * 1.1 if 'loss' in metric_name.lower() else best_performance * 0.9),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    fontsize=12)
        
        plt.title('Learning Rate Analysis', fontsize=16)
        plt.xlabel('Learning Rate', fontsize=14)
        plt.ylabel(metric_name, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, 'learning_rate_analysis.png'), dpi=300)
        plt.close()
        
        logger.info(f"Learning rate analysis plot saved to {self.visualization_dir}")
    
    def generate_comparison_visualizations(self):
        """
        Generate all visualizations for model comparisons.
        Includes visualizations for sections 5.2 and 5.3 of the thesis.
        """
        if not self.comparison_results:
            logger.warning("No comparison results available for visualization")
            return
        
        # Create a timestamp for this visualization run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Metrics Radar Chart (Section 5.2)
        self.plot_metrics_radar_chart()
        
        # 2. Confusion Matrix Heatmap (Section 5.2)
        if 'model' in self.comparison_results:
            self.plot_confusion_matrix_heatmap(self.comparison_results['model'])
        
        # 3. Position Error Distribution (Section 5.2)
        if 'model' in self.comparison_results:
            self.plot_position_error_distribution(self.comparison_results['model'])
        
        # 4. Bar Chart Comparison (Section 5.3)
        self.plot_bar_chart_comparison()
        
        # 5. ROC Curves (Section 5.3)
        self.plot_roc_curves()
        
        # 6. Cumulative Position Error (Section 5.3)
        self.plot_cumulative_position_error()
        
        logger.info(f"All comparison visualizations saved to {self.visualization_dir}")
    
    def plot_metrics_radar_chart(self):
        """
        Create a radar/spider chart showing multiple performance metrics simultaneously.
        This addresses the first visualization requirement for section 5.2.
        """
        if not self.comparison_results or 'model' not in self.comparison_results:
            logger.warning("No model results available for radar chart")
            return
        
        # Define metrics to include in the radar chart
        metrics = [
            'top3_f1', 'top5_f1', 'position_within_1', 
            'position_within_2', 'exact_accuracy'
        ]
        
        # Get values for the model
        model_values = []
        for metric in metrics:
            if metric in self.comparison_results['model']:
                # Invert MAE so higher is better for all metrics
                if metric == 'position_mae':
                    # Calculate a normalized score between 0 and 1
                    val = max(0, min(1, 1 - self.comparison_results['model'][metric] / 10))
                else:
                    val = self.comparison_results['model'][metric]
                model_values.append(val)
            else:
                model_values.append(0)
        
        # Get values for the baselines
        baseline_names = [name for name in self.comparison_results.keys() if name != 'model']
        baseline_values = []
        
        for baseline in baseline_names:
            baseline_data = []
            for metric in metrics:
                if metric in self.comparison_results[baseline]:
                    # Invert MAE so higher is better for all metrics
                    if metric == 'position_mae':
                        val = max(0, min(1, 1 - self.comparison_results[baseline][metric] / 10))
                    else:
                        val = self.comparison_results[baseline][metric]
                    baseline_data.append(val)
                else:
                    baseline_data.append(0)
            baseline_values.append(baseline_data)
        
        # Prepare radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of metrics (variables)
        N = len(metrics)
        
        # Compute angle for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the model data
        model_values += model_values[:1]  # Close the loop
        ax.plot(angles, model_values, 'o-', linewidth=2, label='Model', color='#1E88E5')
        ax.fill(angles, model_values, alpha=0.25, color='#1E88E5')
        
        # Add the baseline data
        colors = ['#D81B60', '#2E7D32', '#FFC107', '#9C27B0']
        for i, baseline in enumerate(baseline_names):
            values = baseline_values[i] + baseline_values[i][:1]  # Close the loop
            ax.plot(angles, values, 'o-', linewidth=1.5, label=baseline, 
                   color=colors[i % len(colors)], alpha=0.8)
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        
        # Set labels and styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=12)
        
        # Add metric value ranges
        plt.ylim(0, 1)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], color='grey', fontsize=10)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
        
        plt.title('Performance Metrics Comparison', fontsize=16)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, 'metrics_radar_chart.png'), dpi=300)
        plt.close()
        
        logger.info(f"Metrics radar chart saved to {self.visualization_dir}")
    
    def plot_confusion_matrix_heatmap(self, model_metrics: Dict):
        """
        Visualize true vs. predicted qualifying positions as a color-coded heatmap.
        This addresses the second visualization requirement for section 5.2.
        """
        if not 'combined_y_true' in model_metrics or not 'combined_y_pred' in model_metrics:
            logger.warning("No position data available for confusion matrix")
            return
        
        y_true = model_metrics['combined_y_true']
        y_pred = model_metrics['combined_y_pred']
        
        # Create a confusion matrix for positions 1-20
        max_pos = max(max(y_true), max(y_pred))
        cm = np.zeros((max_pos, max_pos), dtype=int)
        
        for true_pos, pred_pos in zip(y_true, y_pred):
            cm[true_pos-1, pred_pos-1] += 1
        
        # Create a normalized confusion matrix (by true position)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaNs with zeros
        
        # Create the heatmap
        plt.figure(figsize=(12, 10))
        
        # Use a heatmap with custom coloring
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                  cbar_kws={'label': 'Percentage of True Position'})
        
        plt.xlabel('Predicted Position', fontsize=14)
        plt.ylabel('True Position', fontsize=14)
        plt.title('Qualifying Position Confusion Matrix', fontsize=16)
        
        # Adjust axis labels to start from 1 instead of 0
        plt.xticks(np.arange(0.5, max_pos+0.5), range(1, max_pos+1))
        plt.yticks(np.arange(0.5, max_pos+0.5), range(1, max_pos+1))
        
        # Highlight the diagonal (perfect predictions)
        plt.plot(np.arange(0.5, max_pos+0.5), np.arange(0.5, max_pos+0.5), 'r--', linewidth=1.5)
        
        # Add annotations for top-3 and top-5 areas
        plt.axhline(y=3, color='r', linestyle='-', linewidth=1.5, alpha=0.5)
        plt.axvline(x=3, color='r', linestyle='-', linewidth=1.5, alpha=0.5)
        plt.text(max_pos-1, 1.5, 'Top-3', color='r', fontsize=12, rotation=90, ha='center', va='center')
        
        plt.axhline(y=5, color='g', linestyle='-', linewidth=1.5, alpha=0.5)
        plt.axvline(x=5, color='g', linestyle='-', linewidth=1.5, alpha=0.5)
        plt.text(max_pos-0.5, 4, 'Top-5', color='g', fontsize=12, rotation=90, ha='center', va='center')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, 'confusion_matrix_heatmap.png'), dpi=300)
        plt.close()
        
        logger.info(f"Confusion matrix heatmap saved to {self.visualization_dir}")
    
    def plot_position_error_distribution(self, model_metrics: Dict):
        """
        Create a histogram showing the distribution of position errors.
        This addresses the third visualization requirement for section 5.2.
        """
        if not 'combined_raw_errors' in model_metrics:
            logger.warning("No error data available for histogram")
            return
        
        errors = model_metrics['combined_raw_errors']
        
        plt.figure(figsize=(10, 6))
        
        # Create histogram with custom styling
        n, bins, patches = plt.hist(errors, bins=range(min(errors)-1, max(errors)+2), 
                                  align='mid', color='#1E88E5', edgecolor='black', alpha=0.7)
        
        # Color positive and negative errors differently
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor('#D81B60')  # red for overestimates (predicted position < actual)
            elif bins[i] > 0:
                patch.set_facecolor('#2E7D32')  # green for underestimates (predicted position > actual)
            else:
                patch.set_facecolor('#FFC107')  # yellow for exact matches
        
        # Add statistics annotations
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        
        plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.2f}')
        plt.axvline(median_error, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_error:.2f}')
        
        # Add zero line
        plt.axvline(0, color='black', linestyle='-', linewidth=1.5)
        
        # Add annotations
        stats_text = (f'Mean Error: {mean_error:.2f}\n'
                    f'Median Error: {median_error:.2f}\n'
                    f'Std Dev: {std_error:.2f}\n'
                    f'Exact Matches: {errors.count(0)} ({100 * errors.count(0) / len(errors):.1f}%)')
        
        plt.annotate(stats_text, xy=(0.97, 0.97), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                   ha='right', va='top', fontsize=12)
        
        # Add styling
        plt.title('Distribution of Position Prediction Errors', fontsize=16)
        plt.xlabel('Prediction Error (Predicted - Actual)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#D81B60', edgecolor='black', alpha=0.7, label='Overestimated Position\n(Predicted Better Than Actual)'),
            Patch(facecolor='#FFC107', edgecolor='black', alpha=0.7, label='Exact Match'),
            Patch(facecolor='#2E7D32', edgecolor='black', alpha=0.7, label='Underestimated Position\n(Predicted Worse Than Actual)')
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, 'position_error_distribution.png'), dpi=300)
        plt.close()
        
        logger.info(f"Position error distribution plot saved to {self.visualization_dir}")
    
    def plot_bar_chart_comparison(self):
        """
        Create a bar chart comparing key metrics between model and baselines.
        This addresses the first visualization requirement for section 5.3.
        """
        if not self.comparison_results:
            logger.warning("No comparison results available for bar chart")
            return
        
        # Select key metrics to compare
        metrics = ['top3_f1', 'top5_f1', 'position_within_1', 'exact_accuracy']
        metric_labels = ['Top-3 F1', 'Top-5 F1', 'Within 1 Position', 'Exact Position']
        
        # Get model names
        models = list(self.comparison_results.keys())
        
        # Create data structure for plotting
        data = []
        for metric in metrics:
            metric_data = []
            for model in models:
                if metric in self.comparison_results[model]:
                    metric_data.append(self.comparison_results[model][metric])
                else:
                    metric_data.append(0)
            data.append(metric_data)
        
        # Set up plot
        x = np.arange(len(metric_labels))
        width = 0.8 / len(models)  # Width of bars, adjusted for number of models
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Colors for different models
        colors = ['#1E88E5', '#D81B60', '#2E7D32', '#FFC107', '#9C27B0']
        
        # Create bars for each model
        bars = []
        for i, model in enumerate(models):
            offset = (i - len(models)/2 + 0.5) * width
            bar = ax.bar(x + offset, [d[i] for d in data], width, label=model,
                       color=colors[i % len(colors)], edgecolor='black', alpha=0.8)
            bars.append(bar)
        
        # Add labels and styling
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Model Performance Comparison', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=12)
        ax.legend(fontsize=12)
        
        # Format y-axis as percentage
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_yticklabels([f'{int(val*100)}%' for val in np.arange(0, 1.1, 0.1)])
        
        # Add value labels on top of bars
        for bar_group in bars:
            for bar in bar_group:
                height = bar.get_height()
                ax.annotate(f'{height:.1%}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=10)
        
        # Add grid for readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, 'bar_chart_comparison.png'), dpi=300)
        plt.close()
        
        logger.info(f"Bar chart comparison saved to {self.visualization_dir}")
    
    def plot_roc_curves(self):
        """
        Create ROC curves for comparing Top-N classification performance across models.
        This addresses the second visualization requirement for section 5.3.
        """
        if not self.comparison_results:
            logger.warning("No comparison results available for ROC curves")
            return
        
        # We'll create separate ROC curves for Top-3 and Top-5 classification
        for n in [3, 5]:
            plt.figure(figsize=(10, 8))
            
            # Store AUC values for legend
            auc_values = {}
            
            # Colors for different models
            colors = ['#1E88E5', '#D81B60', '#2E7D32', '#FFC107', '#9C27B0']
            
            # Plot ROC curve for each model
            for i, model_name in enumerate(self.comparison_results.keys()):
                model_metrics = self.comparison_results[model_name]
                
                if f'combined_top{n}_true' in model_metrics and f'combined_top{n}_pred' in model_metrics:
                    y_true = model_metrics[f'combined_top{n}_true']
                    y_pred = model_metrics[f'combined_top{n}_pred']
                    
                    # Calculate ROC curve
                    try:
                        # For binary classification like this, we can use the raw predictions
                        # But to be more robust, we can treat the binary predictions as probabilities
                        # by adding a small random noise
                        y_pred_proba = np.array(y_pred).astype(float)
                        
                        # Calculate ROC curve and ROC area
                        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                        roc_auc = auc(fpr, tpr)
                        auc_values[model_name] = roc_auc
                        
                        # Plot ROC curve
                        plt.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                               label=f'{model_name} (AUC = {roc_auc:.3f})')
                    except Exception as e:
                        logger.error(f"Error calculating ROC curve for {model_name}: {e}")
            
            # Plot diagonal reference line (random classifier)
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
            
            # Add styling
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title(f'ROC Curve for Top-{n} Qualification Prediction', fontsize=16)
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add explanatory annotation
            plt.annotate('Better performance →', 
                       xy=(0.4, 0.6), 
                       xytext=(0.6, 0.4),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                       fontsize=12)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualization_dir, f'roc_curve_top{n}.png'), dpi=300)
            plt.close()
            
            logger.info(f"ROC curve for Top-{n} saved to {self.visualization_dir}")
    
    def plot_cumulative_position_error(self):
        """
        Create a line chart showing cumulative percentage of predictions within X positions.
        This addresses the third visualization requirement for section 5.3.
        """
        if not self.comparison_results:
            logger.warning("No comparison results available for cumulative error plot")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Colors for different models
        colors = ['#1E88E5', '#D81B60', '#2E7D32', '#FFC107', '#9C27B0']
        
        # Calculate cumulative error distributions for each model
        for i, model_name in enumerate(self.comparison_results.keys()):
            model_metrics = self.comparison_results[model_name]
            
            if 'combined_position_errors' in model_metrics:
                errors = model_metrics['combined_position_errors']
                
                # Calculate percentages within each error threshold
                max_error = max(errors)
                thresholds = list(range(0, int(max_error) + 2))
                percentages = []
                
                for threshold in thresholds:
                    count_within = sum(1 for error in errors if error <= threshold)
                    percentages.append(count_within / len(errors) * 100)
                
                # Plot cumulative distribution
                plt.plot(thresholds, percentages, 'o-', color=colors[i % len(colors)],
                       linewidth=2, label=model_name)
                
                # Add annotations for key thresholds
                for threshold in [0, 1, 2]:
                    if threshold < len(percentages):
                        plt.annotate(f'{percentages[threshold]:.1f}%', 
                                   xy=(threshold, percentages[threshold]),
                                   xytext=(3, 0),
                                   textcoords="offset points",
                                   fontsize=10)
        
        # Add styling
        plt.xlabel('Position Error Threshold', fontsize=14)
        plt.ylabel('Percentage of Predictions', fontsize=14)
        plt.title('Cumulative Position Error Distribution', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Set x-axis to integers only
        plt.xticks(range(0, int(max_error) + 2))
        
        # Format y-axis as percentage
        plt.ylim(0, 100)
        plt.yticks(range(0, 101, 10))
        
        # Add horizontal lines at important thresholds
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        plt.axhline(y=90, color='gray', linestyle='--', alpha=0.7)
        
        # Add vertical lines at position error thresholds
        plt.axvline(x=1, color='green', linestyle='--', alpha=0.7, label='Within 1 Position')
        plt.axvline(x=2, color='orange', linestyle='--', alpha=0.7, label='Within 2 Positions')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, 'cumulative_position_error.png'), dpi=300)
        plt.close()
        
        logger.info(f"Cumulative position error plot saved to {self.visualization_dir}")

# Helper functions for using the evaluator
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