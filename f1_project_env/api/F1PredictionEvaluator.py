import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class F1PredictionEvaluator:
    
    def __init__(self, visualization_dir: str = 'evaluation_plots'):
        self.metrics_history = []
        self.training_history = {}
        self.visualization_dir = visualization_dir
        self.comparison_results = {}
        self.learning_rate_results = {}
        
        os.makedirs(visualization_dir, exist_ok=True)
    
    def evaluate_prediction(self, 
                       predicted_positions: Dict[str, int], 
                       actual_positions: Dict[str, int]) -> Dict[str, float]:
        metrics = {}
        
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
        
        drivers = sorted(list(common_drivers))
        y_pred = [predicted_positions[d] for d in drivers]
        y_true = [actual_positions[d] for d in drivers]
        
        exact_matches = sum(1 for p, a in zip(y_pred, y_true) if p == a)
        metrics["exact_accuracy"] = exact_matches / len(drivers)
        
        position_errors = [abs(p - a) for p, a in zip(y_pred, y_true)]
        metrics["position_mae"] = sum(position_errors) / len(position_errors)
        
        metrics["position_rmse"] = np.sqrt(sum(e**2 for e in position_errors) / len(position_errors))
        
        metrics["position_errors"] = position_errors
        metrics["raw_position_errors"] = [p - a for p, a in zip(y_pred, y_true)]
        
        for n in [3, 5, 10]:
            y_true_topn = [1 if pos <= n else 0 for pos in y_true]
            y_pred_topn = [1 if pos <= n else 0 for pos in y_pred]
            
            try:
                if all(y_true_topn) or not any(y_true_topn):
                    if all(y_true_topn) == all(y_pred_topn):
                        metrics[f"top{n}_precision"] = 1.0
                        metrics[f"top{n}_recall"] = 1.0
                        metrics[f"top{n}_f1"] = 1.0
                    else:
                        metrics[f"top{n}_precision"] = 0.0
                        metrics[f"top{n}_recall"] = 0.0
                        metrics[f"top{n}_f1"] = 0.0
                else:
                    metrics[f"top{n}_precision"] = precision_score(y_true_topn, y_pred_topn, zero_division=0)
                    metrics[f"top{n}_recall"] = recall_score(y_true_topn, y_pred_topn, zero_division=0)
                    metrics[f"top{n}_f1"] = f1_score(y_true_topn, y_pred_topn, zero_division=0)
                
                metrics[f"top{n}_true"] = y_true_topn
                metrics[f"top{n}_pred"] = y_pred_topn
                
            except Exception as e:
                logger.error(f"Error calculating Top-{n} metrics: {e}")
                metrics[f"top{n}_precision"] = 0.0
                metrics[f"top{n}_recall"] = 0.0
                metrics[f"top{n}_f1"] = 0.0
        
        for n in [1, 2, 3]:
            within_n = sum(1 for p, a in zip(y_pred, y_true) if abs(p - a) <= n)
            metrics[f"position_within_{n}"] = within_n / len(drivers)
        
        for n in [3, 5, 10]:
            try:
                metrics[f"top{n}_confusion_matrix"] = self._calculate_confusion_matrix(
                    metrics[f"top{n}_true"], metrics[f"top{n}_pred"])
            except Exception as e:
                logger.error(f"Error calculating confusion matrix for Top-{n}: {e}")
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def evaluate_qualifying_predictions(self, 
                                       predictions: List[Dict], 
                                       actual_results: List[Dict],
                                       model_name: str = "model") -> Dict[str, float]:
        all_metrics = []
        combined_position_errors = []
        combined_raw_errors = []
        combined_top3_true = []
        combined_top3_pred = []
        combined_top5_true = []
        combined_top5_pred = []
        
        for pred in predictions:
            circuit = pred.get('circuit', '')
            
            matching_actuals = [a for a in actual_results if a.get('circuit', '') == circuit]
            
            if not matching_actuals:
                logger.warning(f"No matching actual results found for circuit: {circuit}")
                continue
                
            actual = matching_actuals[0]
            
            pred_positions = {}
            for p in pred.get('top5', []):
                driver_code = p.get('driver_code', '')
                position = p.get('position', 0)
                if driver_code and position > 0:
                    pred_positions[driver_code] = position
            
            actual_positions = {}
            for p in actual.get('results', []):
                driver_code = p.get('driver_code', '')
                position = p.get('position', 0)
                if driver_code and position > 0:
                    actual_positions[driver_code] = position
            
            if pred_positions and actual_positions:
                metrics = self.evaluate_prediction(pred_positions, actual_positions)
                metrics['circuit'] = circuit
                all_metrics.append(metrics)
                
                if 'position_errors' in metrics:
                    combined_position_errors.extend(metrics['position_errors'])
                if 'raw_position_errors' in metrics:
                    combined_raw_errors.extend(metrics['raw_position_errors'])
                if 'top3_true' in metrics and 'top3_pred' in metrics:
                    combined_top3_true.extend(metrics['top3_true'])
                    combined_top3_pred.extend(metrics['top3_pred'])
                if 'top5_true' in metrics and 'top5_pred' in metrics:
                    combined_top5_true.extend(metrics['top5_true'])
                    combined_top5_pred.extend(metrics['top5_pred'])
        
        if not all_metrics:
            logger.warning("No valid prediction-actual pairs to evaluate")
            return {}
        
        aggregate_metrics = {}
        metric_keys = [k for k in all_metrics[0].keys() if k != 'circuit' and not k.endswith('confusion_matrix') 
                      and k not in ['position_errors', 'raw_position_errors',
                                  'top3_true', 'top3_pred', 'top5_true', 'top5_pred', 'top10_true', 'top10_pred']]
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            aggregate_metrics[key] = sum(values) / len(values)
            aggregate_metrics[f"{key}_min"] = min(values)
            aggregate_metrics[f"{key}_max"] = max(values)
        
        aggregate_metrics["races_evaluated"] = len(all_metrics)
        
        aggregate_metrics["combined_position_errors"] = combined_position_errors
        aggregate_metrics["combined_raw_errors"] = combined_raw_errors
        aggregate_metrics["combined_top3_true"] = combined_top3_true
        aggregate_metrics["combined_top3_pred"] = combined_top3_pred
        aggregate_metrics["combined_top5_true"] = combined_top5_true
        aggregate_metrics["combined_top5_pred"] = combined_top5_pred
        
        self.comparison_results[model_name] = aggregate_metrics
        
        return aggregate_metrics
    
    def compare_with_baselines(self, 
                              model_predictions: List[Dict], 
                              actual_results: List[Dict],
                              random_baseline: bool = True,
                              previous_quali_baseline: bool = True) -> Dict[str, Dict[str, float]]:
        comparison = {}
        
        comparison["model"] = self.evaluate_qualifying_predictions(model_predictions, actual_results, "model")
        
        if random_baseline:
            random_predictions = []
            for actual in actual_results:
                circuit = actual.get('circuit', '')
                drivers = [r.get('driver_code', '') for r in actual.get('results', [])]
                
                if drivers:
                    positions = list(range(1, len(drivers) + 1))
                    np.random.shuffle(positions)
                    
                    random_pred = {
                        'circuit': circuit,
                        'top5': [
                            {'driver_code': driver, 'position': pos}
                            for driver, pos in zip(drivers, positions)
                            if pos <= 5
                        ]
                    }
                    random_predictions.append(random_pred)
            
            comparison["random_baseline"] = self.evaluate_qualifying_predictions(
                random_predictions, actual_results, "random_baseline")
        
        if previous_quali_baseline and len(actual_results) > 1:
            prev_quali_predictions = []
            
            for i in range(1, len(actual_results)):
                prev_actual = actual_results[i-1]
                current_actual = actual_results[i]
                
                prev_quali_pred = {
                    'circuit': current_actual.get('circuit', ''),
                    'top5': [
                        {
                            'driver_code': r.get('driver_code', ''),
                            'position': r.get('position', 0)
                        }
                        for r in prev_actual.get('results', [])
                        if r.get('position', 0) <= 5
                    ]
                }
                prev_quali_predictions.append(prev_quali_pred)
            
            comparison["previous_quali_baseline"] = self.evaluate_qualifying_predictions(
                prev_quali_predictions, actual_results[1:], "previous_quali_baseline")
        
        self.generate_comparison_visualizations()
        
        return comparison
    
    def _calculate_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
        cm_dict = {
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_positive": 0
        }
        
        unique_true = set(y_true)
        unique_pred = set(y_pred)
        
        if len(unique_true) == 1 or len(unique_pred) == 1 or len(y_true) < 10:
            for true_val, pred_val in zip(y_true, y_pred):
                if true_val == 0 and pred_val == 0:
                    cm_dict["true_negative"] += 1
                elif true_val == 0 and pred_val == 1:
                    cm_dict["false_positive"] += 1
                elif true_val == 1 and pred_val == 0:
                    cm_dict["false_negative"] += 1
                elif true_val == 1 and pred_val == 1:
                    cm_dict["true_positive"] += 1
        else:
            try:
                cm = confusion_matrix(y_true, y_pred)
                cm_dict["true_negative"] = int(cm[0, 0])
                cm_dict["false_positive"] = int(cm[0, 1])
                cm_dict["false_negative"] = int(cm[1, 0])
                cm_dict["true_positive"] = int(cm[1, 1])
            except Exception as e:
                logger.warning(f"Falling back to manual confusion matrix calculation: {e}")
                for true_val, pred_val in zip(y_true, y_pred):
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
        report = "F1 QUALIFYING PREDICTION EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += "KEY METRICS SUMMARY:\n"
        report += "-" * 30 + "\n"
        
        models = list(metrics.keys())
        
        key_metrics = [
            "top3_f1", "top5_f1", 
            "position_within_1", "position_within_2",
            "position_mae","position_rmse", "exact_accuracy"
        ]
        
        header = f"{'Metric':<20}"
        for model in models:
            header += f"{model:<15}"
        report += header + "\n"
        report += "-" * (20 + 15 * len(models)) + "\n"
        
        for metric in key_metrics:
            line = f"{metric:<20}"
            for model in models:
                if model in metrics and metric in metrics[model]:
                    value = metrics[model][metric]
                    if metric.startswith("position_mae") or metric.startswith("position_rmse"):
                        line += f"{value:<15.2f}"
                    else:
                        line += f"{value*100:<14.1f}%"
                else:
                    line += f"{'N/A':<15}"
            report += line + "\n"
        
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
            
            if "position_within_1" in metrics["model"]:
                within1 = metrics["model"]["position_within_1"] * 100
                report += f"- {within1:.1f}% of predictions are within 1 position of actual result\n"
            
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
        
        report += f"\nVISUALIZATIONS:\n"
        report += "-" * 30 + "\n"
        report += f"Visualizations have been saved to: {os.path.abspath(self.visualization_dir)}\n"
        
        return report
    
    def store_training_history(self, history: Dict[str, List[float]]):
        self.training_history = history
        
        self.plot_training_convergence()
    
    def store_learning_rate_results(self, lr_values: List[float], performances: List[float], 
                                  metric_name: str = 'val_loss'):
        self.learning_rate_results = {
            'lr_values': lr_values,
            'performances': performances,
            'metric_name': metric_name
        }
        
        self.plot_learning_rate_analysis()
    
    def plot_training_convergence(self):
        if not self.training_history:
            logger.warning("No training history available to plot")
            return
        
        plt.figure(figsize=(10, 6))
        
        if 'train_loss' in self.training_history:
            plt.plot(self.training_history['train_loss'], label='Training Loss', color='#1E88E5', linewidth=2)
        
        if 'val_loss' in self.training_history:
            plt.plot(self.training_history['val_loss'], label='Validation Loss', color='#D81B60', linewidth=2)
        
        for key, values in self.training_history.items():
            if key not in ['train_loss', 'val_loss'] and isinstance(values, list):
                plt.plot(values, label=key, linewidth=1.5, alpha=0.7)
        
        plt.title('Training Convergence', fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        if 'val_loss' in self.training_history:
            val_loss = self.training_history['val_loss']
            min_val_loss = min(val_loss)
            min_epoch = val_loss.index(min_val_loss)
            plt.annotate(f'Min Val Loss: {min_val_loss:.4f}',
                        xy=(min_epoch, min_val_loss),
                        xytext=(min_epoch + 1, min_val_loss * 1.2),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                        fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, 'training_convergence.png'), dpi=300)
        plt.close()
        
        logger.info(f"Training convergence plot saved to {self.visualization_dir}")
    
    def plot_learning_rate_analysis(self):
        if not self.learning_rate_results:
            logger.warning("No learning rate results available to plot")
            return
        
        lr_values = self.learning_rate_results['lr_values']
        performances = self.learning_rate_results['performances']
        metric_name = self.learning_rate_results['metric_name']
        
        plt.figure(figsize=(10, 6))
        
        plt.semilogx(lr_values, performances, 'o-', color='#1E88E5', linewidth=2, markersize=8)
        
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
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, 'learning_rate_analysis.png'), dpi=300)
        plt.close()
        
        logger.info(f"Learning rate analysis plot saved to {self.visualization_dir}")
    
    def generate_comparison_visualizations(self):
        if not self.comparison_results:
            logger.warning("No comparison results available for visualization")
            return
        
        self.plot_bar_chart_comparison()
        
        if 'model' in self.comparison_results:
            self.plot_position_error_distribution(self.comparison_results['model'])
            self.plot_metrics_radar_chart()
        
        logger.info(f"All comparison visualizations saved to {self.visualization_dir}")
    
    def plot_metrics_radar_chart(self):
        if not self.comparison_results or 'model' not in self.comparison_results:
            logger.warning("No model results available for radar chart")
            return
        
        metrics = [
            'top3_f1', 'top5_f1', 'position_within_1', 
            'position_within_2', 'exact_accuracy'
        ]
        
        model_values = []
        for metric in metrics:
            if metric in self.comparison_results['model']:
                if metric == 'position_mae':
                    val = max(0, min(1, 1 - self.comparison_results['model'][metric] / 10))
                else:
                    val = self.comparison_results['model'][metric]
                model_values.append(val)
            else:
                model_values.append(0)
        
        baseline_names = [name for name in self.comparison_results.keys() if name != 'model']
        baseline_values = []
        
        for baseline in baseline_names:
            baseline_data = []
            for metric in metrics:
                if metric in self.comparison_results[baseline]:
                    if metric == 'position_mae':
                        val = max(0, min(1, 1 - self.comparison_results[baseline][metric] / 10))
                    else:
                        val = self.comparison_results[baseline][metric]
                    baseline_data.append(val)
                else:
                    baseline_data.append(0)
            baseline_values.append(baseline_data)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        N = len(metrics)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        model_values += model_values[:1]
        ax.plot(angles, model_values, 'o-', linewidth=2, label='Model', color='#1E88E5')
        ax.fill(angles, model_values, alpha=0.25, color='#1E88E5')
        
        colors = ['#D81B60', '#2E7D32', '#FFC107', '#9C27B0']
        for i, baseline in enumerate(baseline_names):
            values = baseline_values[i] + baseline_values[i][:1]
            ax.plot(angles, values, 'o-', linewidth=1.5, label=baseline, 
                   color=colors[i % len(colors)], alpha=0.8)
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=12)
        
        plt.ylim(0, 1)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], color='grey', fontsize=10)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
        
        plt.title('Performance Metrics Comparison', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, 'metrics_radar_chart.png'), dpi=300)
        plt.close()
        
        logger.info(f"Metrics radar chart saved to {self.visualization_dir}")
    
    def plot_position_error_distribution(self, model_metrics: Dict):
        if not 'combined_raw_errors' in model_metrics:
            logger.warning("No error data available for histogram")
            return
        
        errors = model_metrics['combined_raw_errors']
        
        plt.figure(figsize=(10, 6))
        
        n, bins, patches = plt.hist(errors, bins=range(min(errors)-1, max(errors)+2), 
                                  align='mid', color='#1E88E5', edgecolor='black', alpha=0.7)
        
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor('#D81B60')
            elif bins[i] > 0:
                patch.set_facecolor('#2E7D32')
            else:
                patch.set_facecolor('#FFC107')
        
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        
        plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.2f}')
        plt.axvline(median_error, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_error:.2f}')
        
        plt.axvline(0, color='black', linestyle='-', linewidth=1.5)
        
        stats_text = (f'Mean Error: {mean_error:.2f}\n'
                    f'Median Error: {median_error:.2f}\n'
                    f'Std Dev: {std_error:.2f}\n'
                    f'Exact Matches: {errors.count(0)} ({100 * errors.count(0) / len(errors):.1f}%)')
        
        plt.annotate(stats_text, xy=(0.97, 0.97), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                   ha='right', va='top', fontsize=12)
        
        plt.title('Distribution of Position Prediction Errors', fontsize=16)
        plt.xlabel('Prediction Error (Predicted - Actual)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#D81B60', edgecolor='black', alpha=0.7, label='Overestimated Position\n(Predicted Better Than Actual)'),
            Patch(facecolor='#FFC107', edgecolor='black', alpha=0.7, label='Exact Match'),
            Patch(facecolor='#2E7D32', edgecolor='black', alpha=0.7, label='Underestimated Position\n(Predicted Worse Than Actual)')
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, 'position_error_distribution.png'), dpi=300)
        plt.close()
        
        logger.info(f"Position error distribution plot saved to {self.visualization_dir}")
    
    def plot_bar_chart_comparison(self):
        if not self.comparison_results:
            logger.warning("No comparison results available for bar chart")
            return
        
        metrics = ['top3_f1', 'top5_f1', 'position_within_1', 'exact_accuracy']
        metric_labels = ['Top-3 F1', 'Top-5 F1', 'Within 1 Position', 'Exact Position']
        
        models = list(self.comparison_results.keys())
        
        data = []
        for metric in metrics:
            metric_data = []
            for model in models:
                if metric in self.comparison_results[model]:
                    metric_data.append(self.comparison_results[model][metric])
                else:
                    metric_data.append(0)
            data.append(metric_data)
        
        x = np.arange(len(metric_labels))
        width = 0.8 / len(models)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['#1E88E5', '#D81B60', '#2E7D32', '#FFC107', '#9C27B0']
        
        bars = []
        for i, model in enumerate(models):
            offset = (i - len(models)/2 + 0.5) * width
            bar = ax.bar(x + offset, [d[i] for d in data], width, label=model,
                       color=colors[i % len(colors)], edgecolor='black', alpha=0.8)
            bars.append(bar)
        
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Model Performance Comparison', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=12)
        ax.legend(fontsize=12)
        
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_yticklabels([f'{int(val*100)}%' for val in np.arange(0, 1.1, 0.1)])
        
        for bar_group in bars:
            for bar in bar_group:
                height = bar.get_height()
                ax.annotate(f'{height:.1%}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=10)
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, 'bar_chart_comparison.png'), dpi=300)
        plt.close()
        
        logger.info(f"Bar chart comparison saved to {self.visualization_dir}")

def convert_predictions_to_evaluator_format(predictions: List[Dict]) -> List[Dict]:
    formatted_predictions = []
    
    for pred in predictions:
        if isinstance(pred, dict) and 'top5' in pred:
            formatted_pred = {
                'circuit': pred.get('circuit', ''),
                'top5': []
            }
            
            for driver_pred in pred['top5']:
                formatted_pred['top5'].append({
                    'driver_code': driver_pred.get('driver_code', ''),
                    'position': driver_pred.get('position', 0)
                })
                
            formatted_predictions.append(formatted_pred)
    
    return formatted_predictions
