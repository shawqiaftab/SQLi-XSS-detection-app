# ============================================================================
# FILE: visualization.py (IMPROVED VERSION)
# DESCRIPTION: Generate improved publication-ready visualizations
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import json
from pathlib import Path
from matplotlib.patches import Patch
from config import GLOBAL_CONFIG
import warnings
warnings.filterwarnings('ignore')

# Modern styling
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3
})

# Color scheme
MODEL_COLORS = {
    'Classical ML': '#3498db',
    'Deep Learning': '#e74c3c',
    'Transformer': '#2ecc71',
    'Hybrid': '#9b59b6',
    'GNN': '#f39c12'
}

# ============================================================================
# VISUALIZATION GENERATOR
# ============================================================================

class VisualizationGenerator:
    """Generate improved publication-ready visualizations"""

    def __init__(self, config=GLOBAL_CONFIG):
        self.config = config
        self.viz_dir = config['viz_dir']

    def plot_training_curves(self, model_name, history, model_type='deep_learning'):
        """Enhanced training curves"""
        save_dir = f"{self.viz_dir}/individual_models/{model_name}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if isinstance(history, dict):
            history_dict = history
        else:
            history_dict = history.history

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        epochs = range(1, len(history_dict['loss']) + 1)

        # Loss curve
        ax1.plot(epochs, history_dict['loss'], 'o-', color='#3498db', 
                linewidth=2, markersize=4, label='Training Loss', markevery=max(1, len(epochs)//15))
        if 'val_loss' in history_dict:
            ax1.plot(epochs, history_dict['val_loss'], 's-', color='#e74c3c', 
                    linewidth=2, markersize=4, label='Validation Loss', markevery=max(1, len(epochs)//15))

        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title(f'{model_name} - Loss', fontweight='bold', pad=15)
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Accuracy curve
        acc_key = 'accuracy' if 'accuracy' in history_dict else ('train_accuracy' if 'train_accuracy' in history_dict else None)

        if acc_key:
            ax2.plot(epochs, history_dict[acc_key], 'o-', color='#2ecc71', 
                    linewidth=2, markersize=4, label='Training Accuracy', markevery=max(1, len(epochs)//15))
            if 'val_accuracy' in history_dict:
                ax2.plot(epochs, history_dict['val_accuracy'], 's-', color='#f39c12', 
                        linewidth=2, markersize=4, label='Validation Accuracy', markevery=max(1, len(epochs)//15))

            ax2.set_xlabel('Epoch', fontweight='bold')
            ax2.set_ylabel('Accuracy', fontweight='bold')
            ax2.set_title(f'{model_name} - Accuracy', fontweight='bold', pad=15)
            ax2.legend(frameon=True, fancybox=True, shadow=True)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim([0, 1.05])

        plt.tight_layout()
        save_path = f"{save_dir}/training_curves.png"
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    def plot_confusion_matrix(self, model_name, y_true, y_pred):
        """Improved confusion matrix"""
        save_dir = f"{self.viz_dir}/individual_models/{model_name}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create annotations
        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]:,}\n({cm_percent[i, j]:.1f}%)'

        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=['Benign', 'Attack'],
                   yticklabels=['Benign', 'Attack'],
                   cbar_kws={'label': 'Count'},
                   linewidths=2, linecolor='white',
                   ax=ax, square=True,
                   annot_kws={'size': 13, 'weight': 'bold'})

        ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=14)
        ax.set_ylabel('True Label', fontweight='bold', fontsize=14)
        ax.set_title(f'{model_name} - Confusion Matrix', fontweight='bold', fontsize=16, pad=20)

        plt.tight_layout()
        save_path = f"{save_dir}/confusion_matrix.png"
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    def plot_roc_curve(self, model_name, y_true, y_pred_proba):
        """Improved ROC curve"""
        save_dir = f"{self.viz_dir}/individual_models/{model_name}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
            y_scores = y_pred_proba[:, 1]
        else:
            y_scores = y_pred_proba.flatten()

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(9, 7))

        ax.plot(fpr, tpr, color='#2c3e50', lw=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax.fill_between(fpr, tpr, alpha=0.2, color='#3498db')
        ax.plot([0, 1], [0, 1], color='#e74c3c', lw=2, linestyle='--', label='Random Classifier')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title(f'{model_name} - ROC Curve', fontweight='bold', pad=15)
        ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = f"{save_dir}/roc_curve.png"
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    def plot_precision_recall_curve(self, model_name, y_true, y_pred_proba):
        """Improved Precision-Recall curve"""
        save_dir = f"{self.viz_dir}/individual_models/{model_name}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
            y_scores = y_pred_proba[:, 1]
        else:
            y_scores = y_pred_proba.flatten()

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=(9, 7))

        ax.plot(recall, precision, color='#27ae60', lw=3, label=f'PR Curve (AUC = {pr_auc:.4f})')
        ax.fill_between(recall, precision, alpha=0.2, color='#2ecc71')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontweight='bold')
        ax.set_ylabel('Precision', fontweight='bold')
        ax.set_title(f'{model_name} - Precision-Recall Curve', fontweight='bold', pad=15)
        ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = f"{save_dir}/pr_curve.png"
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    def plot_feature_importance(self, model_name, model, feature_names=None, top_k=20):
        """Improved feature importance"""
        if not hasattr(model, 'feature_importances_'):
            return

        save_dir = f"{self.viz_dir}/individual_models/{model_name}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        importances = model.feature_importances_
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(len(importances))]

        indices = np.argsort(importances)[-top_k:]

        fig, ax = plt.subplots(figsize=(12, 9))

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_k))
        bars = ax.barh(range(top_k), importances[indices], color=colors, edgecolor='black', linewidth=1)

        ax.set_yticks(range(top_k))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score', fontweight='bold')
        ax.set_title(f'{model_name} - Top {top_k} Features', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        save_path = f"{save_dir}/feature_importance.png"
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    def plot_prediction_distribution(self, model_name, y_pred_proba, y_true):
        """Improved prediction distribution"""
        save_dir = f"{self.viz_dir}/individual_models/{model_name}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
            y_scores = y_pred_proba[:, 1]
        else:
            y_scores = y_pred_proba.flatten()

        fig, ax = plt.subplots(figsize=(12, 6))

        benign_scores = y_scores[y_true == 0]
        attack_scores = y_scores[y_true == 1]

        ax.hist(benign_scores, bins=50, alpha=0.65, label='Benign', color='#3498db', edgecolor='black')
        ax.hist(attack_scores, bins=50, alpha=0.65, label='Attack', color='#e74c3c', edgecolor='black')
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')

        ax.set_xlabel('Prediction Probability (Attack)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{model_name} - Prediction Distribution', fontweight='bold', pad=15)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = f"{save_dir}/prediction_distribution.png"
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    # ========================================================================
    # COMPARATIVE VISUALIZATIONS
    # ========================================================================

    def plot_comparative_accuracy(self, results_df):
        """Improved accuracy comparison"""
        save_dir = f"{self.viz_dir}/comparative"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 10))

        results_sorted = results_df.sort_values('accuracy', ascending=True)
        colors = [MODEL_COLORS.get(mt, '#95a5a6') for mt in results_sorted['model_type']]

        bars = ax.barh(range(len(results_sorted)), results_sorted['accuracy'], 
                      color=colors, edgecolor='black', linewidth=1)
        ax.set_yticks(range(len(results_sorted)))
        ax.set_yticklabels(results_sorted['model_name'])
        ax.set_xlabel('Accuracy', fontweight='bold', fontsize=13)
        ax.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=16, pad=20)
        ax.set_xlim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='x')

        # Value labels
        for i, (idx, row) in enumerate(results_sorted.iterrows()):
            ax.text(row['accuracy'] + 0.01, i, f"{row['accuracy']:.4f}", va='center', fontsize=9)

        # Legend
        legend_elements = [Patch(facecolor=color, label=label, edgecolor='black') 
                          for label, color in MODEL_COLORS.items()]
        ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True)

        plt.tight_layout()
        save_path = f"{save_dir}/accuracy_comparison.png"
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    def plot_comparative_f1_score(self, results_df):
        """Improved F1-score comparison"""
        save_dir = f"{self.viz_dir}/comparative"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 10))

        results_sorted = results_df.sort_values('f1_score', ascending=True)
        colors = [MODEL_COLORS.get(mt, '#95a5a6') for mt in results_sorted['model_type']]

        bars = ax.barh(range(len(results_sorted)), results_sorted['f1_score'], 
                      color=colors, edgecolor='black', linewidth=1)
        ax.set_yticks(range(len(results_sorted)))
        ax.set_yticklabels(results_sorted['model_name'])
        ax.set_xlabel('F1-Score', fontweight='bold', fontsize=13)
        ax.set_title('Model F1-Score Comparison', fontweight='bold', fontsize=16, pad=20)
        ax.set_xlim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='x')

        # Value labels
        for i, (idx, row) in enumerate(results_sorted.iterrows()):
            ax.text(row['f1_score'] + 0.01, i, f"{row['f1_score']:.4f}", va='center', fontsize=9)

        # Legend
        legend_elements = [Patch(facecolor=color, label=label, edgecolor='black') 
                          for label, color in MODEL_COLORS.items()]
        ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True)

        plt.tight_layout()
        save_path = f"{save_dir}/f1_score_comparison.png"
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    def plot_error_metrics_comparison(self, results_df):
        """NEW: Compare error metrics (MSE, RMSE, MAE)"""
        save_dir = f"{self.viz_dir}/comparative"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        error_metrics = ['mse', 'rmse', 'mae']
        available = [m for m in error_metrics if m in results_df.columns]

        if not available:
            return

        fig, axes = plt.subplots(1, len(available), figsize=(6*len(available), 8))
        if len(available) == 1:
            axes = [axes]

        for ax, metric in zip(axes, available):
            sorted_df = results_df.sort_values(metric, ascending=True).head(15)
            colors = [MODEL_COLORS.get(mt, '#95a5a6') for mt in sorted_df['model_type']]

            bars = ax.barh(range(len(sorted_df)), sorted_df[metric], 
                          color=colors, edgecolor='black', linewidth=1)
            ax.set_yticks(range(len(sorted_df)))
            ax.set_yticklabels(sorted_df['model_name'], fontsize=10)
            ax.set_xlabel(metric.upper(), fontweight='bold')
            ax.set_title(f'Top 15 Models by {metric.upper()}\n(Lower is Better)', 
                        fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, axis='x')

            # Value labels
            for i, val in enumerate(sorted_df[metric]):
                ax.text(val + max(sorted_df[metric])*0.02, i, f'{val:.4f}', va='center', fontsize=8)

        plt.tight_layout()
        save_path = f"{save_dir}/error_metrics_comparison.png"
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    def plot_metrics_heatmap(self, results_df):
        """NEW: Heatmap of key metrics for top models"""
        save_dir = f"{self.viz_dir}/comparative"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Select metrics
        metric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'mse', 'rmse']
        available_cols = [col for col in metric_cols if col in results_df.columns]

        if len(available_cols) < 3:
            return

        # Top 15 models by F1
        top_models = results_df.nlargest(15, 'f1_score')
        heatmap_data = top_models[['model_name'] + available_cols].set_index('model_name')

        # Normalize MSE and RMSE (invert so higher is better for visualization)
        for col in ['mse', 'rmse', 'mae']:
            if col in heatmap_data.columns:
                max_val = heatmap_data[col].max()
                if max_val > 0:
                    heatmap_data[col] = 1 - (heatmap_data[col] / max_val)

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   linewidths=1.5, linecolor='white',
                   cbar_kws={'label': 'Normalized Score'},
                   vmin=0, vmax=1, ax=ax, annot_kws={'size': 9})

        ax.set_xlabel('Metrics', fontweight='bold', fontsize=13)
        ax.set_ylabel('Model', fontweight='bold', fontsize=13)
        ax.set_title('Performance Heatmap - Top 15 Models', fontweight='bold', fontsize=16, pad=20)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        save_path = f"{save_dir}/metrics_heatmap.png"
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    def plot_radar_chart_top_models(self, results_df, top_n=8):
        """Improved radar chart"""
        save_dir = f"{self.viz_dir}/comparative"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        top_models = results_df.nlargest(top_n, 'f1_score')

        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        if 'roc_auc' in results_df.columns:
            metrics.append('roc_auc')

        available = [m for m in metrics if m in results_df.columns and results_df[m].notna().any()]

        if len(available) < 3:
            return

        num_vars = len(available)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

        colors = plt.cm.Set3(np.linspace(0, 1, top_n))

        for idx, (_, row) in enumerate(top_models.iterrows()):
            values = [row[metric] if not np.isnan(row[metric]) else 0 for metric in available]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2.5, label=row['model_name'], 
                   color=colors[idx], markersize=6)
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in available], 
                          fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title(f'Top {top_n} Models - Performance Radar', fontweight='bold', fontsize=16, pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=10, frameon=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = f"{save_dir}/radar_chart_top_models.png"
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    def plot_precision_recall_scatter(self, results_df):
        """Improved precision vs recall scatter"""
        save_dir = f"{self.viz_dir}/comparative"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 9))

        for model_type in results_df['model_type'].unique():
            subset = results_df[results_df['model_type'] == model_type]
            ax.scatter(subset['recall'], subset['precision'],
                      c=MODEL_COLORS.get(model_type, '#95a5a6'),
                      label=model_type, s=120, alpha=0.7, edgecolors='black', linewidth=1.5)

        # Annotate best models
        top_models = results_df.nlargest(5, 'f1_score')
        for _, row in top_models.iterrows():
            ax.annotate(row['model_name'], (row['recall'], row['precision']),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

        ax.set_xlabel('Recall', fontweight='bold', fontsize=13)
        ax.set_ylabel('Precision', fontweight='bold', fontsize=13)
        ax.set_title('Precision vs Recall - All Models', fontweight='bold', fontsize=16, pad=20)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = f"{save_dir}/precision_recall_scatter.png"
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    def plot_training_time_comparison(self, results_df):
        """Training time comparison"""
        save_dir = f"{self.viz_dir}/comparative"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        training_times = []
        model_names = []

        for idx, row in results_df.iterrows():
            model_name = row['model_name']
            model_type = row['model_type']

            try:
                if model_type == 'Classical ML':
                    metadata_path = f"{self.config['models_dir']}/classical_ml/{model_name}_metadata.json"
                elif model_type == 'Deep Learning':
                    metadata_path = f"{self.config['models_dir']}/deep_learning/{model_name}_metadata.json"
                elif model_type == 'Transformer':
                    metadata_path = f"{self.config['models_dir']}/transformers/{model_name}/metadata.json"
                elif model_type == 'Hybrid':
                    metadata_path = f"{self.config['models_dir']}/hybrid/{model_name}_metadata.json"
                else:
                    continue

                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    training_times.append(metadata['training_time'])
                    model_names.append(model_name)
            except:
                continue

        if not training_times:
            print("  ⚠ No training time data available")
            return

        fig, ax = plt.subplots(figsize=(14, 10))

        sorted_indices = np.argsort(training_times)
        sorted_times = [training_times[i] for i in sorted_indices]
        sorted_names = [model_names[i] for i in sorted_indices]

        bars = ax.barh(range(len(sorted_times)), sorted_times, color='#3498db', 
                      edgecolor='black', linewidth=1)
        ax.set_yticks(range(len(sorted_times)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Training Time (seconds, log scale)', fontweight='bold')
        ax.set_xscale('log')
        ax.set_title('Training Time Comparison', fontweight='bold', fontsize=16, pad=20)
        ax.grid(True, alpha=0.3, axis='x')

        # Value labels
        for i, time in enumerate(sorted_times):
            ax.text(time * 1.1, i, f"{time:.1f}s", va='center', fontsize=9)

        plt.tight_layout()
        save_path = f"{save_dir}/training_time_comparison.png"
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    def generate_all_visualizations(self, results_df):
        """Generate all comparative visualizations"""
        print("\n" + "="*70)
        print("GENERATING IMPROVED VISUALIZATIONS")
        print("="*70 + "\n")

        self.plot_comparative_accuracy(results_df)
        self.plot_comparative_f1_score(results_df)
        self.plot_error_metrics_comparison(results_df)  # NEW
        self.plot_metrics_heatmap(results_df)  # NEW
        self.plot_radar_chart_top_models(results_df, top_n=8)
        self.plot_precision_recall_scatter(results_df)
        self.plot_training_time_comparison(results_df)

        print("\n" + "="*70)
        print("✓ All visualizations generated successfully!")
        print("="*70)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results_path = f"{GLOBAL_CONFIG['results_dir']}/metrics/all_models_metrics.csv"
    results_df = pd.read_csv(results_path)

    viz_gen = VisualizationGenerator(GLOBAL_CONFIG)
    viz_gen.generate_all_visualizations(results_df)

    print("\nVisualization complete!")
