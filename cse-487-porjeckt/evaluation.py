# ============================================================================
# FILE: evaluation.py (MODIFIED WITH ERROR METRICS)
# DESCRIPTION: Comprehensive model evaluation with SSE, MSE, RMSE, MAE, WSSE, WMSE
# ============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, classification_report
)
import time
import json
from pathlib import Path
import joblib
import tensorflow as tf
import torch
import pickle
from config import (
    GLOBAL_CONFIG,
    set_seed,
    CLASSICAL_ML_MODELS,
    DEEP_LEARNING_MODELS,
    TRANSFORMER_MODELS,
    GNN_MODELS,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch_geometric.data import DataLoader
from models_gnn import GCN, GAT

# ============================================================================
# COMPREHENSIVE EVALUATOR
# ============================================================================

class ComprehensiveEvaluator:
    """Evaluate all models with comprehensive metrics"""

    def __init__(self, config=GLOBAL_CONFIG):
        self.config = config

    def compute_error_metrics(self, y_true, y_pred):
        """
        Compute error-based metrics: SSE, MSE, RMSE, MAE, WSSE, WMSE
        """
        # Convert to binary arrays
        y_true_binary = np.array(y_true).flatten()
        y_pred_binary = np.array(y_pred).flatten()

        # Calculate errors
        squared_errors = (y_true_binary - y_pred_binary) ** 2
        absolute_errors = np.abs(y_true_binary - y_pred_binary)

        # 1. SSE - Sum of Squared Errors
        sse = float(np.sum(squared_errors))

        # 2. MSE - Mean Squared Error
        mse = float(np.mean(squared_errors))

        # 3. RMSE - Root Mean Squared Error
        rmse = float(np.sqrt(mse))

        # 4. MAE - Mean Absolute Error
        mae = float(np.mean(absolute_errors))

        # 5. WSSE - Weighted Sum of Squared Errors
        # Weight errors by inverse class frequency (higher weight for minority class)
        unique, counts = np.unique(y_true_binary, return_counts=True)
        class_weights = {int(cls): len(y_true_binary) / count for cls, count in zip(unique, counts)}
        weights = np.array([class_weights.get(int(yt), 1.0) for yt in y_true_binary])
        weighted_squared_errors = weights * squared_errors
        wsse = float(np.sum(weighted_squared_errors))

        # 6. WMSE - Weighted Mean Squared Error
        wmse = float(np.mean(weighted_squared_errors))

        return {
            'sse': sse,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'wsse': wsse,
            'wmse': wmse
        }

    def compute_all_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Compute all evaluation metrics including error metrics
        Returns:
            Dictionary with all metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'mcc': float(mcc),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'fpr': float(fpr),
            'fnr': float(fnr),
        }

        # Add probability-based metrics
        if y_pred_proba is not None:
            try:
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                    y_scores = y_pred_proba[:, 1]
                else:
                    y_scores = y_pred_proba

                roc_auc = roc_auc_score(y_true, y_scores)
                pr_auc = average_precision_score(y_true, y_scores)

                metrics['roc_auc'] = float(roc_auc)
                metrics['pr_auc'] = float(pr_auc)
            except Exception:
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None
        else:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None

        # Add error-based metrics (NEW!)
        error_metrics = self.compute_error_metrics(y_true, y_pred)
        metrics.update(error_metrics)

        return metrics

    def evaluate_classical_ml(self, model_name, X_test, y_test):
        """Evaluate classical ML model"""
        print(f"\n  Evaluating {model_name}...")

        model_path = f"{self.config['models_dir']}/classical_ml/{model_name}.pkl"
        model = joblib.load(model_path)

        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time

        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred_proba = None

        metrics = self.compute_all_metrics(y_test, y_pred, y_pred_proba)

        metrics['total_inference_time'] = float(inference_time)
        metrics['inference_time_per_sample'] = float(inference_time / len(X_test) * 1000.0)
        metrics['throughput'] = float(len(X_test) / inference_time)

        print(f"    Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | MSE: {metrics['mse']:.6f}")

        return metrics, y_pred, y_pred_proba

    def evaluate_deep_learning(self, model_name, X_test, y_test):
        """Evaluate deep learning model"""
        print(f"\n  Evaluating {model_name}...")

        model_path = f"{self.config['models_dir']}/deep_learning/{model_name}.h5"
        model = tf.keras.models.load_model(model_path)

        if model_name == 'CNN':
            X_test_prepared = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        elif model_name in ['RNN', 'LSTM', 'BiLSTM', 'CNN_LSTM', 'CNN_RNN']:
            n_features = X_test.shape[1]
            timesteps = min(n_features, 50)
            features_per_timestep = n_features // timesteps
            truncated_features = timesteps * features_per_timestep
            X_test_truncated = X_test[:, :truncated_features]
            X_test_prepared = X_test_truncated.reshape(
                X_test.shape[0], timesteps, features_per_timestep
            )
        else:
            X_test_prepared = X_test

        start_time = time.time()
        y_pred_proba = model.predict(X_test_prepared, verbose=0)
        inference_time = time.time() - start_time

        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        metrics = self.compute_all_metrics(y_test, y_pred, y_pred_proba)

        metrics['total_inference_time'] = float(inference_time)
        metrics['inference_time_per_sample'] = float(inference_time / len(X_test) * 1000.0)
        metrics['throughput'] = float(len(X_test) / inference_time)
        metrics['model_parameters'] = int(model.count_params())

        print(f"    Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | MSE: {metrics['mse']:.6f}")

        return metrics, y_pred, y_pred_proba

    def evaluate_transformer(self, model_name, X_test_text, y_test):
        """Evaluate transformer model"""
        print(f"\n  Evaluating {model_name}...")

        model_dir = f"{self.config['models_dir']}/transformers/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        y_pred = []
        y_pred_proba = []
        batch_size = 32

        start_time = time.time()

        with torch.no_grad():
            for i in range(0, len(X_test_text), batch_size):
                batch_texts = list(X_test_text[i:i+batch_size])
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config['max_seq_length'],
                    return_tensors='pt'
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}

                outputs = model(**encoded)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                y_pred.extend(preds)
                y_pred_proba.extend(probs)

        inference_time = time.time() - start_time

        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)

        metrics = self.compute_all_metrics(y_test, y_pred, y_pred_proba)

        metrics['total_inference_time'] = float(inference_time)
        metrics['inference_time_per_sample'] = float(inference_time / len(X_test_text) * 1000.0)
        metrics['throughput'] = float(len(X_test_text) / inference_time)
        metrics['model_parameters'] = int(sum(p.numel() for p in model.parameters()))

        print(f"    Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | MSE: {metrics['mse']:.6f}")

        return metrics, y_pred, y_pred_proba

    def evaluate_gnn(self, model_name, test_graphs):
        """Evaluate a GNN model (GCN/GAT)"""
        print(f"\n  Evaluating {model_name}...")

        input_dim = test_graphs[0].x.shape[1]

        if model_name == 'GCN':
            model = GCN(
                input_dim=input_dim,
                hidden_dims=self.config['gnn_hidden_dims'],
                num_classes=2,
                dropout=self.config['gnn_dropout']
            )
        elif model_name == 'GAT':
            model = GAT(
                input_dim=input_dim,
                hidden_dims=self.config['gnn_hidden_dims'],
                num_classes=2,
                heads=self.config['gat_heads'],
                dropout=self.config['gnn_dropout']
            )
        else:
            raise ValueError(f"Unknown GNN model: {model_name}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = f"{self.config['models_dir']}/gnn/{model_name}.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        loader = DataLoader(test_graphs, batch_size=self.config['gnn_batch_size'], shuffle=False)

        all_preds = []
        all_labels = []

        start_time = time.time()

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                preds = out.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        inference_time = time.time() - start_time

        y_pred = np.array(all_preds)
        y_true = np.array(all_labels)

        metrics = self.compute_all_metrics(y_true, y_pred, y_pred_proba=None)

        # GNN doesn't have probability predictions, so set these to None
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None

        metrics['total_inference_time'] = float(inference_time)
        metrics['inference_time_per_sample'] = float(inference_time / len(y_true) * 1000.0)
        metrics['throughput'] = float(len(y_true) / inference_time)
        metrics['model_parameters'] = int(sum(p.numel() for p in model.parameters()))

        print(f"    Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | MSE: {metrics['mse']:.6f}")

        return metrics, y_pred, None

    def evaluate_all_models(self, test_data):
        """
        Evaluate all trained models
        Args:
            test_data: dict with test features and labels
        Returns:
            DataFrame with all results
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL EVALUATION (WITH ERROR METRICS)")
        print("="*70)

        all_results = []

        # Classical ML
        print("\n" + "-"*70)
        print("EVALUATING CLASSICAL ML MODELS")
        print("-"*70)

        for model_name in CLASSICAL_ML_MODELS:
            try:
                metrics, y_pred, y_pred_proba = self.evaluate_classical_ml(
                    model_name, test_data['X_test_tfidf'], test_data['y_test']
                )
                metrics['model_name'] = model_name
                metrics['model_type'] = 'Classical ML'
                all_results.append(metrics)
                self.save_predictions(model_name, y_pred, y_pred_proba, test_data['y_test'])
            except Exception as e:
                print(f"    Error evaluating {model_name}: {str(e)}")

        # Deep Learning
        print("\n" + "-"*70)
        print("EVALUATING DEEP LEARNING MODELS")
        print("-"*70)

        for model_name in DEEP_LEARNING_MODELS:
            try:
                metrics, y_pred, y_pred_proba = self.evaluate_deep_learning(
                    model_name, test_data['X_test_uniembed'], test_data['y_test']
                )
                metrics['model_name'] = model_name
                metrics['model_type'] = 'Deep Learning'
                all_results.append(metrics)
                self.save_predictions(model_name, y_pred, y_pred_proba, test_data['y_test'])
            except Exception as e:
                print(f"    Error evaluating {model_name}: {str(e)}")

        # Transformers
        print("\n" + "-"*70)
        print("EVALUATING TRANSFORMER MODELS")
        print("-"*70)

        for model_name in TRANSFORMER_MODELS:
            try:
                metrics, y_pred, y_pred_proba = self.evaluate_transformer(
                    model_name, test_data['X_test_text'], test_data['y_test']
                )
                metrics['model_name'] = model_name
                metrics['model_type'] = 'Transformer'
                all_results.append(metrics)
                self.save_predictions(model_name, y_pred, y_pred_proba, test_data['y_test'])
            except Exception as e:
                print(f"    Error evaluating {model_name}: {str(e)}")

        # GNN
        print("\n" + "-"*70)
        print("EVALUATING GNN MODELS")
        print("-"*70)

        if 'test_graphs' in test_data:
            test_graphs = test_data['test_graphs']
            print(f"  Loaded {len(test_graphs)} test graphs for GNN evaluation")

            for model_name in ['GCN', 'GAT']:
                model_path = f"{self.config['models_dir']}/gnn/{model_name}.pt"
                if not Path(model_path).exists():
                    continue

                try:
                    metrics, y_pred, _ = self.evaluate_gnn(model_name, test_graphs)
                    metrics['model_name'] = model_name
                    metrics['model_type'] = 'GNN'
                    all_results.append(metrics)

                    y_true = np.array([g.y.item() for g in test_graphs])
                    self.save_predictions(model_name, y_pred, None, y_true)
                except Exception as e:
                    print(f"    Error evaluating {model_name}: {str(e)}")
        else:
            print("  No test_graphs found in test_data; skipping GNN evaluation")

        # Results DataFrame
        results_df = pd.DataFrame(all_results)

        if len(results_df) > 0 and 'f1_score' in results_df.columns:
            results_df = results_df.sort_values('f1_score', ascending=False)

        results_path = f"{self.config['results_dir']}/metrics/all_models_metrics.csv"
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved: {results_path}")

        print("\n" + "="*70)
        print("EVALUATION SUMMARY - ALL MODELS")
        print("="*70)

        if len(results_df) > 0:
            # Show key metrics including new error metrics
            display_cols = ['model_name', 'model_type', 'accuracy', 'f1_score', 'mse', 'rmse', 'roc_auc']
            display_cols = [col for col in display_cols if col in results_df.columns]
            print(results_df[display_cols].to_string(index=False))
        else:
            print("No models were successfully evaluated.")

        return results_df

    def save_predictions(self, model_name, y_pred, y_pred_proba, y_test):
        """Save model predictions"""
        pred_dir = f"{self.config['results_dir']}/predictions"
        Path(pred_dir).mkdir(parents=True, exist_ok=True)

        predictions = {
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'correct': (y_pred == y_test).tolist()
        }

        if y_pred_proba is not None:
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                predictions['y_pred_proba_class_1'] = y_pred_proba[:, 1].tolist()
            else:
                predictions['y_pred_proba_class_1'] = y_pred_proba.flatten().tolist()

        pred_path = f"{pred_dir}/{model_name}_predictions.json"
        with open(pred_path, 'w') as f:
            json.dump(predictions, f, indent=2)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    set_seed(GLOBAL_CONFIG['random_seed'])

    feature_dir = f"{GLOBAL_CONFIG['base_dir']}/features"
    data_dir = f"{GLOBAL_CONFIG['base_dir']}/data"

    test_data = {
        'X_test_tfidf': np.load(f"{feature_dir}/X_test_tfidf.npy"),
        'X_test_uniembed': np.load(f"{feature_dir}/X_test_uniembed.npy"),
        'X_test_text': np.load(f"{data_dir}/X_test.npy", allow_pickle=True),
        'y_test': np.load(f"{data_dir}/y_test.npy")
    }

    # Optional GNN graphs
    gnn_test_path = f"{data_dir}/test_graphs.pkl"
    if Path(gnn_test_path).exists():
        with open(gnn_test_path, 'rb') as f:
            test_data['test_graphs'] = pickle.load(f)

    evaluator = ComprehensiveEvaluator(GLOBAL_CONFIG)
    results_df = evaluator.evaluate_all_models(test_data)

    print("\nEvaluation complete!")
