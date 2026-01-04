# ============================================================================
# FILE: master_runner.py
# DESCRIPTION: Master pipeline orchestrator (FIXED)
# ============================================================================

import sys
import time
import traceback
from pathlib import Path
import numpy as np
import os
import pickle

from config import (
    GLOBAL_CONFIG, 
    set_seed, 
    create_directory_structure, 
    check_gpu,
    CLASSICAL_ML_MODELS,
    DEEP_LEARNING_MODELS,
    TRANSFORMER_MODELS,
    HYBRID_MODELS,
    GNN_MODELS
)

class MasterPipeline:
    """Master pipeline for training all models"""
    
    def __init__(self, config=None):
        if config is None:
            config = GLOBAL_CONFIG
        self.config = config
        
    def print_step(self, step, total, description):
        """Print formatted step header"""
        print(f"\n[STEP {step}/{total}] {description}")
        print("-"*70)
    
    def run_setup(self):
        """Step 1: Setup and configuration"""
        self.print_step(1, 10, "SETUP AND CONFIGURATION")
        set_seed(self.config['random_seed'])
        create_directory_structure()
        check_gpu()
        print("Setup complete!")
    
    def run_data_preprocessing(self):
        """Step 2: Data preprocessing"""
        self.print_step(2, 10, "DATA PREPROCESSING")
        
        from data_preprocessing import DataLoader
        
        loader = DataLoader(data_dir=self.config['data_dir'], config=self.config)
        merged_df, _ = loader.load_all_datasets()
        processed_df = loader.preprocess_data(merged_df)
        splits = loader.create_splits(processed_df)
        
        data_dir = f"{self.config['base_dir']}/data"
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
        for key, val in splits.items():
            np.save(f"{data_dir}/{key}.npy", val)
        
        processed_df.to_csv(f"{data_dir}/processed_data.csv", index=False)
        
        print("Data preprocessing complete!")
        return splits
    
    def run_feature_extraction(self, splits=None):
        """Step 3: Feature extraction"""
        self.print_step(3, 10, "FEATURE EXTRACTION")
        
        from feature_engineering import extract_all_features
        
        if splits is None:
            data_dir = f"{self.config['base_dir']}/data"
            splits = {
                'X_train': np.load(f"{data_dir}/X_train.npy", allow_pickle=True),
                'y_train': np.load(f"{data_dir}/y_train.npy"),
                'X_val': np.load(f"{data_dir}/X_val.npy", allow_pickle=True),
                'y_val': np.load(f"{data_dir}/y_val.npy"),
                'X_test': np.load(f"{data_dir}/X_test.npy", allow_pickle=True),
                'y_test': np.load(f"{data_dir}/y_test.npy"),
            }
        
        features = extract_all_features(splits, self.config)
        print("Feature extraction complete!")
        return features
    
    def run_classical_ml_training(self):
        """Step 4: Train classical ML models"""
        self.print_step(4, 10, "TRAINING CLASSICAL ML MODELS")
        
        from models_classical import ClassicalMLTrainer
        
        feature_dir = f"{self.config['base_dir']}/features"
        X_train = np.load(f"{feature_dir}/X_train_tfidf.npy")
        X_val = np.load(f"{feature_dir}/X_val_tfidf.npy")
        
        data_dir = f"{self.config['base_dir']}/data"
        y_train = np.load(f"{data_dir}/y_train.npy")
        y_val = np.load(f"{data_dir}/y_val.npy")
        
        trainer = ClassicalMLTrainer(self.config)
        trained_models = trainer.train_all_classical_models(
            X_train, y_train, X_val, y_val, feature_type='tfidf'
        )
        
        print(f"Classical ML training complete! Trained {len(trained_models)} models.")
    
    def run_deep_learning_training(self):
        """Step 5: Train deep learning models"""
        self.print_step(5, 10, "TRAINING DEEP LEARNING MODELS")
        
        from models_deep_learning import DeepLearningTrainer
        
        feature_dir = f"{self.config['base_dir']}/features"
        X_train = np.load(f"{feature_dir}/X_train_uniembed.npy")
        X_val = np.load(f"{feature_dir}/X_val_uniembed.npy")
        
        data_dir = f"{self.config['base_dir']}/data"
        y_train = np.load(f"{data_dir}/y_train.npy")
        y_val = np.load(f"{data_dir}/y_val.npy")
        
        trainer = DeepLearningTrainer(self.config)
        trained_models = trainer.train_all_deep_learning_models(
            X_train, y_train, X_val, y_val, feature_type='uniembed'
        )
        
        print(f"Deep learning training complete! Trained {len(trained_models)} models.")
    
    def run_transformer_training(self):
        """Step 6: Train transformer models"""
        self.print_step(6, 10, "FINE-TUNING TRANSFORMER MODELS")
        
        from models_transformers import TransformerTrainer
        
        data_dir = f"{self.config['base_dir']}/data"
        X_train = np.load(f"{data_dir}/X_train.npy", allow_pickle=True)
        X_val = np.load(f"{data_dir}/X_val.npy", allow_pickle=True)
        y_train = np.load(f"{data_dir}/y_train.npy")
        y_val = np.load(f"{data_dir}/y_val.npy")
        
        trainer = TransformerTrainer(self.config)
        trained_models = trainer.train_all_transformer_models(
            X_train, y_train, X_val, y_val
        )
        
        print(f"Transformer training complete! Trained {len(trained_models)} models.")
    
    def run_gnn_training(self):
        """Step 7: Train GNN models"""
        self.print_step(7, 10, "TRAINING GRAPH NEURAL NETWORK MODELS")
        
        from models_gnn import GraphConstructor, GNNTrainer
        from gensim.models import Word2Vec
        import pickle
        
        feature_dir = f"{self.config['base_dir']}/features"
        word2vec_model = Word2Vec.load(f"{feature_dir}/word2vec.model")
        
        data_dir = f"{self.config['base_dir']}/data"
        X_train = np.load(f"{data_dir}/X_train.npy", allow_pickle=True)
        X_val = np.load(f"{data_dir}/X_val.npy", allow_pickle=True)
        X_test = np.load(f"{data_dir}/X_test.npy", allow_pickle=True)
        y_train = np.load(f"{data_dir}/y_train.npy")
        y_val = np.load(f"{data_dir}/y_val.npy")
        y_test = np.load(f"{data_dir}/y_test.npy")
        
        print("\nApplying aggressive sampling for GNN...")
        
        gnn_sample_fraction = 0.05
        max_samples_train = 6000
        max_samples_val = 1200
        max_samples_test = 1200
        
        train_sample_size = min(int(len(X_train) * gnn_sample_fraction), max_samples_train)
        train_indices = np.random.choice(len(X_train), train_sample_size, replace=False)
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        
        val_sample_size = min(int(len(X_val) * gnn_sample_fraction), max_samples_val)
        val_indices = np.random.choice(len(X_val), val_sample_size, replace=False)
        X_val = X_val[val_indices]
        y_val = y_val[val_indices]
        
        test_sample_size = min(int(len(X_test) * gnn_sample_fraction), max_samples_test)
        test_indices = np.random.choice(len(X_test), test_sample_size, replace=False)
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]
        
        print(f"\nGNN Dataset sizes:")
        print(f"  Train: {len(X_train)}")
        print(f"  Val: {len(X_val)}")
        print(f"  Test: {len(X_test)}\n")
        
        constructor = GraphConstructor(word2vec_model)
        
        print("Constructing training graphs...")
        train_graphs = constructor.texts_to_graphs(X_train, y_train)
        
        print("Constructing validation graphs...")
        val_graphs = constructor.texts_to_graphs(X_val, y_val)
        
        print("Constructing test graphs...")
        test_graphs = constructor.texts_to_graphs(X_test, y_test)
        
        print("\nSaving graphs...")
        with open(f"{data_dir}/train_graphs.pkl", 'wb') as f:
            pickle.dump(train_graphs, f)
        with open(f"{data_dir}/val_graphs.pkl", 'wb') as f:
            pickle.dump(val_graphs, f)
        with open(f"{data_dir}/test_graphs.pkl", 'wb') as f:
            pickle.dump(test_graphs, f)
        
        print("Graphs saved!")
        
        del X_train, X_val, X_test
        import gc
        gc.collect()
        
        trainer = GNNTrainer(self.config)
        trained_models = trainer.train_all_gnn_models(train_graphs, val_graphs)
        
        print(f"GNN training complete! Trained {len(trained_models)} models.")
    
    def run_hybrid_training(self):
        """Step 8: Train hybrid models"""
        self.print_step(8, 10, "TRAINING HYBRID ENSEMBLE MODELS")
        
        from models_hybrid import HybridModelTrainer
        
        feature_dir = f"{self.config['base_dir']}/features"
        data_dir = f"{self.config['base_dir']}/data"
        
        X_train = np.load(f"{feature_dir}/X_train_uniembed.npy")
        X_val = np.load(f"{feature_dir}/X_val_uniembed.npy")
        y_train = np.load(f"{data_dir}/y_train.npy")
        y_val = np.load(f"{data_dir}/y_val.npy")
        
        X_bert_train = X_bert_val = None
        bert_train_path = f"{feature_dir}/X_train_bert.npy"
        bert_val_path = f"{feature_dir}/X_val_bert.npy"
        
        if os.path.exists(bert_train_path) and os.path.exists(bert_val_path):
            X_bert_train = np.load(bert_train_path)
            X_bert_val = np.load(bert_val_path)
            print("Loaded BERT embeddings.")
        else:
            print("BERT embeddings not found, skipping BERT_XGBoost hybrid.")
        
        models_to_train = []
        for name in HYBRID_MODELS:
            if name == 'BERT_XGBoost' and X_bert_train is None:
                continue
            models_to_train.append(name)
        
        trainer = HybridModelTrainer(self.config)
        trained_models = trainer.train_all_hybrid_models(
            X_train, y_train, X_val, y_val,
            X_bert_train=X_bert_train, X_bert_val=X_bert_val,
            models_override=models_to_train
        )
        
        print(f"Hybrid training complete! Trained {len(trained_models)} models.")
    
    def run_evaluation(self):
        """Step 9: Evaluate all models"""
        self.print_step(9, 10, "EVALUATING ALL MODELS")
        
        from evaluation import ComprehensiveEvaluator
        
        feature_dir = f"{self.config['base_dir']}/features"
        data_dir = f"{self.config['base_dir']}/data"
        
        X_test_text = np.load(f"{data_dir}/X_test.npy", allow_pickle=True)
        y_test = np.load(f"{data_dir}/y_test.npy")
        
        test_data = {
            'X_test_tfidf': np.load(f"{feature_dir}/X_test_tfidf.npy"),
            'X_test_uniembed': np.load(f"{feature_dir}/X_test_uniembed.npy"),
            'X_test_text': X_test_text,
            'y_test': y_test
        }
        
        import pickle
        gnn_test_path = f"{data_dir}/test_graphs.pkl"
        if os.path.exists(gnn_test_path):
            with open(gnn_test_path, 'rb') as f:
                test_data['test_graphs'] = pickle.load(f)
        
        evaluator = ComprehensiveEvaluator(self.config)
        all_results = evaluator.evaluate_all_models(test_data)
        
        print(f"Evaluation complete! Evaluated {len(all_results)} models.")
        return all_results
    
    def run_visualization(self):
        """Step 10: Generate visualizations"""
        self.print_step(10, 10, "GENERATING VISUALIZATIONS")
        
        from visualization import VisualizationGenerator
        import pandas as pd
        
        results_path = f"{self.config['results_dir']}/metrics/all_models_metrics.csv"
        results_df = pd.read_csv(results_path)
        
        viz_gen = VisualizationGenerator(self.config)
        viz_gen.generate_all_visualizations(results_df)
        
        print("Visualization complete!")
    
    def run_full_pipeline(self, skip_steps=None):
        """Run complete pipeline"""
        if skip_steps is None:
            skip_steps = []
        
        print("="*70)
        print("    COMPREHENSIVE WEB ATTACK DETECTION RESEARCH PIPELINE    ")
        print("="*70)
        print(f"\nStart: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Random seed: {self.config['random_seed']}")
        print(f"Output: {self.config['base_dir']}\n")
        
        try:
            if 1 not in skip_steps:
                self.run_setup()
            
            if 2 not in skip_steps:
                splits = self.run_data_preprocessing()
            else:
                splits = None
            
            if 3 not in skip_steps:
                self.run_feature_extraction(splits if 2 not in skip_steps else None)
            
            if 4 not in skip_steps and self.config.get('train_classical_ml', True):
                self.run_classical_ml_training()
            
            if 5 not in skip_steps and self.config.get('train_deep_learning', True):
                self.run_deep_learning_training()
            
            if 6 not in skip_steps and self.config.get('train_transformers', False):
                self.run_transformer_training()
            
            if 7 not in skip_steps and self.config.get('train_gnn', False):
                self.run_gnn_training()
            
            if 8 not in skip_steps and self.config.get('train_hybrid', True):
                self.run_hybrid_training()
            
            if 9 not in skip_steps:
                all_results = self.run_evaluation()
            
            if 10 not in skip_steps:
                self.run_visualization()
            
            print("\n" + "="*70)
            print("    PIPELINE COMPLETE!    ")
            print("="*70)
            print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        except Exception as e:
            print("\n" + "="*70)
            print("ERROR: Pipeline failed!")
            print("="*70)
            print(f"Error: {str(e)}\n")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ML pipeline')
    parser.add_argument('--skip-steps', nargs='+', type=int, default=[],
                        help='Steps to skip (1-10)')
    args = parser.parse_args()
    
    pipeline = MasterPipeline(GLOBAL_CONFIG)
    pipeline.run_full_pipeline(skip_steps=args.skip_steps)

