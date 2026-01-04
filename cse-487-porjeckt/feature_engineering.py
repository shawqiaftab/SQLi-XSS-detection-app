# ============================================================================
# FILE: feature_engineering.py
# DESCRIPTION: Extract all feature representations (MEMORY OPTIMIZED)
# ============================================================================

from config import GLOBAL_CONFIG

import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from transformers import BertTokenizer, BertModel
import torch
from pathlib import Path
import pickle
import warnings
import gc
warnings.filterwarnings('ignore')

# ============================================================================
# UNIEMBED FEATURE EXTRACTION (MEMORY OPTIMIZED)
# ============================================================================

class UniEmbedExtractor:
    """Extract UniEmbed features with dynamic dimensions"""
    
    def __init__(self, config=None):
        if config is None:
            config = GLOBAL_CONFIG
        self.config = config
        self.word2vec_model = None
        self.fasttext_model = None
        self.use_model = None
        
    def tokenize_texts(self, texts):
        """Tokenize texts for training word embedding models"""
        return [text.split() for text in texts]
    
    def train_word2vec(self, texts):
        """Train Word2Vec model"""
        print("Training Word2Vec model...")
        
        tokenized = self.tokenize_texts(texts)
        vector_size = self.config['word2vec_dim']
        
        self.word2vec_model = Word2Vec(
            sentences=tokenized,
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=2,
            sg=0,
            seed=self.config['random_seed']
        )
        
        print(f"  Word2Vec trained: {len(self.word2vec_model.wv)} tokens")
        return self.word2vec_model
    
    def train_fasttext(self, texts):
        """Train FastText model"""
        print("Training FastText model...")
        
        tokenized = self.tokenize_texts(texts)
        vector_size = self.config['fasttext_dim']
        
        self.fasttext_model = FastText(
            sentences=tokenized,
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=2,
            min_n=3,
            max_n=6,
            seed=self.config['random_seed']
        )
        
        print(f"  FastText trained: {len(self.fasttext_model.wv)} tokens")
        return self.fasttext_model
    
    def load_use_model(self):
        """Load Universal Sentence Encoder"""
        if self.config.get('skip_use', False):
            print("Skipping USE (disabled in config)")
            return None
            
        print("Loading Universal Sentence Encoder...")
        self.use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        print("  USE loaded successfully")
        return self.use_model
    
    def get_word2vec_embedding(self, text):
        """Get Word2Vec embedding for a text"""
        tokens = text.split()
        vectors = []
        
        for token in tokens:
            if token in self.word2vec_model.wv:
                vectors.append(self.word2vec_model.wv[token])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.config['word2vec_dim'])
    
    def get_fasttext_embedding(self, text):
        """Get FastText embedding for a text"""
        tokens = text.split()
        vectors = []
        
        for token in tokens:
            vectors.append(self.fasttext_model.wv[token])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.config['fasttext_dim'])
    
    def get_use_embedding(self, texts):
        """Get USE embeddings for texts"""
        if self.use_model is None:
            return np.zeros((len(texts), self.config['use_dim']))
        embeddings = self.use_model(texts)
        return embeddings.numpy()
    
    def extract_uniembed_features(self, texts, batch_size=1000):
        """Extract complete UniEmbed features (dynamic dimensions)"""
        
        w2v_dim = self.config['word2vec_dim']
        ft_dim = self.config['fasttext_dim']
        use_dim = self.config['use_dim']
        
        # Calculate total based on whether USE is used
        if self.config.get('skip_use', False):
            total_dim = w2v_dim + ft_dim
        else:
            total_dim = w2v_dim + ft_dim + use_dim
        
        print("\n" + "="*70)
        print(f"EXTRACTING UNIEMBED FEATURES ({total_dim}D)")
        print(f"  Word2Vec: {w2v_dim}D")
        print(f"  FastText: {ft_dim}D")
        if not self.config.get('skip_use', False):
            print(f"  USE: {use_dim}D")
        print("="*70 + "\n")
        
        n_samples = len(texts)
        uniembed_features = np.zeros((n_samples, total_dim))
        
        # Word2Vec
        print("Extracting Word2Vec features...")
        for i, text in enumerate(texts):
            uniembed_features[i, :w2v_dim] = self.get_word2vec_embedding(text)
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{n_samples} samples")
        
        # FastText
        print("\nExtracting FastText features...")
        for i, text in enumerate(texts):
            uniembed_features[i, w2v_dim:w2v_dim+ft_dim] = self.get_fasttext_embedding(text)
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{n_samples} samples")
        
        # USE (if not skipped)
        if not self.config.get('skip_use', False):
            print("\nExtracting USE features...")
            use_start = w2v_dim + ft_dim
            use_end = use_start + use_dim
            
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_texts = list(texts[i:batch_end])
                
                batch_embeddings = self.get_use_embedding(batch_texts)
                uniembed_features[i:batch_end, use_start:use_end] = batch_embeddings
                
                if (batch_end) % 10000 == 0 or batch_end == n_samples:
                    print(f"  Processed {batch_end}/{n_samples} samples")
        
        print(f"\nUniEmbed features extracted: {uniembed_features.shape}")
        return uniembed_features
    
    def fit_transform(self, train_texts, val_texts, test_texts):
        """Train models on train set and transform all sets"""
        self.train_word2vec(train_texts)
        self.train_fasttext(train_texts)
        self.load_use_model()
        
        X_train_uniembed = self.extract_uniembed_features(train_texts)
        gc.collect()
        
        X_val_uniembed = self.extract_uniembed_features(val_texts)
        gc.collect()
        
        X_test_uniembed = self.extract_uniembed_features(test_texts)
        gc.collect()
        
        return X_train_uniembed, X_val_uniembed, X_test_uniembed
    
    def save(self, save_dir):
        """Save trained models"""
        print(f"\nSaving UniEmbed models to {save_dir}...")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        self.word2vec_model.save(f"{save_dir}/word2vec.model")
        self.fasttext_model.save(f"{save_dir}/fasttext.model")
        
        print("Models saved successfully!")

# ============================================================================
# TF-IDF FEATURE EXTRACTION
# ============================================================================

class TFIDFExtractor:
    """Extract TF-IDF features"""
    
    def __init__(self, config=None):
        if config is None:
            config = GLOBAL_CONFIG
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=config['tfidf_max_features'],
            ngram_range=config['tfidf_ngram_range'],
            analyzer='word',
            lowercase=True,
            stop_words='english'
        )
    
    def fit_transform(self, train_texts, val_texts, test_texts):
        """Fit on train and transform all sets"""
        print("\n" + "="*70)
        print(f"EXTRACTING TF-IDF FEATURES ({self.config['tfidf_max_features']}D)")
        print("="*70 + "\n")
        
        print("Fitting TF-IDF vectorizer...")
        X_train_tfidf = self.vectorizer.fit_transform(train_texts)
        
        print("Transforming validation set...")
        X_val_tfidf = self.vectorizer.transform(val_texts)
        
        print("Transforming test set...")
        X_test_tfidf = self.vectorizer.transform(test_texts)
        
        print(f"\nTF-IDF features extracted:")
        print(f"   Train: {X_train_tfidf.shape}")
        print(f"   Val: {X_val_tfidf.shape}")
        print(f"   Test: {X_test_tfidf.shape}")
        
        X_train_tfidf = X_train_tfidf.toarray()
        X_val_tfidf = X_val_tfidf.toarray()
        X_test_tfidf = X_test_tfidf.toarray()
        
        return X_train_tfidf, X_val_tfidf, X_test_tfidf
    
    def save(self, save_dir):
        """Save vectorizer"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{save_dir}/tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print("TF-IDF vectorizer saved!")

# ============================================================================
# MAIN FEATURE EXTRACTION PIPELINE
# ============================================================================

def extract_all_features(splits, config=None):
    """Extract all feature representations"""
    
    if config is None:
        config = GLOBAL_CONFIG
    
    X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
    y_train = splits['y_train']
    
    features = {}
    feature_dir = f"{config['base_dir']}/features"
    Path(feature_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. UniEmbed Features
    print("\n" + "="*70)
    print("STAGE 1: UNIEMBED FEATURES")
    print("="*70)
    
    uniembed_extractor = UniEmbedExtractor(config)
    X_train_uni, X_val_uni, X_test_uni = uniembed_extractor.fit_transform(
        X_train, X_val, X_test
    )
    uniembed_extractor.save(feature_dir)
    
    features['uniembed'] = {
        'X_train': X_train_uni,
        'X_val': X_val_uni,
        'X_test': X_test_uni
    }
    
    np.save(f"{feature_dir}/X_train_uniembed.npy", X_train_uni)
    np.save(f"{feature_dir}/X_val_uniembed.npy", X_val_uni)
    np.save(f"{feature_dir}/X_test_uniembed.npy", X_test_uni)
    
    del X_train_uni, X_val_uni, X_test_uni
    gc.collect()
    
    # 2. TF-IDF Features
    print("\n" + "="*70)
    print("STAGE 2: TF-IDF FEATURES")
    print("="*70)
    
    tfidf_extractor = TFIDFExtractor(config)
    X_train_tfidf, X_val_tfidf, X_test_tfidf = tfidf_extractor.fit_transform(
        X_train, X_val, X_test
    )
    tfidf_extractor.save(feature_dir)
    
    features['tfidf'] = {
        'X_train': X_train_tfidf,
        'X_val': X_val_tfidf,
        'X_test': X_test_tfidf
    }
    
    np.save(f"{feature_dir}/X_train_tfidf.npy", X_train_tfidf)
    np.save(f"{feature_dir}/X_val_tfidf.npy", X_val_tfidf)
    np.save(f"{feature_dir}/X_test_tfidf.npy", X_test_tfidf)
    
    del X_train_tfidf, X_val_tfidf, X_test_tfidf
    gc.collect()
    
    print("\n" + "="*70)
    print("ALL FEATURES EXTRACTED AND SAVED!")
    print("="*70)
    
    return features

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    from config import GLOBAL_CONFIG, set_seed
    
    set_seed(GLOBAL_CONFIG['random_seed'])
    
    data_dir = f"{GLOBAL_CONFIG['base_dir']}/data"
    
    splits = {
        'X_train': np.load(f"{data_dir}/X_train.npy", allow_pickle=True),
        'y_train': np.load(f"{data_dir}/y_train.npy"),
        'X_val': np.load(f"{data_dir}/X_val.npy", allow_pickle=True),
        'y_val': np.load(f"{data_dir}/y_val.npy"),
        'X_test': np.load(f"{data_dir}/X_test.npy", allow_pickle=True),
        'y_test': np.load(f"{data_dir}/y_test.npy"),
    }
    
    features = extract_all_features(splits, GLOBAL_CONFIG)
    
    print("\nFeature extraction pipeline complete!")

