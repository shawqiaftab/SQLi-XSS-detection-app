import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import tensorflow as tf
from tensorflow import keras
import torch
from pathlib import Path
import re
import os
from gensim.models import Word2Vec, FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# SQL and XSS Keywords
SQLKEYWORDS = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'UNION', 
               'WHERE', 'FROM', 'JOIN', 'AND', 'OR', 'NOT', 'NULL', 'ORDER', 'GROUP', 
               'HAVING', 'LIMIT', 'OFFSET', 'AS', 'ON', 'EXEC', 'EXECUTE', 'DECLARE', 
               'TABLE', 'DATABASE', 'COLUMN', 'BETWEEN', 'LIKE', 'IN', 'EXISTS', 
               'CASE', 'WHEN', 'THEN', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX']

XSSKEYWORDS = ['script', 'iframe', 'object', 'embed', 'applet', 'meta', 'link', 'style', 
               'img', 'svg', 'video', 'audio', 'canvas', 'input', 'button', 'form', 
               'body', 'html', 'onerror', 'onload', 'onclick', 'onmouseover', 'onfocus', 
               'onblur', 'alert', 'prompt', 'confirm', 'eval', 'expression', 'javascript', 
               'vbscript', 'document', 'window', 'location', 'cookie', 'localstorage', 
               'sessionstorage']

class ContentMatchingPreprocessor:
    """Preprocess text input"""
    def __init__(self):
        self.sqlkeywords = set([kw.lower() for kw in SQLKEYWORDS])
        self.xsskeywords = set([kw.lower() for kw in XSSKEYWORDS])

    def digital_generalization(self, text):
        return re.sub(r'\d+', 'NUM', text)

    def url_replacement(self, text):
        url_pattern = r'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        return re.sub(url_pattern, 'URL', text)

    def preserve_keywords(self, text):
        for keyword in self.sqlkeywords:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            text = pattern.sub(f'SQL_{keyword.upper()}', text)
        for keyword in self.xsskeywords:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            text = pattern.sub(f'XSS_{keyword.upper()}', text)
        return text

    def normalize_text(self, text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def special_character_mapping(self, text):
        char_mappings = {
            "'": "SQUOTE", '"': "DQUOTE", ';': "SEMICOLON",
            '--': "COMMENT", '/*': "BLOCKCOMMENT_START", '*/': "BLOCKCOMMENT_END",
            '=': "EQUALS", '<': "LT", '>': "GT",
            '(': "LPAREN", ')': "RPAREN", '{': "LBRACE", '}': "RBRACE",
            '[': "LBRACKET", ']': "RBRACKET"
        }
        for char, token in char_mappings.items():
            text = text.replace(char, f' {token} ')
        return text

    def preprocess(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = self.digital_generalization(text)
        text = self.url_replacement(text)
        text = self.preserve_keywords(text)
        text = self.normalize_text(text)
        text = self.special_character_mapping(text)
        return text

# Initialize session state
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {}
if 'feature_extractors' not in st.session_state:
    st.session_state.feature_extractors = {}

@st.cache_resource
def load_single_model(model_name, model_type):
    """Load a single model on demand from HuggingFace"""
    repo_id = "Dr-KeK/sqli-xss-models"
    
    try:
        if model_type == "classical":
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"models/classical_ml/{model_name}.pkl",
                repo_type="model"
            )
            return joblib.load(file_path)
        
        elif model_type == "deep_learning":
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"models/deep_learning/{model_name}.h5",
                repo_type="model"
            )
            return keras.models.load_model(file_path)
        
        elif model_type == "transformer":
            if model_name == 'DistilBERT':
                tokenizer = DistilBertTokenizer.from_pretrained(repo_id, subfolder=f"models/transformers/{model_name}")
                model = DistilBertForSequenceClassification.from_pretrained(repo_id, subfolder=f"models/transformers/{model_name}")
            else:
                tokenizer = BertTokenizer.from_pretrained(repo_id, subfolder=f"models/transformers/{model_name}")
                model = BertForSequenceClassification.from_pretrained(repo_id, subfolder=f"models/transformers/{model_name}")
            return {'model': model, 'tokenizer': tokenizer}
        
        elif model_type == "hybrid":
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"models/hybrid/{model_name}.pkl",
                repo_type="model"
            )
            return joblib.load(file_path)
            
    except Exception as e:
        st.error(f"Failed to load {model_name}: {str(e)}")
        return None

@st.cache_resource
def load_feature_extractors_full():
    """Load ALL feature extractors including Word2Vec and FastText"""
    repo_id = "Dr-KeK/sqli-xss-models"
    extractors = {}
    
    try:
        # Download entire features folder to get all Gensim files
        with st.spinner("📥 Downloading feature models from HuggingFace..."):
            cache_dir = snapshot_download(
                repo_id=repo_id,
                allow_patterns="features/*",
                repo_type="model"
            )
        
        features_dir = os.path.join(cache_dir, "features")
        
        # Load Word2Vec
        w2v_path = os.path.join(features_dir, "word2vec.model")
        if os.path.exists(w2v_path):
            extractors['word2vec'] = Word2Vec.load(w2v_path)
            st.success("✅ Word2Vec loaded")
        
        # Load FastText
        ft_path = os.path.join(features_dir, "fasttext.model")
        if os.path.exists(ft_path):
            extractors['fasttext'] = FastText.load(ft_path)
            st.success("✅ FastText loaded")
        
        # Load TF-IDF
        tfidf_path = os.path.join(features_dir, "tfidf_vectorizer.pkl")
        if os.path.exists(tfidf_path):
            with open(tfidf_path, 'rb') as f:
                extractors['tfidf'] = pickle.load(f)
            st.success("✅ TF-IDF loaded")
            
    except Exception as e:
        st.error(f"❌ Error loading feature extractors: {str(e)}")
    
    return extractors

def extract_tfidf_features(text, extractors):
    """Extract TF-IDF features for classical ML models"""
    if 'tfidf' in extractors:
        features = extractors['tfidf'].transform([text]).toarray()[0]
        return features
    else:
        st.error("❌ TF-IDF vectorizer not available!")
        return None

def extract_uniembed_features(text, extractors, w2v_dim=50, ft_dim=50):
    """Extract UniEmbed features (Word2Vec + FastText) for deep learning models"""
    tokens = text.split()
    
    # Word2Vec embeddings
    w2v_vectors = []
    if 'word2vec' in extractors:
        for token in tokens:
            if token in extractors['word2vec'].wv:
                w2v_vectors.append(extractors['word2vec'].wv[token])
    w2v_emb = np.mean(w2v_vectors, axis=0) if w2v_vectors else np.zeros(w2v_dim)
    
    # FastText embeddings
    ft_vectors = []
    if 'fasttext' in extractors:
        for token in tokens:
            if token in extractors['fasttext'].wv:
                ft_vectors.append(extractors['fasttext'].wv[token])
    ft_emb = np.mean(ft_vectors, axis=0) if ft_vectors else np.zeros(ft_dim)
    
    # Concatenate: Word2Vec (50) + FastText (50) = 100D
    # NOTE: Your training used Word2Vec(50) + FastText(50) + USE(512) = 612D
    # If USE was used, we need to add it here too
    uniembed = np.concatenate([w2v_emb, ft_emb])
    
    return uniembed

def prepare_deep_learning_input(features, model_name):
    """Reshape features for deep learning models"""
    if model_name == 'MLP':
        return features.reshape(1, -1)
    elif model_name == 'CNN':
        return features.reshape(1, features.shape[0], 1)
    elif model_name in ['LSTM', 'BiLSTM', 'CNN_LSTM']:
        n_features = features.shape[0]
        timesteps = min(n_features, 50)
        features_per_timestep = n_features // timesteps
        truncated_features = timesteps * features_per_timestep
        features_truncated = features[:truncated_features]
        return features_truncated.reshape(1, timesteps, features_per_timestep)
    return features.reshape(1, -1)

def predict_with_transformer(text, model_dict, max_length=64):
    """Predict with transformer model"""
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']

    encoding = tokenizer(text, add_special_tokens=True, max_length=max_length,
                        padding='max_length', truncation=True,
                        return_attention_mask=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(input_ids=encoding['input_ids'],
                       attention_mask=encoding['attention_mask'])
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()
        confidence = probs[0][pred].item()

    return pred, confidence

# Streamlit App
st.set_page_config(page_title="SQLi & XSS Detection System", layout="wide", page_icon="🛡️")

st.title("🛡️ SQL Injection & XSS Attack Detection System")
st.markdown("### Memory-Optimized Version - Select Models to Use")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Select Models")
    st.warning("⚠️ Select only models you want to use (saves memory)")
    
    st.markdown("#### Classical ML (Small, TF-IDF)")
    classical_models = {
        'Logistic_Regression': st.checkbox("Logistic Regression", value=True),
        'Random_Forest': st.checkbox("Random Forest", value=True),
        'XGBoost': st.checkbox("XGBoost", value=True),
        'SVM': st.checkbox("SVM"),
        'Decision_Tree': st.checkbox("Decision Tree"),
        'KNN': st.checkbox("KNN"),
        'Gradient_Boosting': st.checkbox("Gradient Boosting"),
        'Extra_Trees': st.checkbox("Extra Trees"),
        'Gaussian_Naive_Bayes': st.checkbox("Naive Bayes"),
    }
    
    st.markdown("#### Deep Learning (Large, UniEmbed)")
    st.caption("⚠️ These use Word2Vec + FastText features")
    dl_models = {
        'MLP': st.checkbox("MLP", value=False),
        'CNN': st.checkbox("CNN", value=False),
        'LSTM': st.checkbox("LSTM", value=False),
        'BiLSTM': st.checkbox("BiLSTM", value=False),
        'CNN_LSTM': st.checkbox("CNN-LSTM", value=False),
    }
    
    st.markdown("#### Transformers (Large, Raw Text)")
    st.caption("✨ These use raw text directly")
    transformer_models = {
        'DistilBERT': st.checkbox("DistilBERT", value=False),
        'BERT': st.checkbox("BERT", value=False),
    }
    
    st.markdown("#### Hybrid (Medium, UniEmbed)")
    hybrid_models = {
        'StackingEnsemble': st.checkbox("Stacking Ensemble", value=False),
        'SoftVoting': st.checkbox("Soft Voting", value=False),
        'HardVoting': st.checkbox("Hard Voting", value=False),
    }
    
    st.markdown("---")
    selected_count = (sum(classical_models.values()) + sum(dl_models.values()) + 
                     sum(transformer_models.values()) + sum(hybrid_models.values()))
    
    if selected_count > 10:
        st.error(f"⚠️ {selected_count} models selected! May crash.")
    elif selected_count > 5:
        st.warning(f"⚠️ {selected_count} models selected.")
    else:
        st.metric("Selected Models", selected_count)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Input Query")
    user_input = st.text_area("Enter your query/payload to analyze:",
                               height=150,
                               placeholder="Example: ' OR 1=1 -- ")
    analyze_button = st.button("🔍 Analyze Query", type="primary", use_container_width=True)

with col2:
    st.header("📋 Example Attacks")
    st.code("' OR '1'='1", language="sql")
    st.caption("SQL Injection")
    st.code("<script>alert('XSS')</script>", language="html")
    st.caption("XSS Attack")
    st.code("admin'--", language="sql")
    st.caption("SQL Comment Injection")

if analyze_button and user_input:
    if selected_count == 0:
        st.warning("⚠️ Please select at least one model from the sidebar!")
        st.stop()
    
    with st.spinner("🔄 Loading features and analyzing..."):
        preprocessor = ContentMatchingPreprocessor()
        processed_text = preprocessor.preprocess(user_input)

        try:
            # Load feature extractors (cached)
            if not st.session_state.feature_extractors:
                st.session_state.feature_extractors = load_feature_extractors_full()
            
            extractors = st.session_state.feature_extractors
            
            if not extractors:
                st.error("❌ Failed to load feature extractors!")
                st.stop()
            
            # Extract features based on what's needed
            tfidf_features = None
            uniembed_features = None
            
            # Check if we need TF-IDF (for classical ML)
            if any(classical_models.values()):
                tfidf_features = extract_tfidf_features(processed_text, extractors)
                if tfidf_features is not None:
                    st.info(f"📊 TF-IDF features: {tfidf_features.shape[0]} dimensions")
            
            # Check if we need UniEmbed (for deep learning & hybrid)
            if any(dl_models.values()) or any(hybrid_models.values()):
                uniembed_features = extract_uniembed_features(processed_text, extractors)
                if uniembed_features is not None:
                    st.info(f"📊 UniEmbed features: {uniembed_features.shape[0]} dimensions (Word2Vec + FastText)")

            with st.expander("🔍 Preprocessed Text"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.text("Original:")
                    st.code(user_input)
                with col_b:
                    st.text("Processed:")
                    st.code(processed_text)

            st.markdown("---")
            st.header("🎯 Prediction Results")

            results = []

            # Classical ML Models - USE TF-IDF
            selected_classical = [k for k, v in classical_models.items() if v]
            if selected_classical and tfidf_features is not None:
                st.subheader("🔹 Classical Machine Learning Models")
                classical_cols = st.columns(min(3, len(selected_classical)))
                
                for idx, model_name in enumerate(selected_classical):
                    with st.spinner(f"Loading {model_name}..."):
                        if model_name not in st.session_state.loaded_models:
                            model = load_single_model(model_name, "classical")
                            if model:
                                st.session_state.loaded_models[model_name] = model
                        
                        if model_name in st.session_state.loaded_models:
                            try:
                                model = st.session_state.loaded_models[model_name]
                                pred = model.predict(tfidf_features.reshape(1, -1))[0]
                                if hasattr(model, 'predict_proba'):
                                    proba = model.predict_proba(tfidf_features.reshape(1, -1))[0]
                                    confidence = proba[pred] * 100
                                else:
                                    confidence = 100 if pred == 1 else 0

                                label = "🚨 ATTACK" if pred == 1 else "✅ SAFE"

                                with classical_cols[idx % 3]:
                                    st.metric(label=model_name.replace('_', ' '), value=label,
                                            delta=f"{confidence:.1f}% confidence")

                                results.append({
                                    'Model': model_name.replace('_', ' '),
                                    'Category': 'Classical ML',
                                    'Prediction': label,
                                    'Confidence': f'{confidence:.1f}%'
                                })
                            except Exception as e:
                                st.warning(f"⚠️ {model_name}: {str(e)}")

            # Deep Learning Models - USE UNIEMBED
            selected_dl = [k for k, v in dl_models.items() if v]
            if selected_dl and uniembed_features is not None:
                st.subheader("🔹 Deep Learning Models")
                dl_cols = st.columns(min(3, len(selected_dl)))
                
                for idx, model_name in enumerate(selected_dl):
                    with st.spinner(f"Loading {model_name}..."):
                        if model_name not in st.session_state.loaded_models:
                            model = load_single_model(model_name, "deep_learning")
                            if model:
                                st.session_state.loaded_models[model_name] = model
                        
                        if model_name in st.session_state.loaded_models:
                            try:
                                model = st.session_state.loaded_models[model_name]
                                # USE UNIEMBED FEATURES
                                input_data = prepare_deep_learning_input(uniembed_features, model_name)
                                proba = model.predict(input_data, verbose=0)[0][0]
                                pred = 1 if proba > 0.5 else 0
                                confidence = proba * 100 if pred == 1 else (1 - proba) * 100

                                label = "🚨 ATTACK" if pred == 1 else "✅ SAFE"

                                with dl_cols[idx % 3]:
                                    st.metric(label=model_name, value=label,
                                            delta=f"{confidence:.1f}% confidence")

                                results.append({
                                    'Model': model_name,
                                    'Category': 'Deep Learning',
                                    'Prediction': label,
                                    'Confidence': f'{confidence:.1f}%'
                                })
                            except Exception as e:
                                st.warning(f"⚠️ {model_name}: {str(e)}")

            # Transformer Models - USE RAW TEXT
            selected_transformers = [k for k, v in transformer_models.items() if v]
            if selected_transformers:
                st.subheader("🔹 Transformer Models")
                trans_cols = st.columns(min(2, len(selected_transformers)))
                
                for idx, model_name in enumerate(selected_transformers):
                    with st.spinner(f"Loading {model_name}..."):
                        if model_name not in st.session_state.loaded_models:
                            model = load_single_model(model_name, "transformer")
                            if model:
                                st.session_state.loaded_models[model_name] = model
                        
                        if model_name in st.session_state.loaded_models:
                            try:
                                # USE RAW TEXT
                                pred, conf = predict_with_transformer(user_input, st.session_state.loaded_models[model_name])
                                confidence = conf * 100

                                label = "🚨 ATTACK" if pred == 1 else "✅ SAFE"

                                with trans_cols[idx % 2]:
                                    st.metric(label=model_name, value=label,
                                            delta=f"{confidence:.1f}% confidence")

                                results.append({
                                    'Model': model_name,
                                    'Category': 'Transformer',
                                    'Prediction': label,
                                    'Confidence': f'{confidence:.1f}%'
                                })
                            except Exception as e:
                                st.warning(f"⚠️ {model_name}: {str(e)}")

            # Hybrid Models - USE UNIEMBED
            selected_hybrid = [k for k, v in hybrid_models.items() if v]
            if selected_hybrid and uniembed_features is not None:
                st.subheader("🔹 Hybrid Ensemble Models")
                hybrid_cols = st.columns(min(3, len(selected_hybrid)))
                
                for idx, model_name in enumerate(selected_hybrid):
                    with st.spinner(f"Loading {model_name}..."):
                        if model_name not in st.session_state.loaded_models:
                            model = load_single_model(model_name, "hybrid")
                            if model:
                                st.session_state.loaded_models[model_name] = model
                        
                        if model_name in st.session_state.loaded_models:
                            try:
                                model = st.session_state.loaded_models[model_name]
                                # USE UNIEMBED FEATURES
                                pred = model.predict(uniembed_features.reshape(1, -1))[0]
                                if hasattr(model, 'predict_proba'):
                                    proba = model.predict_proba(uniembed_features.reshape(1, -1))[0]
                                    confidence = proba[pred] * 100
                                else:
                                    confidence = 100 if pred == 1 else 0

                                label = "🚨 ATTACK" if pred == 1 else "✅ SAFE"

                                with hybrid_cols[idx % 3]:
                                    st.metric(label=model_name, value=label,
                                            delta=f"{confidence:.1f}% confidence")

                                results.append({
                                    'Model': model_name,
                                    'Category': 'Hybrid',
                                    'Prediction': label,
                                    'Confidence': f'{confidence:.1f}%'
                                })
                            except Exception as e:
                                st.warning(f"⚠️ {model_name}: {str(e)}")

            # Summary
            if results:
                st.markdown("---")
                st.header("📊 Summary")
                df_results = pd.DataFrame(results)

                attack_count = sum(1 for r in results if "ATTACK" in r['Prediction'])
                safe_count = len(results) - attack_count

                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Models Used", len(results))
                with col_s2:
                    st.metric("Attack Detected", attack_count, 
                             delta=f"{attack_count/len(results)*100:.1f}%")
                with col_s3:
                    st.metric("Safe Detected", safe_count,
                             delta=f"{safe_count/len(results)*100:.1f}%")

                if attack_count > safe_count:
                    st.error(f"⚠️ **CONSENSUS: POTENTIAL ATTACK DETECTED** ({attack_count}/{len(results)} models)")
                elif safe_count > attack_count:
                    st.success(f"✅ **CONSENSUS: QUERY APPEARS SAFE** ({safe_count}/{len(results)} models)")
                else:
                    st.warning(f"⚖️ **CONSENSUS: UNCERTAIN** (Split decision)")

                with st.expander("📋 Detailed Results Table"):
                    st.dataframe(df_results, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)

elif analyze_button:
    st.warning("⚠️ Please enter a query to analyze")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 Advanced Web Attack Detection System | Memory-Optimized Version</p>
    <p>Models load on-demand from HuggingFace • Full Feature Extraction</p>
</div>
""", unsafe_allow_html=True)
