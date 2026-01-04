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
from gensim.models import Word2Vec, FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# SQL and XSS Keywords (keep as is)
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

@st.cache_resource
def load_models(models_dir='web_attack_detection/models'):
    """Load all trained models from repository"""
    models = {}
    loaded_count = 0

    # Classical ML models
    classical_path = Path(models_dir) / 'classical_ml'
    if classical_path.exists():
        for model_file in classical_path.glob('*.pkl'):
            try:
                models[model_file.stem] = joblib.load(model_file)
                loaded_count += 1
            except Exception as e:
                st.warning(f"Could not load {model_file.stem}: {str(e)}")

    # Deep Learning models
    dl_path = Path(models_dir) / 'deep_learning'
    if dl_path.exists():
        for model_file in dl_path.glob('*.h5'):
            try:
                models[model_file.stem] = keras.models.load_model(model_file)
                loaded_count += 1
            except Exception as e:
                st.warning(f"Could not load {model_file.stem}: {str(e)}")

    # Transformer models
    transformer_models = ['DistilBERT', 'BERT']
    for model_name in transformer_models:
        model_dir = Path(models_dir) / 'transformers' / model_name
        if model_dir.exists():
            try:
                if model_name == 'DistilBERT':
                    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
                    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
                else:
                    tokenizer = BertTokenizer.from_pretrained(model_dir)
                    model = BertForSequenceClassification.from_pretrained(model_dir)
                models[model_name] = {'model': model, 'tokenizer': tokenizer}
                loaded_count += 1
            except Exception as e:
                st.warning(f"Could not load {model_name}: {str(e)}")

    # Hybrid models
    hybrid_path = Path(models_dir) / 'hybrid'
    if hybrid_path.exists():
        for model_file in hybrid_path.glob('*.pkl'):
            try:
                models[model_file.stem] = joblib.load(model_file)
                loaded_count += 1
            except Exception as e:
                st.warning(f"Could not load {model_file.stem}: {str(e)}")

    return models, loaded_count

@st.cache_resource
def load_feature_extractors(features_dir='web_attack_detection/features'):
    """Load feature extraction models"""
    extractors = {}

    # Word2Vec
    w2v_path = Path(features_dir) / 'word2vec.model'
    if w2v_path.exists():
        extractors['word2vec'] = Word2Vec.load(str(w2v_path))

    # FastText
    ft_path = Path(features_dir) / 'fasttext.model'
    if ft_path.exists():
        extractors['fasttext'] = FastText.load(str(ft_path))

    # TF-IDF Vectorizer - CRITICAL FOR CLASSICAL ML!
    tfidf_path = Path(features_dir) / 'tfidf_vectorizer.pkl'
    if tfidf_path.exists():
        with open(tfidf_path, 'rb') as f:
            extractors['tfidf'] = pickle.load(f)
    else:
        st.error("⚠️ TF-IDF vectorizer not found! Classical ML models will fail.")

    return extractors

def extract_tfidf_features(text, extractors):
    """Extract TF-IDF features for classical ML models"""
    if 'tfidf' not in extractors:
        st.error("TF-IDF vectorizer not loaded!")
        return None
    
    # Transform single text sample
    tfidf_features = extractors['tfidf'].transform([text]).toarray()
    return tfidf_features[0]

def extract_uniembed_features(text, extractors, w2v_dim=50, ft_dim=50):
    """Extract UniEmbed features for deep learning models"""
    tokens = text.split()

    # Word2Vec
    w2v_vectors = []
    for token in tokens:
        if 'word2vec' in extractors and token in extractors['word2vec'].wv:
            w2v_vectors.append(extractors['word2vec'].wv[token])
    w2v_emb = np.mean(w2v_vectors, axis=0) if w2v_vectors else np.zeros(w2v_dim)

    # FastText
    ft_vectors = []
    for token in tokens:
        if 'fasttext' in extractors:
            ft_vectors.append(extractors['fasttext'].wv[token])
    ft_emb = np.mean(ft_vectors, axis=0) if ft_vectors else np.zeros(ft_dim)

    return np.concatenate([w2v_emb, ft_emb])

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

# Define which models use which features
CLASSICAL_ML_MODELS = ['Logistic_Regression', 'SVM', 'Gaussian_Naive_Bayes', 
                       'Decision_Tree', 'KNN', 'Random_Forest', 'XGBoost', 
                       'Gradient_Boosting', 'Extra_Trees']

DEEP_LEARNING_MODELS = ['MLP', 'CNN', 'LSTM', 'BiLSTM', 'CNN_LSTM']

TRANSFORMER_MODELS = ['DistilBERT', 'BERT']

HYBRID_MODELS = ['StackingEnsemble', 'SoftVoting', 'HardVoting', 
                'LightGBM_BiLSTM', 'BERT_XGBoost']

# Streamlit App
st.set_page_config(page_title="SQLi & XSS Detection System", layout="wide", page_icon="🛡️")

st.title("🛡️ SQL Injection & XSS Attack Detection System")
st.markdown("### Powered by 17 Machine Learning Models")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    models_dir = st.text_input("Models Directory", value="web_attack_detection/models")
    features_dir = st.text_input("Features Directory", value="web_attack_detection/features")

    st.markdown("---")
    st.header("📊 Model Categories")
    show_classical = st.checkbox("Classical ML (9 models)", value=True)
    show_dl = st.checkbox("Deep Learning (5 models)", value=True)
    show_transformer = st.checkbox("Transformers (2 models)", value=True)
    show_hybrid = st.checkbox("Hybrid Models (3-5 models)", value=True)

    st.markdown("---")
    st.info("💡 Enter a query below to test for SQLi or XSS attacks")

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
    with st.spinner("🔄 Loading models and analyzing..."):
        preprocessor = ContentMatchingPreprocessor()
        processed_text = preprocessor.preprocess(user_input)

        try:
            models, loaded_count = load_models(models_dir)
            extractors = load_feature_extractors(features_dir)

            if not models:
                st.error("❌ No models found! Please check the models directory path.")
                st.stop()

            st.success(f"✅ Loaded {loaded_count} models successfully!")

            # Extract BOTH feature types
            tfidf_features = extract_tfidf_features(processed_text, extractors)
            uniembed_features = extract_uniembed_features(processed_text, extractors)

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

            # Classical ML Models - USE TF-IDF FEATURES!
            if show_classical and tfidf_features is not None:
                st.subheader("🔹 Classical Machine Learning Models")
                classical_cols = st.columns(3)
                col_idx = 0

                for model_name in CLASSICAL_ML_MODELS:
                    if model_name in models:
                        try:
                            model = models[model_name]
                            # USE TF-IDF FEATURES HERE!
                            pred = model.predict(tfidf_features.reshape(1, -1))[0]
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(tfidf_features.reshape(1, -1))[0]
                                confidence = proba[pred] * 100
                            else:
                                confidence = 100 if pred == 1 else 0

                            label = "🚨 ATTACK" if pred == 1 else "✅ SAFE"

                            with classical_cols[col_idx % 3]:
                                st.metric(label=model_name.replace('_', ' '), value=label,
                                        delta=f"{confidence:.1f}% confidence")

                            results.append({
                                'Model': model_name.replace('_', ' '),
                                'Category': 'Classical ML',
                                'Prediction': label,
                                'Confidence': f'{confidence:.1f}%'
                            })
                            col_idx += 1
                        except Exception as e:
                            st.warning(f"⚠️ {model_name}: {str(e)}")

            # Deep Learning Models - USE UNIEMBED FEATURES!
            if show_dl and uniembed_features is not None:
                st.subheader("🔹 Deep Learning Models")
                dl_cols = st.columns(3)
                col_idx = 0

                for model_name in DEEP_LEARNING_MODELS:
                    if model_name in models:
                        try:
                            model = models[model_name]
                            # USE UNIEMBED FEATURES HERE!
                            input_data = prepare_deep_learning_input(uniembed_features, model_name)
                            proba = model.predict(input_data, verbose=0)[0][0]
                            pred = 1 if proba > 0.5 else 0
                            confidence = proba * 100 if pred == 1 else (1 - proba) * 100

                            label = "🚨 ATTACK" if pred == 1 else "✅ SAFE"

                            with dl_cols[col_idx % 3]:
                                st.metric(label=model_name, value=label,
                                        delta=f"{confidence:.1f}% confidence")

                            results.append({
                                'Model': model_name,
                                'Category': 'Deep Learning',
                                'Prediction': label,
                                'Confidence': f'{confidence:.1f}%'
                            })
                            col_idx += 1
                        except Exception as e:
                            st.warning(f"⚠️ {model_name}: {str(e)}")

            # Transformer Models - USE RAW TEXT!
            if show_transformer:
                st.subheader("🔹 Transformer Models")
                trans_cols = st.columns(2)
                col_idx = 0

                for model_name in TRANSFORMER_MODELS:
                    if model_name in models:
                        try:
                            # USE ORIGINAL RAW TEXT HERE!
                            pred, confidence = predict_with_transformer(user_input, models[model_name])
                            confidence = confidence * 100

                            label = "🚨 ATTACK" if pred == 1 else "✅ SAFE"

                            with trans_cols[col_idx % 2]:
                                st.metric(label=model_name, value=label,
                                        delta=f"{confidence:.1f}% confidence")

                            results.append({
                                'Model': model_name,
                                'Category': 'Transformer',
                                'Prediction': label,
                                'Confidence': f'{confidence:.1f}%'
                            })
                            col_idx += 1
                        except Exception as e:
                            st.warning(f"⚠️ {model_name}: {str(e)}")

            # Hybrid Models - USE UNIEMBED FEATURES (or check metadata)
            if show_hybrid and uniembed_features is not None:
                st.subheader("🔹 Hybrid Ensemble Models")
                hybrid_cols = st.columns(3)
                col_idx = 0

                for model_name in HYBRID_MODELS:
                    if model_name in models:
                        try:
                            model = models[model_name]
                            pred = model.predict(uniembed_features.reshape(1, -1))[0]
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(uniembed_features.reshape(1, -1))[0]
                                confidence = proba[pred] * 100
                            else:
                                confidence = 100 if pred == 1 else 0

                            label = "🚨 ATTACK" if pred == 1 else "✅ SAFE"

                            with hybrid_cols[col_idx % 3]:
                                st.metric(label=model_name, value=label,
                                        delta=f"{confidence:.1f}% confidence")

                            results.append({
                                'Model': model_name,
                                'Category': 'Hybrid',
                                'Prediction': label,
                                'Confidence': f'{confidence:.1f}%'
                            })
                            col_idx += 1
                        except Exception as e:
                            st.warning(f"⚠️ {model_name}: {str(e)}")

            # Summary
            st.markdown("---")
            st.header("📊 Summary")

            if results:
                df_results = pd.DataFrame(results)

                attack_count = sum(1 for r in results if "ATTACK" in r['Prediction'])
                safe_count = len(results) - attack_count

                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Total Models", len(results))
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
    <p>🔬 Advanced Web Attack Detection System | Powered by 17 ML Models</p>
    <p>Classical ML • Deep Learning • Transformers • Hybrid Ensembles</p>
</div>
""", unsafe_allow_html=True)
