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
from huggingface_hub import hf_hub_download
import warnings
import gc
warnings.filterwarnings('ignore')

# HuggingFace repo
HF_REPO = "Dr-KeK/sqli-xss-models"

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

# Model categories and their feature requirements
CLASSICAL_ML_MODELS = {
    'Logistic_Regression': 'tfidf',
    'SVM': 'tfidf',
    'Gaussian_Naive_Bayes': 'tfidf',
    'Decision_Tree': 'tfidf',
    'KNN': 'tfidf',
    'Random_Forest': 'tfidf',
    'XGBoost': 'tfidf',
    'Gradient_Boosting': 'tfidf',
    'Extra_Trees': 'tfidf'
}

DEEP_LEARNING_MODELS = {
    'MLP': 'uniembed',
    'CNN': 'uniembed',
    'LSTM': 'uniembed',
    'BiLSTM': 'uniembed',
    'CNN_LSTM': 'uniembed'
}

TRANSFORMER_MODELS = {
    'DistilBERT': 'raw_text',
    'BERT': 'raw_text'
}

HYBRID_MODELS = {
    'StackingEnsemble': 'uniembed',
    'SoftVoting': 'uniembed',
    'HardVoting': 'uniembed',
    'LightGBM_BiLSTM': 'uniembed',
    'BERT_XGBoost': 'uniembed'
}

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
if 'extractors_loaded' not in st.session_state:
    st.session_state.extractors_loaded = False
if 'extractors' not in st.session_state:
    st.session_state.extractors = {}

@st.cache_resource
def load_feature_extractors_from_hf():
    """Load feature extractors from HuggingFace (cached)"""
    extractors = {}
    
    try:
        # Word2Vec - IN features/ FOLDER
        with st.spinner("Downloading Word2Vec model..."):
            w2v_file = hf_hub_download(repo_id=HF_REPO, filename='features/word2vec.model')
            extractors['word2vec'] = Word2Vec.load(w2v_file)
        st.success("✅ Word2Vec loaded")

        # FastText - IN features/ FOLDER
        with st.spinner("Downloading FastText model..."):
            ft_file = hf_hub_download(repo_id=HF_REPO, filename='features/fasttext.model')
            extractors['fasttext'] = FastText.load(ft_file)
        st.success("✅ FastText loaded")

        # TF-IDF Vectorizer - IN features/ FOLDER - CRITICAL!
        with st.spinner("Downloading TF-IDF vectorizer..."):
            tfidf_file = hf_hub_download(repo_id=HF_REPO, filename='features/tfidf_vectorizer.pkl')
            with open(tfidf_file, 'rb') as f:
                extractors['tfidf'] = pickle.load(f)
        st.success("✅ TF-IDF vectorizer loaded")
            
    except Exception as e:
        st.error(f"❌ Error loading feature extractors: {str(e)}")
        st.error("Make sure the following files exist in your HuggingFace repo:")
        st.code("- features/word2vec.model\n- features/fasttext.model\n- features/tfidf_vectorizer.pkl")
    
    return extractors

def load_model_from_hf(model_name, model_category):
    """Load a single model from HuggingFace"""
    cache_key = f"{model_category}_{model_name}"
    
    if cache_key in st.session_state.loaded_models:
        return st.session_state.loaded_models[cache_key]
    
    try:
        if model_category == 'classical':
            # IN models/classical_ml/ FOLDER
            model_file = hf_hub_download(repo_id=HF_REPO, filename=f'models/classical_ml/{model_name}.pkl')
            model = joblib.load(model_file)
                
        elif model_category == 'deep_learning':
            # IN models/deep_learning/ FOLDER
            model_file = hf_hub_download(repo_id=HF_REPO, filename=f'models/deep_learning/{model_name}.h5')
            model = keras.models.load_model(model_file)
                
        elif model_category == 'transformer':
            # IN models/transformers/ FOLDER
            if model_name == 'DistilBERT':
                tokenizer = DistilBertTokenizer.from_pretrained(f"{HF_REPO}", subfolder=f"models/transformers/{model_name}")
                model_obj = DistilBertForSequenceClassification.from_pretrained(f"{HF_REPO}", subfolder=f"models/transformers/{model_name}")
            else:
                tokenizer = BertTokenizer.from_pretrained(f"{HF_REPO}", subfolder=f"models/transformers/{model_name}")
                model_obj = BertForSequenceClassification.from_pretrained(f"{HF_REPO}", subfolder=f"models/transformers/{model_name}")
            model = {'model': model_obj, 'tokenizer': tokenizer}
                
        elif model_category == 'hybrid':
            # IN models/hybrid/ FOLDER
            model_file = hf_hub_download(repo_id=HF_REPO, filename=f'models/hybrid/{model_name}.pkl')
            model = joblib.load(model_file)
        else:
            return None
            
        st.session_state.loaded_models[cache_key] = model
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading {model_name}: {str(e)}")
        st.info(f"Expected path: models/{model_category}/{model_name}")
        return None

def unload_model(model_name, model_category):
    """Unload a model from memory"""
    cache_key = f"{model_category}_{model_name}"
    if cache_key in st.session_state.loaded_models:
        del st.session_state.loaded_models[cache_key]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def extract_tfidf_features(text, extractors):
    """Extract TF-IDF features for classical ML models"""
    if 'tfidf' not in extractors:
        return None
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

# Streamlit App
st.set_page_config(page_title="SQLi & XSS Detection System", layout="wide", page_icon="🛡️")

st.title("🛡️ SQL Injection & XSS Attack Detection System")
st.markdown("### Powered by 17 Machine Learning Models")
st.markdown(f"📦 Models loaded from: [{HF_REPO}](https://huggingface.co/{HF_REPO})")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Select Individual Models")
    st.warning("⚠️ **Memory Warning**: Loading many models may crash the page. Select only what you need!")
    
    # INDIVIDUAL MODEL SELECTION
    selected_models = []
    
    with st.expander("🔹 Classical ML (9 models)", expanded=True):
        for model_name in CLASSICAL_ML_MODELS.keys():
            display_name = model_name.replace('_', ' ')
            if st.checkbox(display_name, key=f"classical_{model_name}"):
                selected_models.append((model_name, 'classical'))
            else:
                unload_model(model_name, 'classical')
    
    with st.expander("🔹 Deep Learning (5 models)"):
        for model_name in DEEP_LEARNING_MODELS.keys():
            if st.checkbox(model_name, key=f"dl_{model_name}"):
                selected_models.append((model_name, 'deep_learning'))
            else:
                unload_model(model_name, 'deep_learning')
    
    with st.expander("🔹 Transformers (2 models)"):
        for model_name in TRANSFORMER_MODELS.keys():
            if st.checkbox(model_name, key=f"trans_{model_name}"):
                selected_models.append((model_name, 'transformer'))
            else:
                unload_model(model_name, 'transformer')
    
    with st.expander("🔹 Hybrid Models (3-5 models)"):
        for model_name in HYBRID_MODELS.keys():
            if st.checkbox(model_name, key=f"hybrid_{model_name}"):
                selected_models.append((model_name, 'hybrid'))
            else:
                unload_model(model_name, 'hybrid')
    
    # Model count warnings
    model_count = len(selected_models)
    if model_count == 0:
        st.info("ℹ️ No models selected")
    elif model_count > 10:
        st.error(f"⚠️ {model_count} models selected! This WILL cause memory issues.")
    elif model_count > 5:
        st.warning(f"⚠️ {model_count} models selected. Page may slow down.")
    else:
        st.success(f"✅ {model_count} models selected")

    st.markdown("---")
    loaded_count = len(st.session_state.loaded_models)
    st.caption(f"🔢 Currently in memory: {loaded_count} models")
    
    if st.button("🗑️ Clear All Models from Memory"):
        st.session_state.loaded_models = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.success("Memory cleared!")
        st.rerun()

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
    if len(selected_models) == 0:
        st.warning("⚠️ Please select at least one model from the sidebar")
    else:
        with st.spinner("🔄 Loading feature extractors from HuggingFace..."):
            preprocessor = ContentMatchingPreprocessor()
            processed_text = preprocessor.preprocess(user_input)

            try:
                # Load feature extractors from HuggingFace (cached)
                extractors = load_feature_extractors_from_hf()

                if not extractors:
                    st.error("❌ Failed to load feature extractors. Cannot proceed.")
                    st.stop()

                # Check what features we need
                needs_tfidf = any(cat == 'classical' or (cat == 'hybrid' and HYBRID_MODELS.get(name) == 'tfidf') 
                                 for name, cat in selected_models)
                needs_uniembed = any(cat in ['deep_learning', 'hybrid'] for name, cat in selected_models)
                
                # Extract features
                tfidf_features = None
                uniembed_features = None
                
                if needs_tfidf:
                    if 'tfidf' not in extractors:
                        st.error("❌ TF-IDF vectorizer not loaded. Classical ML models cannot run.")
                        st.stop()
                    tfidf_features = extract_tfidf_features(processed_text, extractors)
                    
                if needs_uniembed:
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
                
                # Group models by category for display
                classical_results = []
                dl_results = []
                trans_results = []
                hybrid_results = []

                for model_name, model_category in selected_models:
                    with st.spinner(f"📥 Loading {model_name} from HuggingFace..."):
                        model = load_model_from_hf(model_name, model_category)
                        
                        if not model:
                            st.warning(f"⚠️ Could not load {model_name}")
                            continue

                        try:
                            label = None
                            confidence = 0
                            
                            # Classical ML
                            if model_category == 'classical':
                                if tfidf_features is None:
                                    st.error(f"Cannot run {model_name}: TF-IDF features not available")
                                    continue
                                pred = model.predict(tfidf_features.reshape(1, -1))[0]
                                if hasattr(model, 'predict_proba'):
                                    proba = model.predict_proba(tfidf_features.reshape(1, -1))[0]
                                    confidence = proba[pred] * 100
                                else:
                                    confidence = 100 if pred == 1 else 0
                                label = "🚨 ATTACK" if pred == 1 else "✅ SAFE"
                                classical_results.append((model_name, label, confidence))
                                
                            # Deep Learning
                            elif model_category == 'deep_learning':
                                if uniembed_features is None:
                                    st.error(f"Cannot run {model_name}: UniEmbed features not available")
                                    continue
                                input_data = prepare_deep_learning_input(uniembed_features, model_name)
                                proba = model.predict(input_data, verbose=0)[0][0]
                                pred = 1 if proba > 0.5 else 0
                                confidence = proba * 100 if pred == 1 else (1 - proba) * 100
                                label = "🚨 ATTACK" if pred == 1 else "✅ SAFE"
                                dl_results.append((model_name, label, confidence))
                                
                            # Transformers
                            elif model_category == 'transformer':
                                pred, conf = predict_with_transformer(user_input, model)
                                confidence = conf * 100
                                label = "🚨 ATTACK" if pred == 1 else "✅ SAFE"
                                trans_results.append((model_name, label, confidence))
                                
                            # Hybrid
                            elif model_category == 'hybrid':
                                if uniembed_features is None:
                                    st.error(f"Cannot run {model_name}: UniEmbed features not available")
                                    continue
                                pred = model.predict(uniembed_features.reshape(1, -1))[0]
                                if hasattr(model, 'predict_proba'):
                                    proba = model.predict_proba(uniembed_features.reshape(1, -1))[0]
                                    confidence = proba[pred] * 100
                                else:
                                    confidence = 100 if pred == 1 else 0
                                label = "🚨 ATTACK" if pred == 1 else "✅ SAFE"
                                hybrid_results.append((model_name, label, confidence))

                            if label:
                                results.append({
                                    'Model': model_name.replace('_', ' '),
                                    'Category': model_category.replace('_', ' ').title(),
                                    'Prediction': label,
                                    'Confidence': f'{confidence:.1f}%'
                                })
                                
                        except Exception as e:
                            st.error(f"⚠️ Error running {model_name}: {str(e)}")

                # Display results by category
                if classical_results:
                    st.subheader("🔹 Classical ML Results")
                    cols = st.columns(min(3, len(classical_results)))
                    for idx, (name, label, conf) in enumerate(classical_results):
                        with cols[idx % 3]:
                            st.metric(name.replace('_', ' '), label, f"{conf:.1f}%")

                if dl_results:
                    st.subheader("🔹 Deep Learning Results")
                    cols = st.columns(min(3, len(dl_results)))
                    for idx, (name, label, conf) in enumerate(dl_results):
                        with cols[idx % 3]:
                            st.metric(name, label, f"{conf:.1f}%")

                if trans_results:
                    st.subheader("🔹 Transformer Results")
                    cols = st.columns(min(2, len(trans_results)))
                    for idx, (name, label, conf) in enumerate(trans_results):
                        with cols[idx % 2]:
                            st.metric(name, label, f"{conf:.1f}%")

                if hybrid_results:
                    st.subheader("🔹 Hybrid Results")
                    cols = st.columns(min(3, len(hybrid_results)))
                    for idx, (name, label, conf) in enumerate(hybrid_results):
                        with cols[idx % 3]:
                            st.metric(name, label, f"{conf:.1f}%")

                # Summary
                if results:
                    st.markdown("---")
                    st.header("📊 Summary")

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
    <p>📦 All models loaded from HuggingFace</p>
</div>
""", unsafe_allow_html=True)
