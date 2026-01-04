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
import gc
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

# Initialize session state for lazy loading
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {}
if 'loaded_extractors' not in st.session_state:
    st.session_state.loaded_extractors = {}

def load_model_lazy(model_name, model_category, models_dir='web_attack_detection/models', hf_repo=None):
    """Lazy load a single model only when needed"""
    cache_key = f"{model_category}_{model_name}"
    
    if cache_key in st.session_state.loaded_models:
        return st.session_state.loaded_models[cache_key]
    
    try:
        if model_category == 'classical':
            model_path = Path(models_dir) / 'classical_ml' / f'{model_name}.pkl'
            if model_path.exists():
                model = joblib.load(model_path)
            elif hf_repo:
                # Load from HuggingFace
                from huggingface_hub import hf_hub_download
                model_file = hf_hub_download(repo_id=hf_repo, filename=f'classical_ml/{model_name}.pkl')
                model = joblib.load(model_file)
            else:
                return None
                
        elif model_category == 'deep_learning':
            model_path = Path(models_dir) / 'deep_learning' / f'{model_name}.h5'
            if model_path.exists():
                model = keras.models.load_model(model_path)
            elif hf_repo:
                from huggingface_hub import hf_hub_download
                model_file = hf_hub_download(repo_id=hf_repo, filename=f'deep_learning/{model_name}.h5')
                model = keras.models.load_model(model_file)
            else:
                return None
                
        elif model_category == 'transformer':
            model_dir = Path(models_dir) / 'transformers' / model_name
            if model_dir.exists():
                if model_name == 'DistilBERT':
                    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
                    model_obj = DistilBertForSequenceClassification.from_pretrained(model_dir)
                else:
                    tokenizer = BertTokenizer.from_pretrained(model_dir)
                    model_obj = BertForSequenceClassification.from_pretrained(model_dir)
                model = {'model': model_obj, 'tokenizer': tokenizer}
            elif hf_repo:
                # Load from HuggingFace
                hf_model_path = f"{hf_repo}/transformers/{model_name}"
                if model_name == 'DistilBERT':
                    tokenizer = DistilBertTokenizer.from_pretrained(hf_model_path)
                    model_obj = DistilBertForSequenceClassification.from_pretrained(hf_model_path)
                else:
                    tokenizer = BertTokenizer.from_pretrained(hf_model_path)
                    model_obj = BertForSequenceClassification.from_pretrained(hf_model_path)
                model = {'model': model_obj, 'tokenizer': tokenizer}
            else:
                return None
                
        elif model_category == 'hybrid':
            model_path = Path(models_dir) / 'hybrid' / f'{model_name}.pkl'
            if model_path.exists():
                model = joblib.load(model_path)
            elif hf_repo:
                from huggingface_hub import hf_hub_download
                model_file = hf_hub_download(repo_id=hf_repo, filename=f'hybrid/{model_name}.pkl')
                model = joblib.load(model_file)
            else:
                return None
        else:
            return None
            
        st.session_state.loaded_models[cache_key] = model
        return model
        
    except Exception as e:
        st.error(f"Error loading {model_name}: {str(e)}")
        return None

def unload_model(model_name, model_category):
    """Unload a model from memory"""
    cache_key = f"{model_category}_{model_name}"
    if cache_key in st.session_state.loaded_models:
        del st.session_state.loaded_models[cache_key]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@st.cache_resource
def load_feature_extractors(features_dir='web_attack_detection/features', hf_repo=None):
    """Load feature extraction models (cached)"""
    extractors = {}

    try:
        # Word2Vec
        w2v_path = Path(features_dir) / 'word2vec.model'
        if w2v_path.exists():
            extractors['word2vec'] = Word2Vec.load(str(w2v_path))
        elif hf_repo:
            from huggingface_hub import hf_hub_download
            w2v_file = hf_hub_download(repo_id=hf_repo, filename='features/word2vec.model')
            extractors['word2vec'] = Word2Vec.load(w2v_file)

        # FastText
        ft_path = Path(features_dir) / 'fasttext.model'
        if ft_path.exists():
            extractors['fasttext'] = FastText.load(str(ft_path))
        elif hf_repo:
            from huggingface_hub import hf_hub_download
            ft_file = hf_hub_download(repo_id=hf_repo, filename='features/fasttext.model')
            extractors['fasttext'] = FastText.load(ft_file)

        # TF-IDF Vectorizer
        tfidf_path = Path(features_dir) / 'tfidf_vectorizer.pkl'
        if tfidf_path.exists():
            with open(tfidf_path, 'rb') as f:
                extractors['tfidf'] = pickle.load(f)
        elif hf_repo:
            from huggingface_hub import hf_hub_download
            tfidf_file = hf_hub_download(repo_id=hf_repo, filename='features/tfidf_vectorizer.pkl')
            with open(tfidf_file, 'rb') as f:
                extractors['tfidf'] = pickle.load(f)
    except Exception as e:
        st.warning(f"Some feature extractors could not be loaded: {str(e)}")

    return extractors

def extract_tfidf_features(text, extractors):
    """Extract TF-IDF features for classical ML models"""
    if 'tfidf' not in extractors:
        st.error("⚠️ TF-IDF vectorizer not loaded! Classical ML models will fail.")
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

def get_features_for_model(model_name, processed_text, raw_text, tfidf_features, uniembed_features, all_models_dict):
    """Get the correct features for a specific model"""
    # Determine feature type needed
    feature_type = None
    for models_dict in [CLASSICAL_ML_MODELS, DEEP_LEARNING_MODELS, TRANSFORMER_MODELS, HYBRID_MODELS]:
        if model_name in models_dict:
            feature_type = models_dict[model_name]
            break
    
    if feature_type == 'tfidf':
        return tfidf_features, feature_type
    elif feature_type == 'uniembed':
        return uniembed_features, feature_type
    elif feature_type == 'raw_text':
        return raw_text, feature_type
    else:
        return None, None

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
    hf_repo = st.text_input("HuggingFace Repo (optional)", value="", placeholder="username/repo-name")

    st.markdown("---")
    st.header("📊 Model Categories")
    st.warning("⚠️ **Memory Warning**: Loading many models simultaneously may crash the page. Select only the models you need.")
    
    show_classical = st.checkbox("Classical ML (9 models)", value=True)
    show_dl = st.checkbox("Deep Learning (5 models)", value=False)
    show_transformer = st.checkbox("Transformers (2 models)", value=False)
    show_hybrid = st.checkbox("Hybrid Models (3-5 models)", value=False)

    # Count active models
    active_count = 0
    if show_classical: active_count += 9
    if show_dl: active_count += 5
    if show_transformer: active_count += 2
    if show_hybrid: active_count += 5
    
    if active_count > 10:
        st.error(f"⚠️ {active_count} models selected! This may cause memory issues.")
    elif active_count > 5:
        st.warning(f"ℹ️ {active_count} models selected. Page may slow down.")
    
    # Cleanup unchecked models
    if not show_classical:
        for model_name in CLASSICAL_ML_MODELS.keys():
            unload_model(model_name, 'classical')
    if not show_dl:
        for model_name in DEEP_LEARNING_MODELS.keys():
            unload_model(model_name, 'deep_learning')
    if not show_transformer:
        for model_name in TRANSFORMER_MODELS.keys():
            unload_model(model_name, 'transformer')
    if not show_hybrid:
        for model_name in HYBRID_MODELS.keys():
            unload_model(model_name, 'hybrid')

    st.markdown("---")
    st.info("💡 Enter a query below to test for SQLi or XSS attacks")
    
    # Memory stats
    loaded_count = len(st.session_state.loaded_models)
    st.caption(f"🔢 Currently loaded: {loaded_count} models")

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
    with st.spinner("🔄 Loading feature extractors..."):
        preprocessor = ContentMatchingPreprocessor()
        processed_text = preprocessor.preprocess(user_input)

        try:
            # Load feature extractors (cached)
            extractors = load_feature_extractors(features_dir, hf_repo if hf_repo else None)

            # Extract features ONCE for all models
            tfidf_features = None
            uniembed_features = None
            
            # Only extract features that are needed
            needs_tfidf = show_classical or (show_hybrid and any(HYBRID_MODELS.get(m) == 'tfidf' for m in HYBRID_MODELS))
            needs_uniembed = show_dl or show_hybrid
            
            if needs_tfidf:
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

            # Classical ML Models
            if show_classical:
                st.subheader("🔹 Classical Machine Learning Models")
                classical_cols = st.columns(3)
                col_idx = 0

                for model_name in CLASSICAL_ML_MODELS.keys():
                    with st.spinner(f"Loading {model_name}..."):
                        model = load_model_lazy(model_name, 'classical', models_dir, hf_repo if hf_repo else None)
                        
                        if model and tfidf_features is not None:
                            try:
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
                                st.error(f"⚠️ {model_name}: {str(e)}")

            # Deep Learning Models
            if show_dl:
                st.subheader("🔹 Deep Learning Models")
                dl_cols = st.columns(3)
                col_idx = 0

                for model_name in DEEP_LEARNING_MODELS.keys():
                    with st.spinner(f"Loading {model_name}..."):
                        model = load_model_lazy(model_name, 'deep_learning', models_dir, hf_repo if hf_repo else None)
                        
                        if model and uniembed_features is not None:
                            try:
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
                                st.error(f"⚠️ {model_name}: {str(e)}")

            # Transformer Models
            if show_transformer:
                st.subheader("🔹 Transformer Models")
                trans_cols = st.columns(2)
                col_idx = 0

                for model_name in TRANSFORMER_MODELS.keys():
                    with st.spinner(f"Loading {model_name}..."):
                        model = load_model_lazy(model_name, 'transformer', models_dir, hf_repo if hf_repo else None)
                        
                        if model:
                            try:
                                pred, confidence = predict_with_transformer(user_input, model)
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
                                st.error(f"⚠️ {model_name}: {str(e)}")

            # Hybrid Models
            if show_hybrid:
                st.subheader("🔹 Hybrid Ensemble Models")
                hybrid_cols = st.columns(3)
                col_idx = 0

                for model_name in HYBRID_MODELS.keys():
                    with st.spinner(f"Loading {model_name}..."):
                        model = load_model_lazy(model_name, 'hybrid', models_dir, hf_repo if hf_repo else None)
                        
                        if model and uniembed_features is not None:
                            try:
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
                                st.error(f"⚠️ {model_name}: {str(e)}")

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
