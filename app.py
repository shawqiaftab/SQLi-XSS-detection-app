import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
from pathlib import Path
import re
from huggingface_hub import hf_hub_download
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
import warnings
import gc
warnings.filterwarnings('ignore')

# SQL and XSS Keywords (from your training)
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
    """EXACT preprocessing from training"""
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
if 'tfidf_vectorizer' not in st.session_state:
    st.session_state.tfidf_vectorizer = None

@st.cache_resource
def load_tfidf_vectorizer():
    """Load TF-IDF vectorizer from HuggingFace"""
    repo_id = "Dr-KeK/sqli-xss-models"
    try:
        tfidf_file = hf_hub_download(
            repo_id=repo_id,
            filename="features/tfidf_vectorizer.pkl",
            repo_type="model"
        )
        with open(tfidf_file, 'rb') as f:
            vectorizer = pickle.load(f)
        st.success("✅ TF-IDF vectorizer loaded")
        return vectorizer
    except Exception as e:
        st.error(f"❌ Failed to load TF-IDF: {str(e)}")
        return None

@st.cache_resource
def load_single_model(model_name):
    """Load a single Classical ML model from HuggingFace"""
    repo_id = "Dr-KeK/sqli-xss-models"
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"models/classical_ml/{model_name}.pkl",
            repo_type="model"
        )
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Failed to load {model_name}: {str(e)}")
        return None

def extract_tfidf_features(text, vectorizer):
    """Extract TF-IDF features"""
    if vectorizer is None:
        return None
    features = vectorizer.transform([text]).toarray()[0]
    return features

# Streamlit App
st.set_page_config(page_title="SQLi & XSS Detection", layout="wide", page_icon="🛡️")

st.title("🛡️ SQL Injection & XSS Attack Detection")
st.markdown("### Classical ML Models (TF-IDF Features)")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Select Models")
    st.info("💡 Using only Classical ML (proven accuracy)")
    
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
    
    selected_count = sum(classical_models.values())
    st.metric("Selected Models", selected_count)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Input Query")
    user_input = st.text_area("Enter your query/payload:",
                               height=150,
                               placeholder="Example: ' OR 1=1 -- ")
    analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)

with col2:
    st.header("📋 Examples")
    st.code("' OR '1'='1", language="sql")
    st.caption("SQL Injection")
    st.code("<script>alert('XSS')</script>", language="html")
    st.caption("XSS Attack")

if analyze_button and user_input:
    if selected_count == 0:
        st.warning("⚠️ Select at least one model!")
        st.stop()
    
    with st.spinner("🔄 Analyzing..."):
        # Preprocess
        preprocessor = ContentMatchingPreprocessor()
        processed_text = preprocessor.preprocess(user_input)
        
        # Load TF-IDF
        if st.session_state.tfidf_vectorizer is None:
            st.session_state.tfidf_vectorizer = load_tfidf_vectorizer()
        
        vectorizer = st.session_state.tfidf_vectorizer
        if vectorizer is None:
            st.error("❌ TF-IDF not loaded!")
            st.stop()
        
        # Extract features
        features = extract_tfidf_features(processed_text, vectorizer)
        
        with st.expander("🔍 Preprocessing"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.code(user_input, language="text")
                st.caption("Original")
            with col_b:
                st.code(processed_text, language="text")
                st.caption("Processed")
        
        st.markdown("---")
        st.header("🎯 Results")
        
        results = []
        selected_models = [k for k, v in classical_models.items() if v]
        
        cols = st.columns(min(3, len(selected_models)))
        
        for idx, model_name in enumerate(selected_models):
            model = load_single_model(model_name)
            
            if model:
                try:
                    pred = model.predict(features.reshape(1, -1))[0]
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features.reshape(1, -1))[0]
                        confidence = proba[pred] * 100
                    else:
                        confidence = 100 if pred == 1 else 0
                    
                    # pred==1 means ATTACK, pred==0 means SAFE (from training)
                    label = "🚨 ATTACK" if pred == 1 else "✅ SAFE"
                    
                    with cols[idx % 3]:
                        st.metric(
                            label=model_name.replace('_', ' '),
                            value=label,
                            delta=f"{confidence:.1f}%"
                        )
                    
                    results.append({
                        'Model': model_name.replace('_', ' '),
                        'Prediction': label,
                        'Confidence': f'{confidence:.1f}%'
                    })
                except Exception as e:
                    st.error(f"Error with {model_name}: {str(e)}")
        
        # Summary
        if results:
            st.markdown("---")
            attack_count = sum(1 for r in results if "ATTACK" in r['Prediction'])
            safe_count = len(results) - attack_count
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Models", len(results))
            with col_s2:
                st.metric("Attack", attack_count)
            with col_s3:
                st.metric("Safe", safe_count)
            
            if attack_count > safe_count:
                st.error(f"⚠️ **ATTACK DETECTED** ({attack_count}/{len(results)} models)")
            elif safe_count > attack_count:
                st.success(f"✅ **SAFE** ({safe_count}/{len(results)} models)")
            else:
                st.warning(f"⚖️ **UNCERTAIN**")
            
            with st.expander("📋 Details"):
                st.dataframe(pd.DataFrame(results), use_container_width=True)

elif analyze_button:
    st.warning("⚠️ Please enter a query")

st.markdown("---")
st.caption("🔬 Classical ML Models trained on 174K samples | TF-IDF Features")
