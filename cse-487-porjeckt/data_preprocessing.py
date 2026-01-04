# ============================================================================
# FILE: data_preprocessing.py
# DESCRIPTION: Data loading, cleaning, and preprocessing (MEMORY OPTIMIZED)
# ============================================================================

from config import GLOBAL_CONFIG

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings
import gc
warnings.filterwarnings('ignore')

# ============================================================================
# SQL/XSS KEYWORDS AND PATTERNS
# ============================================================================

SQL_KEYWORDS = [
    'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
    'UNION', 'WHERE', 'FROM', 'JOIN', 'AND', 'OR', 'NOT', 'NULL',
    'ORDER', 'GROUP', 'HAVING', 'LIMIT', 'OFFSET', 'AS', 'ON',
    'EXEC', 'EXECUTE', 'DECLARE', 'TABLE', 'DATABASE', 'COLUMN',
    'BETWEEN', 'LIKE', 'IN', 'EXISTS', 'CASE', 'WHEN', 'THEN',
    'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX'
]

XSS_KEYWORDS = [
    'script', 'iframe', 'object', 'embed', 'applet', 'meta', 'link',
    'style', 'img', 'svg', 'video', 'audio', 'canvas', 'input',
    'button', 'form', 'body', 'html', 'onerror', 'onload', 'onclick',
    'onmouseover', 'onfocus', 'onblur', 'alert', 'prompt', 'confirm',
    'eval', 'expression', 'javascript', 'vbscript', 'document',
    'window', 'location', 'cookie', 'localstorage', 'sessionstorage'
]

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

class ContentMatchingPreprocessor:
    """Implements content matching preprocessing"""
    
    def __init__(self):
        self.sql_keywords = set([kw.lower() for kw in SQL_KEYWORDS])
        self.xss_keywords = set([kw.lower() for kw in XSS_KEYWORDS])
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def digital_generalization(self, text):
        """Replace numbers with <NUM> token"""
        return re.sub(r'\b\d+\b', '<NUM>', text)
    
    def url_replacement(self, text):
        """Replace URLs with <URL> token"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '<URL>', text)
    
    def preserve_keywords(self, text):
        """Preserve SQL and XSS keywords as atomic units"""
        for keyword in self.sql_keywords:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            text = pattern.sub(f'<SQL_{keyword.upper()}>', text)
        
        for keyword in self.xss_keywords:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            text = pattern.sub(f'<XSS_{keyword.upper()}>', text)
        
        return text
    
    def normalize_text(self, text):
        """Lowercase and remove excessive whitespace"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def special_character_mapping(self, text):
        """Map special characters to tokens"""
        char_mappings = {
            '\'': '<SQUOTE>',
            '"': '<DQUOTE>',
            ';': '<SEMICOLON>',
            '--': '<COMMENT>',
            '/*': '<BLOCKCOMMENT_START>',
            '*/': '<BLOCKCOMMENT_END>',
            '=': '<EQUALS>',
            '<': '<LT>',
            '>': '<GT>',
            '(': '<LPAREN>',
            ')': '<RPAREN>',
            '{': '<LBRACE>',
            '}': '<RBRACE>',
            '[': '<LBRACKET>',
            ']': '<RBRACKET>',
        }
        
        for char, token in char_mappings.items():
            text = text.replace(char, f' {token} ')
        
        return text
    
    def preprocess(self, text, apply_stemming=False, remove_stopwords=False):
        """Complete preprocessing pipeline"""
        if not isinstance(text, str):
            text = str(text)
        
        text = self.digital_generalization(text)
        text = self.url_replacement(text)
        text = self.preserve_keywords(text)
        text = self.normalize_text(text)
        text = self.special_character_mapping(text)
        
        if apply_stemming or remove_stopwords:
            tokens = word_tokenize(text)
            
            if remove_stopwords:
                tokens = [t for t in tokens if t not in self.stop_words]
            
            if apply_stemming:
                tokens = [self.stemmer.stem(t) for t in tokens]
            
            text = ' '.join(tokens)
        
        return text

# ============================================================================
# DATA LOADING AND MERGING
# ============================================================================

class DataLoader:
    """Load and merge all datasets with memory optimization"""
    
    def __init__(self, data_dir='/content', config=None):
        self.data_dir = data_dir
        self.preprocessor = ContentMatchingPreprocessor()
        if config is None:
            config = GLOBAL_CONFIG
        self.config = config
        
    def load_dataset(self, filename, text_col=None, label_col='Label'):
        """Load a single dataset"""
        filepath = f"{self.data_dir}/{filename}"
        print(f"Loading {filename}...")
        
        try:
            df = pd.read_csv(filepath, engine='python')
            
            if text_col is None:
                text_cols = ['Sentence', 'Query', 'text', 'Text', 'payload', 'Payload']
                for col in text_cols:
                    if col in df.columns:
                        text_col = col
                        break
                
                if text_col is None:
                    text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            df = df.rename(columns={text_col: 'text', label_col: 'label'})
            
            if 'text' in df.columns and 'label' in df.columns:
                df = df[['text', 'label']]
            
            df = df.dropna(subset=['label'])
            df['text'] = df['text'].fillna('')
            df['label'] = df['label'].astype(int)
            
            print(f"  Loaded {len(df)} samples")
            print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
            
            return df
        
        except Exception as e:
            print(f"  Error loading {filename}: {str(e)}")
            return None
    
    def load_all_datasets(self):
        """Load all datasets and merge them"""
        print("\n" + "="*70)
        print("LOADING ALL DATASETS")
        print("="*70 + "\n")
        
        datasets = {}
        
        xss_df = self.load_dataset('XSS_dataset.csv', text_col='Sentence')
        if xss_df is not None:
            xss_df['dataset_source'] = 'xss'
            datasets['xss'] = xss_df
        
        train_df = self.load_dataset('Train.csv', text_col='Query')
        if train_df is not None:
            train_df['dataset_source'] = 'sql_train'
            datasets['sql_train'] = train_df
        
        test_df = self.load_dataset('Test.csv', text_col='Query')
        if test_df is not None:
            test_df['dataset_source'] = 'sql_test'
            datasets['sql_test'] = test_df
        
        val_df = self.load_dataset('Validation.csv', text_col='Query')
        if val_df is not None:
            val_df['dataset_source'] = 'sql_val'
            datasets['sql_val'] = val_df
        
        modified_sql_df = self.load_dataset('Modified_SQL_Dataset.csv', text_col='Query')
        if modified_sql_df is not None:
            modified_sql_df['dataset_source'] = 'sql_modified'
            datasets['sql_modified'] = modified_sql_df
        
        print("\n" + "="*70)
        print("MERGING DATASETS")
        print("="*70 + "\n")
        
        all_dfs = [df for df in datasets.values() if df is not None]
        
        if not all_dfs:
            raise ValueError("No datasets were loaded successfully!")
        
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        print(f"Total samples: {len(merged_df)}")
        print(f"Total features: {merged_df.shape[1]}")
        print(f"Label distribution:")
        print(merged_df['label'].value_counts())
        print(f"\nDataset sources:")
        print(merged_df['dataset_source'].value_counts())
        
        duplicates = merged_df.duplicated(subset=['text', 'label']).sum()
        print(f"\nChecking for duplicates...")
        print(f"  Found {duplicates} duplicate rows")
        
        if duplicates > 0:
            merged_df = merged_df.drop_duplicates(subset=['text', 'label'], keep='first')
            print(f"  Removed duplicates, {len(merged_df)} samples remaining")
        
        # MEMORY OPTIMIZATION: Apply sampling if configured
        if self.config.get('use_sample_data', False):
            sample_fraction = self.config.get('sample_fraction', 0.3)
            max_samples = self.config.get('max_samples', 50000)
            
            target_size = int(len(merged_df) * sample_fraction)
            target_size = min(target_size, max_samples)
            
            print(f"\nUSING SAMPLE DATA FOR MEMORY OPTIMIZATION")
            print(f"  Original size: {len(merged_df)}")
            print(f"  Sample size: {target_size}")
            
            merged_df = merged_df.sample(n=target_size, random_state=self.config['random_seed'])
            print(f"  Sampled dataset: {len(merged_df)} samples\n")
        
        return merged_df, datasets
    
    def preprocess_data(self, df, apply_stemming=False, remove_stopwords=False):
        """Apply content matching preprocessing to all text"""
        print("\n" + "="*70)
        print("PREPROCESSING DATA (Content Matching)")
        print("="*70 + "\n")
        
        df = df.copy()
        df['text_original'] = df['text']
        
        print("Applying content matching preprocessing...")
        df['text'] = df['text'].apply(
            lambda x: self.preprocessor.preprocess(
                x, 
                apply_stemming=apply_stemming,
                remove_stopwords=remove_stopwords
            )
        )
        
        print("Preprocessing complete!")
        
        print("\nExample transformations:")
        for i in range(min(3, len(df))):
            print(f"\n  Original: {df['text_original'].iloc[i][:100]}...")
            print(f"  Processed: {df['text'].iloc[i][:100]}...")
        
        return df
    
    def create_splits(self, df, config=None):
        """Create train/val/test splits with stratification"""
        if config is None:
            config = self.config
            
        print("\n" + "="*70)
        print("CREATING DATA SPLITS")
        print("="*70 + "\n")
        
        X = df['text'].values
        y = df['label'].values
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(config['val_split'] + config['test_split']),
            random_state=config['random_seed'],
            stratify=y if config['stratify'] else None
        )
        
        val_ratio = config['val_split'] / (config['val_split'] + config['test_split'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio),
            random_state=config['random_seed'],
            stratify=y_temp if config['stratify'] else None
        )
        
        print(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Class distribution: {np.bincount(y_train)}")
        print(f"\nValidation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Class distribution: {np.bincount(y_val)}")
        print(f"\nTest set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"   Class distribution: {np.bincount(y_test)}")
        
        splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
        
        return splits

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    from config import GLOBAL_CONFIG, set_seed
    from pathlib import Path
    
    set_seed(GLOBAL_CONFIG['random_seed'])
    
    loader = DataLoader(data_dir=GLOBAL_CONFIG['data_dir'], config=GLOBAL_CONFIG)
    merged_df, individual_datasets = loader.load_all_datasets()
    
    processed_df = loader.preprocess_data(merged_df)
    
    splits = loader.create_splits(processed_df)
    
    print("\nSaving processed data...")
    save_dir = f"{GLOBAL_CONFIG['base_dir']}/data"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    np.save(f"{save_dir}/X_train.npy", splits['X_train'])
    np.save(f"{save_dir}/y_train.npy", splits['y_train'])
    np.save(f"{save_dir}/X_val.npy", splits['X_val'])
    np.save(f"{save_dir}/y_val.npy", splits['y_val'])
    np.save(f"{save_dir}/X_test.npy", splits['X_test'])
    np.save(f"{save_dir}/y_test.npy", splits['y_test'])
    
    processed_df.to_csv(f"{save_dir}/processed_data.csv", index=False)
    
    print("Data preprocessing complete!")

