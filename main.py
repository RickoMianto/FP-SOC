"""
DGA Analysis Engine - All-in-One Solution
AIOps SOC Project for Domain Generation Algorithm Detection
"""

import pandas as pd
import numpy as np
import re
import string
import pickle
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import *

class DGAAnalyzer:
    """
    Advanced DGA Detection System using Deep Learning
    Combines LSTM and CNN for maximum classification performance
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self.is_trained = False
        
    def extract_features(self, domains):
        """
        Advanced feature engineering for domain analysis
        """
        features = []
        
        for domain in tqdm(domains, desc="Extracting features"):
            domain_clean = domain.lower().strip()
            
            # Basic features
            length = len(domain_clean)
            digit_count = sum(c.isdigit() for c in domain_clean)
            alpha_count = sum(c.isalpha() for c in domain_clean)
            special_count = len([c for c in domain_clean if c in '.-_'])
            
            # Character frequency analysis
            char_freq = Counter(domain_clean)
            vowels = 'aeiou'
            vowel_count = sum(char_freq.get(v, 0) for v in vowels)
            consonant_count = alpha_count - vowel_count
            
            # Entropy calculation
            entropy = -sum((count/length) * np.log2(count/length) 
                          for count in char_freq.values() if count > 0)
            
            # N-gram analysis
            bigrams = [domain_clean[i:i+2] for i in range(len(domain_clean)-1)]
            trigrams = [domain_clean[i:i+3] for i in range(len(domain_clean)-2)]
            
            # Lexical diversity
            unique_chars = len(set(domain_clean))
            lexical_diversity = unique_chars / length if length > 0 else 0
            
            # Pattern analysis
            has_numbers = any(c.isdigit() for c in domain_clean)
            has_hyphens = '-' in domain_clean
            starts_with_number = domain_clean[0].isdigit() if domain_clean else False
            
            # Randomness indicators
            max_consecutive_chars = max(len(list(group)) for _, group in 
                                      __import__('itertools').groupby(domain_clean)) if domain_clean else 0
            
            # Compile features
            feature_vector = [
                length,
                digit_count,
                alpha_count,
                special_count,
                vowel_count,
                consonant_count,
                entropy,
                len(bigrams),
                len(trigrams),
                lexical_diversity,
                int(has_numbers),
                int(has_hyphens),
                int(starts_with_number),
                max_consecutive_chars,
                digit_count / length if length > 0 else 0,
                vowel_count / alpha_count if alpha_count > 0 else 0,
                unique_chars / length if length > 0 else 0
            ]
            
            features.append(feature_vector)
        
        self.feature_names = [
            'length', 'digit_count', 'alpha_count', 'special_count',
            'vowel_count', 'consonant_count', 'entropy', 'bigram_count',
            'trigram_count', 'lexical_diversity', 'has_numbers', 'has_hyphens',
            'starts_with_number', 'max_consecutive_chars', 'digit_ratio',
            'vowel_ratio', 'unique_char_ratio'
        ]
        
        return np.array(features)
    
    def create_hybrid_model(self, vocab_size, max_length, num_features, num_classes):
        """
        Create hybrid CNN-LSTM model for DGA detection
        """
        # Text input branch (LSTM + CNN)
        text_input = Input(shape=(max_length,), name='text_input')
        
        # Embedding layer
        embedding = Embedding(vocab_size, MODEL_CONFIG['embedding_dim'], 
                            input_length=max_length)(text_input)
        
        # LSTM branch
        lstm_out = LSTM(MODEL_CONFIG['lstm_units'], return_sequences=True, 
                       dropout=MODEL_CONFIG['dropout_rate'])(embedding)
        lstm_out = LSTM(MODEL_CONFIG['lstm_units']//2, 
                       dropout=MODEL_CONFIG['dropout_rate'])(lstm_out)
        
        # CNN branch
        cnn_out = Conv1D(MODEL_CONFIG['cnn_filters'], 3, activation='relu')(embedding)
        cnn_out = MaxPooling1D(2)(cnn_out)
        cnn_out = Conv1D(MODEL_CONFIG['cnn_filters']//2, 3, activation='relu')(cnn_out)
        cnn_out = GlobalMaxPooling1D()(cnn_out)
        
        # Numerical features input
        numeric_input = Input(shape=(num_features,), name='numeric_input')
        numeric_dense = Dense(64, activation='relu')(numeric_input)
        numeric_dense = Dropout(MODEL_CONFIG['dropout_rate'])(numeric_dense)
        
        # Combine all branches
        combined = Concatenate()([lstm_out, cnn_out, numeric_dense])
        
        # Dense layers
        dense = Dense(MODEL_CONFIG['dense_units'], activation='relu')(combined)
        dense = BatchNormalization()(dense)
        dense = Dropout(MODEL_CONFIG['dropout_rate'])(dense)
        
        dense = Dense(MODEL_CONFIG['dense_units']//2, activation='relu')(dense)
        dense = Dropout(MODEL_CONFIG['dropout_rate']//2)(dense)
        
        # Output layer
        if num_classes == 2:
            output = Dense(1, activation='sigmoid', name='output')(dense)
            loss = 'binary_crossentropy'
        else:
            output = Dense(num_classes, activation='softmax', name='output')(dense)
            loss = 'categorical_crossentropy'
        
        # Create model
        model = Model(inputs=[text_input, numeric_input], outputs=output)
        
        # Compile model
        optimizer = Adam(learning_rate=MODEL_CONFIG['learning_rate'])
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
        return model
    
    def prepare_data(self, df):
        """
        Prepare data for training
        """
        print("Preparing data...")
        
        # Extract features
        domains = df['domain'].values
        features = self.extract_features(domains)
        
        # Prepare text data
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(char_level=True, 
                                     num_words=MODEL_CONFIG['vocab_size'])
            self.tokenizer.fit_on_texts(domains)
        
        sequences = self.tokenizer.texts_to_sequences(domains)
        X_text = pad_sequences(sequences, maxlen=MODEL_CONFIG['max_length'])
        
        # Scale numerical features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_numeric = self.scaler.fit_transform(features)
        else:
            X_numeric = self.scaler.transform(features)
        
        # Prepare labels
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            # Secara eksplisit menentukan urutan kelas untuk memastikan
            # 'legitimate' menjadi 0 dan 'dga' menjadi 1.
            self.label_encoder.classes_ = np.array(['legitimate', 'dga'])
            y = self.label_encoder.transform(df['isDGA'].values)
        else:
            y = self.label_encoder.transform(df['isDGA'].values)
        
        return X_text, X_numeric, y
    
    def train(self, df):
        """
        Train the DGA detection model
        """
        print("Starting DGA model training...")
        
        # Prepare data
        X_text, X_numeric, y = self.prepare_data(df)
        
        # Split data
        X_text_train, X_text_test, X_numeric_train, X_numeric_test, y_train, y_test = \
            train_test_split(X_text, X_numeric, y, test_size=0.2, 
                           random_state=42, stratify=y)
        
        # Create model
        vocab_size = len(self.tokenizer.word_index) + 1
        num_features = X_numeric.shape[1]
        num_classes = len(np.unique(y))
        
        self.model = self.create_hybrid_model(vocab_size, MODEL_CONFIG['max_length'], 
                                            num_features, num_classes)
        
        print(f"Model created with vocab_size: {vocab_size}, features: {num_features}")
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, 
                                     restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                    patience=5, min_lr=1e-6)
        
        # Train model
        history = self.model.fit(
            [X_text_train, X_numeric_train], y_train,
            batch_size=MODEL_CONFIG['batch_size'],
            epochs=MODEL_CONFIG['epochs'],
            validation_data=([X_text_test, X_numeric_test], y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        predictions = self.model.predict([X_text_test, X_numeric_test])
        if len(np.unique(y)) == 2:
            y_pred = (predictions > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        self.is_trained = True
        return history
    
    def predict(self, domains):
        """
        Predict DGA probability for domains
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare input data
        if isinstance(domains, str):
            domains = [domains]
        
        # Extract features
        features = self.extract_features(domains)
        X_numeric = self.scaler.transform(features)
        
        # Prepare text data
        sequences = self.tokenizer.texts_to_sequences(domains)
        X_text = pad_sequences(sequences, maxlen=MODEL_CONFIG['max_length'])
        
        # Make predictions
        predictions = self.model.predict([X_text, X_numeric])
        
        if len(self.label_encoder.classes_) == 2:
            probabilities = predictions.flatten()
            predicted_classes = (probabilities > 0.5).astype(int)
        else:
            probabilities = np.max(predictions, axis=1)
            predicted_classes = np.argmax(predictions, axis=1)
        
        results = []
        for i, domain in enumerate(domains):
            result = {
                'domain': domain,
                'is_dga': bool(predicted_classes[i]),
                'dga_probability': float(probabilities[i]),
                'predicted_class': self.label_encoder.inverse_transform([predicted_classes[i]])[0],
                'risk_level': self._get_risk_level(probabilities[i])
            }
            results.append(result)
        
        return results
    
    def _get_risk_level(self, probability):
        """
        Determine risk level based on DGA probability
        """
        if probability >= ALERT_CONFIG['high_risk_threshold']:
            return 'HIGH'
        elif probability >= ALERT_CONFIG['medium_risk_threshold']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def save_model(self, model_dir=MODEL_PATH):
        """
        Save trained model and preprocessors
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save model
        model_path = os.path.join(model_dir, MODEL_FILES['main_model'])
        self.model.save(model_path)
        
        # Save preprocessors
        with open(os.path.join(model_dir, MODEL_FILES['tokenizer']), 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        with open(os.path.join(model_dir, MODEL_FILES['scaler']), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(os.path.join(model_dir, MODEL_FILES['label_encoder']), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir=MODEL_PATH):
        """
        Load trained model and preprocessors
        """
        try:
            # Load model
            model_path = os.path.join(model_dir, MODEL_FILES['main_model'])
            self.model = load_model(model_path)
            
            # Load preprocessors
            with open(os.path.join(model_dir, MODEL_FILES['tokenizer']), 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            with open(os.path.join(model_dir, MODEL_FILES['scaler']), 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(os.path.join(model_dir, MODEL_FILES['label_encoder']), 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            self.is_trained = True
            print(f"Model loaded from {model_dir}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def create_visualizations(analyzer, df):
    """
    Create comprehensive visualizations for DGA analysis
    """
    # Sample predictions for visualization
    sample_domains = df['domain'].sample(min(1000, len(df))).tolist()
    predictions = analyzer.predict(sample_domains)
    
    # Create prediction DataFrame
    pred_df = pd.DataFrame(predictions)
    
    # 1. DGA Distribution
    fig1 = px.pie(pred_df, names='predicted_class', 
                  title='DGA vs Legitimate Domain Distribution',
                  color_discrete_map={'dga': '#ff4444', 'legitimate': '#44ff44'})
    
    # 2. Risk Level Distribution
    fig2 = px.bar(pred_df.groupby('risk_level').size().reset_index(name='count'),
                  x='risk_level', y='count',
                  title='Risk Level Distribution',
                  color='risk_level',
                  color_discrete_map={'HIGH': '#ff0000', 'MEDIUM': '#ffaa00', 'LOW': '#00ff00'})
    
    # 3. DGA Probability Distribution
    fig3 = px.histogram(pred_df, x='dga_probability', 
                       title='DGA Probability Distribution',
                       bins=50, color='predicted_class')
    
    return fig1, fig2, fig3

def main():
    """
    Main function for training and testing DGA analyzer
    """
    print("🛡️ DGA Analysis System - AIOps SOC")
    print("="*50)
    
    # Initialize analyzer
    analyzer = DGAAnalyzer()
    
    # Try to load existing model
    if analyzer.load_model():
        print("✅ Pre-trained model loaded successfully!")
    else:
        print("🔄 No pre-trained model found. Training new model...")
        
        # Load and prepare data
        if os.path.exists(DGA_DATA_FILE):
            df = pd.read_csv(DGA_DATA_FILE)
            print(f"📊 Loaded {len(df)} domain records")

            # --- TAMBAHKAN KODE INI ---
            # Membersihkan data: Hapus baris dengan domain kosong (NaN) dan pastikan semua domain adalah string
            print("🧹 Cleaning data...")
            df.dropna(subset=['domain'], inplace=True)
            df['domain'] = df['domain'].astype(str)
            print(f"📊 {len(df)} valid domain records after cleaning.")
            # --------------------------

            # --- TAMBAHKAN KODE INI UNTUK NORMALISASI LABEL ---
            print("🔬 Normalizing labels...")
            # Mengubah semua label yang bukan 'dga' menjadi 'legitimate' untuk konsistensi.
            # Ini akan menangani 'legit', 'Legitimate', dll.
            df['isDGA'] = df['isDGA'].apply(lambda x: 'dga' if x == 'dga' else 'legitimate')
            print(f"Labels after normalization: {df['isDGA'].unique()}")
            # ----------------------------------------------------

            # Train model
            history = analyzer.train(df)
            
            # Save model
            analyzer.save_model()
            print("✅ Model training completed and saved!")
        else:
            print(f"❌ Data file not found: {DGA_DATA_FILE}")
            return
    
    # Interactive testing
    print("\n🧪 Interactive DGA Testing")
    print("Enter domains to analyze (or 'quit' to exit):")
    
    while True:
        domain = input("\nDomain: ").strip()
        if domain.lower() == 'quit':
            break
        
        if domain:
            try:
                results = analyzer.predict([domain])
                result = results[0]
                
                print(f"\n📊 Analysis Results for: {result['domain']}")
                print(f"   DGA Status: {'🚨 DGA DETECTED' if result['is_dga'] else '✅ LEGITIMATE'}")
                print(f"   Probability: {result['dga_probability']:.4f}")
                print(f"   Risk Level: {result['risk_level']}")
                print(f"   Classification: {result['predicted_class']}")
                
                if result['risk_level'] == 'HIGH':
                    print("   🚨 HIGH RISK ALERT - Immediate attention required!")
                
            except Exception as e:
                print(f"❌ Error analyzing domain: {e}")

if __name__ == "__main__":
    main()