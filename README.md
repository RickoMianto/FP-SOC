# 🛡️ DGA Analysis - AIOps SOC Project

Advanced Domain Generation Algorithm (DGA) detection system using Deep Learning for Security Operations Center (SOC) environments.

## Overview

This project implements a comprehensive DGA detection system that combines:
- **Deep Learning**: Hybrid CNN-LSTM architecture for maximum accuracy
- **Real-time Analysis**: Live domain monitoring and alerting
- **Interactive Dashboard**: User-friendly Streamlit interface
- **SOC Integration**: Ready for production SOC environments

## Features

### Core Capabilities
- **High Accuracy**: >95% detection accuracy using deep learning
- **Real-time Processing**: Live domain analysis and monitoring
- **Batch Analysis**: Process thousands of domains efficiently
- **Interactive Dashboard**: Web-based interface for SOC analysts
- **Alert System**: Configurable risk-based alerting
- **Model Training**: Custom model training with your data

### Advanced Features
- **Hybrid Deep Learning**: CNN + LSTM for pattern recognition
- **Feature Engineering**: 17+ advanced domain features
- **Rich Visualizations**: Interactive charts and graphs
- **Responsive Design**: Works on desktop and mobile
- **Model Persistence**: Save and load trained models
- **Performance Monitoring**: Track model performance over time

## Architecture

### Model Architecture
```
Input Layer (Domain Text + Numerical Features)
    ↓
Embedding Layer (128-dim character embeddings)
    ↓
┌─────────────────┐    ┌─────────────────┐
│ LSTM Branch     │    │ CNN Branch      │
│ - LSTM (256)    │    │ - Conv1D (128)  │
│ - LSTM (128)    │    │ - MaxPool1D     │
│ - Dropout       │    │ - Conv1D (64)   │
└─────────────────┘    └─────────────────┘
    ↓                      ↓
┌─────────────────────────────────────────┐
│ Feature Fusion Layer                    │
│ - Concatenate CNN + LSTM + Numeric      │
│ - Dense (512) + BatchNorm + Dropout     │
│ - Dense (256) + Dropout                 │
│ - Output Layer (Sigmoid/Softmax)        │
└─────────────────────────────────────────┘
```

### Feature Engineering
- **Basic Features**: Length, character counts, special characters
- **Linguistic Features**: Vowel/consonant ratio, lexical diversity
- **Entropy Analysis**: Character frequency distribution
- **N-gram Analysis**: Bigrams and trigrams
- **Pattern Recognition**: Consecutive characters, randomness indicators

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.13+
- 4GB+ RAM (8GB+ recommended for training)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/RickoMianto/FP-SOC.git
cd FP-SOC
```

2. **Create virtual environment**
```bash
python -m venv venv
# On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create directory structure**
```bash
mkdir -p data models static
```

5. **Add your dataset**
   - Place your `dga_data.csv` file in the `data/` directory
   - Format: `isDGA,domain,host,subclass`
   - you can get the dataset by following this [link](https://www.kaggle.com/datasets/gtkcyber/dga-dataset)

## Usage

### 1. Training the Model
```bash
python main.py
```

### 2. Running the Dashboard
```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### 3. Analysis Modes

#### Single Domain Analysis
- Enter individual domains for immediate analysis
- Get detailed risk assessment and probability scores
- View feature analysis and recommendations

#### Batch Analysis
- Upload CSV or TXT files with domain lists
- Process thousands of domains efficiently
- Download comprehensive analysis reports

#### Real-time Monitoring
- Monitor network traffic in real-time
- Set up automated alerts for high-risk domains
- Track trends and patterns over time

## 📊 Dataset Format

Your training dataset should be a CSV file with the following columns:

```csv
isDGA,domain,host,subclass
dga,6xzxsw3sokvg1tc752y1a6p0af,6xzxsw3sokvg1tc752y1a6p0af.com,gameoverdga
legitimate,google,google.com,legitimate
dga,glbtlxwwhbnpxs,glbtlxwwhbnpxs.ru,cryptolocker
legitimate,facebook,facebook.com,legitimate
```

### Required Columns:
- **isDGA**: `dga` or `legitimate` (or any binary classification)
- **domain**: The domain name to analyze
- **host**: Full hostname (optional, can be same as domain)
- **subclass**: DGA family or category (optional)

## ⚙️ Configuration

### Model Parameters (`config.py`)
```python
MODEL_CONFIG = {
    'max_length': 64,          # Maximum domain length
    'embedding_dim': 128,      # Embedding dimension
    'lstm_units': 256,         # LSTM units
    'cnn_filters': 128,        # CNN filters
    'learning_rate': 0.001,    # Learning rate
    'batch_size': 128,         # Batch size
    'epochs': 100,             # Training epochs
    'dropout_rate': 0.3,       # Dropout rate
    'validation_split': 0.2,   # Validation data ratio
}

FEATURE_CONFIG = {
    'use_ngrams': True,        # Enable n-gram features
    'ngram_range': (2, 3),     # Bigrams and trigrams
    'calculate_entropy': True,  # Calculate domain entropy
    'vowel_consonant_ratio': True,  # Linguistic features
    'special_chars': True,     # Special character analysis
}

TRAINING_CONFIG = {
    'early_stopping': True,    # Stop when no improvement
    'patience': 10,            # Early stopping patience
    'monitor': 'val_accuracy', # Metric to monitor
    'save_best_model': True,   # Save best performing model
    'model_checkpoint': True,  # Save model checkpoints
}
```

### Dashboard Configuration (`config.py`)
```python
DASHBOARD_CONFIG = {
    'title': 'DGA Detection SOC Dashboard',
    'theme': 'dark',
    'refresh_interval': 30,    # Seconds
    'max_results': 1000,       # Maximum displayed results
    'alert_threshold': 0.8,    # DGA probability threshold
    'colors': {
        'dga': '#FF4444',      # Red for DGA
        'legitimate': '#44FF44', # Green for legitimate
        'warning': '#FFAA44',   # Orange for warnings
    }
}
```

## 📈 Performance Metrics

The model is designed to achieve the following performance targets:

### Target Performance
- **Accuracy**: >95%
- **Precision**: >94%
- **Recall**: >96%
- **F1-Score**: >95%
- **False Positive Rate**: <2%

### Evaluation Metrics
- **Confusion Matrix**: Detailed classification results
- **ROC Curve**: Receiver Operating Characteristic analysis
- **Precision-Recall Curve**: Precision vs Recall trade-offs
- **Cross-Validation**: K-fold validation results

## 🔧 Advanced Usage

### Feature Engineering Components

#### Text Preprocessing
```python
# Domain length analysis
domain_length = len(domain)

# Character frequency distribution
char_freq = Counter(domain.lower())

# Entropy calculation
entropy = -sum((freq/len(domain)) * math.log2(freq/len(domain)) 
               for freq in char_freq.values())
```

#### N-gram Analysis
```python
# Bigrams and trigrams
bigrams = [''.join(bg) for bg in zip(domain[:-1], domain[1:])]
trigrams = [''.join(tg) for tg in zip(domain[:-2], domain[1:-1], domain[2:])]
```

#### Linguistic Features
```python
# Vowel/consonant ratio
vowels = 'aeiou'
vowel_count = sum(1 for char in domain.lower() if char in vowels)
consonant_count = sum(1 for char in domain.lower() if char.isalpha() and char not in vowels)
vowel_consonant_ratio = vowel_count / max(consonant_count, 1)

# Lexical diversity
lexical_diversity = len(set(domain.lower())) / len(domain)
```

### Model Architecture Details

#### LSTM Branch
```python
# Sequential pattern recognition
lstm_input = Embedding(vocab_size, embedding_dim)(input_layer)
lstm_1 = LSTM(256, return_sequences=True, dropout=0.3)(lstm_input)
lstm_2 = LSTM(128, dropout=0.3)(lstm_1)
```

#### CNN Branch
```python
# Local pattern detection
cnn_input = Embedding(vocab_size, embedding_dim)(input_layer)
conv_1 = Conv1D(128, 3, activation='relu')(cnn_input)
pool_1 = MaxPooling1D(2)(conv_1)
conv_2 = Conv1D(64, 3, activation='relu')(pool_1)
global_pool = GlobalMaxPooling1D()(conv_2)
```

#### Feature Fusion
```python
# Combine all features
concat_features = Concatenate()([lstm_2, global_pool, numeric_features])
dense_1 = Dense(512, activation='relu')(concat_features)
batch_norm = BatchNormalization()(dense_1)
dropout_1 = Dropout(0.3)(batch_norm)
dense_2 = Dense(256, activation='relu')(dropout_1)
dropout_2 = Dropout(0.3)(dense_2)
output = Dense(1, activation='sigmoid')(dropout_2)
```

## 📁 Project Structure

```
project-aiops-dga/
│
├── data/
│   └── dga_data.csv           # Dataset utama
│
├── models/                    # Model yang sudah dilatih disimpan disini
│   ├── best_model.h5         # Best performing model
│   ├── model_weights.h5      # Model weights
│   └── tokenizer.pkl         # Text tokenizer
│
├── static/
│   └── style.css             # Styling untuk Streamlit
│
├── main.py                   # File utama - All-in-one solution
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── config.py                 # Konfigurasi project
└── README.md                 # This file
```

## 🧩 Main Components

### `main.py` - All-in-One Solution
Contains all core functionality:

1. **Data Preprocessing & Feature Engineering**
   - Text preprocessing untuk domain names
   - Feature extraction (panjang domain, entropy, karakter khusus, dll)
   - N-gram analysis
   - Statistical features

2. **Deep Learning Model**
   - LSTM-based Neural Network untuk sequence analysis
   - CNN + LSTM Hybrid untuk pattern recognition
   - Transformer-based model (optional) untuk advanced analysis
   - Pre-trained embeddings untuk domain representation

3. **Model Training & Evaluation**
   - Cross-validation
   - Hyperparameter tuning
   - Performance metrics (Accuracy, Precision, Recall, F1-score)
   - Confusion matrix analysis

4. **Visualization & Analytics**
   - Real-time classification dashboard
   - Performance metrics visualization
   - Domain analysis charts
   - Alert system untuk suspicious domains

### `app.py` - Streamlit Dashboard
Interactive web interface featuring:
- Real-time domain analysis
- Batch processing capability
- Alert generation
- Historical trend analysis
- Model performance monitoring
- False positive reduction

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow/Keras, PyTorch
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Web Interface**: Streamlit
- **Model Serving**: TensorFlow Serving (optional)

## ✨ Key Advantages

1. **Single File Solution** - Mudah maintain dan deploy
2. **High Performance** - Deep learning untuk akurasi maksimal
3. **Real-time Analysis** - Cocok untuk SOC environment
4. **Scalable Architecture** - Bisa handle volume tinggi
5. **Interactive Dashboard** - User-friendly interface

## 🚨 SOC Integration Features

### Real-time Monitoring
```python
# Continuous domain monitoring
monitor = DGAMonitor(model_path='models/best_model.h5')
monitor.start_real_time_analysis(
    threshold=0.8,
    alert_callback=send_alert_to_soc
)
```

### Alert System
```python
# Configurable alerting
alert_config = {
    'high_risk_threshold': 0.9,    # Immediate alert
    'medium_risk_threshold': 0.7,  # Warning alert
    'batch_alert_interval': 300,   # 5 minutes
    'max_alerts_per_hour': 100,    # Rate limiting
}
```

### Batch Processing
```python
# Process large domain lists
batch_processor = BatchDGAAnalyzer(model_path='models/best_model.h5')
results = batch_processor.analyze_domains(
    domains=domain_list,
    output_format='csv',
    include_features=True
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues During Training**
```python
# Reduce batch size
MODEL_CONFIG['batch_size'] = 64  # Instead of 128

# Use gradient checkpointing
tf.keras.utils.get_custom_objects()['gradient_checkpointing'] = True
```

2. **Overfitting**
```python
# Increase dropout rate
MODEL_CONFIG['dropout_rate'] = 0.5

# Add more regularization
from tensorflow.keras.regularizers import l2
Dense(512, activation='relu', kernel_regularizer=l2(0.001))
```

3. **Slow Inference**
```python
# Use model quantization
converter = tf.lite.TFLiteConverter.from_saved_model('models/best_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

## 📚 Dependencies

### Core Requirements (`requirements.txt`)
```
tensorflow>=2.13.0
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Optional Dependencies
```
torch>=2.0.0          # For PyTorch implementation
transformers>=4.30.0   # For transformer models
optuna>=3.2.0         # For hyperparameter tuning
mlflow>=2.5.0         # For experiment tracking
```

## 🔒 Security Considerations

- Implement input validation for domain names
- Use secure file upload mechanisms
- Sanitize user inputs in the dashboard
- Implement rate limiting for API endpoints
- Regular model updates with new threat intelligence

## 📞 Support & Contact

For questions, issues, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/RickoMianto/FP-SOC/issues)
- **Email**: [Your contact email]
- **Documentation**: Check the code comments in `main.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Institut Teknologi Sepuluh Nopember (ITS) for educational support
- SOC community for best practices and feedback
- TensorFlow and Streamlit communities for excellent tools

---

⭐ **Star this repository if it helped you detect DGA domains effectively!** ⭐
