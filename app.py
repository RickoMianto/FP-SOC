"""
Streamlit Dashboard for DGA Analysis
AIOps SOC - Real-time Domain Analysis Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import io
import base64

from main import DGAAnalyzer
from config import *

# Page configuration
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout=STREAMLIT_CONFIG['layout'],
    initial_sidebar_state=STREAMLIT_CONFIG['sidebar_state']
)

# Custom CSS
def load_css():
    with open(os.path.join(STATIC_PATH, 'style.css'), 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = DGAAnalyzer()
    st.session_state.analyzer.load_model()

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def main():
    """
    Main Streamlit application
    """
    # try:
    #     load_css()
    # except:
    #     pass  # CSS file optional
    
    # Header
    st.title("🛡️ DGA Analysis Dashboard")
    st.markdown("**AIOps SOC - Real-time Domain Generation Algorithm Detection**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Control Panel")
        
        # Model status
        if st.session_state.analyzer.is_trained:
            st.success("✅ Model Ready")
        else:
            st.error("❌ Model Not Loaded")
        
        st.markdown("---")
        
        # Analysis mode selection
        analysis_mode = st.selectbox(
            "Select Analysis Mode",
            ["Single Domain", "Batch Analysis", "Real-time Monitor", "Model Training"]
        )
        
        st.markdown("---")
        
        # Alert settings
        st.subheader("🚨 Alert Settings")
        high_threshold = st.slider("High Risk Threshold", 0.5, 1.0, 
                                 ALERT_CONFIG['high_risk_threshold'], 0.05)
        medium_threshold = st.slider("Medium Risk Threshold", 0.3, 0.8, 
                                   ALERT_CONFIG['medium_risk_threshold'], 0.05)
        
        # Update alert config
        ALERT_CONFIG['high_risk_threshold'] = high_threshold
        ALERT_CONFIG['medium_risk_threshold'] = medium_threshold
    
    # Main content based on mode
    if analysis_mode == "Single Domain":
        single_domain_analysis()
    elif analysis_mode == "Batch Analysis":
        batch_analysis()
    elif analysis_mode == "Real-time Monitor":
        realtime_monitor()
    elif analysis_mode == "Model Training":
        model_training()
    
    # Footer
    st.markdown("---")
    st.markdown("*DGA Analysis System powered by Deep Learning | AIOps SOC*")

def single_domain_analysis():
    """
    Single domain analysis interface
    """
    st.header("🔍 Single Domain Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        domain_input = st.text_input("Enter domain to analyze:", placeholder="example.com")
        
        if st.button("🔍 Analyze Domain", type="primary"):
            if domain_input and st.session_state.analyzer.is_trained:
                with st.spinner("Analyzing domain..."):
                    try:
                        results = st.session_state.analyzer.predict([domain_input])
                        result = results[0]
                        
                        # Add to history
                        result['timestamp'] = datetime.now()
                        st.session_state.analysis_history.append(result)
                        
                        # Display results
                        display_single_result(result)
                        
                    except Exception as e:
                        st.error(f"Error analyzing domain: {e}")
            elif not domain_input:
                st.warning("Please enter a domain to analyze")
            else:
                st.error("Model not loaded. Please check model training.")
    
    with col2:
        st.subheader("📊 Quick Stats")
        if st.session_state.analysis_history:
            recent_analyses = st.session_state.analysis_history[-10:]
            dga_count = sum(1 for r in recent_analyses if r['is_dga'])
            
            st.metric("Recent Analyses", len(recent_analyses))
            st.metric("DGA Detected", dga_count)
            st.metric("Detection Rate", f"{(dga_count/len(recent_analyses)*100):.1f}%")
        else:
            st.info("No analyses performed yet")

def display_single_result(result):
    """
    Display single domain analysis result
    """
    st.subheader("📊 Analysis Results")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🚨" if result['is_dga'] else "✅"
        status_text = "DGA DETECTED" if result['is_dga'] else "LEGITIMATE"
        st.metric("Status", f"{status_color} {status_text}")
    
    with col2:
        st.metric("DGA Probability", f"{result['dga_probability']:.4f}")
    
    with col3:
        risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        st.metric("Risk Level", f"{risk_color[result['risk_level']]} {result['risk_level']}")
    
    with col4:
        st.metric("Classification", result['predicted_class'])
    
    # Alert if high risk
    if result['risk_level'] == 'HIGH':
        st.error("🚨 **HIGH RISK ALERT** - This domain shows strong indicators of being DGA-generated!")
    elif result['risk_level'] == 'MEDIUM':
        st.warning("⚠️ **MEDIUM RISK** - This domain should be investigated further.")
    else:
        st.success("✅ **LOW RISK** - This domain appears to be legitimate.")
    
    # Detailed analysis
    with st.expander("🔬 Detailed Analysis"):
        st.write(f"**Domain:** {result['domain']}")
        st.write(f"**Analysis Timestamp:** {result.get('timestamp', 'N/A')}")
        st.write(f"**Confidence Score:** {result['dga_probability']:.6f}")
        
        # Probability gauge
        fig = create_probability_gauge(result['dga_probability'])
        st.plotly_chart(fig, use_container_width=True)

def batch_analysis():
    """
    Batch domain analysis interface
    """
    st.header("📂 Batch Domain Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload domain list (CSV or TXT)",
        type=['csv', 'txt'],
        help="CSV: should have 'domain' column. TXT: one domain per line."
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if 'domain' not in df.columns:
                    st.error("CSV file must contain a 'domain' column")
                    return
                domains = df['domain'].tolist()
            else:  # TXT file
                content = str(uploaded_file.read(), "utf-8")
                domains = [line.strip() for line in content.split('\n') if line.strip()]
            
            st.success(f"✅ Loaded {len(domains)} domains for analysis")
            
            # Analysis button
            if st.button("🚀 Start Batch Analysis", type="primary"):
                if st.session_state.analyzer.is_trained:
                    perform_batch_analysis(domains)
                else:
                    st.error("Model not loaded. Please check model training.")
        
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    # Display previous batch results
    if st.session_state.analysis_history:
        display_batch_history()

def perform_batch_analysis(domains):
    """
    Perform batch analysis on domains
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    batch_size = 50  # Process in batches
    
    for i in range(0, len(domains), batch_size):
        batch = domains[i:i+batch_size]
        status_text.text(f"Processing batch {i//batch_size + 1}/{(len(domains)-1)//batch_size + 1}")
        
        try:
            batch_results = st.session_state.analyzer.predict(batch)
            results.extend(batch_results)
            
            # Update progress
            progress = min((i + batch_size) / len(domains), 1.0)
            progress_bar.progress(progress)
            
        except Exception as e:
            st.error(f"Error processing batch: {e}")
            break
    
    status_text.text("Analysis complete!")
    
    if results:
        # Create results DataFrame
        df_results = pd.DataFrame(results)
        df_results['timestamp'] = datetime.now()
        
        # Display summary
        display_batch_results(df_results)
        
        # Add to history
        st.session_state.analysis_history.extend(results)

def display_batch_results(df_results):
    """
    Display batch analysis results
    """
    st.subheader("📊 Batch Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_domains = len(df_results)
    dga_count = df_results['is_dga'].sum()
    high_risk = (df_results['risk_level'] == 'HIGH').sum()
    avg_probability = df_results['dga_probability'].mean()
    
    with col1:
        st.metric("Total Domains", total_domains)
    with col2:
        st.metric("DGA Detected", dga_count)
    with col3:
        st.metric("High Risk", high_risk)
    with col4:
        st.metric("Avg DGA Probability", f"{avg_probability:.3f}")
    
    # Alerts
    if high_risk > ALERT_CONFIG['batch_alert_count']:
        st.error(f"🚨 **BATCH ALERT**: {high_risk} high-risk domains detected!")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # DGA Distribution Pie Chart
        fig_pie = px.pie(df_results, names='predicted_class', 
                        title='DGA vs Legitimate Distribution',
                        color_discrete_map={'dga': '#ff4444', 'legitimate': '#44ff44'})
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Risk Level Bar Chart
        risk_counts = df_results['risk_level'].value_counts()
        fig_bar = px.bar(x=risk_counts.index, y=risk_counts.values,
                        title='Risk Level Distribution',
                        color=risk_counts.index,
                        color_discrete_map={'HIGH': '#ff0000', 'MEDIUM': '#ffaa00', 'LOW': '#00ff00'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Probability distribution
    fig_hist = px.histogram(df_results, x='dga_probability', 
                           title='DGA Probability Distribution',
                           bins=30, color='predicted_class')
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Top high-risk domains
    high_risk_domains = df_results[df_results['risk_level'] == 'HIGH'].sort_values(
        'dga_probability', ascending=False).head(10)
    
    if not high_risk_domains.empty:
        st.subheader("🚨 Top High-Risk Domains")
        st.dataframe(high_risk_domains[['domain', 'dga_probability', 'risk_level']], 
                    use_container_width=True)
    
    # Download results
    csv = df_results.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name=f"dga_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def realtime_monitor():
    """
    Real-time monitoring interface
    """
    st.header("📡 Real-time DGA Monitor")
    
    # Monitor controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        monitor_active = st.checkbox("🔄 Enable Monitoring")
    with col2:
        refresh_interval = st.slider("Refresh Interval (sec)", 1, 60, 5)
    with col3:
        max_domains = st.number_input("Max Domains to Show", 10, 1000, 100)
    
    # Monitor display area
    monitor_container = st.container()
    
    if monitor_active:
        # Simulate real-time monitoring
        placeholder = st.empty()
        
        while monitor_active:
            with placeholder.container():
                # Generate sample real-time data (in real implementation, this would come from network logs)
                sample_domains = generate_sample_domains(10)
                
                if st.session_state.analyzer.is_trained:
                    try:
                        results = st.session_state.analyzer.predict(sample_domains)
                        
                        # Add timestamps
                        for result in results:
                            result['timestamp'] = datetime.now()
                        
                        # Display real-time results
                        display_realtime_results(results)
                        
                    except Exception as e:
                        st.error(f"Error in real-time analysis: {e}")
                
                time.sleep(refresh_interval)
    else:
        st.info("🔄 Click 'Enable Monitoring' to start real-time DGA detection")

def display_realtime_results(results):
    """
    Display real-time monitoring results
    """
    st.subheader("📊 Real-time Analysis Feed")
    
    # Recent detections
    df_realtime = pd.DataFrame(results)
    
    # Alerts for high-risk domains
    high_risk_domains = df_realtime[df_realtime['risk_level'] == 'HIGH']
    if not high_risk_domains.empty:
        st.error(f"🚨 {len(high_risk_domains)} HIGH RISK domains detected!")
        for _, domain in high_risk_domains.iterrows():
            st.error(f"🚨 {domain['domain']} - Probability: {domain['dga_probability']:.4f}")
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Domains Analyzed", len(results))
    with col2:
        dga_count = sum(1 for r in results if r['is_dga'])
        st.metric("DGA Detected", dga_count)
    with col3:
        high_risk_count = sum(1 for r in results if r['risk_level'] == 'HIGH')
        st.metric("High Risk", high_risk_count)
    with col4:
        avg_prob = np.mean([r['dga_probability'] for r in results])
        st.metric("Avg Probability", f"{avg_prob:.3f}")
    
    # Real-time data table
    st.dataframe(df_realtime[['domain', 'dga_probability', 'risk_level', 'timestamp']], 
                use_container_width=True)

def model_training():
    """
    Model training interface
    """
    st.header("🤖 Model Training")
    
    # Training data upload
    st.subheader("📤 Upload Training Data")
    uploaded_file = st.file_uploader(
        "Upload training dataset (CSV)",
        type=['csv'],
        help="CSV file should contain columns: isDGA, domain, host, subclass"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_columns = ['isDGA', 'domain']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return
            
            st.success(f"✅ Training data loaded: {len(df)} samples")
            
            # Data overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Data Overview")
                st.write(f"**Total samples:** {len(df)}")
                st.write(f"**DGA samples:** {(df['isDGA'] == 'dga').sum()}")
                st.write(f"**Legitimate samples:** {(df['isDGA'] != 'dga').sum()}")
            
            with col2:
                # Class distribution
                class_counts = df['isDGA'].value_counts()
                fig = px.pie(values=class_counts.values, names=class_counts.index,
                           title="Class Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Training parameters
            st.subheader("⚙️ Training Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                epochs = st.slider("Epochs", 10, 100, MODEL_CONFIG['epochs'])
                batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], 
                                        index=2 if MODEL_CONFIG['batch_size'] == 128 else 0)
            
            with col2:
                learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 
                                        MODEL_CONFIG['learning_rate'], format="%.4f")
                validation_split = st.slider("Validation Split", 0.1, 0.3, 
                                           MODEL_CONFIG['validation_split'])
            
            with col3:
                dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 
                                       MODEL_CONFIG['dropout_rate'])
            
            # Update config
            MODEL_CONFIG.update({
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'validation_split': validation_split,
                'dropout_rate': dropout_rate
            })
            
            # Start training
            if st.button("🚀 Start Training", type="primary"):
                start_model_training(df)
        
        except Exception as e:
            st.error(f"Error loading training data: {e}")

def start_model_training(df):
    """
    Start model training process
    """
    st.subheader("🔄 Training in Progress...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize new analyzer
        analyzer = DGAAnalyzer()
        
        status_text.text("Preparing training data...")
        progress_bar.progress(0.1)
        
        # Custom training with progress updates
        status_text.text("Training model...")
        progress_bar.progress(0.3)
        
        # Train the model
        history = analyzer.train(df)
        
        status_text.text("Saving model...")
        progress_bar.progress(0.9)
        
        # Save model
        analyzer.save_model()
        
        # Update session state
        st.session_state.analyzer = analyzer
        
        progress_bar.progress(1.0)
        status_text.text("Training completed successfully!")
        
        st.success("✅ Model training completed and saved!")
        
        # Display training history
        display_training_history(history)
        
    except Exception as e:
        st.error(f"Training failed: {e}")

def display_training_history(history):
    """
    Display training history and metrics
    """
    st.subheader("📈 Training History")
    
    # Training curves
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Accuracy', 'Model Loss')
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(y=history.history['accuracy'], name='Training Accuracy'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy'),
        row=1, col=1
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(y=history.history['loss'], name='Training Loss'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=history.history['val_loss'], name='Validation Loss'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def create_probability_gauge(probability):
    """
    Create probability gauge chart
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "DGA Probability"},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.6], 'color': "lightgray"},
                {'range': [0.6, 0.8], 'color': "yellow"},
                {'range': [0.8, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def generate_sample_domains(count=10):
    """
    Generate sample domains for real-time monitoring demo
    """
    import random
    import string
    
    domains = []
    
    # Generate mix of legitimate and DGA-like domains
    legitimate_domains = [
        "google.com", "facebook.com", "twitter.com", "youtube.com", 
        "amazon.com", "microsoft.com", "apple.com", "github.com"
    ]
    
    for _ in range(count):
        if random.random() < 0.7:  # 70% legitimate
            domain = random.choice(legitimate_domains)
        else:  # 30% DGA-like
            length = random.randint(8, 20)
            domain = ''.join(random.choices(string.ascii_lowercase + string.digits, 
                                          k=length)) + ".com"
        
        domains.append(domain)
    
    return domains

def display_batch_history():
    """
    Display history of batch analyses
    """
    if st.session_state.analysis_history:
        st.subheader("📋 Analysis History")
        
        # Recent analyses summary
        recent_count = min(100, len(st.session_state.analysis_history))
        recent_analyses = st.session_state.analysis_history[-recent_count:]
        
        df_history = pd.DataFrame(recent_analyses)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Analyses", len(st.session_state.analysis_history))
        with col2:
            dga_count = df_history['is_dga'].sum()
            st.metric("DGA Detected", dga_count)
        with col3:
            detection_rate = (dga_count / len(df_history) * 100) if len(df_history) > 0 else 0
            st.metric("Detection Rate", f"{detection_rate:.1f}%")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.success("History cleared!")

if __name__ == "__main__":
    main()