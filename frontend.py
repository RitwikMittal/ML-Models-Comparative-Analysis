import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import os
from sklearn.preprocessing import StandardScaler
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Model Comparison Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A6741;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2F4858;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
    }
    .metric-card {
        background-color: #f0f7f0;
        border-left: 5px solid #4A6741;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: #2F4858;
    }
    .result-header {
        color: #2F4858;
        font-weight: 600;
        margin-bottom: 10px;
        text-align: center;
    }
    .stButton>button {
        background-color: #4A6741;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #385233;
    }
    .highlight {
        background-color: #f0f7f0;
        padding: 5px;
        border-radius: 3px;
        color: #2F4858;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #ffffff;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def preprocess_data(df):
    """Preprocess the input data for model prediction."""
    # This function should be customized based on your specific preprocessing needs
    # Example: Basic preprocessing
    X = df.iloc[:, :-1].sample(n=100,replace = True)  # All columns except the last one (assuming last column is target)
    y = df.iloc[:, -1].sample(n=100,replace = True)  # Last column as target
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def get_model_predictions(X, models):
    """Get predictions from all loaded models."""
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X)
    return predictions

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Generate and plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def get_download_link(buf, filename):
    """Generate a download link for the plot."""
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download {filename} Plot</a>'

def calculate_metrics(y_true, y_pred):
    """Calculate model performance metrics."""
    acc = accuracy_score(y_true, y_pred)
    # Add more metrics as needed
    return {
        'accuracy': acc
    }
