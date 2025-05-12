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


# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">Model Comparison Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for model loading
    with st.sidebar:
        st.markdown('<h2 class="sub-header">Model Configuration</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Load your trained models")
        
        # Model 1
        model1_file = st.file_uploader("Upload Model 1 (.pkl or .joblib)", type=["pkl", "joblib"], key="model1")
        model1_name = st.text_input("Model 1 Name", "Model 1")
        
        # Model 2
        model2_file = st.file_uploader("Upload Model 2 (.pkl or .joblib)", type=["pkl", "joblib"], key="model2")
        model2_name = st.text_input("Model 2 Name", "Model 2")
        
        # Model 3
        model3_file = st.file_uploader("Upload Model 3 (.pkl or .joblib)", type=["pkl", "joblib"], key="model3")
        model3_name = st.text_input("Model 3 Name", "Model 3")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional settings
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Display Settings")
        show_detailed_metrics = st.checkbox("Show Detailed Metrics", True)
        plot_size = st.slider("Plot Size", min_value=1, max_value=10, value=6)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Information
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.info("Upload your dataset in CSV or TXT format. The last column is assumed to be the target variable.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Upload Your Dataset</h3>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader("Upload dataset (CSV or TXT)", type=["csv", "txt"])
        
        # File format settings
        if uploaded_file is not None:
            file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": f"{uploaded_file.size / 1024:.2f} KB"}
            st.write(file_details)
            
            # File parsing options
            st.markdown("#### File Parsing Options")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                delimiter = st.selectbox("Delimiter", [",", ";", "\t", " ", "|"], index=0)
            
            with col_b:
                header = st.selectbox("Header", ["Yes", "No"], index=0)
                has_header = header == "Yes"
            
            with col_c:
                index_col = st.checkbox("First column as index", False)
                
            try:
                # Read data based on file extension
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, delimiter=delimiter, header=0 if has_header else None, 
                                   index_col=0 if index_col else None)
                else:  # txt file
                    df = pd.read_csv(uploaded_file, delimiter=delimiter, header=0 if has_header else None, 
                                   index_col=0 if index_col else None)
                
                # Display dataset
                st.markdown("#### Dataset Preview")
                st.dataframe(df.head(5))
                st.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Data validation alert
                if df.isnull().sum().sum() > 0:
                    st.warning(f"Warning: Dataset contains {df.isnull().sum().sum()} missing values.")
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Quick Statistics</h3>', unsafe_allow_html=True)
        
        if 'df' in locals():
            # Basic statistics
            st.markdown("#### Numerical Columns")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe())
            else:
                st.write("No numerical columns found.")
            
            # Target variable distribution
            if len(df.columns) > 0:
                st.markdown("#### Target Variable Distribution")
                try:
                    target = df.iloc[:, -1]
                    if target.nunique() < 10:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        target.value_counts().plot(kind='bar', ax=ax)
                        plt.title("Target Distribution")
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.write("Target has too many unique values for visualization.")
                except Exception as e:
                    st.write(f"Could not visualize target: {e}")
        else:
            st.write("Upload a dataset to see statistics.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model comparison section
    st.markdown('<h2 class="sub-header">Model Comparison</h2>', unsafe_allow_html=True)
    
    # Check if all required components are ready
    models_ready = False
    
    # Load models if files are uploaded
    models = {}
    try:
        if model1_file is not None:
            models[model1_name] = joblib.load(model1_file)
        
        if model2_file is not None:
            models[model2_name] = joblib.load(model2_file)
            
        if model3_file is not None:
            models[model3_name] = joblib.load(model3_file)
        
        if len(models) > 0:
            models_ready = True
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_btn = st.button("Run Comparison Analysis", use_container_width=True)
    
    # Run analysis if button is clicked and all prerequisites are met
    if process_btn and 'df' in locals() and models_ready:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        with st.spinner('Processing data and generating predictions...'):
            try:
                # Preprocess data
                X, y_true = preprocess_data(df)
                
                # Get predictions from all models
                predictions = get_model_predictions(X, models)
                
                # Calculate metrics and display results
                st.markdown('<h3 class="result-header">Analysis Results</h3>', unsafe_allow_html=True)
                
                # Create tabs for different types of results
                tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Confusion Matrices", "Detailed Comparison"])
                
                with tab1:
                    # Performance metrics comparison
                    metrics_data = []
                    
                    for model_name, y_pred in predictions.items():
                        metrics = calculate_metrics(y_true, y_pred)
                        metrics_data.append({
                            'Model': model_name,
                            'Accuracy': f"{metrics['accuracy']:.4f}"
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    # Display metrics as a stylish table
                    st.markdown("### Model Performance Summary")
                    
                    # Find best model based on accuracy
                    metrics_df['Accuracy'] = metrics_df['Accuracy'].astype(float)
                    best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax()]['Model']
                    
                    # Create styled display
                    for _, row in metrics_df.iterrows():
                        model = row['Model']
                        acc = float(row['Accuracy'])
                        
                        is_best = model == best_model
                        
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 5px solid {'#4CAF50' if is_best else '#1E88E5'}">
                            <h4>{model} {' 🏆 Best Model' if is_best else ''}</h4>
                            <p>Accuracy: <b>{acc:.4f}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualize metrics
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bar_colors = ['#4CAF50' if model == best_model else '#1E88E5' for model in metrics_df['Model']]
                        ax.bar(metrics_df['Model'], metrics_df['Accuracy'].astype(float), color=bar_colors)
                        ax.set_ylim(0, 1)
                        ax.set_ylabel('Accuracy Score')
                        ax.set_title('Model Accuracy Comparison')
                        
                        for i, v in enumerate(metrics_df['Accuracy'].astype(float)):
                            ax.text(i, v + 0.01, f'{v:.4f}', ha='center')
                            
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("""
                        ### Key Observations
                        
                        - The model with the highest accuracy is highlighted in green
                        - Higher bars indicate better performance
                        - Accuracy ranges from 0 to 1
                        """)
                
                with tab2:
                    # Confusion matrices
                    st.markdown("### Confusion Matrices")
                    
                    conf_cols = st.columns(len(predictions))
                    
                    for i, (model_name, y_pred) in enumerate(predictions.items()):
                        with conf_cols[i]:
                            st.markdown(f"#### {model_name}")
                            
                            # Generate confusion matrix
                            cm_buf = plot_confusion_matrix(y_true, y_pred, model_name)
                            
                            # Display the confusion matrix
                            st.image(cm_buf, caption=f"Confusion Matrix - {model_name}")
                            
                            # Download link
                            st.markdown(get_download_link(cm_buf, f"{model_name}_cm"), unsafe_allow_html=True)
                
                with tab3:
                    # Detailed comparison
                    if show_detailed_metrics:
                        st.markdown("### Detailed Performance Analysis")
                        
                        # Sample-by-sample comparison for a subset
                        sample_size = min(10, len(y_true))
                        
                        comparison_data = {
                            'True Value': y_true[:sample_size]
                        }
                        
                        for model_name, y_pred in predictions.items():
                            comparison_data[f'{model_name} Prediction'] = y_pred[:sample_size]
                            comparison_data[f'{model_name} Correct'] = y_pred[:sample_size] == y_true[:sample_size]
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Additional insights
                        st.markdown("### Model Agreement Analysis")
                        
                        if len(models) >= 2:
                            # Calculate agreement between models
                            agreement_matrix = np.zeros((len(models), len(models)))
                            model_names = list(predictions.keys())
                            
                            for i, model1 in enumerate(model_names):
                                for j, model2 in enumerate(model_names):
                                    if i <= j:
                                        agreement = np.mean(predictions[model1] == predictions[model2])
                                        agreement_matrix[i, j] = agreement
                                        agreement_matrix[j, i] = agreement
                            
                            # Plot agreement matrix
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(agreement_matrix, annot=True, fmt='.2f', cmap='YlGnBu',
                                        xticklabels=model_names, yticklabels=model_names)
                            plt.title('Model Prediction Agreement')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Find samples where all models agree/disagree
                            all_agree = np.all([predictions[m] == predictions[model_names[0]] 
                                                for m in model_names], axis=0)
                            
                            st.write(f"All models agree on {np.sum(all_agree)} out of {len(y_true)} samples ({np.mean(all_agree)*100:.1f}%)")
                            
                            # Check if the models agree on correct predictions
                            correct_agreements = np.logical_and(all_agree, predictions[model_names[0]] == y_true)
                            st.write(f"All models correctly predict {np.sum(correct_agreements)} samples ({np.mean(correct_agreements)*100:.1f}%)")
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.code(str(e))
        
        st.markdown('</div>', unsafe_allow_html=True)
    elif process_btn:
        if 'df' not in locals():
            st.warning("Please upload a dataset before running the analysis.")
        if not models_ready:
            st.warning("Please upload at least one model before running the analysis.")
    # Instructions section
    with st.expander("How to Use This Dashboard"):
        st.markdown("""
        ### Instructions
        
        1. **Upload your models**: Use the sidebar to upload your trained ML models (.pkl or .joblib files)
        2. **Upload your dataset**: Upload a CSV or TXT file containing your test data
        3. **Configure parsing options**: Set delimiter and header options as needed
        4. **Run comparison**: Click the "Run Comparison Analysis" button to process the data
        5. **Explore results**: View performance metrics, confusion matrices, and detailed comparisons
        
        ### Notes
        
        - Your dataset should be preprocessed the same way as your training data
        - The last column in your dataset is assumed to be the target variable
        - You can upload 1-3 models for comparison
        """)
    
    # Footer
    st.markdown('<div class="footer">Made by Ritwik Mittal & Yashas Raina<b><b></div>', 
                unsafe_allow_html=True)