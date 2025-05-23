# ML Models Comparative Analysis Dashboard

This project provides a Streamlit-based web application for comparing the performance of different machine learning models. Users can upload their trained models (in `.pkl` or `.joblib` format) and a test dataset (CSV or TXT) to visualize and compare model metrics, confusion matrices, and detailed prediction agreements.

## Features

-   **Dynamic Model Loading:** Upload up to three pre-trained machine learning models.
-   **Flexible Dataset Upload:** Supports CSV and TXT file formats for test data.
-   **Customizable Parsing Options:** Configure delimiter, header presence, and index column for the uploaded dataset.
-   **Interactive Dashboard:**
    -   Dataset preview and quick statistics (numerical summaries, target variable distribution).
    -   Model performance comparison (e.g., accuracy) with visual bar charts and highlighted best model.
    -   Individual confusion matrices for each model, with download option.
    -   Detailed sample-by-sample prediction comparison.
    -   Model agreement analysis heatmap to see how often models concur.
-   **User-Friendly Interface:** Intuitive sidebar for configurations and main area for results, styled with custom CSS.

## Project Structure

![alt text](image.png)

## Getting Started

### Prerequisites

-   Python 3.7+
-   Pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install required Python packages:**
    ```bash
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib
    ```

### Running the Application

1.  **Ensure you have pre-trained models:**
    * You can use the `ml_models.py` script to train and save example models (KNN, Random Forest, SVM) using the provided `train_data.txt`. Running this script will generate `.pkl` files (e.g., `knn_parkinson.pkl`, `rf_parkinson.pkl`, `SVM_parkinson.pkl`).
    ```bash
    python ml_models.py
    ```
    * Alternatively, use your own pre-trained models saved in `.pkl` or `.joblib` format.

2.  **Launch the Streamlit dashboard:**
    ```bash
    streamlit run frontend.py
    ```
    This will open the dashboard in your default web browser.

## How to Use the Dashboard

1.  **Upload Your Models:**
    * On the sidebar, use the file uploaders to select your trained model files (`.pkl` or `.joblib`).
    * Optionally, provide a custom name for each model.

2.  **Upload Your Dataset:**
    * In the main section, upload your test dataset (CSV or TXT format).
    * The application assumes the last column of your dataset is the target variable.

3.  **Configure Parsing Options:**
    * Select the appropriate delimiter for your dataset (e.g., comma, semicolon, tab).
    * Indicate if your dataset has a header row.
    * Specify if the first column should be used as the index.

4.  **Run Comparison Analysis:**
    * Once models and data are uploaded, click the "Run Comparison Analysis" button.

5.  **Explore Results:**
    * **Performance Metrics Tab:** View key metrics like accuracy for each model. The best model is highlighted, and a bar chart provides a visual comparison.
    * **Confusion Matrices Tab:** Examine the confusion matrix for each model. You can download the plot for each matrix.
    * **Detailed Comparison Tab:**
        * See a sample-by-sample comparison of true values versus predictions from each model.
        * If "Show Detailed Metrics" is enabled in the sidebar and multiple models are loaded, a model agreement heatmap is displayed, showing the percentage of predictions on which pairs of models agree.
        * Statistics on overall model agreement and correct agreements are also provided.

## Code Overview

### `frontend.py`

-   Handles the Streamlit user interface, file uploads, and display of results.
-   Includes helper functions for:
    -   `preprocess_data()`: Basic preprocessing for the input dataset (currently samples 100 rows and scales features; **customize this for your specific data**).
    -   `get_model_predictions()`: Loads uploaded models and generates predictions.
    -   `plot_confusion_matrix()`: Creates confusion matrix plots.
    -   `calculate_metrics()`: Computes performance metrics like accuracy.
-   Uses custom CSS for enhanced visual styling.

### `ml_models.py`

-   A script to demonstrate training and saving three types of machine learning models:
    -   K-Nearest Neighbors (KNN)
    -   Random Forest
    -   Support Vector Machine (SVM)
-   Uses `train_data.txt` for training.
-   Performs basic hyperparameter tuning for KNN and Random Forest.
-   Saves the trained models using `joblib` into `.pkl` files.
-   Generates and saves a correlation matrix heatmap (`Correlation Matrix Test Data.png`).

### Data

-   `train_data.txt`: A plain text file where each row represents a sample and columns are feature values, with the last column being the target variable. The delimiter appears to be a comma.
-   `Correlation Matrix Test Data.jpg`: An image showing a heatmap of feature correlations, likely generated during the model training/exploration phase.

## Customization

-   **Preprocessing:** The `preprocess_data` function in `frontend.py` is a basic example. You **must** adapt this function to match the preprocessing steps used when training your original models (e.g., handling missing values, encoding categorical features, feature scaling specific to your training setup).
-   **Metrics:** Add more performance metrics (e.g., precision, recall, F1-score, ROC AUC) to the `calculate_metrics` function in `frontend.py` and update the display accordingly.
-   **Model Support:** While the dashboard uses `joblib` to load models (common for scikit-learn), you might need to adjust loading mechanisms for models from other libraries (e.g., TensorFlow, PyTorch).
-   **Styling:** Further customize the appearance using the CSS section in `frontend.py`.

## Authors

-   Ritwik Mittal
-   Yashas Raina

## License

This project is open-source. Please refer to the `LICENSE` file if one is included in the repository (otherwise, assume standard open-source licensing or specify one if needed).