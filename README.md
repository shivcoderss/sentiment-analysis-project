# Sentiment Analysis Project

## Overview

This project performs **Sentiment Analysis** on the IMDb movie reviews dataset using two different models:
1. **Logistic Regression** (Traditional Machine Learning model)
2. **LSTM** (Deep Learning model)

The purpose of this project is to compare the performance of traditional machine learning models with deep learning models in the context of text classification.

## Project Workflow

### 1. Data Preprocessing
- The raw movie reviews are cleaned by removing special characters, converting text to lowercase, and eliminating stopwords.
- Reviews are then tokenized for use in the LSTM model, and TF-IDF vectorization is applied for the Logistic Regression model.

### 2. Logistic Regression Model
- A **Logistic Regression** model is implemented using the **TF-IDF** features extracted from the reviews.
- This model is trained and evaluated based on accuracy, precision, recall, and F1-score.

### 3. LSTM Model
- The **LSTM** (Long Short-Term Memory) model is implemented to better capture the sequential nature of the text.
- Reviews are tokenized and padded to fixed-length sequences before being fed into the LSTM network.
- The model consists of an Embedding layer, followed by an LSTM layer and a Dense output layer.

### 4. Model Comparison
- The performance of the **Logistic Regression** and **LSTM** models is compared using the following metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
  
Visualizations are created to compare these metrics and the confusion matrices for each model.

## Technologies Used
- **Python** for scripting and model development.
- **Keras** and **TensorFlow** for deep learning model (LSTM).
- **Scikit-learn** for traditional machine learning (Logistic Regression) and evaluation metrics.
- **Matplotlib** and **Seaborn** for data visualization.
- **NLTK** for text preprocessing (stopword removal, tokenization).

## How to Run the Project
1. Clone the repository from GitHub:
   ```bash
   git clone https://github.com/shivcoderss/sentiment-analysis-project.git
   cd sentiment-analysis-project
