# TruthLens AI

### Intelligent Fake News Detection System

**Developed by Sidharth**

## Project Overview

TruthLens AI is a machine-learning mini-project that classifies news articles as Fake or Real based on patterns learned from a labeled news dataset.

The application uses Natural Language Processing (NLP) and provides an interactive web dashboard.

## Features

- Fake and Real news classification
- Model confidence percentage
- Modern Streamlit dashboard
- Interactive pie and bar charts
- Prediction history
- Prediction statistics
- CSV history download

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit
- Plotly

## Dataset

Kaggle Fake and Real News Dataset (ISOT).

The dataset contains labeled fake and real news articles used for model training and evaluation.

## Model Performance

The model achieved approximately 98.61% accuracy on a held-out random test split.

This score does not guarantee reliable classification of new or current news.

## How to Run

1. Clone or download the repository.
2. Install the dependencies:

   pip install -r requirements.txt

3. Ensure `model.pkl` and `vectorizer.pkl` are available in the project folder.

4. Start the application:

   streamlit run app.py

## Project Limitations

TruthLens AI is a machine-learning classifier, not an independent fact-checking service.

It does not verify claims using live news sources and may make incorrect predictions.

## Developers

Sidharth

## Project Type

B.Tech Artificial Intelligence & Data Science Mini-Project