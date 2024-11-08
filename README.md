# Movie Recommendation System

An end-to-end machine learning pipeline for personalized movie recommendations using the MovieLens 20M Dataset. This project combines collaborative filtering, content-based filtering, and hybrid systems to provide tailored movie suggestions.

## Motivation

This project is part of my ML Engineering portfolio, showcasing skills in data preprocessing, model building, evaluation, hyperparameter tuning, and deployment in a production-ready environment.

## Project Goals

- Build a recommendation system using collaborative, content-based, and hybrid filtering methods.
- Implement an evaluation pipeline with RMSE and Precision@5 as primary metrics.
- Perform hyperparameter tuning for optimized model selection.
- Deploy the system as a web application.

## Dataset

The project uses the [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/), which contains 20 million ratings for 27,000 movies from 138,000 users.

## Models

### Collaborative Filtering

A model using collaborative filtering with k-Nearest Neighbors to find and recommend movies based on user-item interactions. 

#### Hyperparameter Tuning Results

**Collaborative Filtering Results**

- Metric: Cosine
- n_neighbors: 6

| Metric     | n_neighbors | RMSE     | Precision@5 |
|------------|-------------|----------|--------------|
| Cosine     | 5           | 0.963661 | 0.220       |
| Cosine     | 10          | 0.944672 | 0.132       |
| Cosine     | 20          | 0.913900 | 0.140       |
| Cosine     | 30          | 0.915411 | 0.100       |
| Cosine     | 50          | 0.900560 | 0.160       |
| Euclidean  | 5           | 0.924139 | 0.020       |
| Euclidean  | 10          | 0.926824 | 0.020       |
| Euclidean  | 20          | 0.970004 | 0.020       |
| Euclidean  | 30          | 0.958882 | 0.028       |
| Euclidean  | 50          | 0.975207 | 0.040       |

### Content-Based Filtering

A model using cosine similarity and euclidean distance metrics for content-based recommendations based on movie genres.

### Hybrid System

Combines collaborative filtering with content-based filtering for a more personalized recommendation experience.

#### Example Results

**Metric**: Euclidean
**RMSE**: 1.3071
**Precision@5**: 0.0120

## Setup

- Clone this repository.
- Create and activate a virtual environment.
- Install the dependencies with `pip install -r requirements.txt`.

## Files and Directories

- `src/`: Source code for model training, evaluation, and serving.
- `data/processed`: Processed dataset files.
- `models/`: Saved models and model artifacts.
- `templates/`: HTML templates for the web application.
- `app.py`: Flask application entry point.
- `Dockerfile`: Docker setup for containerized deployment.
- `requirements.txt`: List of required Python packages.

## Requirements

- Python 3.9 or higher
- pandas
- numpy
- scikit-learn
- scipy
- flask
- gunicorn
- surprise
- sqlalchemy
- joblib

## Docker Setup

Build and run the application using Docker:

```bash
docker build -t movie-recommendation-system .
docker run -p 5000:5000 movie-recommendation-system