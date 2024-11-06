import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
import os
import joblib

# Import get_content_recommendations from content_based_filter.py
from src.models.content_based_filter import get_content_recommendations

# Define paths
processed_data_path = os.path.join('data', 'processed')
model_save_path = os.path.join('models')
movies = pd.read_csv(os.path.join(processed_data_path, 'movies_content.csv'))

# Load the cleaned data
data = pd.read_csv(os.path.join(processed_data_path, 'ratings_movies_cleaned.csv'))

# Load the cosine similarity matrix and indices
cosine_sim = joblib.load(os.path.join(model_save_path, 'cosine_sim.pkl'))
indices = joblib.load(os.path.join(model_save_path, 'indices.pkl'))

# Map userId and movieId to indices
user_ids = data['userId'].unique()
movie_ids = data['movieId'].unique()
user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

# Add index columns to the data
data['user_index'] = data['userId'].map(user_id_to_index)
data['movie_index'] = data['movieId'].map(movie_id_to_index)

# Create a smaller dataset for evaluation if needed
data = data.sample(n=100000, random_state=42)

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Function to create sparse matrix
def create_sparse_matrix(df):
    row = df['user_index'].values
    col = df['movie_index'].values
    values = df['rating'].values
    return coo_matrix((values, (row, col)), shape=(len(user_ids), len(movie_ids))).tocsr()

# Create training and test sparse matrices
train_sparse = create_sparse_matrix(train_data)
test_sparse = create_sparse_matrix(test_data)

# Build the kNN model
item_user_sparse = train_sparse.T  # Transpose to get item-user matrix
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(item_user_sparse)

def predict_rating(user_index, movie_index, model, train_sparse, k=20):
    # Use the transposed matrix
    item_user_sparse = train_sparse.T

    # Find k similar movies
    distances, indices = model.kneighbors(item_user_sparse[movie_index], n_neighbors=k+1)
    
    # Exclude the input movie itself
    similar_movie_indices = indices.flatten()[1:]
    
    # Get ratings the user gave to these similar movies
    user_ratings = train_sparse[user_index, similar_movie_indices].toarray().flatten()
    
    # Filter out zero ratings
    non_zero_ratings = user_ratings[user_ratings > 0]
    
    if len(non_zero_ratings) == 0:
        # Return the user's average rating if no ratings are found
        user_mean_rating = train_data[train_data['user_index'] == user_index]['rating'].mean()
        return user_mean_rating if not np.isnan(user_mean_rating) else train_data['rating'].mean()
    else:
        # Return the average rating for similar movies rated by the user
        return non_zero_ratings.mean()

# Initialize lists to store actual and predicted ratings
actual_ratings = []
predicted_ratings = []

# Limit the number of predictions for time considerations
n_predictions = 10000  # Adjust as needed

for idx in range(n_predictions):
    user_idx = test_data['user_index'].iloc[idx]
    movie_idx = test_data['movie_index'].iloc[idx]
    actual_rating = test_data['rating'].iloc[idx]
    
    pred_rating = predict_rating(user_idx, movie_idx, model_knn, train_sparse)
    
    actual_ratings.append(actual_rating)
    predicted_ratings.append(pred_rating)

# Calculate RMSE
rmse = sqrt(mean_squared_error(actual_ratings, predicted_ratings))
print(f"Collaborative Filtering RMSE: {rmse}")

def get_user_recommendations(user_id, n_recommendations=10):
    # Get movies the user has already rated
    user_movies = data[data['userId'] == user_id]['title'].tolist()
    
    # Choose one movie the user has rated
    if len(user_movies) == 0:
        return []
    seed_movie = user_movies[0]
    
    # Get content-based recommendations
    recommendations = get_content_recommendations(
        title=seed_movie,
        cosine_sim=cosine_sim,
        df=movies,
        indices=indices,
        n_recommendations=n_recommendations
    )
    rec_titles = recommendations['title'].tolist()
    
    return rec_titles

def precision_at_k(user_id, k=5):
    # Get the user's relevant items (movies rated 4 or higher)
    relevant_items = test_data[(test_data['userId'] == user_id) & (test_data['rating'] >= 4)]['title'].tolist()
    
    # Get recommendations
    recommended_items = get_user_recommendations(user_id, n_recommendations=k)
    
    if not recommended_items:
        return 0
    
    # Calculate precision
    relevant_and_recommended = set(relevant_items).intersection(set(recommended_items))
    precision = len(relevant_and_recommended) / k
    
    return precision

# Calculate average Precision@K for multiple users
user_ids_sample = test_data['userId'].unique()[:100]  # Sample of users
precision_scores = []

for user_id in user_ids_sample:
    precision = precision_at_k(user_id, k=5)
    precision_scores.append(precision)

average_precision = np.mean(precision_scores)
print(f"Content-Based Filtering Average Precision@5: {average_precision}")