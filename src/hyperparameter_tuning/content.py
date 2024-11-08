import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import os

# Define paths
processed_data_path = os.path.join('data', 'processed')

# Load datasets
movies = pd.read_csv(os.path.join(processed_data_path, 'movies_content.csv'))
data = pd.read_csv(os.path.join(processed_data_path, 'ratings_movies_cleaned.csv'))

# Optionally filter out users with few ratings to increase density
user_counts = data['userId'].value_counts()
active_users = user_counts[user_counts >= 20].index.tolist()
data = data[data['userId'].isin(active_users)]

# Recompute movie and user IDs after filtering
user_ids = data['userId'].unique()
movie_ids = data['movieId'].unique()

# Map userId and movieId to indices
user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

# Add index columns to the data
data['user_index'] = data['userId'].map(user_id_to_index)
data['movie_index'] = data['movieId'].map(movie_id_to_index)

# Split data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Process movie features (genres)
genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX',
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# Ensure genre columns are present
for genre in genre_cols:
    if genre not in movies.columns:
        movies[genre] = 0  # Assign 0 if genre column is missing

# Create movie feature matrix
movie_features = movies.set_index('movieId')[genre_cols]
movie_features = movie_features.loc[movie_ids]  # Ensure alignment with filtered movie_ids
movie_features = movie_features.fillna(0)

# Function to compute similarity matrix
def compute_similarity_matrix(metric='cosine'):
    if metric == 'cosine':
        similarity = cosine_similarity(movie_features)
    elif metric == 'euclidean':
        similarity = 1 / (1 + euclidean_distances(movie_features))  # Convert distances to similarity
    else:
        raise ValueError("Unsupported similarity metric.")
    return similarity

# Function to build user profiles
def build_user_profiles(train_data):
    user_profiles = {}
    for user_id in train_data['userId'].unique():
        user_ratings = train_data[train_data['userId'] == user_id]
        user_movies = user_ratings['movieId'].tolist()
        user_movie_indices = [movie_id_to_index[m_id] for m_id in user_movies]
        user_ratings_values = user_ratings['rating'].values.reshape(-1, 1)

        # Get movie features for the movies the user has rated
        user_movie_features = movie_features.iloc[user_movie_indices]

        # Create user profile by weighted average of movie features
        user_profile = np.average(user_movie_features, axis=0, weights=user_ratings_values.flatten())
        user_profiles[user_id] = user_profile
    return user_profiles

# Define the evaluation functions

def predict_rating(user_id, movie_index, user_profiles, metric='cosine'):
    user_profile = user_profiles.get(user_id)
    if user_profile is None:
        # If user profile is not found, return global average rating
        return train_data['rating'].mean()

    # Get the movie features
    movie_feature = movie_features.iloc[movie_index]

    # Compute similarity between user profile and movie feature
    if metric == 'cosine':
        similarity = cosine_similarity(user_profile.reshape(1, -1), movie_feature.values.reshape(1, -1))
        similarity_score = similarity.flatten()[0]
    elif metric == 'euclidean':
        distance = euclidean_distances(user_profile.reshape(1, -1), movie_feature.values.reshape(1, -1))
        similarity_score = 1 / (1 + distance.flatten()[0])
    else:
        raise ValueError("Unsupported similarity metric.")

    # Predict rating as normalized similarity score scaled to rating scale (1-5)
    predicted_rating = similarity_score * 4 + 1  # Scale similarity score to rating scale
    return predicted_rating

def get_user_recommendations(user_id, user_profiles, n_recommendations=5, metric='cosine'):
    user_profile = user_profiles.get(user_id)
    if user_profile is None:
        return []

    # Compute similarity between user profile and all movies
    if metric == 'cosine':
        similarities = cosine_similarity(user_profile.reshape(1, -1), movie_features).flatten()
    elif metric == 'euclidean':
        distances = euclidean_distances(user_profile.reshape(1, -1), movie_features).flatten()
        similarities = 1 / (1 + distances)
    else:
        raise ValueError("Unsupported similarity metric.")

    # Get movies the user has already rated
    user_rated_movie_indices = train_data[train_data['userId'] == user_id]['movie_index'].tolist()

    # Exclude movies the user has already rated
    similarities[user_rated_movie_indices] = -np.inf  # Set similarity to negative infinity to exclude

    # Get top N recommendations
    top_indices = similarities.argsort()[-n_recommendations:][::-1]
    rec_movie_ids = [movie_ids[idx] for idx in top_indices]

    return rec_movie_ids

def precision_at_k(user_id, user_profiles, k=5, metric='cosine'):
    # Relevant items (movies rated 4 or higher) in the test set
    relevant_items = test_data[(test_data['userId'] == user_id) & (test_data['rating'] >= 4)]['movieId'].tolist()

    # Get recommendations
    recommended_movie_ids = get_user_recommendations(user_id, user_profiles, n_recommendations=k, metric=metric)

    if not recommended_movie_ids:
        return 0

    # Calculate precision
    relevant_and_recommended = set(relevant_items).intersection(set(recommended_movie_ids))
    precision = len(relevant_and_recommended) / min(k, len(recommended_movie_ids))

    return precision

def evaluate_model(metric='cosine'):
    # Build user profiles
    user_profiles = build_user_profiles(train_data)

    # Evaluate RMSE
    actual_ratings = []
    predicted_ratings = []

    n_predictions = 500  # Adjust for time considerations
    test_sample = test_data.sample(n=n_predictions, random_state=42).reset_index(drop=True)

    for idx in range(n_predictions):
        user_id = test_sample['userId'].iloc[idx]
        movie_id = test_sample['movieId'].iloc[idx]
        user_index = user_id_to_index.get(user_id)
        movie_index = movie_id_to_index.get(movie_id)
        actual_rating = test_sample['rating'].iloc[idx]

        pred_rating = predict_rating(user_id, movie_index, user_profiles, metric=metric)

        actual_ratings.append(actual_rating)
        predicted_ratings.append(pred_rating)

    rmse = sqrt(mean_squared_error(actual_ratings, predicted_ratings))

    # Evaluate Precision@5
    user_ids_sample = test_data['userId'].unique()[:50]  # Sample of users
    precision_scores = []

    for user_id in user_ids_sample:
        precision = precision_at_k(user_id, user_profiles, k=5, metric=metric)
        precision_scores.append(precision)

    average_precision = np.mean(precision_scores)

    return rmse, average_precision

# Hyperparameter grid
metrics = ['cosine', 'euclidean']

results = []

for metric in metrics:
    print(f"Evaluating model with metric={metric}")
    rmse, average_precision = evaluate_model(metric=metric)
    print(f"RMSE: {rmse:.4f}, Precision@5: {average_precision:.4f}")
    results.append({
        'metric': metric,
        'rmse': rmse,
        'precision_at_5': average_precision
    })

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results)
print(results_df)