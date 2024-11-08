import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
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

# Function to create sparse matrix
def create_sparse_matrix(df):
    row = df['movie_index'].values
    col = df['user_index'].values
    values = df['rating'].values
    return coo_matrix((values, (row, col)), shape=(len(movie_ids), len(user_ids))).tocsr()

# Create sparse matrices for training
train_sparse = create_sparse_matrix(train_data)
item_user_sparse = train_sparse  # Item-user matrix

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

# Normalize movie features
from sklearn.preprocessing import normalize
movie_features_normalized = normalize(movie_features, axis=1)

# Define the evaluation functions

def predict_rating(user_index, movie_index, model_knn, train_sparse, alpha, k):
    # Collaborative Filtering Component
    distances_cf, indices_cf = model_knn.kneighbors(train_sparse[movie_index], n_neighbors=k+1)
    similar_movie_indices_cf = indices_cf.flatten()[1:]  # Exclude the input movie itself

    # Get ratings the user gave to these similar movies
    user_ratings_cf = train_sparse[similar_movie_indices_cf, user_index].toarray().flatten()

    # Filter out zero ratings and corresponding distances
    mask_cf = user_ratings_cf > 0
    non_zero_ratings_cf = user_ratings_cf[mask_cf]
    weights_cf = distances_cf.flatten()[1:][mask_cf]

    # Content-Based Filtering Component
    movie_feature = movie_features_normalized[movie_index]
    movie_features_similar = movie_features_normalized[similar_movie_indices_cf[mask_cf]]

    # Compute content-based similarities
    similarities_cb = movie_features_similar.dot(movie_feature.T).flatten()

    # Combine similarities
    combined_weights = alpha * (1 - weights_cf) + (1 - alpha) * similarities_cb

    if len(non_zero_ratings_cf) == 0:
        # Return the user's average rating if no ratings are found
        user_mean_rating = train_data[train_data['user_index'] == user_index]['rating'].mean()
        return user_mean_rating if not np.isnan(user_mean_rating) else train_data['rating'].mean()
    else:
        if combined_weights.sum() == 0:
            # All weights are zero, return unweighted average
            return non_zero_ratings_cf.mean()
        else:
            # Return the weighted average rating
            return np.average(non_zero_ratings_cf, weights=combined_weights)

def get_user_recommendations(user_id, model_knn, item_user_sparse, alpha, n_recommendations=5, k=5):
    user_index = user_id_to_index.get(user_id)
    if user_index is None:
        return []

    # Get movies the user has not rated
    user_rated_movie_indices = train_data[train_data['userId'] == user_id]['movie_index'].tolist()
    unrated_movies = list(set(range(len(movie_ids))) - set(user_rated_movie_indices))

    # Limit the number of unrated movies to 50
    if len(unrated_movies) > 50:
        unrated_movies = np.random.choice(unrated_movies, size=50, replace=False)

    # Predict ratings for unrated movies
    predicted_ratings = []
    for idx, movie_idx in enumerate(unrated_movies):
        pred_rating = predict_rating(user_index, movie_idx, model_knn, train_sparse, alpha, k)
        predicted_ratings.append((movie_idx, pred_rating))
        # Progress update (optional)
        # if (idx + 1) % 50 == 0:
        #     print(f"Processed {idx + 1}/{len(unrated_movies)} unrated movies for user {user_id}")

    # Get top N recommendations
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)
    top_movie_indices = [idx for idx, _ in predicted_ratings[:n_recommendations]]
    rec_movie_ids = [movie_ids[idx] for idx in top_movie_indices]

    return rec_movie_ids

def precision_at_k(user_id, model_knn, item_user_sparse, alpha, k_recommendations=5, k_neighbors=5):
    # Relevant items (movies rated 4 or higher) in the test set
    relevant_items = test_data[(test_data['userId'] == user_id) & (test_data['rating'] >= 4)]['movieId'].tolist()

    # Get recommendations
    recommended_movie_ids = get_user_recommendations(user_id, model_knn, item_user_sparse, alpha, n_recommendations=k_recommendations, k=k_neighbors)

    if not recommended_movie_ids:
        return 0

    # Calculate precision
    relevant_and_recommended = set(relevant_items).intersection(set(recommended_movie_ids))
    precision = len(relevant_and_recommended) / min(k_recommendations, len(recommended_movie_ids))

    return precision

def evaluate_model(metric='cosine', n_neighbors=5, alpha=0.5):
    # Build kNN model with given hyperparameters
    model_knn = NearestNeighbors(metric=metric, algorithm='brute', n_neighbors=n_neighbors+1, n_jobs=-1)
    model_knn.fit(item_user_sparse)

    # Evaluate RMSE
    actual_ratings = []
    predicted_ratings = []

    n_predictions = 50  # Reduced number of predictions
    test_sample = test_data.sample(n=n_predictions, random_state=42).reset_index(drop=True)

    for idx in range(n_predictions):
        user_idx = test_sample['user_index'].iloc[idx]
        movie_idx = test_sample['movie_index'].iloc[idx]
        actual_rating = test_sample['rating'].iloc[idx]

        pred_rating = predict_rating(user_idx, movie_idx, model_knn, train_sparse, alpha, k=n_neighbors)

        actual_ratings.append(actual_rating)
        predicted_ratings.append(pred_rating)

        # Optional: Print progress
        # if (idx + 1) % 50 == 0:
        #     print(f"Processed {idx + 1}/{n_predictions} predictions")

    rmse = sqrt(mean_squared_error(actual_ratings, predicted_ratings))

    # Evaluate Precision@5
    user_ids_sample = test_data['userId'].unique()[:10]  # Reduced number of users
    precision_scores = []

    for idx, user_id in enumerate(user_ids_sample):
        precision = precision_at_k(user_id, model_knn, item_user_sparse, alpha, k_recommendations=5, k_neighbors=n_neighbors)
        precision_scores.append(precision)

        # Optional: Print progress
        # if (idx + 1) % 5 == 0:
        #     print(f"Evaluated precision for {idx + 1}/{len(user_ids_sample)} users")

    average_precision = np.mean(precision_scores)

    return rmse, average_precision

# Hyperparameter grid
metrics = ['cosine']  # Limited to one metric
n_neighbors_list = [5, 10]  # Reduced list
alpha_list = [0.5]  # Single alpha value

results = []

for metric in metrics:
    for n_neighbors in n_neighbors_list:
        for alpha in alpha_list:
            print(f"Evaluating model with metric={metric}, n_neighbors={n_neighbors}, alpha={alpha}")
            rmse, average_precision = evaluate_model(metric=metric, n_neighbors=n_neighbors, alpha=alpha)
            print(f"RMSE: {rmse:.4f}, Precision@5: {average_precision:.4f}")
            results.append({
                'metric': metric,
                'n_neighbors': n_neighbors,
                'alpha': alpha,
                'rmse': rmse,
                'precision_at_5': average_precision
            })

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results)
print(results_df)