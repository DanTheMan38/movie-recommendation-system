import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
import os

def collaborative_pipeline():
    # Define paths
    processed_data_path = os.path.join('data', 'processed')

    # Load datasets
    movies = pd.read_csv(os.path.join(processed_data_path, 'movies_content.csv'))
    data = pd.read_csv(os.path.join(processed_data_path, 'ratings_movies_cleaned.csv'))

    # Sample data before any processing (Increase sample size if possible)
    # data = data.sample(n=100000, random_state=42)  # Commenting this out for now

    # Optionally filter out users and movies with few ratings to increase density
    # For example, keep users who have rated at least 20 movies
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

    # Split data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create sparse matrix function
    def create_sparse_matrix(df):
        row = df['movie_index'].values
        col = df['user_index'].values
        values = df['rating'].values
        return coo_matrix((values, (row, col)), shape=(len(movie_ids), len(user_ids))).tocsr()

    # Create training sparse matrix
    train_sparse = create_sparse_matrix(train_data)
    item_user_sparse = train_sparse  # Item-user matrix

    # Check sparsity of the train_sparse matrix
    non_zero_count = train_sparse.nnz
    total_elements = train_sparse.shape[0] * train_sparse.shape[1]
    density = non_zero_count / total_elements
    print(f"Train Sparse Matrix Density: {density:.6f}")

    # Build the kNN model (Adjust n_neighbors)
    n_neighbors = 5  # Adjusted based on data size
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors+1, n_jobs=-1)
    model_knn.fit(item_user_sparse)

    def predict_rating(user_index, movie_index, model, train_sparse, k=n_neighbors):
        # Find k similar movies
        distances, indices = model.kneighbors(train_sparse[movie_index], n_neighbors=k+1)
        similar_movie_indices = indices.flatten()[1:]  # Exclude the input movie itself

        # Get ratings the user gave to these similar movies
        user_ratings = train_sparse[similar_movie_indices, user_index].toarray().flatten()

        # Filter out zero ratings and corresponding distances
        mask = user_ratings > 0
        non_zero_ratings = user_ratings[mask]
        weights = distances.flatten()[1:][mask]

        if len(non_zero_ratings) == 0:
            # Return the user's average rating if no ratings are found
            user_mean_rating = train_data[train_data['user_index'] == user_index]['rating'].mean()
            return user_mean_rating if not np.isnan(user_mean_rating) else train_data['rating'].mean()
        else:
            # Return the weighted average rating
            adjusted_weights = 1 - weights
            if adjusted_weights.sum() == 0:
                # All weights are zero, return unweighted average
                return non_zero_ratings.mean()
            else:
                return np.average(non_zero_ratings, weights=adjusted_weights)

    # Evaluate RMSE
    actual_ratings = []
    predicted_ratings = []

    n_predictions = 1000  # Adjust for time considerations

    for idx in range(n_predictions):
        user_idx = test_data['user_index'].iloc[idx]
        movie_idx = test_data['movie_index'].iloc[idx]
        actual_rating = test_data['rating'].iloc[idx]

        pred_rating = predict_rating(user_idx, movie_idx, model_knn, train_sparse)

        actual_ratings.append(actual_rating)
        predicted_ratings.append(pred_rating)

        # Print progress every 100 predictions
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{n_predictions} predictions")

    rmse = sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    print(f"Collaborative Filtering RMSE: {rmse:.4f}")

    def get_user_recommendations(user_id, n_recommendations=10):
        # Get movies the user has rated in training data
        user_movies_train = train_data[train_data['userId'] == user_id]['movieId'].tolist()
        user_index = user_id_to_index.get(user_id)

        if user_index is None or len(user_movies_train) == 0:
            return []

        # Randomly select a seed movie the user has rated
        seed_movie_id = np.random.choice(user_movies_train)
        seed_movie_index = movie_id_to_index[seed_movie_id]

        # Get collaborative recommendations
        distances, indices = model_knn.kneighbors(train_sparse[seed_movie_index], n_neighbors=n_recommendations + len(user_movies_train))
        rec_movie_indices = indices.flatten()

        # Exclude movies the user has already rated
        rec_movie_ids = [movie_ids[idx] for idx in rec_movie_indices if movie_ids[idx] not in user_movies_train]

        # Return top N recommendations
        return rec_movie_ids[:n_recommendations]

    def precision_at_k(user_id, k=5):
        # Relevant items (movies rated 4 or higher) in the test set
        relevant_items = test_data[(test_data['userId'] == user_id) & (test_data['rating'] >= 4)]['movieId'].tolist()

        # Get recommendations
        recommended_movie_ids = get_user_recommendations(user_id, n_recommendations=k)

        if not recommended_movie_ids:
            return 0

        # Calculate precision
        relevant_and_recommended = set(relevant_items).intersection(set(recommended_movie_ids))
        precision = len(relevant_and_recommended) / min(k, len(recommended_movie_ids))

        return precision

    # Calculate average Precision@K for multiple users
    user_ids_sample = test_data['userId'].unique()[:100]  # Sample of users
    precision_scores = []

    for user_id in user_ids_sample:
        precision = precision_at_k(user_id, k=5)
        precision_scores.append(precision)

    average_precision = np.mean(precision_scores)
    print(f"Collaborative Filtering Average Precision@5: {average_precision:.4f}")

    # Similarly for Recall@K and F1 Score
    def recall_at_k(user_id, k=5):
        relevant_items = test_data[(test_data['userId'] == user_id) & (test_data['rating'] >= 4)]['movieId'].tolist()
        recommended_movie_ids = get_user_recommendations(user_id, n_recommendations=k)

        if not relevant_items:
            return 0

        relevant_and_recommended = set(relevant_items).intersection(set(recommended_movie_ids))
        recall = len(relevant_and_recommended) / len(relevant_items)

        return recall

    def f1_score_at_k(user_id, k=5):
        precision = precision_at_k(user_id, k)
        recall = recall_at_k(user_id, k)
        if precision + recall == 0:
            return 0
        else:
            return 2 * (precision * recall) / (precision + recall)

    # Calculate average Recall@K and F1 Score@K
    recall_scores = []
    f1_scores = []

    for user_id in user_ids_sample:
        recall = recall_at_k(user_id, k=5)
        f1 = f1_score_at_k(user_id, k=5)
        recall_scores.append(recall)
        f1_scores.append(f1)

    average_recall = np.mean(recall_scores)
    average_f1 = np.mean(f1_scores)

    print(f"Collaborative Filtering Average Recall@5: {average_recall:.4f}")
    print(f"Collaborative Filtering Average F1 Score@5: {average_f1:.4f}")

    # Check sparsity of the train_sparse matrix after adjustments
    non_zero_count = train_sparse.nnz
    total_elements = train_sparse.shape[0] * train_sparse.shape[1]
    density = non_zero_count / total_elements
    print(f"Train Sparse Matrix Density after adjustments: {density:.6f}")