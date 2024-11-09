import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
import os
import joblib

def collaborative_filtering_pipeline():
    print("Defining paths...")
    # Define paths
    processed_data_path = os.path.join('data', 'processed')
    model_save_path = os.path.join('models')

    print("Loading cleaned data...")
    # Load the cleaned data
    data = pd.read_csv(os.path.join(processed_data_path, 'ratings_movies_cleaned.csv'))

    print("Mapping userId and movieId to indices...")
    # Map userId and movieId to indices
    user_ids = data['userId'].unique()
    movie_ids = data['movieId'].unique()
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    print("Converting userId and movieId to indices...")
    # Convert userId and movieId to indices
    data['user_index'] = data['userId'].map(user_id_to_index)
    data['movie_index'] = data['movieId'].map(movie_id_to_index)

    print("Creating sparse matrix...")
    # Create a sparse matrix
    row = data['movie_index'].values
    col = data['user_index'].values
    values = data['rating'].values
    item_user_sparse = coo_matrix((values, (row, col)), shape=(len(movie_ids), len(user_ids))).tocsr()

    print("Building kNN model...")
    # Build the kNN model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
    model_knn.fit(item_user_sparse)

    print("Saving the model...")
    # Save the model
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    joblib.dump(model_knn, os.path.join(model_save_path, 'knn_model.pkl'))

def get_movie_recommendations(movie_id, data, model, n_recommendations=5):
    print("Mapping userId and movieId to indices...")
    user_ids = data['userId'].unique()
    movie_ids = data['movieId'].unique()
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    print("Converting userId and movieId to indices...")
    # Convert userId and movieId to indices
    data['user_index'] = data['userId'].map(user_id_to_index)
    data['movie_index'] = data['movieId'].map(movie_id_to_index)

    print("Creating sparse matrix...")
    # Create a sparse matrix
    row = data['movie_index'].values
    col = data['user_index'].values
    values = data['rating'].values
    item_user_sparse = coo_matrix((values, (row, col)), shape=(len(movie_ids), len(user_ids))).tocsr()

    if movie_id not in movie_id_to_index:
        print("Movie ID not found in the dataset.")
        return "Movie ID not found in the dataset."
    
    print("Finding nearest neighbors...")
    movie_idx = movie_id_to_index[movie_id]
    distances, indices = model.kneighbors(item_user_sparse[movie_idx], n_neighbors=n_recommendations+1)
    rec_indices = indices.flatten()[1:]  # Exclude the input movie itself
    rec_distances = distances.flatten()[1:]  # Corresponding distances
    
    print("Mapping indices back to movie IDs...")
    # Map indices back to movie IDs
    rec_movie_ids = [movie_ids[i] for i in rec_indices]
    
    # Get movie titles
    movie_titles = data[['movieId', 'title']].drop_duplicates()
    recommended_movies = movie_titles[movie_titles['movieId'].isin(rec_movie_ids)].copy()
    recommended_movies['distance'] = rec_distances
    # Convert distances to similarity scores (1 - distance)
    recommended_movies['similarity_score'] = 1 - recommended_movies['distance']
    
    print("Returning recommended movies...")
    return recommended_movies[['title', 'similarity_score']]

# Test the function
if __name__ == "__main__":
    print("Starting collaborative filtering pipeline...")
    collaborative_filtering_pipeline()
    
    print("Loading cleaned data for recommendations...")
    # Load the cleaned data again for recommendations
    processed_data_path = os.path.join('data', 'processed')
    data = pd.read_csv(os.path.join(processed_data_path, 'ratings_movies_cleaned.csv'))

    print("Loading saved kNN model...")
    # Load the saved model
    model_save_path = os.path.join('models')
    model_knn = joblib.load(os.path.join(model_save_path, 'knn_model.pkl'))
    
    # Example movie ID to test
    test_movie_id = data['movieId'].iloc[0]
    print(f"Generating recommendations for Movie ID {test_movie_id}...")
    recommendations = get_movie_recommendations(test_movie_id, data, model_knn)
    print(f"Recommendations for Movie ID {test_movie_id}:")
    print(recommendations)