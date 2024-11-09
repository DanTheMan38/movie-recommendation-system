import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib

def hybrid_pipeline():
    # Define paths
    processed_data_path = os.path.join('data', 'processed')
    model_save_path = os.path.join('models')

    # Load the cleaned data
    data = pd.read_csv(os.path.join(processed_data_path, 'ratings_movies_cleaned.csv'))

    # Load movie content data
    movies = pd.read_csv(os.path.join(processed_data_path, 'movies_content.csv'))

    # Map userId and movieId to indices
    user_ids = data['userId'].unique()
    movie_ids = data['movieId'].unique()
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    # Convert userId and movieId to indices
    data['user_index'] = data['userId'].map(user_id_to_index)
    data['movie_index'] = data['movieId'].map(movie_id_to_index)

    # Create a sparse matrix for collaborative filtering
    row = data['movie_index'].values
    col = data['user_index'].values
    values = data['rating'].values
    item_user_sparse = coo_matrix((values, (row, col)), shape=(len(movie_ids), len(user_ids))).tocsr()

    # Build the collaborative filtering kNN model
    cf_model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
    cf_model_knn.fit(item_user_sparse)

    # Process movie features for content-based filtering
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

    # Compute the content-based similarity matrix
    content_similarity = cosine_similarity(movie_features)
    content_similarity_df = pd.DataFrame(content_similarity, index=movie_features.index, columns=movie_features.index)

    # Save the models and similarity matrix
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    joblib.dump(cf_model_knn, os.path.join(model_save_path, 'cf_knn_model.pkl'))
    joblib.dump(content_similarity_df, os.path.join(model_save_path, 'content_similarity.pkl'))

    print("Hybrid model training completed.")
