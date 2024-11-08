import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib

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

# Function to get hybrid movie recommendations
def get_hybrid_recommendations(movie_id, data, cf_model, content_sim_df, alpha=0.5, n_recommendations=5):
    if movie_id not in movie_id_to_index:
        return "Movie ID not found in the dataset."
    
    movie_idx = movie_id_to_index[movie_id]
    
    # Collaborative Filtering Similarities
    distances_cf, indices_cf = cf_model.kneighbors(item_user_sparse[movie_idx], n_neighbors=n_recommendations+1)
    rec_indices_cf = indices_cf.flatten()[1:]  # Exclude the input movie itself
    sim_scores_cf = 1 - distances_cf.flatten()[1:]  # Convert distances to similarity scores
    
    # Content-Based Similarities
    sim_scores_content = content_sim_df.loc[movie_id].drop(movie_id)  # Exclude the input movie itself
    top_content = sim_scores_content.sort_values(ascending=False).head(n_recommendations)
    rec_movie_ids_content = top_content.index.values
    sim_scores_content = top_content.values
    
    # Combine Recommendations
    combined_movie_ids = np.concatenate((movie_ids[rec_indices_cf], rec_movie_ids_content))
    combined_sim_scores_cf = np.concatenate((sim_scores_cf, np.zeros(len(sim_scores_content))))
    combined_sim_scores_content = np.concatenate((np.zeros(len(sim_scores_cf)), sim_scores_content))
    
    # Aggregate the similarities with weighting factor alpha
    combined_sim_scores = alpha * combined_sim_scores_cf + (1 - alpha) * combined_sim_scores_content
    
    # Create a DataFrame to hold the recommendations
    recommendations_df = pd.DataFrame({
        'movieId': combined_movie_ids,
        'similarity_score': combined_sim_scores
    })
    
    # Remove duplicates and the input movie itself
    recommendations_df = recommendations_df.drop_duplicates(subset='movieId')
    recommendations_df = recommendations_df[recommendations_df['movieId'] != movie_id]
    
    # Sort by similarity score
    recommendations_df = recommendations_df.sort_values(by='similarity_score', ascending=False)
    
    # Get top N recommendations
    recommendations_df = recommendations_df.head(n_recommendations)
    
    # Get movie titles
    movie_titles = data[['movieId', 'title']].drop_duplicates()
    recommendations_df = recommendations_df.merge(movie_titles, on='movieId', how='left')
    
    return recommendations_df[['title', 'similarity_score']]

# Test the function
if __name__ == "__main__":
    # Example movie ID to test
    test_movie_id = data['movieId'].iloc[0]
    recommendations = get_hybrid_recommendations(test_movie_id, data, cf_model_knn, content_similarity_df, alpha=0.5, n_recommendations=5)
    print(f"Hybrid Recommendations for Movie ID {test_movie_id}:\n")
    print(recommendations)
    
    # Save the models and similarity matrix
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    joblib.dump(cf_model_knn, os.path.join(model_save_path, 'cf_knn_model.pkl'))
    joblib.dump(content_similarity_df, os.path.join(model_save_path, 'content_similarity.pkl'))