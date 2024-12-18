import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib

def content_based_filtering_pipeline():
    print("Defining paths...")
    # Define paths
    processed_data_path = os.path.join('data', 'processed')
    model_save_path = os.path.join('models')

    print("Loading movies data with content features...")
    # Load the movies data with content features
    movies = pd.read_csv(os.path.join(processed_data_path, 'movies_content.csv'))

    print("Checking if genre columns exist...")
    # Check if 'movies_content.csv' has genre columns; if not, process genres
    if 'Action' not in movies.columns:
        # Assuming 'genres' column exists and is a string of genres separated by '|'
        genres = set()
        for genre_list in movies['genres']:
            genres.update(genre_list.split('|'))

        for genre in genres:
            movies[genre] = movies['genres'].apply(lambda x: int(genre in x))

    print("Creating movie features DataFrame...")
    # Create a movie features DataFrame
    movie_features = movies.set_index('movieId').drop(columns=['title', 'genres'])

    print("Computing cosine similarity matrix...")
    # Compute the cosine similarity matrix
    content_similarity = cosine_similarity(movie_features.values)
    content_similarity_df = pd.DataFrame(content_similarity, index=movie_features.index, columns=movie_features.index)

    print("Saving the similarity matrix...")
    # Save the similarity matrix for future use
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    joblib.dump(content_similarity_df, os.path.join(model_save_path, 'content_similarity.pkl'))

def get_content_based_recommendations(movie_id, movies_df, similarity_df, n_recommendations=5):
    if movie_id not in similarity_df.index:
        print("Movie ID not found in the dataset.")
        return "Movie ID not found in the dataset."
    
    print("Getting similarity scores for the given movie...")
    # Get similarity scores for the given movie
    sim_scores = similarity_df[movie_id]
    
    print("Excluding the movie itself from recommendations...")
    # Exclude the movie itself
    sim_scores = sim_scores.drop(index=movie_id)
    
    print("Getting the top N similar movies...")
    # Get the top N similar movies
    top_n_movies = sim_scores.nlargest(n_recommendations)
    
    print("Getting movie titles...")
    # Get movie titles
    recommended_movies = movies_df[movies_df['movieId'].isin(top_n_movies.index)][['title']]
    recommended_movies['similarity_score'] = top_n_movies.values
    
    print("Returning recommended movies...")
    return recommended_movies[['title', 'similarity_score']]

def content_based_pipeline():
    print("Starting content-based filtering pipeline...")
    content_based_filtering_pipeline()
    
    print("Loading movies data for recommendations...")
    # Load the movies data again for recommendations
    processed_data_path = os.path.join('data', 'processed')
    movies = pd.read_csv(os.path.join(processed_data_path, 'movies_content.csv'))

    print("Loading saved similarity matrix...")
    # Load the saved similarity matrix
    model_save_path = os.path.join('models')
    content_similarity_df = joblib.load(os.path.join(model_save_path, 'content_similarity.pkl'))
    
    # Example movie ID to test
    test_movie_id = movies['movieId'].iloc[0]
    print(f"Generating content-based recommendations for Movie ID {test_movie_id}...")
    recommendations = get_content_based_recommendations(test_movie_id, movies, content_similarity_df)
    print(f"Content-Based Recommendations for Movie ID {test_movie_id}:")
    print(recommendations)

# Test the function
if __name__ == "__main__":
    content_based_pipeline()