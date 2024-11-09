import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

def data_processing_pipeline():
    def define_paths():
        # Define paths
        raw_data_path = os.path.join('data', 'raw')
        processed_data_path = os.path.join('data', 'processed')
        print(f"Defined paths: raw_data_path={raw_data_path}, processed_data_path={processed_data_path}")
        return raw_data_path, processed_data_path

    def load_datasets(raw_data_path):
        # Load datasets
        print(f"Loading datasets from {raw_data_path}")
        movies = pd.read_csv(os.path.join(raw_data_path, 'movies.csv'))
        ratings = pd.read_csv(os.path.join(raw_data_path, 'ratings.csv'))
        print(f"Loaded movies and ratings datasets with shapes {movies.shape} and {ratings.shape} respectively")
        return movies, ratings

    def explore_datasets(movies, ratings):
        # Explore datasets
        print("Movies DataFrame:")
        print(movies.head())
        print(movies.info())

        print("\nRatings DataFrame:")
        print(ratings.head())
        print(ratings.info())

    def merge_dataframes(movies, ratings):
        # Merge DataFrames
        print("Merging movies and ratings DataFrames")
        ratings_movies = pd.merge(ratings, movies, on='movieId')
        print(f"Merged DataFrame shape: {ratings_movies.shape}")
        return ratings_movies

    def feature_engineering(ratings_movies):
        # Split genres and one-hot encode
        print("Performing feature engineering on genres and ratings")
        ratings_movies['genres'] = ratings_movies['genres'].str.split('|')

        mlb = MultiLabelBinarizer()
        genres_encoded = mlb.fit_transform(ratings_movies['genres'])
        genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
        ratings_movies = pd.concat([ratings_movies, genres_df], axis=1)

        # Normalize ratings
        scaler = MinMaxScaler()
        ratings_movies['rating_norm'] = scaler.fit_transform(ratings_movies[['rating']])
        print(f"Feature engineering completed. DataFrame shape: {ratings_movies.shape}")
        return ratings_movies

    def save_processed_data(ratings_movies, processed_data_path):
        # Save processed ratings data
        print(f"Saving processed ratings data to {processed_data_path}")
        if not os.path.exists(processed_data_path):
            os.makedirs(processed_data_path)
        ratings_movies.to_csv(os.path.join(processed_data_path, 'ratings_movies_cleaned.csv'), index=False)
        print(f"Processed ratings data saved.")

    def load_processed_data(processed_data_path):
        # Load the processed data
        print(f"Loading processed data from {processed_data_path}")
        data = pd.read_csv(os.path.join(processed_data_path, 'ratings_movies_cleaned.csv'))
        print(f"Loaded processed data with shape {data.shape}")
        return data

    def select_relevant_columns(data):
        # Select the relevant columns for unique movies and genres
        print("Selecting relevant columns for movies and genres")
        genre_columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                         'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX',
                         'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

        movies = data[['movieId', 'title'] + genre_columns].drop_duplicates(subset='movieId')
        # Reset index for the movies DataFrame
        movies.reset_index(drop=True, inplace=True)
        print(f"Selected relevant columns. Movies DataFrame shape: {movies.shape}")
        return movies

    def save_movies_data(movies, processed_data_path):
        # Save the movies DataFrame to CSV
        print(f"Saving movies data to {processed_data_path}")
        movies.to_csv(os.path.join(processed_data_path, 'movies_content.csv'), index=False)
        print("Movies data saved.")

    raw_data_path, processed_data_path = define_paths()
    movies, ratings = load_datasets(raw_data_path)
    explore_datasets(movies, ratings)
    ratings_movies = merge_dataframes(movies, ratings)
    ratings_movies = feature_engineering(ratings_movies)
    save_processed_data(ratings_movies, processed_data_path)
    data = load_processed_data(processed_data_path)
    movies = select_relevant_columns(data)
    save_movies_data(movies, processed_data_path)

if __name__ == "__main__":
    data_processing_pipeline()