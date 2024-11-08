import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

# Define paths
raw_data_path = os.path.join('data', 'raw')
processed_data_path = os.path.join('data', 'processed')

# Load datasets
movies = pd.read_csv(os.path.join(raw_data_path, 'movies.csv'))
ratings = pd.read_csv(os.path.join(raw_data_path, 'ratings.csv'))

# Explore datasets
print("Movies DataFrame:")
print(movies.head())
print(movies.info())

print("\nRatings DataFrame:")
print(ratings.head())
print(ratings.info())

# Merge DataFrames
ratings_movies = pd.merge(ratings, movies, on='movieId')

# Feature engineering
# Split genres and one-hot encode
ratings_movies['genres'] = ratings_movies['genres'].str.split('|')

mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(ratings_movies['genres'])
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
ratings_movies = pd.concat([ratings_movies, genres_df], axis=1)

# Normalize ratings
scaler = MinMaxScaler()
ratings_movies['rating_norm'] = scaler.fit_transform(ratings_movies[['rating']])

# Save processed ratings data
if not os.path.exists(processed_data_path):
    os.makedirs(processed_data_path)
ratings_movies.to_csv(os.path.join(processed_data_path, 'ratings_movies_cleaned.csv'), index=False)

# Load the processed data
data = pd.read_csv(os.path.join(processed_data_path, 'ratings_movies_cleaned.csv'))

# Select the relevant columns for unique movies and genres
genre_columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX',
                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = data[['movieId', 'title'] + genre_columns].drop_duplicates(subset='movieId')

# Reset index for the movies DataFrame
movies.reset_index(drop=True, inplace=True)

# Save the movies DataFrame to CSV
movies.to_csv(os.path.join(processed_data_path, 'movies_content.csv'), index=False)