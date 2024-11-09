from flask import Flask, render_template, request
import pandas as pd
import os
import joblib
from scipy.sparse import csr_matrix
import sys
import difflib
import re  # Import re for regular expressions

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root directory to sys.path
sys.path.append(PROJECT_ROOT)

# Initialize the Flask application
app = Flask(__name__)

# Load models and data using absolute paths
processed_data_path = os.path.join(PROJECT_ROOT, 'data', 'processed')
model_save_path = os.path.join(PROJECT_ROOT, 'models')

# Load movie data and collaborative filtering model
movies = pd.read_csv(os.path.join(processed_data_path, 'movies_content.csv'))
knn_model = joblib.load(os.path.join(model_save_path, 'knn_model.pkl'))
ratings = pd.read_csv(os.path.join(processed_data_path, 'ratings_movies_cleaned.csv'))

# Map userId and movieId to indices
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()
user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

# Map userId and movieId in ratings DataFrame
ratings['user_index'] = ratings['userId'].map(user_id_to_index)
ratings['movie_index'] = ratings['movieId'].map(movie_id_to_index)

# Create the sparse item-user matrix
row = ratings['movie_index'].values  # Movie indices (rows)
col = ratings['user_index'].values   # User indices (columns)
data = ratings['rating'].values      # Ratings

item_user_sparse = csr_matrix((data, (row, col)), shape=(len(movie_ids), len(user_ids)))

# Function to preprocess movie titles
def preprocess_title(title):
    # Convert to lowercase
    title = title.lower()
    # Remove year in parentheses
    title = re.sub(r'\(\d{4}\)', '', title)
    # Remove punctuation
    title = re.sub(r'[^\w\s]', '', title)
    # Remove common articles
    title = re.sub(r'\b(the|a|an)\b', '', title)
    # Remove extra whitespace
    title = ' '.join(title.split())
    return title

# Preprocess movie titles and add a new column
movies['processed_title'] = movies['title'].apply(preprocess_title)

def get_collaborative_recommendations(movie_title, n_recommendations=5):
    # Preprocess user input
    processed_input = preprocess_title(movie_title)
    
    # Try to find an exact match first
    exact_matches = movies[movies['processed_title'] == processed_input]
    if not exact_matches.empty:
        movie_id = exact_matches.iloc[0]['movieId']
    else:
        # Use difflib to find close matches
        movie_titles = movies['processed_title'].tolist()
        close_matches = difflib.get_close_matches(processed_input, movie_titles, n=1, cutoff=0.4)
        if close_matches:
            # Use the closest match
            matched_title = close_matches[0]
            movie_id = movies[movies['processed_title'] == matched_title].iloc[0]['movieId']
        else:
            return ["Movie not found in the dataset."]
    
    # Check if movie_id exists in collaborative filtering data
    if movie_id not in movie_id_to_index:
        return ["Movie ID not found in the dataset."]
    
    # Get index in item-user matrix
    movie_idx = movie_id_to_index[movie_id]
    
    # Find k similar movies
    distances, indices = knn_model.kneighbors(item_user_sparse[movie_idx], n_neighbors=n_recommendations+1)
    
    # Exclude the input movie itself from recommendations
    rec_indices = indices.flatten()[1:]
    rec_movie_ids = [movie_ids[i] for i in rec_indices]
    
    # Get movie titles
    recommended_movies = movies[movies['movieId'].isin(rec_movie_ids)]
    return recommended_movies['title'].tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input from the form
        movie_title = request.form.get('movie_title').strip()
        rec_titles = get_collaborative_recommendations(movie_title, n_recommendations=5)

        return render_template('recommendations.html', recommendations=rec_titles, movie_title=movie_title)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)