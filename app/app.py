from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import joblib
from scipy.sparse import csr_matrix

# Initialize the Flask application
app = Flask(__name__)

# Load models and data
processed_data_path = os.path.join('data', 'processed')
model_save_path = os.path.join('models')

# Load the content-based filtering data
movies = pd.read_csv(os.path.join(processed_data_path, 'movies_content.csv'))
cosine_sim = joblib.load(os.path.join(model_save_path, 'cosine_sim.pkl'))
indices = joblib.load(os.path.join(model_save_path, 'indices.pkl'))

# Load the collaborative filtering model
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

# Import the recommendation functions
from src.models.content_based_filter import get_content_recommendations

def get_collaborative_recommendations(movie_id, n_recommendations=5):
    if movie_id not in movie_id_to_index:
        return ["Movie ID not found in the dataset."]
    
    movie_idx = movie_id_to_index[movie_id]
    
    # Get similar movies
    distances, indices_knn = knn_model.kneighbors(item_user_sparse[movie_idx], n_neighbors=n_recommendations+1)
    
    # Exclude the input movie itself
    rec_indices = indices_knn.flatten()[1:]
    
    # Map indices back to movie IDs
    rec_movie_ids = [movie_ids[i] for i in rec_indices]
    
    # Get movie titles
    recommended_movies = movies[movies['movieId'].isin(rec_movie_ids)]
    return recommended_movies['title'].tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input from the form
        movie_title = request.form.get('movie_title')
        recommendation_type = request.form.get('recommendation_type')

        if recommendation_type == 'content':
            recommendations = get_content_recommendations(
                title=movie_title,
                cosine_sim=cosine_sim,
                df=movies,
                indices=indices,
                n_recommendations=5
            )
            rec_titles = recommendations['title'].tolist()
        else:
            # Get movieId from the title
            movie_id = movies[movies['title'] == movie_title]['movieId'].values
            if len(movie_id) == 0:
                rec_titles = ["Movie not found in the dataset."]
            else:
                rec_titles = get_collaborative_recommendations(movie_id[0], n_recommendations=5)

        return render_template('recommendations.html', recommendations=rec_titles, movie_title=movie_title)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)