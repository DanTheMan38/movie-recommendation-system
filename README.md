# Movie Recommendation System

An end-to-end machine learning pipeline for personalized movie recommendations using the MovieLens 20M Dataset. This project combines collaborative filtering, content-based filtering, and hybrid systems to provide tailored movie suggestions.

---

## Summary

This project demonstrates the development of a movie recommendation system using three primary approaches: collaborative filtering, content-based filtering, and a hybrid of the two. Using the MovieLens 20M dataset, it offers insights into model performance through metrics such as RMSE and Precision@5. The project also includes a web application for an interactive recommendation experience.

---

## Motivation

This project is part of my ML Engineering portfolio, showcasing skills in data preprocessing, model building, evaluation, hyperparameter tuning, and deployment in a production-ready environment.

## Project Goals

- Build a recommendation system using collaborative, content-based, and hybrid filtering methods.
- Implement an evaluation pipeline with RMSE and Precision@5 as primary metrics.
- Perform hyperparameter tuning for optimized model selection.
- Deploy the system as a web application.

## Features

- **Collaborative Filtering**: Recommends movies based on user-item interactions.
- **Content-Based Filtering**: Utilizes movie metadata like genres for recommendations.
- **Hybrid System**: Combines both collaborative and content-based methods for improved performance.
- **Evaluation Metrics**: Includes RMSE and Precision@5 for model evaluation.
- **Hyperparameter Tuning**: Systematic tuning of model parameters to enhance performance.
- **Web Application**: A user-friendly interface for obtaining movie recommendations.

## Dataset

The project uses the [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/), which contains 20 million ratings for 27,000 movies from 138,000 users.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/DanTheMan38/movie-recommendation-system.git
   cd movierec-system
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Processing**:

   ```bash
   python src/data_processing.py
   ```

2. **Train Models**:

   - Collaborative Filtering:

     ```bash
     python src/make_models/collaborative_filter.py
     ```

   - Content-Based Filtering:

     ```bash
     python src/make_models/content_based_filter.py
     ```

   - Hybrid System:

     ```bash
     python src/make_models/hybrid_system.py
     ```

3. **Evaluate Models**:

   ```bash
   python src/evaluation/evaluate.py
   ```

4. **Run the Web Application**:

   To deploy the application locally, simply run:

   ```bash
   python app.py
   ```

   The application will be accessible at `http://localhost:5000`.

## Project Structure

- `src/`: Source code for data processing, model training, and evaluation.
  - `data_processing.py`: Script for data cleaning and preprocessing.
  - `make_models/`: Contains scripts for building different recommendation models.
  - `evaluation/`: Scripts for evaluating model performance.
- `data/processed/`: Processed dataset files.
- `models/`: Saved models and artifacts.
- `templates/`: HTML templates for the web application.
- `static/`: Static files like CSS and JavaScript.
- `app.py`: Flask application entry point.
- `requirements.txt`: List of required Python packages.

## Results

### Collaborative Filtering

**Hyperparameter Tuning Results**

| Metric     | n_neighbors |   RMSE    | Precision@5 |
|------------|-------------|-----------|-------------|
| Cosine     | 5           | 0.963661  | 0.220       |
| Cosine     | 10          | 0.944672  | 0.132       |
| Cosine     | 20          | 0.913900  | 0.140       |
| Cosine     | 30          | 0.915411  | 0.100       |
| Cosine     | 50          | 0.900560  | 0.160       |
| Euclidean  | 5           | 0.924139  | 0.020       |
| Euclidean  | 10          | 0.926824  | 0.020       |
| Euclidean  | 20          | 0.970004  | 0.020       |
| Euclidean  | 30          | 0.958882  | 0.028       |
| Euclidean  | 50          | 0.975207  | 0.040       |

### Content-Based Filtering

**Results**

| Metric     |   RMSE    | Precision@5 |
|------------|-----------|-------------|
| Cosine     | 1.307100  | 0.012       |
| Euclidean  | 1.342158  | 0.008       |

### Hybrid System

**Example Results**

| Metric     | n_neighbors | Alpha |   RMSE    | Precision@5 |
|------------|-------------|-------|-----------|-------------|
| Cosine     | 5           | 0.5   | 1.006312  | 0.020       |
| Cosine     | 10          | 0.5   | 1.008018  | 0.000       |

## Key Findings and Takeaways

### What Did I Learn?

- **Implementation Skills**: Gained hands-on experience in implementing various recommendation algorithms.
- **Data Preprocessing**: Learned the importance of data cleaning and feature engineering.
- **Model Evaluation**: Understood the use of different metrics like RMSE and Precision@5 to evaluate model performance.
- **Hyperparameter Tuning**: Recognized how tuning parameters can significantly impact model results.

### Are the Results What I Expected?

The collaborative filtering model performed better than the content-based model, which aligns with expectations given the richness of user-rating data compared to genre information alone. The hybrid model showed potential but didn't outperform collaborative filtering, indicating room for further improvement.

### Key Takeaways

- **Data Quality Matters**: High-quality and rich datasets lead to better-performing models.
- **Model Complexity**: Sometimes simpler models perform just as well or better than complex ones.
- **Continuous Evaluation**: Regularly evaluating models helps in understanding their strengths and limitations.

## Conclusion and Next Steps

### Future-Looking Takeaways and Limitations

- **Enhancing Content Features**: Incorporate more content features like director, cast, and plot summaries to improve content-based filtering.
- **Advanced Algorithms**: Experiment with deep learning techniques such as autoencoders or neural collaborative filtering.
- **User Interface Improvements**: Develop a more interactive and user-friendly web interface.

### If I Were to Do This Project Again

- **More Data Exploration**: Spend more time understanding the data distribution and user behaviors.
- **Cross-Validation**: Implement cross-validation to get more reliable evaluation metrics.
- **Scalability**: Consider the scalability of the models for deployment in a production environment.

## Communicating Findings

Efforts were made to present the results clearly through tables and structured documentation. Visualizations can be added to further illustrate model performance and comparisons.

- **Visualizations**: Creating plots for RMSE and Precision@5 across different models and hyperparameters.
- **Documentation**: Detailed explanations of each step in the data processing and modeling pipeline.

## Acknowledgments

- [GroupLens Research](https://grouplens.org/) for providing the MovieLens dataset.
- Inspiration from various online resources and open-source projects.