from src.data_processing import data_processing_pipeline
from src.make_models.collaborative_filter import collaborative_filtering_pipeline
from src.make_models.content_based_filter import content_based_pipeline
from src.make_models.hybrid_system import hybrid_pipeline
from src.evaluation.evaluate import collaborative_pipeline

if __name__ == '__main__':
    print("Starting data preprocessing...")
    data_processing_pipeline()
    
    print("Training collaborative filtering model...")
    collaborative_filtering_pipeline()
    
    print("Training content-based filtering model...")
    content_based_pipeline()
    
    print("Training hybrid model...")
    hybrid_pipeline()
    
    print("Evaluating model...")
    collaborative_pipeline()
    
    print("Pipeline execution completed.")