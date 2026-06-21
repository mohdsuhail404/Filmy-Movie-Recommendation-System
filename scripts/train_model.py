import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def train_and_save():
    data_dir = "data/ml-latest-small"
    models_dir = "models"
    
    # 1. Load raw CSVs
    print("Loading data...")
    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")
    tags_path = os.path.join(data_dir, "tags.csv")
    
    if not (os.path.exists(movies_path) and os.path.exists(ratings_path) and os.path.exists(tags_path)):
        raise FileNotFoundError("Dataset CSVs not found in data/ml-latest-small/. Please run scripts/download_data.py first.")
        
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    tags = pd.read_csv(tags_path)
    
    # 2. Filter users with fewer than 20 interactions to reduce noise
    print("Filtering user ratings...")
    user_counts = ratings['userId'].value_counts()
    active_users = user_counts[user_counts >= 20].index
    ratings_filtered = ratings[ratings['userId'].isin(active_users)].copy()
    print(f"Original ratings: {len(ratings)}, Filtered ratings: {len(ratings_filtered)}")
    print(f"Users remaining: {ratings_filtered['userId'].nunique()} (filtered out {ratings['userId'].nunique() - ratings_filtered['userId'].nunique()} users with < 20 ratings)")
    
    # 3. Clean and aggregate tags per movie
    print("Processing movie tags and genres...")
    tags['tag'] = tags['tag'].astype(str)
    # Group tags by movieId and join them with space
    movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    
    # 4. Merge tags and genres with movie metadata
    movies_merged = movies.merge(movie_tags, on='movieId', how='left')
    movies_merged['tag'] = movies_merged['tag'].fillna('')
    movies_merged['genres_clean'] = movies_merged['genres'].str.replace('|', ' ', regex=False)
    
    # Combine title, genres, and user tags for TF-IDF content
    movies_merged['content'] = movies_merged['title'] + ' ' + movies_merged['genres_clean'] + ' ' + movies_merged['tag']
    movies_merged['content'] = movies_merged['content'].str.lower()
    movies_merged['content'] = movies_merged['content'].apply(lambda x: ' '.join(x.split()))
    
    # 5. Fit TfidfVectorizer
    print("Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(stop_words='english', min_df=1)
    tfidf_matrix = tfidf.fit_transform(movies_merged['content'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # 6. Save assets
    os.makedirs(models_dir, exist_ok=True)
    print("Saving processed data and models...")
    
    with open(os.path.join(models_dir, "movies_processed.pkl"), "wb") as f:
        pickle.dump(movies_merged, f)
        
    with open(os.path.join(models_dir, "ratings_filtered.pkl"), "wb") as f:
        pickle.dump(ratings_filtered, f)
        
    with open(os.path.join(models_dir, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)
        
    with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
        
    print("Training and model saving completed successfully!")

if __name__ == "__main__":
    train_and_save()
