import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def get_human_name_for_id(user_id):
    """Deterministically maps a numerical User ID to a unique, relatable human name."""
    first_names = [
        "Aria", "Liam", "Olivia", "Noah", "Emma", "Oliver", "Ava", "Elijah", 
        "Sophia", "James", "Isabella", "Benjamin", "Mia", "Lucas", "Charlotte", 
        "Henry", "Amelia", "Alexander", "Harper", "Mason", "Evelyn", "Michael", 
        "Abigail", "Ethan", "Emily", "Daniel", "Elizabeth", "Jacob", "Sofia", "Logan"
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", 
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", 
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", 
        "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", 
        "Ramirez", "Lewis", "Robinson"
    ]
    idx = int(user_id) - 1
    first_idx = idx % len(first_names)
    last_idx = (idx // len(first_names)) % len(last_names)
    return f"{first_names[first_idx]} {last_names[last_idx]}"

class MovieRecommender:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.movies = None
        self.ratings = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.movie_id_to_idx = {}
        self.load_models()

    def load_models(self):
        """Loads the processed data and trained model components from disk."""
        movies_path = os.path.join(self.models_dir, "movies_processed.pkl")
        ratings_path = os.path.join(self.models_dir, "ratings_filtered.pkl")
        tfidf_matrix_path = os.path.join(self.models_dir, "tfidf_matrix.pkl")
        tfidf_vectorizer_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")

        if not (os.path.exists(movies_path) and os.path.exists(ratings_path) and 
                os.path.exists(tfidf_matrix_path) and os.path.exists(tfidf_vectorizer_path)):
            raise FileNotFoundError(
                "Model files not found. Please run scripts/train_model.py to generate them."
            )

        with open(movies_path, "rb") as f:
            self.movies = pickle.load(f)
            
        with open(ratings_path, "rb") as f:
            self.ratings = pickle.load(f)
            
        with open(tfidf_matrix_path, "rb") as f:
            self.tfidf_matrix = pickle.load(f)
            
        with open(tfidf_vectorizer_path, "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)

        # Create mapping from movieId to matrix index
        self.movie_id_to_idx = {row['movieId']: idx for idx, row in self.movies.iterrows()}

        # Create a list of user profiles with human names
        self.user_profiles = [
            {"userId": uid, "name": get_human_name_for_id(uid)}
            for uid in sorted(self.ratings['userId'].unique())
        ]
        self.user_id_to_name = {p['userId']: p['name'] for p in self.user_profiles}
        self.name_to_user_id = {p['name']: p['userId'] for p in self.user_profiles}

    def find_movie_by_title(self, query):
        """Finds the closest movie matching the query string in titles."""
        if not query:
            return None
        
        query_lower = query.lower().strip()
        
        # 1. Exact match (case insensitive)
        matches = self.movies[self.movies['title'].str.lower() == query_lower]
        if not matches.empty:
            return matches.iloc[0]
            
        # 2. Substring match
        matches = self.movies[self.movies['title'].str.lower().str.contains(query_lower, regex=False)]
        if not matches.empty:
            return matches.iloc[0]
            
        return None

    def get_content_recommendations(self, movie_title, top_n=10):
        """Recommends movies similar to the given movie title."""
        movie = self.find_movie_by_title(movie_title)
        if movie is None:
            return None, []
            
        movie_idx = self.movie_id_to_idx[movie['movieId']]
        
        # Get TF-IDF vector of the query movie
        query_vector = self.tfidf_matrix[movie_idx]
        
        # Compute cosine similarity using optimized dot product (since TF-IDF rows are L2-normalized)
        sim_scores = (self.tfidf_matrix @ query_vector.T).toarray().flatten()
        
        # Sort indices by similarity score descending
        sim_indices = np.argsort(sim_scores)[::-1]
        
        recommendations = []
        for idx in sim_indices:
            # Exclude the query movie itself
            if idx != movie_idx:
                sim_movie = self.movies.iloc[idx]
                recommendations.append({
                    'movieId': sim_movie['movieId'],
                    'title': sim_movie['title'],
                    'genres': sim_movie['genres'],
                    'similarity': float(sim_scores[idx])
                })
                if len(recommendations) >= top_n:
                    break
                    
        return movie, recommendations

    def get_personalized_recommendations(self, user_id, top_n=10):
        """Recommends movies based on the user profile vector constructed from their ratings."""
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        if user_ratings.empty:
            return [], [] # Return empty ratings history and empty recommendations
            
        # Get user's rating history
        rated_movie_ids = user_ratings['movieId'].values
        ratings_val = user_ratings['rating'].values
        
        # Build rating history info to return for display
        history = []
        valid_indices = []
        weights = []
        
        for m_id, r in zip(rated_movie_ids, ratings_val):
            if m_id in self.movie_id_to_idx:
                idx = self.movie_id_to_idx[m_id]
                m_info = self.movies.iloc[idx]
                history.append({
                    'movieId': m_id,
                    'title': m_info['title'],
                    'genres': m_info['genres'],
                    'rating': r
                })
                valid_indices.append(idx)
                # Weight: center around 2.5 (neutral rating)
                weights.append(r - 2.5)
                
        # Sort history by rating descending for display
        history = sorted(history, key=lambda x: x['rating'], reverse=True)
        
        if not valid_indices:
            return history, []
            
        # Retrieve the TF-IDF vectors of the rated movies
        user_movie_vectors = self.tfidf_matrix[valid_indices] # shape: (len(valid_indices), num_features)
        
        # Compute weighted profile vector using optimized sparse matrix multiplication (W @ X)
        weights_arr = np.array(weights).reshape(1, -1)
        profile_vector = weights_arr @ user_movie_vectors
        profile_vector = normalize(profile_vector) # Normalize user profile vector
        
        # Compute similarity between user profile and all movies using dot product (W @ X)
        sim_scores = (self.tfidf_matrix @ profile_vector.T).flatten()
        
        # Sort indices by similarity score descending
        sim_indices = np.argsort(sim_scores)[::-1]
        
        rated_indices_set = set(valid_indices)
        recommendations = []
        for idx in sim_indices:
            # Exclude movies the user has already rated
            if idx not in rated_indices_set:
                rec_movie = self.movies.iloc[idx]
                recommendations.append({
                    'movieId': rec_movie['movieId'],
                    'title': rec_movie['title'],
                    'genres': rec_movie['genres'],
                    'similarity': float(sim_scores[idx])
                })
                if len(recommendations) >= top_n:
                    break
                    
        return history, recommendations

    def get_custom_user_recommendations(self, custom_ratings, top_n=10):
        """Recommends movies for a new user based on their custom rating history.
        
        custom_ratings: List of tuples [(movieId, rating), ...]
        """
        if not custom_ratings:
            return [], []
            
        history = []
        valid_indices = []
        weights = []
        
        for m_id, r in custom_ratings:
            if m_id in self.movie_id_to_idx:
                idx = self.movie_id_to_idx[m_id]
                m_info = self.movies.iloc[idx]
                history.append({
                    'movieId': m_id,
                    'title': m_info['title'],
                    'genres': m_info['genres'],
                    'rating': r
                })
                valid_indices.append(idx)
                # Center weights around 2.5
                weights.append(r - 2.5)
                
        history = sorted(history, key=lambda x: x['rating'], reverse=True)
        
        if not valid_indices:
            return history, []
            
        # Retrieve the TF-IDF vectors of the rated movies
        user_movie_vectors = self.tfidf_matrix[valid_indices] # shape: (len(valid_indices), num_features)
        
        # Compute weighted profile vector using optimized sparse matrix multiplication (W @ X)
        weights_arr = np.array(weights).reshape(1, -1)
        profile_vector = weights_arr @ user_movie_vectors
        profile_vector = normalize(profile_vector) # Normalize user profile vector
        
        # Compute similarity between user profile and all movies using dot product (W @ X)
        sim_scores = (self.tfidf_matrix @ profile_vector.T).flatten()
        
        # Sort indices by similarity score descending
        sim_indices = np.argsort(sim_scores)[::-1]
        
        rated_indices_set = set(valid_indices)
        recommendations = []
        for idx in sim_indices:
            # Exclude movies the user has already rated
            if idx not in rated_indices_set:
                rec_movie = self.movies.iloc[idx]
                recommendations.append({
                    'movieId': rec_movie['movieId'],
                    'title': rec_movie['title'],
                    'genres': rec_movie['genres'],
                    'similarity': float(sim_scores[idx])
                })
                if len(recommendations) >= top_n:
                    break
                    
        return history, recommendations
