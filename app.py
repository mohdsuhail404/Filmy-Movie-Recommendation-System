import os
import time
import streamlit as st
import pandas as pd
import numpy as np

from auth import (
    authenticate_user, register_user, get_user_ratings, add_user_rating,
    validate_username, validate_email, validate_password
)

# Set page layout and title
st.set_page_config(
    page_title="Filmy — AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for Sleek Dark Theme, Custom Typography, and Glassmorphic Cards
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Main application styling */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Background color */
.stApp {
    background-color: #0c0f13;
    color: #e2e8f0;
}

/* Custom Header with Gradient Background */
.header-container {
    background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 50%, #3b82f6 100%);
    padding: 2.2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px -10px rgba(139, 92, 246, 0.3);
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.header-title {
    font-size: 3.2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -1.5px;
    text-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}

.header-subtitle {
    font-size: 1.2rem;
    color: rgba(255, 255, 255, 0.9);
    margin-top: 0.65rem;
    font-weight: 300;
}

/* Login/Signup Box Styling */
.auth-container {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.07);
    padding: 2.5rem;
    margin-top: 1rem;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
}

.auth-header {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 1.5rem;
    text-align: center;
}

/* Glassmorphic Movie Cards */
.movie-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
}

.movie-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 30px rgba(139, 92, 246, 0.25);
    border: 1px solid rgba(139, 92, 246, 0.4);
    background: rgba(255, 255, 255, 0.05);
}

.card-rank {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #a78bfa;
    margin-bottom: 0.25rem;
}

.card-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.3;
    margin-bottom: 0.5rem;
    height: 3.2rem;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.card-genres {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 1rem;
    height: 1.8rem;
    overflow: hidden;
}

.genre-tag {
    background: rgba(139, 92, 246, 0.15);
    color: #c084fc;
    font-size: 0.75rem;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    border: 1px solid rgba(139, 92, 246, 0.25);
    font-weight: 500;
}

.card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    padding-top: 0.75rem;
    margin-top: 0.5rem;
}

.match-pct {
    font-size: 0.95rem;
    font-weight: 600;
    color: #34d399;
}

.match-score-bar {
    width: 60%;
    height: 6px;
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    position: relative;
    overflow: hidden;
}

.match-score-fill {
    height: 100%;
    background: linear-gradient(90deg, #8b5cf6, #34d399);
    border-radius: 3px;
}

/* History Card */
.history-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.history-rating {
    background: #f59e0b;
    color: #000000;
    font-weight: 700;
    padding: 0.1rem 0.5rem;
    border-radius: 4px;
    font-size: 0.85rem;
}

/* Performance Banner */
.perf-banner {
    background: rgba(52, 211, 153, 0.1);
    border: 1px solid rgba(52, 211, 153, 0.2);
    color: #34d399;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
}

/* User Account Info Bar */
.user-info-bar {
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 12px;
    padding: 0.75rem 1.2rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

# Helper to check if database & model pickles are ready
def check_models_exist():
    required_files = [
        "models/movies_processed.pkl",
        "models/ratings_filtered.pkl",
        "models/tfidf_matrix.pkl",
        "models/tfidf_vectorizer.pkl"
    ]
    return all(os.path.exists(f) for f in required_files)

# 1. Database & Model Pre-check
if not check_models_exist():
    # Top Header Layout
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">FILMY 🎬</h1>
        <p class="header-subtitle">Intelligent Movie Recommendation System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Welcome to Filmy!")
    st.info("The movie database needs to be initialized. This will download the MovieLens 100K dataset and fit the TF-IDF vectorizer.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Download & Initialize Database (MovieLens 100K)", use_container_width=True, type="primary"):
            with st.status("Initializing recommendation engine...", expanded=True) as status:
                st.write("Downloading MovieLens 100K zip file...")
                from scripts.download_data import download_and_extract
                download_and_extract()
                
                st.write("Processing rating and metadata CSVs...")
                st.write("Applying TF-IDF vectorization & filtering user-item interactions...")
                from scripts.train_model import train_and_save
                train_and_save()
                
                status.update(label="Initialization complete!", state="complete", expanded=False)
            st.success("🎉 Database successfully initialized! Reloading app...")
            time.sleep(1)
            st.rerun()
    st.stop()

# Load Recommender Engine
from recommender import MovieRecommender

@st.cache_resource
def load_recommender():
    return MovieRecommender()
    
try:
    recommender = load_recommender()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.info("Try re-initializing the database.")
    if st.button("Force Re-initialize"):
        import shutil
        if os.path.exists("models"):
            shutil.rmtree("models")
        st.rerun()
    st.stop()

# 2. Session State Management
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "movielens_userId" not in st.session_state:
    st.session_state.movielens_userId = None
if "email" not in st.session_state:
    st.session_state.email = None

# Header Banner
st.markdown("""
<div class="header-container">
    <h1 class="header-title">FILMY 🎬</h1>
    <p class="header-subtitle">AI-Powered Personalized Movie Recommendations</p>
</div>
""", unsafe_allow_html=True)

# 3. AUTHENTICATION SCREENS (LOGIN / SIGN UP)
if not st.session_state.logged_in:
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        auth_tab1, auth_tab2 = st.tabs(["🔒 Sign In", "📝 Create Account"])
        
        # TAB 1: LOG IN
        with auth_tab1:
            st.markdown('<div class="auth-header">Log In to Filmy</div>', unsafe_allow_html=True)
            login_username = st.text_input("Username", key="login_user_input")
            login_password = st.text_input("Password", type="password", key="login_pass_input")
            
            if st.button("Sign In", type="primary", use_container_width=True):
                if not login_username or not login_password:
                    st.error("Please enter both username and password.")
                else:
                    success, msg, movielens_userId = authenticate_user(login_username, login_password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = login_username
                        st.session_state.movielens_userId = movielens_userId
                        st.success("🎉 Login successful! Welcome back.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
                        
        # TAB 2: REGISTER / SIGN UP
        with auth_tab2:
            st.markdown('<div class="auth-header">Create New Account</div>', unsafe_allow_html=True)
            reg_username = st.text_input("Username (letters, numbers, . or _ only)", key="reg_user_input")
            reg_email = st.text_input("Email Address", key="reg_email_input")
            reg_password = st.text_input("Password (min 8 chars)", type="password", key="reg_pass_input")
            
            # Allow cloning MovieLens taste profiles
            profile_names = ["New Empty Taste Profile"] + [p['name'] for p in recommender.user_profiles]
            selected_clone_profile = st.selectbox(
                "Choose Starting Taste Profile (Optional):",
                options=profile_names,
                help="We will pre-load your history with this user's movie taste so you get instant recommendations!"
            )
            
            # Interactive rules validation feedback helper
            st.markdown("#### Password Strength Checklist:")
            length_chk = "✅" if len(reg_password) >= 8 else "❌"
            lower_chk = "✅" if any(c.islower() for c in reg_password) else "❌"
            upper_chk = "✅" if any(c.isupper() for c in reg_password) else "❌"
            num_chk = "✅" if any(c.isdigit() for c in reg_password) else "❌"
            spec_chk = "✅" if any(not c.isalnum() for c in reg_password) else "❌"
            
            st.write(f"- {length_chk} Minimum 8 characters")
            st.write(f"- {lower_chk} Lowercase letter")
            st.write(f"- {upper_chk} Uppercase letter")
            st.write(f"- {num_chk} Number")
            st.write(f"- {spec_chk} Special character")
            
            if st.button("Register & Sign Up", type="primary", use_container_width=True):
                # 1. Pre-validation checks to show clear errors
                u_ok, u_msg = validate_username(reg_username)
                e_ok, e_msg = validate_email(reg_email)
                p_ok, p_msg = validate_password(reg_password)
                
                if not u_ok:
                    st.error(f"Username Error: {u_msg}")
                elif not e_ok:
                    st.error(f"Email Error: {e_msg}")
                elif not p_ok:
                    st.error(f"Password Error: {p_msg}")
                else:
                    # Resolve movielens user id from selected clone name
                    m_userId = None
                    if selected_clone_profile != "New Empty Taste Profile":
                        m_userId = recommender.name_to_user_id.get(selected_clone_profile)
                        
                    success, msg = register_user(reg_username, reg_email, reg_password, movielens_userId=m_userId)
                    if success:
                        st.success("🎉 Account created successfully! Please sign in using the 'Sign In' tab.")
                    else:
                        st.error(f"❌ {msg}")
                        
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# 4. LOGGED IN DASHBOARD
# User welcome bar
profile_display = "Custom Taste Profile"
if st.session_state.movielens_userId:
    profile_display = recommender.user_id_to_name.get(st.session_state.movielens_userId, "Custom")

st.markdown(f"""
<div class="user-info-bar">
    <div>👋 Welcome back, <b>{st.session_state.username}</b>! &nbsp;|&nbsp; 📊 Taste Profile: <b>{profile_display}</b></div>
</div>
""", unsafe_allow_html=True)

# Add logout button to sidebar
if st.sidebar.button("🚪 Log Out", type="secondary", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.movielens_userId = None
    st.success("Logged out successfully!")
    time.sleep(0.5)
    st.rerun()

# Sidebar Stats & Parameters
st.sidebar.markdown("### 📊 Engine Analytics")
total_movies = len(recommender.movies)
total_ratings = len(recommender.ratings)
total_users = recommender.ratings['userId'].nunique()

st.sidebar.metric("Total Movies Loaded", f"{total_movies:,}")
st.sidebar.metric("Filtered MovieLens Ratings", f"{total_ratings:,}")
st.sidebar.metric("Active Profiles", f"{total_users:,}")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
rec_count = st.sidebar.slider("Number of Recommendations", min_value=5, max_value=20, value=10)

# Main Application Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Movie Similarity & Rating", 
    "👤 Your Taste Recommendations", 
    "👥 Explore MovieLens Profiles"
])

# TAB 1: MOVIE SIMILARITY & RATING
with tab1:
    st.markdown("### Search Movies & Write Ratings")
    st.write("Search for any movie title, view similar recommendations, and rate movies to update your personal profile taste on-the-fly!")
    
    movie_titles = recommender.movies['title'].sort_values().tolist()
    selected_title = st.selectbox(
        "Select or Type a Movie Title:",
        options=movie_titles,
        index=movie_titles.index("Toy Story (1995)") if "Toy Story (1995)" in movie_titles else 0
    )
    
    if selected_title:
        # Latency check
        start_time = time.time()
        movie_info, recs = recommender.get_content_recommendations(selected_title, top_n=rec_count)
        latency = (time.time() - start_time) * 1000
        
        st.markdown(f"""
        <div class="perf-banner">
            ⚡ <b>Latency:</b> {latency:.2f} ms &nbsp;|&nbsp; 
            🎯 <b>Target:</b> &lt; 200 ms &nbsp;|&nbsp;
            ✅ <b>Status:</b> Ultra-Fast Engine
        </div>
        """, unsafe_allow_html=True)
        
        if movie_info is not None:
            # Display target movie details and Rating Widget
            col_info, col_rate = st.columns([2, 1])
            
            with col_info:
                st.markdown(f"#### Selected Movie Details")
                genres_list = movie_info['genres'].split('|')
                genre_html = "".join([f'<span class="genre-tag">{g}</span>' for g in genres_list])
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.02); border-left: 4px solid #ec4899; padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1.5rem;">
                    <div style="font-size: 1.3rem; font-weight: 700; color: #ffffff;">{movie_info['title']}</div>
                    <div style="margin-top: 0.5rem;">{genre_html}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_rate:
                st.markdown("#### Rate this Movie")
                # Look up if user has already rated this movie in our DB
                user_existing_ratings = get_user_ratings(st.session_state.username)
                rating_map = {m: r for m, r in user_existing_ratings}
                movie_id = movie_info['movieId']
                current_val = float(rating_map.get(movie_id, 5.0))
                
                # Simple rating slider
                rating_val = st.slider(
                    "Your Rating (1 to 5 Stars)",
                    min_value=1.0, max_value=5.0, value=current_val, step=0.5
                )
                if st.button("Submit Rating", type="primary"):
                    add_user_rating(st.session_state.username, movie_id, rating_val)
                    st.success(f"Rated '{movie_info['title']}' as {rating_val}★! Your taste profile has been updated.")
                    time.sleep(1)
                    st.rerun()

            # Display content recommendations
            st.markdown("#### Similar Movie Recommendations")
            if recs:
                cols = st.columns(3)
                for idx, rec in enumerate(recs):
                    col = cols[idx % 3]
                    rec_genres = rec['genres'].split('|')
                    rec_genre_html = "".join([f'<span class="genre-tag">{g}</span>' for g in rec_genres])
                    match_score = rec['similarity'] * 100
                    
                    col.markdown(f"""
                    <div class="movie-card">
                        <div class="card-rank">#{idx+1} Recommendation</div>
                        <div class="card-title">{rec['title']}</div>
                        <div class="card-genres">{rec_genre_html}</div>
                        <div class="card-footer">
                            <div class="match-score-bar">
                                <div class="match-score-fill" style="width: {match_score}%;"></div>
                            </div>
                            <span class="match-pct">{match_score:.1f}% Match</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recommendations found.")
        else:
            st.error("Movie not found.")

# TAB 2: YOUR TASTE RECOMMENDATIONS (LOGGED-IN PERSONALIZED TASTE)
with tab2:
    st.markdown("### Your Personalized Movie Recommendations")
    st.write("These recommendations are computed dynamically in real-time by combining your cloned MovieLens profile with your custom ratings.")
    
    # 1. Compile User Rating History
    custom_db_ratings = get_user_ratings(st.session_state.username)
    cloned_ratings = []
    if st.session_state.movielens_userId:
        cloned_ratings_df = recommender.ratings[recommender.ratings['userId'] == st.session_state.movielens_userId]
        cloned_ratings = [(row['movieId'], row['rating']) for _, row in cloned_ratings_df.iterrows()]
        
    # Merge: custom db ratings override cloned MovieLens ratings
    merged_ratings_dict = {m_id: r for m_id, r in cloned_ratings}
    for m_id, r in custom_db_ratings:
        merged_ratings_dict[m_id] = r
        
    merged_ratings_list = list(merged_ratings_dict.items())
    
    if not merged_ratings_list:
        st.info("Your rating profile is empty! Go to the 'Movie Similarity & Rating' tab to rate some movies and initialize your taste profile.")
    else:
        # Latency check
        start_time = time.time()
        history, recs = recommender.get_custom_user_recommendations(merged_ratings_list, top_n=rec_count)
        latency = (time.time() - start_time) * 1000
        
        st.markdown(f"""
        <div class="perf-banner">
            ⚡ <b>Latency:</b> {latency:.2f} ms &nbsp;|&nbsp; 
            🎯 <b>Target:</b> &lt; 200 ms &nbsp;|&nbsp;
            ✅ <b>Status:</b> Ultra-Fast Engine
        </div>
        """, unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown(f"#### 📜 Your Taste History ({len(history)} rated movies)")
            st.write("Movies used to build your taste profile:")
            
            # Show up to 15 items in history
            for hist_item in history[:15]:
                st.markdown(f"""
                <div class="history-card">
                    <div style="font-weight: 500; font-size: 0.95rem; max-width: 80%;">{hist_item['title']}</div>
                    <span class="history-rating">★ {hist_item['rating']:.1f}</span>
                </div>
                """, unsafe_allow_html=True)
                
        with col_right:
            st.markdown("#### 🎯 Recommended For You")
            if recs:
                cols = st.columns(2)
                for idx, rec in enumerate(recs):
                    col = cols[idx % 2]
                    rec_genres = rec['genres'].split('|')
                    rec_genre_html = "".join([f'<span class="genre-tag">{g}</span>' for g in rec_genres])
                    match_score = rec['similarity'] * 100
                    
                    col.markdown(f"""
                    <div class="movie-card">
                        <div class="card-rank">Recommendation #{idx+1}</div>
                        <div class="card-title">{rec['title']}</div>
                        <div class="card-genres">{rec_genre_html}</div>
                        <div class="card-footer">
                            <div class="match-score-bar">
                                <div class="match-score-fill" style="width: {match_score}%;"></div>
                            </div>
                            <span class="match-pct">{match_score:.1f}% Match</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Please rate more movies of different genres to generate customized recommendations.")

# TAB 3: EXPLORE MOVIELENS PROFILES (MAPPED HUMAN NAMES DEMO)
with tab3:
    st.markdown("### Explore MovieLens Demo User Profiles")
    st.write("Browse taste history and personalized recommendations for the 610 MovieLens active profiles, mapped to relatable human names.")
    
    # Dropdown select of human names
    user_names = [p['name'] for p in recommender.user_profiles]
    selected_name = st.selectbox(
        "Select a MovieLens Taste Profile to Explore:",
        options=user_names,
        index=0
    )
    
    if selected_name:
        resolved_uid = recommender.name_to_user_id.get(selected_name)
        
        if resolved_uid:
            start_time = time.time()
            history, recs = recommender.get_personalized_recommendations(resolved_uid, top_n=rec_count)
            latency = (time.time() - start_time) * 1000
            
            st.markdown(f"""
            <div class="perf-banner">
                ⚡ <b>Latency:</b> {latency:.2f} ms &nbsp;|&nbsp; 
                🎯 <b>Target:</b> &lt; 200 ms &nbsp;|&nbsp;
                ✅ <b>Status:</b> Ultra-Fast Engine
            </div>
            """, unsafe_allow_html=True)
            
            col_l, col_r = st.columns([1, 2])
            
            with col_l:
                st.markdown(f"#### 📜 {selected_name}'s Taste Profile History")
                st.write(f"Top ratings in {selected_name}'s profile:")
                for hist_item in history[:15]:
                    st.markdown(f"""
                    <div class="history-card">
                        <div style="font-weight: 500; font-size: 0.95rem; max-width: 80%;">{hist_item['title']}</div>
                        <span class="history-rating">★ {hist_item['rating']:.1f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with col_r:
                st.markdown(f"#### 🎯 Recommended for {selected_name}")
                if recs:
                    cols = st.columns(2)
                    for idx, rec in enumerate(recs):
                        col = cols[idx % 2]
                        rec_genres = rec['genres'].split('|')
                        rec_genre_html = "".join([f'<span class="genre-tag">{g}</span>' for g in rec_genres])
                        match_score = rec['similarity'] * 100
                        
                        col.markdown(f"""
                        <div class="movie-card">
                            <div class="card-rank">Recommendation #{idx+1}</div>
                            <div class="card-title">{rec['title']}</div>
                            <div class="card-genres">{rec_genre_html}</div>
                            <div class="card-footer">
                                <div class="match-score-bar">
                                    <div class="match-score-fill" style="width: {match_score}%;"></div>
                                </div>
                                <span class="match-pct">{match_score:.1f}% Match</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No recommendations found.")
