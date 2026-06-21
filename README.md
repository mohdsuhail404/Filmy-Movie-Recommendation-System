# Filmy 🎬 — AI-Powered Movie Recommendation System

Filmy is a modern, high-performance content-based and personalized movie recommendation web application built in Python using Scikit-Learn, SQLite, and Streamlit. It leverages TF-IDF vectorization and optimized sparse matrix operations to compute and surface movie recommendations in **under 75ms**.

---

## ✨ Features

- **Double-Engine Personalization**:
  - **Movie Similarity Search**: Type or select any movie to find similar titles based on genres, keywords, and tags using TF-IDF and Cosine Similarity.
  - **Dynamic User-Profile Recommendations**: Construct a user taste profile in real-time from rating history (ratings centered and normalized), generating customized recommendations on-the-fly.
- **Secure User Accounts**:
  - Sign-in and Sign-up screens with real-time password strength checklists.
  - Strict validator rules (only alphanumeric, `.` and `_` allowed in usernames).
  - Email authenticity checks (valid formats and domains).
  - Secure password hashing using PBKDF2 with SHA-256 and unique salts.
- **Mapped MovieLens Demo Profiles**:
  - Maps 610 MovieLens active user IDs to unique, relatable human names (e.g., *Aria Smith*, *Liam Jones*) deterministically.
  - Explore their historical ratings and see personalized suggestions matching their taste.
- **Interactive Custom Ratings**:
  - Logged-in users can search for any movie and rate it directly.
  - Custom ratings are saved locally in SQLite and immediately update the user's recommendations feed.
- **Sub-75ms Latency**:
  - Similarity calculations are mathematically optimized using sparse matrix dot products, making searches feel instantaneous.

---

## 🛠️ Tech Stack

- **Core**: Python 3.14+
- **Front-End & UI**: Streamlit (with custom dark glassmorphic styling)
- **Machine Learning**: Scikit-Learn (`TfidfVectorizer`, `normalize`)
- **Data Engineering**: Pandas, NumPy
- **Database**: SQLite3

---

## 📂 Project Structure

```
Filmy/
├── data/                  # Datasets & local SQLite databases (ignored from Git)
│   ├── ml-latest-small/   # Extracted MovieLens CSVs
│   └── users.db           # SQLite database for logins and custom user ratings
├── models/                # Fitted TF-IDF vectorizer and processed matrices (ignored from Git)
├── scripts/
│   ├── download_data.py   # Script to download and unpack MovieLens 100K small
│   └── train_model.py     # Preprocessing, filtering, and fitting vectorizer
├── app.py                 # Main Streamlit web application & UI
├── auth.py                # Database queries, hashing, and credentials validator rules
├── recommender.py         # Recommendation similarity engines & matrix calculations
├── requirements.txt       # List of Python dependencies
├── .gitignore             # Excludes large binaries, databases, and venv from Git
└── README.md              # Project documentation
```

---

## 🚀 Setup & Execution

### 1. Clone & Set Up Environment
```bash
# Clone the repository (if applicable) and navigate to it
cd Filmy

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Database and Models
You can initialize the database using the command line:
```bash
python scripts/download_data.py
python scripts/train_model.py
```
*Alternatively, you can just start the Streamlit app. The UI will detect the database is missing and provide a one-click initialization button!*

### 3. Run the Application
```bash
streamlit run app.py
```
Open your browser and navigate to **`http://localhost:8501`**.

---

## 👤 Validation Rules Checklist

### Username Rules
- Must contain only letters, numbers, dots (`.`), and underscores (`_`).
- Must not already be taken by another user.

### Email Rules
- Must be a valid email format (e.g. `name@domain.com`).
- Domain suffix must be standard and properly structured.

### Password Rules
- Must be at least **8 characters** long.
- Must contain at least:
  - One lowercase letter (`a-z`)
  - One uppercase letter (`A-Z`)
  - One number (`0-9`)
  - One special character (e.g. `!`, `@`, `#`, `$`, `%`, `*`)

---

