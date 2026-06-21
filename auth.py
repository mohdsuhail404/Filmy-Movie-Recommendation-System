import os
import re
import sqlite3
import hashlib
import secrets

DB_PATH = "data/users.db"

def init_db():
    """Initializes the SQLite database and creates the users and custom_ratings tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            movielens_userId INTEGER
        )
    """)
    
    # Custom ratings table for newly registered users who rate movies inside the app
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS custom_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            movieId INTEGER NOT NULL,
            rating REAL NOT NULL,
            timestamp INTEGER NOT NULL,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    
    conn.commit()
    conn.close()

def hash_password(password, salt=None):
    """Hashes a password with SHA-256 and a random or provided salt."""
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Use PBKDF2 for secure hashing
    pw_hash = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000
    ).hex()
    
    return pw_hash, salt

def validate_username(username):
    """Checks if a username satisfies character rules and is available."""
    if not username:
        return False, "Username cannot be empty."
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters long."
        
    if not re.match(r"^[a-zA-Z0-9._]+$", username):
        return False, "Username can only contain alphanumeric characters, dots (.), and underscores (_)."
    
    # Check availability in DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    
    if row is not None:
        return False, "Username is already taken."
        
    return True, "Username is available."

def validate_email(email):
    """Checks email validity and standard email domain authenticity."""
    if not email:
        return False, "Email cannot be empty."
        
    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_regex, email):
        return False, "Invalid email format (e.g. user@domain.com)."
        
    domain = email.split("@")[1].lower()
    
    # Basic check on domain structure (letters, numbers, dots, hyphens only)
    if not re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}$", domain):
        return False, "Invalid email domain structure."
        
    return True, "Email is valid."

def validate_password(password):
    """Checks if password meets length and composition rules."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
        
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
        
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
        
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one number."
        
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)."
        
    return True, "Password is valid."

def register_user(username, email, password, movielens_userId=None):
    """Registers a new user in the database after validating rules."""
    # Run all validations
    ok, msg = validate_username(username)
    if not ok:
        return False, msg
        
    ok, msg = validate_email(email)
    if not ok:
        return False, msg
        
    ok, msg = validate_password(password)
    if not ok:
        return False, msg
        
    # Hash password
    pw_hash, salt = hash_password(password)
    
    # Save to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, salt, movielens_userId) VALUES (?, ?, ?, ?, ?)",
            (username, email, pw_hash, salt, movielens_userId)
        )
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def authenticate_user(username, password):
    """Authenticates a user by checking credentials against the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash, salt, movielens_userId FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return False, "Invalid username or password.", None
        
    saved_hash, salt, movielens_userId = row
    test_hash, _ = hash_password(password, salt)
    
    if test_hash == saved_hash:
        return True, "Login successful!", movielens_userId
    else:
        return False, "Invalid username or password.", None

def get_user_ratings(username):
    """Retrieves custom ratings submitted by a registered user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT movieId, rating FROM custom_ratings WHERE username = ?", (username,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def add_user_rating(username, movie_id, rating):
    """Saves or updates a rating submitted by a registered user."""
    import time
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if rating already exists
    cursor.execute("SELECT id FROM custom_ratings WHERE username = ? AND movieId = ?", (username, movie_id))
    row = cursor.fetchone()
    
    timestamp = int(time.time())
    if row:
        cursor.execute(
            "UPDATE custom_ratings SET rating = ?, timestamp = ? WHERE id = ?",
            (rating, timestamp, row[0])
        )
    else:
        cursor.execute(
            "INSERT INTO custom_ratings (username, movieId, rating, timestamp) VALUES (?, ?, ?, ?)",
            (username, movie_id, rating, timestamp)
        )
    conn.commit()
    conn.close()

# Initialize DB on import
init_db()
