from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import pandas as pd
import joblib
import sqlite3, os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, template_folder="templates")
app.secret_key = "your_secret_key"

DB_NAME = "database.db"

# Database Setup
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

# Model Loading
binary_scaler = joblib.load("binary_scaler.pkl")
binary_model = joblib.load("xgboost_multiclass_model.pkl")
multiclass_model = joblib.load("stacking_multiclass_model.pkl")

binary_features = [
    'Age_day', 'last_funding_at', 'milestones', 'funding_total_usd',
    'first_funding_at', 'first_milestone_at', 'last_milestone_at',
    'relationships', 'funding_rounds', 'lng', 'lat', 'funding_per_round'
]

if hasattr(multiclass_model, "feature_names_in_"):
    multiclass_features = multiclass_model.feature_names_in_
else:
    multiclass_features = []

# Feature Preparation
def prepare_features(data):
    Age_day = (2025 - data['founded_at']) * 365
    funding_per_round = data['funding_total_usd'] / max(data['funding_rounds'], 1)

    features = {
        'founded_at': data['founded_at'],
        'first_funding_at': data['first_funding_at'],
        'last_funding_at': data['last_funding_at'],
        'funding_rounds': data['funding_rounds'],
        'funding_total_usd': data['funding_total_usd'],
        'first_milestone_at': data['first_milestone_at'],
        'last_milestone_at': data['last_milestone_at'],
        'milestones': data['milestones'],
        'relationships': data['relationships'],
        'investment_rounds': data['investment_rounds'],
        'lat': data['lat'],
        'lng': data['lng'],
        'Age_day': Age_day,
        'funding_per_round': funding_per_round,
        f'category_{data["category"]}': 1,
        f'country_{data["country"]}': 1
    }

    return pd.DataFrame([features])

# Auth Routes
@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session['username'] = username
            flash("Login successful!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid credentials", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out successfully!", "info")
    return redirect(url_for('login'))

# Prediction Routes
@app.route("/predict-binary", methods=["POST"])
def predict_binary():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized. Please login first."}), 403

    try:
        data = request.get_json()
        input_df = prepare_features(data)
        X_binary = input_df[binary_features]
        X_scaled = binary_scaler.transform(X_binary)
        binary_pred = binary_model.predict(X_scaled)
        status = "Active" if binary_pred[0] == 1 else "Closed"
        return jsonify({"binary_prediction": status})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------- Prediction Routes ----------------
@app.route("/predict-multiclass", methods=["POST"])
def predict_multiclass():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request. No data received."}), 400

        input_df = prepare_features(data)

        # Handle missing columns for model
        df_dummies = pd.get_dummies(input_df)
        for col in multiclass_features:
            if col not in df_dummies.columns:
                df_dummies[col] = 0
        df_dummies = df_dummies[multiclass_features]

        prediction = multiclass_model.predict(df_dummies)[0]
        status_map = {0: "Acquired", 1: "Operating", 2: "IPO", 3: "Closed"}

        return jsonify({
            "multiclass_prediction": int(prediction),
            "status": status_map.get(int(prediction), "Unknown")
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
