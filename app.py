from flask import Flask, request, jsonify, render_template
from textblob import TextBlob
import os
import sqlite3 # Import the database library

app = Flask(__name__)

# --- Database Setup ---
DB_FILE = "complaints.db"

def init_db():
    """Creates the database table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complain_text TEXT NOT NULL,
            category TEXT,
            submitter_name TEXT,
            roll_number TEXT,
            predicted_priority TEXT,
            detected_emotion TEXT,
            key_tokens TEXT,
            is_anonymous BOOLEAN
        )
    ''')
    conn.commit()
    conn.close()

# --- Lightweight Models ---

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity < -0.2: return "Angry / Negative"
    elif polarity < 0: return "Slightly Negative"
    elif polarity == 0: return "Neutral"
    else: return "Positive"

def extract_key_tokens(text):
    return list(TextBlob(text).noun_phrases)

def intelligent_priority_scorer(record):
    text = record['Complain'].lower()
    score = 0
    
    high_priority_words = ['leakage', 'not working', 'cancelled', 'delay', 'humiliates', 'harassment', 'safety', 'security', 'fire', 'emergency', 'unacceptable', 'fail', 'error']
    medium_priority_words = ['slow', 'equipment', 'responding', 'processed', 'available']
    
    if any(word in text for word in high_priority_words): score += 3
    elif any(word in text for word in medium_priority_words): score += 2
    else: score += 1
        
    emotion = analyze_sentiment(record['Complain'])
    if "Angry" in emotion: score += 2
        
    if score >= 5: priority = "Critical"
    elif score >= 3: priority = "High"
    elif score >= 2: priority = "Medium"
    else: priority = "Low"
        
    if record.get('Category') == 'Academic':
        record['Anonymous'] = True
        record['Name'] = 'Anonymous'
    
    record['Predicted_Priority'] = priority
    record['Detected_Emotion'] = emotion
    record['Key_Tokens'] = extract_key_tokens(record['Complain'])
    
    return record

# --- Web Routes ---

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    processed_data = intelligent_priority_scorer(data)
    
    # --- Save to Database ---
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO complaints (complain_text, category, submitter_name, roll_number, predicted_priority, detected_emotion, key_tokens, is_anonymous)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        processed_data['Complain'],
        processed_data['Category'],
        processed_data['Name'],
        processed_data['Roll'],
        processed_data['Predicted_Priority'],
        processed_data['Detected_Emotion'],
        ", ".join(processed_data['Key_Tokens']),
        processed_data['Anonymous']
    ))
    conn.commit()
    conn.close()
    
    return jsonify({"message": "Success"})

@app.route('/staff')
def staff_dashboard():
    return render_template('staff.html')

@app.route('/api/complaints')
def get_complaints():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row # This allows us to access columns by name
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM complaints ORDER BY id DESC')
    complaints = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(complaints)

# --- Initialize Database on Startup ---
init_db()

if __name__ == '__main__':
    app.run(debug=True)