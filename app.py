from flask import Flask, request, jsonify, render_template
from textblob import TextBlob
import os
import nltk

# --- Automatic Data Download for TextBlob ---
# This runs on server startup to make sure the language models are available.
# This section is now corrected.
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/brown')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except Exception: # Using a general Exception is more robust here
    print("Downloading TextBlob corpora...")
    nltk.download('punkt')
    nltk.download('brown')
    nltk.download('averaged_perceptron_tagger')
    print("Downloads complete.")

app = Flask(__name__)

# --- In-memory "database" ---
complaints_db = []

# --- Lightweight Models (No .pkl files or scikit-learn needed) ---

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
    
    if any(word in text for word in high_priority_words):
        score += 3
    elif any(word in text for word in medium_priority_words):
        score += 2
    else:
        score += 1
        
    emotion = analyze_sentiment(record['Complain'])
    if "Angry" in emotion:
        score += 2
        
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
    complaints_db.append(processed_data)
    return jsonify({"message": "Success"})

@app.route('/staff')
def staff_dashboard():
    return render_template('staff.html')

@app.route('/api/complaints')
def get_complaints():
    return jsonify(complaints_db)

if __name__ == '__main__':
    app.run(debug=True)