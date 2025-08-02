from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import os
from textblob import TextBlob
import spacy

app = Flask(__name__)

# --- In-memory "database" to store complaints ---
# In a real app, this would be a real database.
complaints_db = []

# --- Load All Models ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
priority_model = joblib.load(os.path.join(BASE_DIR, 'complaint_classifier_model.pkl'))
priority_vectorizer = joblib.load(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))
category_model = joblib.load(os.path.join(BASE_DIR, 'category_model.pkl'))
category_vectorizer = joblib.load(os.path.join(BASE_DIR, 'category_vectorizer.pkl'))
nlp = spacy.load("en_core_web_sm")

# --- Define All Helper Functions ---
def analyze_sentiment(complaint_text):
    analysis = TextBlob(complaint_text)
    polarity = analysis.sentiment.polarity
    if polarity < -0.2: return "Angry / Very Negative"
    elif polarity < 0: return "Slightly Negative"
    elif polarity == 0: return "Neutral"
    else: return "Positive"

def extract_key_tokens(complaint_text):
    doc = nlp(complaint_text)
    return [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop]

def improved_anonymizer(complaint_record):
    if complaint_record.get('Category') == 'Academic':
        complaint_record['Anonymous'] = True
        complaint_record['Name'] = 'Anonymous'
    return complaint_record

def intelligent_priority_scorer(complaint_record):
    complaint_text = complaint_record['Complain']
    anonymized_record = improved_anonymizer(complaint_record)
    detected_emotion = analyze_sentiment(complaint_text)
    key_tokens = extract_key_tokens(complaint_text)
    priority_score = 0
    base_prediction = {0: 0, 1: 1, 2: 2, 3: 3}[priority_model.predict(priority_vectorizer.transform([complaint_text]))[0]]
    priority_score += base_prediction
    if detected_emotion == "Angry / Very Negative": priority_score += 2
    elif detected_emotion == "Slightly Negative": priority_score += 1
    critical_keywords = ['safety', 'fire', 'emergency', 'harassment', 'unacceptable', 'legal']
    for token in key_tokens:
        if token.lower() in critical_keywords:
            priority_score += 3
            break
    if priority_score >= 5: final_priority = "Critical"
    elif priority_score >= 3: final_priority = "High"
    elif priority_score >= 2: final_priority = "Medium"
    else: final_priority = "Low"
    anonymized_record['Predicted_Priority'] = final_priority
    anonymized_record['Detected_Emotion'] = detected_emotion
    anonymized_record['Key_Tokens'] = key_tokens
    return anonymized_record

# --- Define Web Routes ---

# Main page for students
@app.route('/')
def home():
    return render_template('index2.html')

# Endpoint to process the form submission
@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    processed_data = intelligent_priority_scorer(data)
    # Add the processed complaint to our "database"
    complaints_db.append(processed_data)
    # Return a success message
    return jsonify({"message": "Success"})

# NEW: A private dashboard for staff
@app.route('/staff')
def staff_dashboard():
    return render_template('staff.html')

# NEW: An API endpoint to provide data for the staff dashboard
@app.route('/api/complaints')
def get_complaints():
    return jsonify(complaints_db)

if __name__ == '__main__':
    app.run(debug=True)