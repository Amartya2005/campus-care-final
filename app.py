
from flask import Flask, request, jsonify, render_template
import joblib
import os
from textblob import TextBlob
import re

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load All Models (No spaCy needed) ---
priority_model = joblib.load(os.path.join(BASE_DIR, 'complaint_classifier_model.pkl'))
priority_vectorizer = joblib.load(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))
category_model = joblib.load(os.path.join(BASE_DIR, 'category_model.pkl'))
category_vectorizer = joblib.load(os.path.join(BASE_DIR, 'category_vectorizer.pkl'))

# --- Define All Helper Functions ---
def analyze_sentiment(complaint_text):
    analysis = TextBlob(complaint_text)
    polarity = analysis.sentiment.polarity
    if polarity < -0.2: return "Angry / Very Negative"
    elif polarity < 0: return "Slightly Negative"
    elif polarity == 0: return "Neutral"
    else: return "Positive"

def lightweight_token_extractor(complaint_text):
    # A simpler way to get important words without a heavy library
    # This finds words that start with a capital letter (likely nouns)
    # or common adjectives.
    blob = TextBlob(complaint_text)
    # We will grab nouns and adjectives
    tokens = [word for (word, tag) in blob.tags if tag.startswith('NN') or tag.startswith('JJ')]
    return list(set(tokens)) # Use set to get unique tokens

def improved_anonymizer(complaint_record):
    if complaint_record.get('Category') == 'Academic':
        complaint_record['Anonymous'] = True
        complaint_record['Name'] = 'Anonymous'
    return complaint_record

def intelligent_priority_scorer(complaint_record):
    complaint_text = complaint_record['Complain']
    anonymized_record = improved_anonymizer(complaint_record)
    detected_emotion = analyze_sentiment(complaint_text)
    key_tokens = lightweight_token_extractor(complaint_text) # Use the new lightweight function
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    return jsonify(intelligent_priority_scorer(data))
