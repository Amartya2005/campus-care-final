import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# --- Step 1: Define file paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'complaints_1k_dataset.csv')

print("--- Starting Model Training ---")

# --- Step 2: Load data and prepare for training ---
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"FATAL ERROR: The dataset file 'complaints_1k_dataset.csv' was not found in your project folder.")
    print("Please add the dataset file and try again.")
    exit()

df.columns = ['Complain', 'Category', 'Name', 'Roll', 'Regd', 'Hostel', 'Room', 'Anonymous']
df.dropna(subset=['Complain'], inplace=True)
print("Dataset loaded successfully.")

# --- Step 3: Train and Save the Priority Model ---
print("Training Priority Model...")
X_priority = df['Complain']
priority_keywords = {
    3: ['leakage', 'not working', 'cancelled', 'delay', 'humiliates', 'harassment', 'safety', 'security', 'fire', 'emergency'],
    2: ['lacks proper equipment', 'not responding', 'not processed yet', 'not available', 'slow', 'portal is not working'],
    1: ['lights are not functioning', 'fan is not working', 'washroom', 'cleanliness', 'mess food', 'syllabus']
}
def create_priority_labels(complaint_text):
    complaint_text = str(complaint_text).lower()
    for priority, keywords in priority_keywords.items():
        for keyword in keywords:
            if keyword in complaint_text:
                return priority
    return 0
y_priority = df['Complain'].apply(create_priority_labels)

priority_vectorizer = TfidfVectorizer(max_features=1000)
X_priority_tfidf = priority_vectorizer.fit_transform(X_priority)
priority_model = LogisticRegression(random_state=42)
priority_model.fit(X_priority_tfidf, y_priority)

joblib.dump(priority_model, os.path.join(BASE_DIR, 'complaint_classifier_model.pkl'))
joblib.dump(priority_vectorizer, os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))
print("✅ Priority model saved.")


# --- Step 4: Train and Save the Category Model ---
print("Training Category Model...")
X_category = df['Complain']
y_category = df['Category'].apply(lambda x: 1 if x == 'academic' else 0)

category_vectorizer = TfidfVectorizer(max_features=1000)
X_category_tfidf = category_vectorizer.fit_transform(X_category)
category_model = LogisticRegression(random_state=42)
category_model.fit(X_category_tfidf, y_category)

joblib.dump(category_model, os.path.join(BASE_DIR, 'category_model.pkl'))
joblib.dump(category_vectorizer, os.path.join(BASE_DIR, 'category_vectorizer.pkl'))
print("✅ Category model saved.")
print("\n--- Model Training Complete ---")