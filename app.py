from flask import Flask, request, jsonify
import joblib
import re
import spacy
import nltk
from nltk.corpus import stopwords
import requests
import csv
import os
import logging
from datetime import datetime

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load SpaCy model for French
nlp = spacy.load('fr_core_news_sm')

# Function to resolve contractions using SpaCy
def resolve_contractions_spacy(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return ' '.join(tokens)

# Basic text cleaning function with POS tagging and lemmatization
def preprocess_text(text):
    # Resolve contractions
    text = resolve_contractions_spacy(text)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Process text with SpaCy for tokenization, POS tagging, and lemmatization
    doc = nlp(text)
    # Remove stopwords and perform lemmatization
    french_stopwords = set(stopwords.words('french'))
    words = [token.lemma_ for token in doc if token.text not in french_stopwords and token.pos_ != 'PUNCT' and len(token.text) > 1]
    # Rejoin words into a single string
    cleaned_text = ' '.join(words)
    # Remove trailing spaces and multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Function to predict the class of a given input text using a specific model and vectorizer
def predict_class(input_text, model, vectorizer):
    # Preprocess text
    input_text_processed = preprocess_text(input_text)
    # Vectorize input text
    input_text_tfidf = vectorizer.transform([input_text_processed])
    # Predict probabilities
    probabilities = model.predict_proba(input_text_tfidf)
    # Predict class
    predicted_class = model.predict(input_text_tfidf)
    return predicted_class[0], probabilities[0]

# Function to predict all classes for a given input text
def predict_all_classes(input_text):
    results = {}
    for label, (model, vectorizer) in model_vectorizer_pairs.items():
        predicted_class, probabilities = predict_class(input_text, model, vectorizer)
        if predicted_class == 1:
            results[label] = "detected"
    return results

# Function to generate user-friendly output based on predictions
def generate_output(input_text):
    predictions = predict_all_classes(input_text)
    output_parts = []
    detected_classes = []

    if 'cv_improver' in predictions:
        output_parts.append("améliorer ton CV")
        detected_classes.append('cv_improver')
    if 'lm_improver' in predictions:
        output_parts.append("ta lettre de motivation")
        detected_classes.append('lm_improver')
    if 'interview_coaching' in predictions:
        output_parts.append("un coaching d'entretien")
        detected_classes.append('interview-coaching')
    if 'best_jobs' in predictions:
        output_parts.append("les meilleures offres d'emploi")
        detected_classes.append('best_jobs')

    if output_parts:
        if len(output_parts) > 1:
            final_output = f"D'accord, j'ai compris que tu as besoin d'aide pour {', '.join(output_parts[:-1])} et {output_parts[-1]}."
        else:
            final_output = f"D'accord, j'ai compris que tu as besoin d'aide pour {output_parts[0]}."
    else:
        final_output = "D'accord, je n'ai pas détecté de besoin spécifique."

    return final_output, detected_classes

# Function to call the external API with streaming and error handling
def call_external_api(input_text):
    url = 'https://llm.thanks-boss.com/ollama'
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': 'thanks-boss-team-test'
    }
    data = {'input_text': input_text}
    
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        api_response = ""
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    api_response += line.decode('utf-8') + "\n"
        return api_response.strip()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling external API: {e}")
        return f"Error: {str(e)}"

# Load models and vectorizers
model_vectorizer_paths = [
    ('models/log_reg_best_jobs_model.pkl', 'vectorizers/tfidf_vectorizer_best_jobs.pkl', 'best_jobs'),
    ('models/log_reg_cv_improver_model.pkl', 'vectorizers/tfidf_vectorizer_cv_improver.pkl', 'cv_improver'),
    ('models/log_reg_lm_improver_model.pkl', 'vectorizers/tfidf_vectorizer_lm_improver.pkl', 'lm_improver'),
    ('models/log_reg_interview_coaching_model.pkl', 'vectorizers/tfidf_vectorizer_interview_coaching.pkl', 'interview-coaching'),
    ('models/log_reg_ko_model.pkl', 'vectorizers/tfidf_vectorizer_ko.pkl', 'KO')
]

model_vectorizer_pairs = {}
for model_path, vectorizer_path, label in model_vectorizer_paths:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    model_vectorizer_pairs[label] = (model, vectorizer)

# Initialize Flask app
app = Flask(__name__)

# CSV file path
csv_file_path = 'predictions.csv'

# Ensure the CSV file exists and has a header
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['input_text', 'detected_classes', 'timestamp'])

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data['input']
    final_output, detected_classes = generate_output(input_text)
    timestamp = datetime.now().isoformat()

    # Save input text, detected classes, and timestamp in CSV
    prediction_data = {
        'input_text': input_text,
        'detected_classes': ', '.join(detected_classes),
        'timestamp': timestamp
    }
    try:
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(prediction_data.values())
        logging.info("Prediction saved in CSV.")
    except Exception as e:
        logging.error(f"Error saving to CSV: {e}")

    # Call the external API if the detected classes list is different from ['ko']
    if detected_classes != ['ko']:
        external_response = call_external_api(input_text)
        return jsonify({
            "output": final_output,
            "detected_classes": detected_classes,
            "external_response": external_response
        })
    else:
        return jsonify({
            "output": final_output,
            "detected_classes": detected_classes
        })

if __name__ == '__main__':
    app.run(debug=True)
