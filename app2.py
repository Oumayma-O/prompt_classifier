from flask import Flask, request, jsonify
import joblib
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from pymongo import MongoClient
import pandas as pd
import requests
import logging

# Configuration du journal
logging.basicConfig(level=logging.INFO)

# Télécharger les ressources nécessaires de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Charger le modèle SpaCy pour le français
nlp = spacy.load('fr_core_news_sm')

# Connecter à MongoDB
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['prompt_classifier_app']
    collection = db['predictions']
    logging.info("Connexion à MongoDB réussie!")
except Exception as e:
    logging.error(f"Erreur de connexion à MongoDB : {e}")

# Fonction pour résoudre les contractions avec SpaCy
def resolve_contractions_spacy(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return ' '.join(tokens)

# Fonction de nettoyage de texte avec étiquetage et lemmatisation
def preprocess_text(text):
    text = resolve_contractions_spacy(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    doc = nlp(text)
    french_stopwords = set(stopwords.words('french'))
    words = [token.lemma_ for token in doc if token.text not in french_stopwords and token.pos_ != 'PUNCT' and len(token.text) > 1]
    cleaned_text = ' '.join(words)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Fonction pour prédire la classe d'un texte donné en utilisant un modèle et un vectoriseur spécifiques
def predict_class(input_text, model, vectorizer):
    input_text_processed = preprocess_text(input_text)
    input_text_tfidf = vectorizer.transform([input_text_processed])
    probabilities = model.predict_proba(input_text_tfidf)
    predicted_class = model.predict(input_text_tfidf)
    return predicted_class[0], probabilities[0]

# Fonction pour prédire toutes les classes pour un texte donné
def predict_all_classes(input_text):
    results = {}
    for label, (model, vectorizer) in model_vectorizer_pairs.items():
        predicted_class, probabilities = predict_class(input_text, model, vectorizer)
        if predicted_class == 1:
            results[label] = "detected"
    return results

# Fonction pour générer une sortie conviviale basée sur les prédictions
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
        detected_classes.append('interview_coaching')
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

# Fonction pour appeler l'API externe
def call_external_api(input_text):
    url = "https://llm.thanks-boss.com/ollama"
    headers = {
        "x-api-key": "thanks-boss-team-test",
        "Content-Type": "application/json"
    }
    payload = {
        "input_text": [
            {
                "role": "assistant",
                "content": input_text
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

# Charger les modèles et les vectoriseurs
model_vectorizer_paths = [
    ('models/log_reg_best_jobs_model.pkl', 'vectorizers/tfidf_vectorizer_best_jobs.pkl', 'best_jobs'),
    ('models/log_reg_cv_improver_model.pkl', 'vectorizers/tfidf_vectorizer_cv_improver.pkl', 'cv_improver'),
    ('models/log_reg_lm_improver_model.pkl', 'vectorizers/tfidf_vectorizer_lm_improver.pkl', 'lm_improver'),
    ('models/log_reg_interview_coaching_model.pkl', 'vectorizers/tfidf_vectorizer_interview_coaching.pkl', 'interview_coaching'),
    ('models/log_reg_ko_model.pkl', 'vectorizers/tfidf_vectorizer_ko.pkl', 'KO')
]

model_vectorizer_pairs = {}
for model_path, vectorizer_path, label in model_vectorizer_paths:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    model_vectorizer_pairs[label] = (model, vectorizer)

# Initialiser l'application Flask
app = Flask(__name__)

@app.route('/home', methods=['POST'])
def predict():
    data = request.json
    input_text = data['input']
    final_output, detected_classes = generate_output(input_text)

    # Sauvegarder le texte d'entrée et les classes prédites dans MongoDB
    prediction_data = {
        'input_text': input_text,
        'detected_classes': detected_classes,
        'output': final_output
    }
    try:
        collection.insert_one(prediction_data)
        logging.info("Prédiction sauvegardée dans MongoDB.")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde dans MongoDB : {e}")

    # Si la liste des classes détectées est différente de ['ko'], appeler l'API externe
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
