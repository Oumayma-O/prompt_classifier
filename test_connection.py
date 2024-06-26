from pymongo import MongoClient

try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['prompt_classifier_app']
    print("Connexion à MongoDB réussie!")
except Exception as e:
    print(f"Erreur de connexion : {e}")
