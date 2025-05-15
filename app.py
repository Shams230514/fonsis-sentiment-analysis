from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import onnxruntime as ort
import pandas as pd
from transformers import AutoTokenizer
import pdfplumber
from tqdm import tqdm
import os
import tempfile
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration avec le chemin réel du modèle dans OpenShift AI
MODEL_PATH = "/opt/app-root/src/model_sentiment/model_fonsis/sentiment_model.onnx"
TOKENIZER_NAME = "ProsusAI/finbert"     # ou le tokenizer de notre modèle

# Chargement du modèle ONNX et du tokenizer
print("Chargement du modèle ONNX...")
try:
    session = ort.InferenceSession(MODEL_PATH)
    print("Modèle ONNX chargé avec succès!")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Fonctions d'analyse
def extract_text_blocks(pdf_path):
    blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                blocks.extend(text.split("\n"))
    return [block.strip() for block in blocks if block.strip()]

def split_into_chunks(text, max_length=512):
    words = text.split()
    return [" ".join(words[i:i+max_length]) for i in range(0, len(words), max_length)]

def analyse_sentiment_onnx(text):
    try:
        # Tokenization
        inputs = tokenizer(
            text, 
            truncation=True, 
            max_length=512, 
            padding=True, 
            return_tensors="np"
        )
        
        # Préparer les inputs pour ONNX (conformément à votre modèle)
        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }
        
        # Inférence
        outputs = session.run(['logits'], ort_inputs)
        logits = outputs[0][0]  # Premier élément du batch
        
        # Appliquer softmax pour obtenir les probabilités
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Obtenir le label et le score
        predicted_class = np.argmax(probabilities)
        score = float(probabilities[predicted_class])
        
        # Mapper les classes aux labels (ajustez selon votre modèle)
        # Pour FinBERT, généralement : 0=positive, 1=negative, 2=neutral
        label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
        label = label_map.get(predicted_class, 'neutral')
        
        return {"label": label, "score": score}
    except Exception as e:
        print(f"Erreur analyse sentiment: {e}")
        return {"label": "neutral", "score": 0.5}

def score_opportunite(sentiment, text=""):
    score = 0
    if sentiment['label'] == 'positive':
        score += 2
    elif sentiment['label'] == 'neutral':
        score += 1
    else:
        score -= 1
    
    # Bonus si le texte contient des mots-clés sectoriels
    secteurs_keywords = ["eau", "énergie", "agrobusiness", "santé", "pharma", "infrastructure", "transport"]
    text_lower = text.lower()
    for keyword in secteurs_keywords:
        if keyword in text_lower:
            score += 1
    
    return score

def analyser_rapport_financier(pdf_path):
    blocs = extract_text_blocks(pdf_path)
    resultats = []
    
    total_blocs = len(blocs)
    print(f"Analyse de {total_blocs} blocs de texte...")
    
    for i, bloc in enumerate(blocs):
        if i % 100 == 0:
            print(f"Progression: {i}/{total_blocs}")
            
        for chunk in split_into_chunks(bloc, max_length=512):
            if len(chunk.strip()) < 10:  # Ignorer les chunks trop courts
                continue
                
            sentiment = analyse_sentiment_onnx(chunk)
            
            # Extraction simple des secteurs mentionnés
            secteurs_keywords = ["eau", "énergie", "agrobusiness", "santé", "pharma", "infrastructure", "transport"]
            secteurs_detectes = [kw for kw in secteurs_keywords if kw in chunk.lower()]
            
            # Calculer le score d'opportunité
            score = score_opportunite(sentiment, chunk)
            
            resultats.append({
                "texte": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                "sentiment": sentiment["label"],
                "score_sentiment": round(sentiment["score"], 3),
                "opportunite_score": score,
                "secteurs": secteurs_detectes
            })
    
    return pd.DataFrame(resultats)

def decisions_par_secteur(df, mots_cles):
    resultats = []
    
    for secteur in mots_cles:
        # Filtrer les lignes qui mentionnent ce secteur
        df_sec = df[df['secteurs'].apply(lambda x: secteur in x)]
        
        if len(df_sec) == 0:
            score_moyen = 0
            sentiment_dominant = "neutral"
            count = 0
        else:
            score_moyen = df_sec['opportunite_score'].mean()
            sentiments_count = df_sec['sentiment'].value_counts()
            sentiment_dominant = sentiments_count.idxmax() if not sentiments_count.empty else "neutral"
            count = len(df_sec)
        
        # Décision d'investissement
        if score_moyen > 2.0:
            decision = "🔼 Considérer investissement"
        elif 1.0 <= score_moyen <= 2.0:
            decision = "🔄 À surveiller"
        else:
            decision = "⛔ Éviter pour l'instant"
        
        resultats.append({
            "name": secteur.capitalize(),
            "score": round(score_moyen, 2),
            "sentiment": sentiment_dominant.capitalize(),
            "occurrences": count,
            "decision": decision
        })
    
    return resultats

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        # Récupérer le fichier PDF
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        print(f"Analyse du fichier: {file.filename}")
        
        # Sauvegarder temporairement le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            file.save(tmp_file.name)
            
            try:
                # Analyser le rapport
                df_resultats = analyser_rapport_financier(tmp_file.name)
                print(f"Analyse terminée: {len(df_resultats)} résultats")
            finally:
                # Supprimer le fichier temporaire
                os.unlink(tmp_file.name)
        
        # Calculer les statistiques
        secteurs_cles = ["eau", "énergie", "agrobusiness", "santé", "pharma", "infrastructure", "transport"]
        secteurs_results = decisions_par_secteur(df_resultats, secteurs_cles)
        
        sentiment_counts = df_resultats['sentiment'].value_counts()
        
        # Calculer le score de confiance moyen
        confidence_avg = df_resultats['score_sentiment'].mean() * 100 if len(df_resultats) > 0 else 0
        
        response = {
            'totalBlocs': len(df_resultats),
            'confidenceScore': round(confidence_avg, 1),
            'opportunities': sum(1 for s in secteurs_results if "Considérer" in s['decision']),
            'sectors': secteurs_results,
            'sentimentDistribution': {
                'positive': int(sentiment_counts.get('positive', 0)),
                'neutral': int(sentiment_counts.get('neutral', 0)),
                'negative': int(sentiment_counts.get('negative', 0))
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Erreur lors de l'analyse: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    # Vérifier que le modèle est chargé
    model_status = "loaded" if 'session' in globals() else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status
    })

if __name__ == '__main__':
    print("Démarrage du serveur Flask...")
    app.run(host='0.0.0.0', port=5000, debug=True)
