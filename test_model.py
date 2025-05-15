import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# Configuration - MODIFIEZ CE CHEMIN
MODEL_PATH = "/opt/app-root/src/model_sentiment/model_fonsis/sentiment_model.onnx"  # Chemin vers le modèle
TOKENIZER_NAME = "ProsusAI/finbert"

# Test de chargement
print("Test de chargement du modèle...")
try:
    session = ort.InferenceSession(MODEL_PATH)
    print("✓ Modèle chargé avec succès")
except Exception as e:
    print(f"✗ Erreur: {e}")
    exit(1)

print("\nTest du tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print("✓ Tokenizer chargé avec succès")
except Exception as e:
    print(f"✗ Erreur: {e}")
    exit(1)

# Test d'inférence
print("\nTest d'inférence...")
test_texts = [
    "Le secteur de l'eau montre une croissance positive",
    "Les pertes financières sont importantes",
    "Les résultats restent stables"
]

for text in test_texts:
    print(f"\nTexte: {text}")
    
    # Tokenization
    inputs = tokenizer(text, truncation=True, max_length=512, padding=True, return_tensors="np")
    
    # Préparer les inputs
    ort_inputs = {
        'input_ids': inputs['input_ids'].astype(np.int64),
        'attention_mask': inputs['attention_mask'].astype(np.int64)
    }
    
    # Inférence
    outputs = session.run(['logits'], ort_inputs)
    logits = outputs[0][0]
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    # Résultat
    predicted_class = np.argmax(probabilities)
    score = float(probabilities[predicted_class])
    
    label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    label = label_map.get(predicted_class, 'unknown')
    
    print(f"  Sentiment: {label}")
    print(f"  Score: {score:.3f}")
    print(f"  Probabilités: {probabilities}")

print("\n✓ Tests terminés avec succès!")