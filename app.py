"""
API Flask + page web pour classifier du texte en hate-speech / non-hate-speech.
"""

import re, string, json
from flask import Flask, request, jsonify, render_template
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# --- 1) Préparation du pré-traitement
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
STEMMER   = SnowballStemmer('english')
MAX_LEN   = 300

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\brt\b', '', text)
    text = re.sub(rf'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = [w for w in text.split() if w not in STOPWORDS]
    return ' '.join(STEMMER.stem(w) for w in words)

# --- 2) Chargement du modèle et du tokenizer
app = Flask(__name__)
model = load_model('hate_speech_model.h5')
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer = tokenizer_from_json(f.read())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_api():
    data = request.get_json()
    text = data.get('text', '')
    cleaned = clean_text(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=MAX_LEN)
    prob    = float(model.predict(padded)[0][0])
    label   = 'hater' if prob >= 0.5 else 'non-hater'
    return jsonify(label=label, probability=prob)

if __name__ == '__main__':
    app.run(debug=True)
