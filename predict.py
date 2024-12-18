
"""
Script d'inférence pour le modèle de détection de hate speech.
Usage exemple :
    python predict.py --input tweets.txt --output results.csv
"""

import re
import string
import argparse
import pickle

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import nltk
nltk.download('stopwords', quiet=True)

# mêmes stopwords + stemmer que pour l'entraînement
STOPWORDS = set(stopwords.words('english'))
STEMMER   = SnowballStemmer('english')

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
    words = [STEMMER.stem(w) for w in words]
    return ' '.join(words)

def predict_texts(texts, model, tokenizer, max_len=300):
    seq = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seq, maxlen=max_len)
    probs = model.predict(padded, verbose=0).flatten()
    classes = (probs >= 0.5).astype(int)
    return classes, probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',     type=str, default='hate_speech_model.keras',
                        help='chemin vers le modèle Keras')
    parser.add_argument('--tokenizer', type=str, default='tokenizer.pkl',
                        help='chemin vers le tokenizer pickle')
    parser.add_argument('--input',     type=str, required=True,
                        help='fichier texte, une phrase par ligne')
    parser.add_argument('--output',    type=str, default='results.csv',
                        help='CSV de sortie (tweet, classe, probabilité)')
    args = parser.parse_args()

    # Chargement
    model     = load_model(args.model)
    tokenizer = pickle.load(open(args.tokenizer, 'rb'))

    # Lecture des données
    df = pd.read_csv(args.input, header=None, names=['tweet'])
    df['clean'] = df['tweet'].apply(clean_text)

    # Prédiction
    classes, probs = predict_texts(df['clean'].tolist(), model, tokenizer)
    df['class']       = classes
    df['probability'] = probs

    # Sauvegarde
    df[['tweet', 'class', 'probability']].to_csv(args.output, index=False)
    print(f"Résultats sauvegardés dans {args.output}")

if __name__ == '__main__':
    main()
