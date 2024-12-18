"""
Modèle LSTM pour détection de hate speech.
Dépendances :
    pip install -r requirements.txt
"""

import re
import string
import json

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1) Télécharger les stopwords une seule fois
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
STEMMER   = SnowballStemmer('english')

def clean_text(text: str) -> str:
    """
    Nettoyage basique d'un tweet :
    - mise en minuscules
    - suppression de liens, balises HTML, retours à la ligne, ponctuation, chiffres
    - suppression des stopwords et stemming
    """
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

def main():
    # --- 1) Chargement des données
    df = pd.read_excel('hate_speech.xlsx', usecols=[0, 1], engine='openpyxl')
    df.columns = ['class', 'tweet']  

    # Conversion safe en int 
    df['class'] = pd.to_numeric(df['class'], errors='coerce')
    df = df.dropna(subset=['class'])
    df['class'] = df['class'].astype(int)

    # Nettoyage des tweets
    df['tweet'] = df['tweet'].apply(clean_text)

    # --- 2) Préparation des features / labels
    X = df['tweet']
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- 3) Tokenization + padding
    MAX_WORDS = 50_000
    MAX_LEN   = 300

    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train)

    seq_train = tokenizer.texts_to_sequences(X_train)
    X_tr_seq  = pad_sequences(seq_train, maxlen=MAX_LEN)
    seq_test  = tokenizer.texts_to_sequences(X_test)
    X_te_seq  = pad_sequences(seq_test, maxlen=MAX_LEN)

    # --- 4) Définition du modèle
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=100, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.summary()

    # --- 5) Callbacks
    es = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )
    mc = ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True
    )

    # --- 6) Entraînement
    model.fit(
        X_tr_seq, y_train,
        batch_size=128,
        epochs=5,
        validation_split=0.1,
        callbacks=[es, mc]
    )

    # --- 7) Évaluation
    loss, acc = model.evaluate(X_te_seq, y_test)
    print(f"Test loss: {loss:.4f}, accuracy: {acc:.4f}")

    # --- 8) Sauvegarde finale du modèle au format HDF5
    model.save('hate_speech_model.h5')
    print("Modèle sauvegardé dans hate_speech_model.h5")

    # --- 9) Export du tokenizer en JSON (écriture brute du JSON)
    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    print("Tokenizer sauvegardé dans tokenizer.json")

if __name__ == '__main__':
    main()
