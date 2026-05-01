import os
import re
import nltk
import pandas as pd
import joblib

# -----------------------------
# ✅ NLTK SETUP (DEPLOY-SAFE)
# -----------------------------
NLTK_PATH = os.path.join(os.getcwd(), "nltk_data")

if NLTK_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_PATH)


def download_nltk_resources():
    """Download required NLTK resources if missing (works in deployment)"""
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']

    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                print(f"[INFO] Downloading {resource}...")
                nltk.download(resource, download_dir=NLTK_PATH)


# 🔥 Call once at import time
download_nltk_resources()

# -----------------------------
# IMPORTS AFTER SETUP
# -----------------------------
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# -----------------------------
# DATA LOADING
# -----------------------------
def load_dataset(csv_path, text_column='review', label_column='sentiment'):
    df = pd.read_csv(csv_path)

    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"Dataset must include '{text_column}' and '{label_column}'")

    df = df[[text_column, label_column]].copy()
    df[text_column] = df[text_column].astype(str)
    df[label_column] = df[label_column].astype(str).str.lower()

    return df


# -----------------------------
# CLEANING
# -----------------------------
def remove_null_duplicates(df, text_column='review', label_column='sentiment'):
    df = df.dropna(subset=[text_column, label_column]).copy()
    df = df.drop_duplicates(subset=[text_column, label_column])
    return df


def clean_text(text):
    text = str(text).lower()

    # remove HTML
    text = re.sub(r'<.*?>', ' ', text)

    # remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # 🔥 NEGATION HANDLING (CRITICAL FIX)
    text = re.sub(r"\b(not|no|never)\s+(\w+)", r"\1_\2", text)

    # remove special characters
    text = re.sub(r'[^a-z\s_]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


# -----------------------------
# TOKENIZATION + LEMMATIZATION
# -----------------------------
def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    # 🔥 Keep negation words
    neg_words = {'not', 'no', 'never'}
    stop_words = stop_words - neg_words

    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words and len(token) > 2
    ]

    return cleaned_tokens


# -----------------------------
# FULL PIPELINE
# -----------------------------
def preprocess_text(text):
    text = clean_text(text)
    tokens = tokenize_and_lemmatize(text)
    return ' '.join(tokens)


def preprocess_series(series):
    return series.fillna('').astype(str).apply(preprocess_text)


# -----------------------------
# TF-IDF (IMPROVED 🔥)
# -----------------------------
def build_tfidf_vectorizer(texts):
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 3),   # unigram + bigram + trigram
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(texts)
    return X, vectorizer


# -----------------------------
# SPLIT
# -----------------------------
def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


# -----------------------------
# SAVE / LOAD
# -----------------------------
def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path):
    return joblib.load(path)