import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.preprocess import (
    load_dataset,
    remove_null_duplicates,
    preprocess_series,
    build_tfidf_vectorizer,
    split_data,
    save_pickle,
)


def create_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def sample_dataset(df, debug=False, sample_size=10000, seed=42):
    if debug and len(df) > sample_size:
        return df.sample(sample_size, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def print_dataset_overview(df):
    print('Dataset shape:', df.shape)
    print('\nDataset info:')
    print(df.info())
    print('\nNull values:')
    print(df.isnull().sum())
    print('\nClass distribution:')
    print(df['sentiment'].value_counts())


def plot_sentiment_count(df, output_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='sentiment', hue='sentiment', legend=False, palette='pastel', ax=ax)
    ax.set_title('Sentiment Distribution')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def generate_wordcloud(text, title, output_path=None):
    if not text:
        return
    try:
        from wordcloud import WordCloud
    except ImportError as error:
        raise ImportError(
            'The wordcloud package is required for generating word clouds. Install it with `pip install wordcloud`.'
        ) from error

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_review_length_distribution(df, output_path=None):
    review_lengths = df['review'].astype(str).apply(lambda x: len(x.split()))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(review_lengths, bins=40, kde=True, color='#4c72b0', ax=ax)
    ax.set_title('Review Length Distribution')
    ax.set_xlabel('Number of Words')
    ax.set_ylabel('Frequency')
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def make_eda_visualizations(df, report_dir):
    create_output_dir(report_dir)
    plot_sentiment_count(df, os.path.join(report_dir, 'sentiment_distribution.png'))
    positive_text = ' '.join(df.loc[df['sentiment'] == 'positive', 'review'].astype(str).tolist())
    negative_text = ' '.join(df.loc[df['sentiment'] == 'negative', 'review'].astype(str).tolist())
    generate_wordcloud(positive_text, 'Positive Reviews Word Cloud', os.path.join(report_dir, 'wordcloud_positive.png'))
    generate_wordcloud(negative_text, 'Negative Reviews Word Cloud', os.path.join(report_dir, 'wordcloud_negative.png'))
    plot_review_length_distribution(df, os.path.join(report_dir, 'review_length_distribution.png'))


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    if hasattr(model, 'predict_proba'):
        positive_index = list(model.classes_).index('positive') if 'positive' in model.classes_ else 1
        y_scores = model.predict_proba(X_test)[:, positive_index]
    elif hasattr(model, 'decision_function'):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = None

    if y_scores is not None:
        y_binary = (y_test == 'positive').astype(int)
        roc_auc = roc_auc_score(y_binary, y_scores)
    else:
        roc_auc = None

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc,
    }


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with balanced class weights for robustness."""
    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a sentiment analysis model on IMDb reviews.')
    parser.add_argument('--dataset', default=os.path.join('data', 'IMDB Dataset.csv'), help='Path to IMDb dataset CSV file.')
    parser.add_argument('--model-path', default=os.path.join('model', 'model.pkl'), help='Path to save the trained model.')
    parser.add_argument('--vectorizer-path', default=os.path.join('model', 'vectorizer.pkl'), help='Path to save the TF-IDF vectorizer.')
    parser.add_argument('--report-dir', default=os.path.join('reports', 'eda'), help='Directory for EDA outputs.')
    parser.add_argument('--debug', action='store_true', help='Use a smaller sample for faster development cycles.')
    parser.add_argument('--sample-size', type=int, default=10000, help='Number of samples to use in debug mode.')
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    print('Training pipeline starting...')
    print('Loading dataset...')
    df = load_dataset(args.dataset)
    df = remove_null_duplicates(df)
    df = sample_dataset(df, debug=args.debug, sample_size=args.sample_size)

    if args.debug:
        print(f'Debug mode enabled: training on {len(df)} samples')

    print_dataset_overview(df)
    make_eda_visualizations(df, args.report_dir)
    print(f'EDA visualizations saved to {args.report_dir}')

    print('Preprocessing text...')
    df['cleaned_review'] = preprocess_series(df['review'])

    print('Vectorizing text...')
    X, vectorizer = build_tfidf_vectorizer(df['cleaned_review'])
    y = df['sentiment']

    print('Splitting data...')
    X_train, X_test, y_train, y_test = split_data(X, y)

    print('Training Logistic Regression model...')
    model = train_logistic_regression(X_train, y_train)

    print('Evaluating model...')
    metrics = evaluate_model(model, X_test, y_test)
    print(f'Accuracy: {metrics["accuracy"]:.4f}')
    if metrics['roc_auc'] is not None:
        print(f'ROC-AUC: {metrics["roc_auc"]:.4f}')
    else:
        print('ROC-AUC: unavailable for this model.')

    save_pickle(model, args.model_path)
    save_pickle(vectorizer, args.vectorizer_path)
    print(f'Model saved to {args.model_path}')
    print(f'Vectorizer saved to {args.vectorizer_path}')
    print('Training pipeline complete.')

    return metrics


if __name__ == '__main__':
    main()
