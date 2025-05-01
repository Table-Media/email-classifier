# /// script
# dependencies = [
#   "scikit-learn",
#   "pandas",
#   "numpy",
#   "joblib",
#   "click",
#   "rich",
# ]
# ///

import click
from pathlib import Path
from rich.console import Console
import logging
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)
console = Console()

def train_classifier(csv_path: Path, model_path: Path):
    """Train and save a text classifier model"""
    try:
        data = pd.read_csv(csv_path)
        X = data['text']
        y = data['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        classifier = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000))
        ])

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        console.print(classification_report(y_test, y_pred))
        joblib.dump(classifier, model_path)
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error training classifier: {e}")
        raise

@click.command()
@click.argument('csv_path', type=click.Path(exists=True, path_type=Path))
@click.argument('model_path', type=click.Path(path_type=Path), default=Path("german_text_classifier.joblib"))
def main(csv_path: Path, model_path: Path):
    """Train a text classifier from CSV data"""
    train_classifier(csv_path, model_path)

if __name__ == "__main__":
    main()
