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
        # Try reading with different encodings and more error-tolerant settings
        try:
            data = pd.read_csv(
                csv_path,
                encoding='utf-8',
                on_bad_lines='warn',
                quotechar='"',
                doublequote=True,
                escapechar='\\'
            )
        except UnicodeDecodeError:
            data = pd.read_csv(
                csv_path,
                encoding='latin1',
                on_bad_lines='warn',
                quotechar='"',
                doublequote=True,
                escapechar='\\'
            )
            
        # Verify required columns exist
        if 'text' not in data.columns or 'label' not in data.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns")
            
        X = data['text']
        y = data['label']
        
        # Log basic stats
        console.print(f"Loaded {len(data)} rows")
        console.print(f"Label distribution:\n{y.value_counts()}")

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
        
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}\nTry checking for unclosed quotes or special characters in the data.")
        raise click.ClickException("Failed to parse CSV file")
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        raise click.ClickException(str(e))
    except Exception as e:
        logger.error(f"Error training classifier: {e}", exc_info=True)
        raise click.ClickException("Training failed - see logs for details")

@click.command()
@click.argument('csv_path', type=click.Path(exists=True, path_type=Path))
@click.argument('model_path', type=click.Path(path_type=Path), default=Path("german_text_classifier.joblib"))
def main(csv_path: Path, model_path: Path):
    """Train a text classifier from CSV data"""
    train_classifier(csv_path, model_path)

if __name__ == "__main__":
    main()
