# /// script
# dependencies = [
#   "joblib",
#   "click",
#   "rich",
#   "pandas",
#   "scikit-learn",
# ]
# ///

import logging
from pathlib import Path
from typing import List, Optional

import click
import joblib
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()

def load_classifier(model_path: Path):
    """Load a trained classifier model from disk.
    
    Args:
        model_path: Path to the saved .joblib model file
    
    Returns:
        The loaded classifier model
    
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    try:
        # Ensure data directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading model from {model_path}")
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def predict_texts(classifier, texts: List[str], batch_size: int = 1000) -> List[str]:
    """Classify a list of German texts using the loaded model.
    
    Args:
        classifier: The trained classifier model
        texts: List of German text strings to classify
        batch_size: Number of texts to process at once
    
    Returns:
        List of predicted classifications
    """
    logger.info(f"Classifying {len(texts)} texts in batches of {batch_size}")
    
    # Use joblib parallel predictions if available
    if hasattr(classifier, 'n_jobs'):
        classifier.n_jobs = -1  # Use all cores
        
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        predictions.extend(classifier.predict(batch))
        logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
    
    return predictions

@click.command()
@click.option(
    "--model-path",
    type=click.Path(path_type=Path),
    default="data/output.joblib",
    help="Path to trained model file (.joblib)"
)
@click.option(
    "--input-csv",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to input CSV file"
)
@click.option(
    "--output-csv",
    type=click.Path(path_type=Path),
    default="data/out.csv",
    help="Path to output CSV file (default: data/out.csv)"
)
@click.option(
    "--text-column",
    default="text",
    help="Name of column containing text to classify"
)
@click.option(
    "--prediction-column",
    default="predicted_label",
    help="Name of column to store predictions"
)
@click.option(
    "--batch-size",
    type=int,
    default=1000,
    help="Number of texts to process at once (default: 1000)"
)
def main(
    model_path: Path,
    input_csv: Path,
    output_csv: Optional[Path],
    text_column: str,
    prediction_column: str,
    batch_size: int
):
    """Classify German texts from a CSV file using a trained model."""
    try:
        # Read input CSV with optimized settings
        logger.info(f"Reading input from {input_csv}")
        df = pd.read_csv(
            input_csv,
            engine='c',  # Use C engine for faster parsing
            dtype={text_column: 'string'},  # Explicit string type
            memory_map=True  # Memory mapping for large files
        )
        
        # Validate text column exists
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")
            
        # Load model and predict
        classifier = load_classifier(model_path)
        texts = df[text_column].tolist()
        predictions = predict_texts(classifier, texts, batch_size)
        
        # Save predictions
        df[prediction_column] = predictions
        output_path = output_csv
        df.to_csv(
            output_path,
            index=False,
            chunksize=10000  # Write in chunks for large files
        )
        logger.info(f"Saved predictions to {output_path}")
        
        # Show sample results
        console.print("\n[bold]Sample Classification Results:[/bold]")
        for _, row in df.head(5).iterrows():
            console.print(f"- [cyan]Text:[/cyan] {row[text_column][:50]}...")
            console.print(f"  [green]Prediction:[/green] {row[prediction_column]}\n")
            
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise click.Abort()

if __name__ == "__main__":
    main()
