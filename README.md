# German Email Classifier

A simple, fast, and efficient classifier for German email texts. This tool can categorize emails into different classes such as:

- autoresponder_generic (automated generic responses)
- menschliche_antwort (human responses)
- new_job (job offers/applications)
- newsletter (newsletter emails)
- not_an_individual (non-individual senders like companies)
- out_of_office (out-of-office replies)
- rechnung (invoices)
- spam (spam messages)
- support_anfrage (support requests)

## Features

- Fast and lightweight implementation using scikit-learn
- Runs completely locally without external dependencies
- Simple CLI for easy integration into existing workflows
- High accuracy with minimal setup
- Includes visualization of model performance

## Files

This package includes the following files:

1. `data/train.csv` - Training dataset with labeled examples (text and label columns)
2. `classify.py` - Script for training the model (uses train/test split)
3. `predict.py` - Script for batch predictions on CSV files
4. `README.md` - This documentation file

## Quick Start

### 1. Train the Model

First, train the classifier using the provided dataset:

```bash
uv run classify.py data/train.csv
```

This will:
- Load the training data
- Train a text classification model
- Evaluate the model on a test split
- Save the trained model to `german_text_classifier.joblib` by default
- Display classification report with performance metrics

### 2. Classify New Texts

#### Classify Texts from a CSV File

```bash
uv run predict.py --input-csv your_texts.csv
```

Optional arguments:
- `--model-path`: Path to model file (default: german_text_classifier.joblib)
- `--output-csv`: Output file path (default: overwrites input)
- `--text-column`: Column name containing texts (default: "text")
- `--prediction-column`: Column name for predictions (default: "predicted_label")

Example with all options:
```bash
uv run predict.py \
  --input-csv emails.csv \
  --output-csv classified_emails.csv \
  --text-column email_body \
  --prediction-column category
```

## Performance

The classifier typically achieves:
- Training time: < 1 second for ~100 examples
- Prediction time: < 5 ms per text
- Accuracy: > 90% on typical email data

## How It Works

The classifier uses a simple but effective approach:
1. Text preprocessing with TF-IDF vectorization
   - Captures unigrams and bigrams (1-2 word sequences)
   - Applies frequency-based filtering
2. Classification with Logistic Regression
   - Fast training and prediction
   - Good explainability (can see which words/phrases are most important)

## Customization

You can easily customize the classifier:

- Add more training examples to `german_email_dataset.csv`
- Adjust the model parameters in `train_model()` function
- Add new categories by including examples in the training data

## Troubleshooting

- **Model not found error**: Make sure you've run the training script first
- **Low accuracy**: Try adding more diverse examples to your training data
- **Slow performance**: Check for very long texts or excessive data volume

## License

This project is provided as open-source software. Feel free to modify and use according to your needs.
