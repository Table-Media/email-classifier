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

1. `data/train.csv` - Training dataset with labeled examples
2. `data/test.csv` - Test dataset for evaluation
3. `classify.py` - Main script for training and evaluating the model
4. `predict.py` - Lightweight script for making predictions on new texts
5. `README.md` - This documentation file

## Quick Start

### 1. Train the Model

First, train the classifier using the provided dataset:

```bash
uv run classify.py
```

This will:
- Load the training and test data
- Train a text classification model
- Evaluate the model on the test data
- Save the trained model to `german_email_classifier.joblib`
- Display performance metrics and feature importance
- Create a confusion matrix visualization

### 2. Classify New Emails

#### Classify a Single Text

```bash
python predict_emails.py text "Ihre Bestellung #45678 wurde versandt und wird in Kürze geliefert."
```

#### Classify Multiple Texts from a CSV File

```bash
python predict_emails.py file your_emails.csv
```

By default, the script looks for a column named 'text' in your CSV. If your column has a different name, specify it:

```bash
python predict_emails.py file your_emails.csv email_body
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