# Product Review Sentiment Analysis using DistilBERT

## Overview
This project performs sentiment analysis on product reviews using a
pretrained DistilBERT model fine-tuned on Amazon product review data.
The model classifies reviews into Positive, Neutral, and Negative sentiments.

## Dataset
Amazon Product Reviews Dataset (Food category used for training).
Columns used:
- `Text` – Review text
- `Score` – Rating (1–5)

Labels mapping:
- 1–2 → Negative
- 3 → Neutral
- 4–5 → Positive
The trained model is generic and can analyze sentiment for any product review.

## Model
-**Base Model:** 'distilbert-base-uncased'
- **Framework:** Hugging Face Transformers
- **Task:** Sequence Classification (3 classes)

## Technologies Used
- Python
- Hugging Face Transformers
- PyTorch
- Scikit-learn
- Pandas

## How to Run
1. Clone the repository
2. Install dependencies:
pip install -r requirements.txt

3. Open the Jupyter Notebook and run all cells

## Example
Input:
> "The product quality was terrible."

Output:
> Negative
