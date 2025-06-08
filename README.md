# Sentiment Analysis on Amazon Reviews

This repository contains a comprehensive Jupyter Notebook for sentiment analysis on Amazon product reviews. The notebook demonstrates the full pipeline from data preprocessing to advanced modeling and evaluation, using both classical machine learning and deep learning techniques.

## Dataset
- **Source:** Amazon Office Products Reviews (TSV format)
- **Features Used:**
  - `review_body`: The text of the review
  - `star_rating`: The rating given by the user (1-5 stars)
- **Preprocessing:**
  - Only reviews and ratings are retained
  - Ratings are converted to integers and missing values are dropped
  - A balanced dataset is created by sampling 50,000 reviews for each rating (1-5)
  - Sentiment labels are created for both binary (positive/negative) and ternary (positive/neutral/negative) classification

## Data Preprocessing
- **Text Cleaning:**
  - Lowercasing
  - Expanding contractions
  - Removing HTML tags, URLs, and non-alphabetic characters
  - Whitespace normalization
- **Tokenization, Stopword Removal, and Lemmatization:**
  - NLTK is used for stopword removal and lemmatization

## Feature Engineering
- **TF-IDF Features:**
  - Used as a baseline for classical models
- **Word Embeddings:**
  - **Pre-trained Word2Vec:** Google News 300-dimensional vectors
  - **Custom Word2Vec:** Trained on the Amazon reviews dataset for domain-specific embeddings
  - Both averaged and concatenated word vectors are used as features

## Modeling Approaches
- **Classical Models:**
  - Perceptron
  - Support Vector Machine (SVM)
- **Feedforward Neural Networks (FFNN):**
  - Two hidden layers with dropout
  - Trained for both binary and ternary classification
  - Input: Averaged or concatenated word vectors
- **Convolutional Neural Network (CNN):**
  - Two 1D convolutional layers followed by a fully connected layer
  - Input: Sequences of word embeddings (truncated/padded to fixed length)

## Evaluation & Results
- **Semantic Similarity:**
  - Pre-trained Word2Vec captures general semantic relationships better
  - Custom Word2Vec is more effective for domain-specific (review) language
- **Model Performance (Test Accuracy):**

| Feature Type              | Model                  | Binary Acc. | Ternary Acc. |
|--------------------------|------------------------|-------------|--------------|
| TF-IDF                   | Perceptron             | 0.8460      | -            |
| TF-IDF                   | SVM                    | 0.8864      | -            |
| Pretrained Word2Vec      | Perceptron             | 0.6518      | -            |
| Pretrained Word2Vec      | SVM                    | 0.8149      | -            |
| Custom Word2Vec          | Perceptron             | 0.6744      | -            |
| Custom Word2Vec          | SVM                    | 0.8426      | -            |
| Pretrained Word2Vec      | FFNN (avg)             | 0.8441      | 0.6823       |
| Custom Word2Vec          | FFNN (avg)             | 0.8623      | 0.6982       |
| Pretrained Word2Vec      | FFNN (concat)          | 0.7660      | 0.6118       |
| Custom Word2Vec          | FFNN (concat)          | 0.7838      | 0.6327       |
| Pretrained Word2Vec      | CNN                    | 0.8545      | 0.7225       |
| Custom Word2Vec          | CNN                    | 0.8641      | 0.6964       |

- **Key Findings:**
  - TF-IDF features with SVM yield the highest accuracy among classical models
  - Custom-trained Word2Vec embeddings outperform pre-trained ones for this domain
  - Neural network models (FFNN, CNN) outperform classical models, especially with custom embeddings
  - CNNs are particularly effective for ternary sentiment classification

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the required data files (Amazon reviews TSV, pre-trained Word2Vec model) and place them in the project directory.
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `code.ipynb` and run all cells sequentially.

## Requirements
- Python 3.11+
- Jupyter Notebook
- pandas, numpy, nltk, gensim, scikit-learn, torch

## File Structure
```
.
├── code.ipynb
├── requirements.txt
└── README.md
```