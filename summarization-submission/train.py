from pathlib import Path
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from tira.rest_api_client import Client
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

if __name__ == "__main__":
    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")

    # Preprocess the text: tokenize, remove stopwords, clean text
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        sentences = sent_tokenize(text)  # Tokenize into sentences
        cleaned_sentences = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence.lower())  # Tokenize into words
            words = [word for word in words if word.isalnum() and word not in stop_words]  # Remove stopwords and non-alphanumeric
            cleaned_sentences.append(' '.join(words))
        return cleaned_sentences

    # Prepare the training data
    train_sentences = []
    for story in text["story"]:
        train_sentences.extend(preprocess_text(story))

    # Fit TF-IDF vectorizer on the training data
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_sentences)

    # Save the TF-IDF vectorizer model
    dump(vectorizer, Path(__file__).parent / "model.joblib")