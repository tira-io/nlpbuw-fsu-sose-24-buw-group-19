from pathlib import Path
from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import json
import re
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")

    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")

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

    def generate_summary(article, top_n=3):
        sentences = sent_tokenize(article)
        preprocessed_sentences = preprocess_text(article)
        if not preprocessed_sentences:
            return ""
        
        tfidf_scores = model.transform(preprocessed_sentences)
        sentence_scores = tfidf_scores.sum(axis=1).A1  # Sum TF-IDF scores for each sentence
        ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores)[-top_n:]]
        summary = ' '.join(ranked_sentences)
        return summary

    # Generate summaries for the dataset
    predictions = []
    for idx, story in df.iterrows():
        summary = generate_summary(story["story"])
        predictions.append({"id": idx, "summary": summary})

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    with open(Path(output_directory) / "predictions.jsonl", 'w') as file:
        for prediction in predictions:
            file.write(json.dumps(prediction) + '\n')