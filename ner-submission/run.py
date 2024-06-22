from pathlib import Path
from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import json
import nltk

nltk.download('punkt')

def preprocess_text(text):
    tokens = text.split()
    return tokens

def extract_features(tokens):
    features = []
    for i, word in enumerate(tokens):
        word_features = {
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'BOS': i == 0,
            'EOS': i == len(tokens) - 1
        }
        features.append(word_features)
    return features

if __name__ == "__main__":
    tira = Client()

    text_data = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    ).set_index("id")

    texts = text_data["sentence"].tolist()
    ids = text_data.index.tolist()

    crf = load(Path(__file__).parent / "model.joblib")

    predictions = []

    for text, id_ in zip(texts, ids):
        tokens = preprocess_text(text)
        features = extract_features(tokens)
        tags = crf.predict([features])[0].tolist()
        predictions.append({"id": id_, "tags": tags})

    output_directory = get_output_directory(str(Path(__file__).parent))
    with open(Path(output_directory) / "predictions.jsonl", 'w') as file:
        for prediction in predictions:
            file.write(json.dumps(prediction) + '\n')
