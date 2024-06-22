from pathlib import Path
from joblib import dump
import nltk
from tira.rest_api_client import Client
import sklearn_crfsuite

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

    label_data = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    ).set_index("id")

    texts = text_data["sentence"].tolist()
    labels = label_data["tags"].tolist()

    X_train = []
    y_train = []

    for text, label in zip(texts, labels):
        tokens = preprocess_text(text)
        tag_list = label
        
        # Ensure tokens and tags are of the same length
        if len(tokens) == len(tag_list):
            X_train.append(extract_features(tokens))
            y_train.append(tag_list)
        else:
            print(f"Skipping mismatched pair (tokens: {len(tokens)}, tags: {len(tag_list)})")

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)

    dump(crf, Path(__file__).parent / "model.joblib")
