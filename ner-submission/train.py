from pathlib import Path
from joblib import dump
import nltk
from tira.rest_api_client import Client
import sklearn_crfsuite

nltk.download('punkt')

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
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

def get_labels(texts):
    labels = []
    for text in texts:
        tokens = nltk.word_tokenize(text)
        labels.append(['O'] * len(tokens))  # generate dummy 'O' labels
    return labels

if __name__ == "__main__":
    tira = Client()
    input_data = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    ).set_index("id")

    texts = input_data["sentence"].tolist()

    X_train = [extract_features(preprocess_text(text)) for text in texts]
    y_train = get_labels(texts)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)

    dump(crf, Path(__file__).parent / "ner_model.joblib")
