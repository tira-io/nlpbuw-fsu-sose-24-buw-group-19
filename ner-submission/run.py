from pathlib import Path
from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import json
import spacy

def preprocess_text(text):
    tokens = text.split()
    return tokens

if __name__ == "__main__":
    tira = Client()

    text_data = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    ).set_index("id")

    texts = text_data["sentence"].tolist()
    ids = text_data.index.tolist()

    # Load the model using joblib
    nlp = load(Path(__file__).parent / "ner_model.joblib")

    predictions = []

    for text, id_ in zip(texts, ids):
        doc = nlp(text)
        tags = [token.ent_iob_ + '-' + token.ent_type_ if token.ent_type_ else 'O' for token in doc]
        predictions.append({"id": id_, "tags": tags})

    output_directory = get_output_directory(str(Path(__file__).parent))
    with open(Path(output_directory) / "predictions.jsonl", 'w') as file:
        for prediction in predictions:
            file.write(json.dumps(prediction) + '\n')
