from pathlib import Path
import spacy
from spacy.tokens import DocBin
from tira.rest_api_client import Client
import nltk

nltk.download('punkt')

def preprocess_text(text):
    tokens = text.split()
    return tokens

def create_training_data_with_spacy(texts, nlp):
    training_data = []
    for text in texts:
        doc = nlp(text)
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        training_data.append((text, {'entities': entities}))
    return training_data

if __name__ == "__main__":
    tira = Client()
    text_data = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    ).set_index("id")

    texts = text_data["sentence"].tolist()

    # Load pre-trained spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Generate initial labels using the pre-trained model
    training_data = create_training_data_with_spacy(texts, nlp)

    # Create a blank spaCy model
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner", last=True)

    for _, annotations in training_data:
        for ent in annotations['entities']:
            ner.add_label(ent[2])

    db = DocBin()
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations['entities']:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
        doc.ents = ents
        db.add(doc)

    db.to_disk(Path(__file__).parent / "training_data.spacy")

    nlp.begin_training()
    for itn in range(20):
        losses = {}
        for text, annotations in training_data:
            doc = nlp.make_doc(text)
            example = spacy.training.Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.5, losses=losses)
        print(losses)

    nlp.to_disk(Path(__file__).parent / "ner_model")

    from joblib import dump
    dump(nlp, Path(__file__).parent / "ner_model.joblib")
