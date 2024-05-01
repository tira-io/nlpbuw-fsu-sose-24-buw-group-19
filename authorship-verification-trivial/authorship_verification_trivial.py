from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def extract_tfidf_features(text_data, tfidf_vectorizer=None):
    tfidf_features = ''

    if tfidf_vectorizer is None:
         tfidfVectorizer = TfidfVectorizer(max_features=1000)
    tfidf_features = tfidf_vectorizer.fit_transform(text_data)
    return tfidf_features

#example usage 
text_train= {'text':['example text 1 ','example tex 2']}

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_features_train = extract_tfidf_features(text_train['text'],tfidf_vectorizer)
print(tfidf_features_train)

if __name__ == "__main__":

    tira = Client()

    # loading train data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    # Extract TF-TDF features 
    tfidf_features_train = extract_tfidf_features(text_train['text'])
    tfidf_features_validation = extract_tfidf_features(text_validation['text'])

    #train classifier 
    classifier = LogisticRegression()

    classifier.fit(tfidf_features_train , targets_train['generated'])

    #predictions 

    predictions = classifier.predict(tfidf_features_validation)

    

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions_df=pd.DataFrame({'id':text_validation['id'] , 'generated':predictions})
    predictions_df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )