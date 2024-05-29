
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump
import json


#reading sentence and their ground truths from both files
with open('text.jsonl', 'r') as f:
    texts = [json.loads(line) for line in f]
with open('labels.jsonl', 'r') as f:
    labels = [json.loads(line) for line in f]


#convert loaded data to data frames to help training and testing
dataframe_texts = pd.DataFrame(texts)
dataframe_labels = pd.DataFrame(labels)

#make sure if the the id are of type int converting them once again to type:int
dataframe_texts['id'] = dataframe_texts['id'].astype(int)
dataframe_labels['id'] = dataframe_labels['id'].astype(int)

#merge data frames based on their ids
data = pd.merge(dataframe_texts, dataframe_labels, on='id')

#dividing dataset to values and their results
X = data[['sentence1', 'sentence2']].copy()
y = data['label']

#to avoid SettingWithCopyWarning
X.loc[:, 'combined'] = X['sentence1'] + ' ' + X['sentence2']

X_train, X_val, y_train, y_val = train_test_split(X['combined'], y, test_size=0.2, random_state=42)

pipeline = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)

pipeline.fit(X_train, y_train)

#dumping the trained model to model.joblib
dump(pipeline, 'model.joblib')
