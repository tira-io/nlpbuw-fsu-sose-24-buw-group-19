import json
from pathlib import Path
import pandas as pd
from joblib import load
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score,matthews_corrcoef
from tira.third_party_integrations import get_output_directory
from tira.rest_api_client import Client


#reading the data to be tested from text.jsonl
# with open('text.jsonl', 'r') as f:
#     texts = [json.loads(line) for line in f]
# texts_df = pd.DataFrame(texts)
if __name__ == "__main__":
    tira = Client()
    texts_df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    )
    texts_df['combined'] = texts_df['sentence1'] + ' ' + texts_df['sentence2']

    #loading the dataset created using train.py
    pipeline = load(Path(__file__).parent / "model.joblib")

    predictions = pipeline.predict(texts_df['combined'])

    output = [{'id': int(row['id']), 'label': int(pred)} for row, pred in zip(texts_df.to_dict('records'), predictions)]


    output_directory = get_output_directory(str(Path(__file__).parent))

    output_file_path = Path(output_directory) / 'predictions.jsonl'
    with open(output_file_path, 'w') as f:
        for pred in output:
            f.write(json.dumps(pred) + '\n')

# #after the predictions are made we need to read both files to calculate the accuracy scr and  mcc        

# #reading truth labels given as input
# with open('labels.jsonl', 'r') as f:
#     true_labels = [json.loads(line) for line in f]
# true_labels_df = pd.DataFrame(true_labels)

# #reading labels predicted by the model
# with open('allpredictions.jsonl', 'r') as f:
#     predictions = [json.loads(line) for line in f]
# predictions_labels_df = pd.DataFrame(predictions)


# #just to make sure if the the id are of type int converting them once again to type:int
# true_labels_df['id'] = true_labels_df['id'].astype(int)
# predictions_labels_df['id'] = predictions_labels_df['id'].astype(int)

# merged_df = pd.merge(true_labels_df, predictions_labels_df, on='id', suffixes=('_true', '_pred'))

# # Calculate the accuracy score
# accuracy = accuracy_score(merged_df['label_true'], merged_df['label_pred'])
# print(f"Accuracy: {accuracy:.4f}")


# #claculating Mathews Correlation Coefficient
# mcc = matthews_corrcoef(merged_df['label_true'], merged_df['label_pred'])
# print(f"Matthews Correlation Coefficient: {mcc:.4f}")