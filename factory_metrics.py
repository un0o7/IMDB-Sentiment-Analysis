import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
import pandas as pd

model = 'qwen2'

file_path = 'llms/' + model + '/' + 'generated_predictions.jsonl'

data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

real = []

pred = []

for d in data:
    real.append(1 if d['label'].startswith('positive') else 0)
    pred.append(1 if d['predict'].startswith('positive') else 0)

df = pd.read_csv("results.csv")

acc = accuracy_score(real, pred)
f1 = f1_score(real, pred, average='binary')
precision = precision_score(real, pred, average='binary')
recall = recall_score(real, pred, average='binary')
mcc = matthews_corrcoef(real, pred)
print("Accuracy: ", acc, " F1: ", f1, " Precision: ", precision, " Recall: ",
      recall, " MCC: ", mcc)
df = df.append(
    {
        'model_name': model + "_llamafactory",
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mcc': mcc
    },
    ignore_index=True)

df.to_csv("results.csv", index=False)
