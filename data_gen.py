from modelscope.msdatasets import MsDataset
import json

train_data = MsDataset.load('imdb', split='train')
test_data = MsDataset.load('imdb', split='test')
unsupervised_data = MsDataset.load('imdb', split='unsupervised')

# print(
#     train_data[0]
# )  # example: {'text':"I've seen this story before but my kids", 'label':1}
# print(unsupervised_data[0])  # label is -1
# print(test_data[0])
instruction = 'Given a movie review, determine whether it is positive or negative.'
fine_tune_data = [{
    'instruction': instruction,
    'input': data['text'],
    'output': 'positive' if data['label'] == 1 else 'negative'
} for data in train_data]

json.dump(fine_tune_data,
          open('LLaMA-Factory/data/imdb_train.json', 'w'),
          indent=2)

unsupervised_labels = json.load(open('output/unsupervised_labels.json'))

unsupervised_data = [{
    'instruction': instruction,
    'input': data['text'],
    'output': 'positive' if label == 1 else 'negative'
} for data, label in zip(unsupervised_data, unsupervised_labels)]

unsupervised_data.extend(fine_tune_data)

json.dump(unsupervised_data,
          open('LLaMA-Factory/data/imdb_semi.json', 'w'),
          indent=2)

instruction = 'Given a movie review, determine whether it is positive or negative.'
fine_tune_data = [{
    'instruction': instruction,
    'input': data['text'],
    'output': 'positive' if data['label'] == 1 else 'negative'
} for data in test_data]

json.dump(fine_tune_data,
          open('LLaMA-Factory/data/imdb_test.json', 'w'),
          indent=2)
