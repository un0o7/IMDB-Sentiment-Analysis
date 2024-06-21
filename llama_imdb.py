from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from modelscope.msdatasets import MsDataset
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
import json

model_dir = 'LLM-Research/Meta-Llama-3-8B-Instruct'
device = 'cuda:5'

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir,
                                             torch_dtype='auto',
                                             device_map=device)


def test():
    test_data = MsDataset.load('imdb', split='test')
    real_labels = [data['label'] for data in test_data]

    texts = [data['text'] for data in test_data]

    print("Total texts: ", len(texts))
    print("Positive labels: ", real_labels.count(1))

    pred_labels = []

    exception_responses = []

    # prompt source: https://github.com/aielte-research/LlamBERT/blob/main/LLM/model_inputs/IMDB/promt_eng_0-shot_prompts.json
    for text in tqdm(texts):
        prompt = text

        messages = [{
            'role':
            'system',
            'content':
            'You are an expert in sentiment analysis who can judge the attitude of a text. Please answer with \'positive\' or \'negative\' only!\n'
        }]
        prompt = f'Decide if the following movie review is positive or negative: \n {prompt} \nIf the movie review is positive please answer \'positive\', if the movie review is negative please answer \'negative\'. Make your decision based on the whole text. If the overall sentiment is not clear, base your decision on whether or not the reviewer recommends the movie for watching. If the sentiment is still not clear, say \'negative\'.'

        messages.append({'role': 'user', 'content': prompt})

        text = tokenizer.apply_chat_template(messages,
                                             tokenize=False,
                                             add_generation_prompt=True)

        model_input = tokenizer([text], return_tensors='pt').to(device)
        attention_mask = torch.ones(model_input.input_ids.shape,
                                    dtype=torch.long,
                                    device=device)
        generated_ids = model.generate(
            model_input.input_ids,
            max_new_tokens=512,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(
                model_input.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids,
                                          skip_special_tokens=True)[0]
        response = response.lower()  # avoid Negative != negative
        print(response)
        if response not in ['positive', 'negative']:
            exception_responses.append(response)
            response = 'negative'
        if response == 'positive':
            pred_labels.append(1)
        else:
            pred_labels.append(0)

        print(f'{response} \n')

    acc = accuracy_score(real_labels, pred_labels)
    f1 = f1_score(real_labels, pred_labels)
    precision = precision_score(real_labels, pred_labels)
    recall = recall_score(real_labels, pred_labels)
    mcc = matthews_corrcoef(real_labels, pred_labels)

    df = pd.read_csv('results.csv')

    df = df._append(
        {
            'model_name': 'llama3-8b',
            'seed': 0,
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'mcc': mcc
        },
        ignore_index=True)
    df.to_csv('results.csv', index=False)

    print("exception_responses: ", exception_responses)


def label_unsupervised():
    unsupervised_data = MsDataset.load('imdb', split='unsupervised')
    texts = [data['text'] for data in unsupervised_data]

    pred_labels = []

    exception_responses = []

    # prompt source:
    for text in tqdm(texts):
        print(f'Enter a prompt to generate a response:')
        prompt = text

        messages = [{
            'role':
            'system',
            'content':
            'You are an expert in sentiment analysis who can judge the attitude of a text. Please answer with \'positive\' or \'negative\' only!\n'
        }]
        prompt = f'Decide if the following movie review is positive or negative: \n {prompt} \nIf the movie review is positive please answer \'positive\', if the movie review is negative please answer \'negative\'. Make your decision based on the whole text. If the overall sentiment is not clear, base your decision on whether or not the reviewer recommends the movie for watching. If the sentiment is still not clear, say \'negative\'.'

        messages.append({'role': 'user', 'content': prompt})

        text = tokenizer.apply_chat_template(messages,
                                             tokenize=False,
                                             add_generation_prompt=True)

        model_input = tokenizer([text], return_tensors='pt').to(device)
        attention_mask = torch.ones(model_input.input_ids.shape,
                                    dtype=torch.long,
                                    device=device)
        generated_ids = model.generate(
            model_input.input_ids,
            max_new_tokens=512,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(
                model_input.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids,
                                          skip_special_tokens=True)[0]
        response = response.lower()  # avoid Negative != negative
        print(response)
        if response not in ['positive', 'negative']:
            exception_responses.append(response)
            response = 'negative'
        if response == 'positive':
            pred_labels.append(1)
        else:
            pred_labels.append(0)

        print(f'{response} \n')

    json.dump(pred_labels,
              open('output/unsupervised_labels.json', 'w'),
              indent=2)
    print("exception_responses: ", exception_responses)


if __name__ == '__main__':
    test()
    label_unsupervised()
