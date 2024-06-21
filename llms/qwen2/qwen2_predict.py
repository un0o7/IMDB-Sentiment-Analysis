from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig, TaskType, PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from modelscope.msdatasets import MsDataset
import pandas as pd
from tqdm import tqdm

test_data = MsDataset.load('imdb', split='test')

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj"
    ],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)

mode_path = 'qwen/Qwen2-7B-Instruct'
lora_path = 'qwen/Qwen2-7B-Instruct-lora'
device = 'cuda:2'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path,
                                             device_map=device,
                                             torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)


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
        prompt = f'Given a movie review, determine whether it is positive or negative. {prompt}'

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
        if response.startswith('positive'):
            pred_labels.append(1)
        elif response.startswith('negative'):
            pred_labels.append(0)
        else:
            pred_labels.append(0)
            exception_responses.append(response)

    acc = accuracy_score(real_labels, pred_labels)
    f1 = f1_score(real_labels, pred_labels)
    precision = precision_score(real_labels, pred_labels)
    recall = recall_score(real_labels, pred_labels)
    mcc = matthews_corrcoef(real_labels, pred_labels)

    df = pd.read_csv('results.csv')

    df = df._append(
        {
            'model_name': 'qwen2-sft',
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


if __name__ == '__main__':
    test()
