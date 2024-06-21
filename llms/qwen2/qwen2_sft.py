from modelscope.msdatasets import MsDataset
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model

train_data = MsDataset.load('imdb', split='train')
device = 'cuda:5'

# 格式转换

train_data = {
    "instruction":
    ["Given a movie review, determine whether it is positive or negative."] *
    len(train_data),
    "input": [data["text"] for data in train_data],
    "output":
    ["positive" if data["label"] == 1 else "negative" for data in train_data]
}

train_data = Dataset.from_pandas(pd.DataFrame(train_data))

tokenizer = AutoTokenizer.from_pretrained('qwen/Qwen2-7B-Instruct',
                                          use_fast=False,
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


def process_func(example):
    MAX_LENGTH = 1024  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\nYou are an expert in sentiment analysis who can judge the attitude of a movie review. Please answer with \'positive\' or \'negative\' only.<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [
        tokenizer.pad_token_id
    ]
    attention_mask = instruction["attention_mask"] + response[
        "attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [
        tokenizer.pad_token_id
    ]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


tokenized_id = train_data.map(process_func,
                              remove_columns=train_data.column_names)

# print(tokenizer.decode(tokenized_id[0]['input_ids']))

# tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))
model = AutoModelForCausalLM.from_pretrained('qwen/Qwen2-7B-Instruct',
                                             device_map=device,
                                             torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

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
model = get_peft_model(model, config)
args = TrainingArguments(output_dir="./output/qwen2",
                         per_device_train_batch_size=1,
                         gradient_accumulation_steps=4,
                         logging_steps=100,
                         num_train_epochs=3,
                         save_steps=1000,
                         learning_rate=1e-4,
                         save_on_each_node=True,
                         gradient_checkpointing=True)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()

peft_model_id = "qwen/Qwen2-7B-Instruct-lora"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
