# Imdb Movie Reviews Sentiment Analysis

This repository utilizes three pre-trained language models—**BERT**, **XLNet**, and **RoBERTa**—and two large models, **Llama3** and **Qwen2**, to perform sentiment classification on the IMDB dataset. Additionally, we reference [LlamBERT](https://github.com/aielte-research/LlamBERT) for semi-supervised learning on the three pre-trained models. Llama3 and Qwen2 are implemented using unsupervised classification, supervised fine-tuning based on [Llama-factory](https://github.com/hiyouga/LLaMA-Factory), and supervised fine-tuning with reference to [self-llm](https://github.com/datawhalechina/self-llm).

## Dataset

The **IMDb Movie Reviews** dataset is a binary sentiment analysis dataset consisting of 50,000 reviews from the Internet Movie Database (IMDb) labeled as positive or negative. The dataset contains an even number of positive and negative reviews. Only highly polarizing reviews are considered. A negative review has a score ≤ 4 out of 10, and a positive review has a score ≥ 7 out of 10. No more than 30 reviews are included per movie. The dataset contains additional unlabeled data.

```python
from modelscope.msdatasets import MsDataset

train_data = MsDataset.load('imdb', split='train')
test_data = MsDataset.load('imdb', split='test')
unsupervised_data = MsDataset.load('imdb', split='unsupervised')
```

## Pretrain models download

### BERT

- https://huggingface.co/google-bert/bert-base-uncased
- https://huggingface.co/FacebookAI/roberta-base
- https://huggingface.co/xlnet/xlnet-base-cased

Download torch files to corresponding folders in `./pretrain_models`

### LLMs: Meta-Llama-3-8B-Instruct & Qwen2-7B-Instruct

```bash
# download llms: llama3 and qwen2
python download.py
```

## Environment Creation

Create two different environment using conda.

### LLaMA-Factory env

```bash
# install llama-factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# optional: install torch-gpu
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Other env

```bash
pip install requirements.txt
```

## LLaMA-Factory

steps:

1. create dataset in `./LLaMA-Factory/data/` through `python data_gen.py ` (add train and test dataset) and `python llama_imdb.py` (add unsupervised dataset labeled by llama3).
2. add dataset in `./LLaMA-Factory/data/dataset_info.json`: ` imdb, imdb_test, imdb_semi`.
3. Move yaml files in `./llms` to LLaMA-Factory and run corresponding train and predict yaml files.

```bash
cd ..
python llama_imdb.py
python data_gen.py
cd LLaMA-Factory

# add yaml files to llama-factory, and then conduct training and prediction
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train yaml/llama3/train.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export yaml/llama3/predict.yaml

CUDA_VISIBLE_DEVICES=1 llamafactory-cli train yaml/qwen2/train.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export yaml/qwen2/predict.yaml
```

Mover Generated `generate_predictions.jsonl` by LLaMA-Factory to `./llm/{model}/` folder.

```bash
python factory_metrics.py # calculate metrics with predictions
```

## Running Pretrain Models in LlamBERT way

![LlamBERT diagram](https://img2023.cnblogs.com/blog/2348945/202406/2348945-20240617100309223-860554157.png)

```bash
# train and predict on bert, roberta, and xlnet
python bert.py --model_name=bert-base-uncased
python bert.py --model_name=roberta-base
python bert.py --model_name=xlnet-base-uncased

# add labeled data using llama3 to annotated unsupervised data
python bert_semi.py --model_name=bert-base-uncased
python bert_semi.py --model_name=roberta-base
python bert_semi.py --model_name=xlnet-base-uncased
```

## Directly Predict with llama3 and qwen2

```
# unsupervised testing on llama3 and qwen2
python llama_imdb.py
python qwen_imdb.py
```

## Supervised Fine-Tuning on llama3 and qwen2 guided by Self-llm

```bash
# self-llm implementation of sft on llama3 and qwen2
python llm/llama3/llama3_sft.py
python llm/qwen2/qwen2_sft.py
python llm/llama3/llama3_predict.py
python llm/qwen2/qwen2_predict.py
```

## Results

|         Model Name          | Accuracy (%) | Precision (%) | Recall (%) |  F1 (%)   |  MCC (%)  |
| :-------------------------: | :----------: | :-----------: | :--------: | :-------: | :-------: |
|    **bert-base-uncased**    |    94.12     |     93.83     |   94.46    |   94.14   |   88.24   |
|      **roberta-base**       |    95.03     |     94.53     |   95.60    |   95.06   |   90.07   |
|   **xlnet-base-uncased**    |    92.31     |     91.91     |   92.79    |   92.35   |   84.63   |
| **bert-base-uncased_semi**  |    93.94     |     94.23     |   93.62    |   93.92   |   87.88   |
|    **roberta-base_semi**    |    94.45     |     95.21     |   93.62    |   94.41   |   88.92   |
| **xlnet-base-uncased_semi** |    94.18     |     95.51     |   92.71    |   94.09   |   88.39   |
|         **fusion**          |    95.00     |     94.73     |   95.30    |   95.01   |   89.99   |
|        **llama3-8b**        |    94.95     |     96.03     |   93.77    |   94.89   |   89.92   |
|   **llama3_llamafactory**   |    97.32     |     97.13     | **97.52**  |   97.33   |   94.64   |
|       **llama3-sft**        |    97.15     |     97.36     |   96.93    |   97.14   |   94.30   |
|        **qwen2-7b**         |    94.03     |     96.62     |   91.26    |   93.86   |   88.20   |
|   **qwen2_llamafactory**    |  **97.33**   |     97.22     |   97.45    | **97.34** | **94.66** |
|        **qwen2-sft**        |    97.24     |   **97.38**   |   97.10    |   97.24   |   94.49   |