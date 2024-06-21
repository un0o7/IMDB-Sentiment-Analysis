#模型下载
from modelscope import snapshot_download

cache_dir = './'

# model_dir = snapshot_download('ZhipuAI/glm-4-9b-chat', cache_dir=cache_dir)

model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct',
                              cache_dir=cache_dir)

model_dir = snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir=cache_dir)
