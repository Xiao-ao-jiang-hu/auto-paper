import os

# API Configuration
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# the file in current directory named api_key
current_work_dir = os.path.dirname(__file__)
API_KEY = open(os.path.join(current_work_dir, "API_KEY"), "r").read().strip()

# Model Configuration
MODEL_VISION = "qwen3-vl-plus"
MODEL_REASONING = "deepseek-v3.2"
MODEL_INSTRUCTION = "qwen3-max"
MODEL_EMBEDDING = "text-embedding-v4"

# Vector Database Configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "planning_papers"

# Search Configuration
TOP_K_RETRIEVAL = 5

# Parallelism Configuration
OCR_PARALLELISM = 32
CODE_GROUNDING_PARALLELISM = 5
