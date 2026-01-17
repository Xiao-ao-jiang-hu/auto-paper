import os

# API Configuration
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = open("./API_KEY", "r").read()

# Model Configuration
MODEL_VISION = "qwen3-vl-plus"
MODEL_REASONING = "kimi-k2-thinking"
MODEL_INSTRUCTION = "qwen3-max"
MODEL_EMBEDDING = "text-embedding-v4"

# Vector Database Configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "planning_papers"

# Search Configuration
TOP_K_RETRIEVAL = 5

# Parallelism Configuration
OCR_PARALLELISM = 3
CODE_GROUNDING_PARALLELISM = 3
