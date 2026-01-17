import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing import List, Dict, Any
import os
from openai import OpenAI
from .config import (
    API_BASE_URL,
    API_KEY,
    MODEL_EMBEDDING,
    CHROMA_PERSIST_DIRECTORY,
    COLLECTION_NAME,
)
from .types import ProcessedEntry
import json


class AliyunEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        # Batching might be needed for large inputs, but kept simple here
        input_text = [text.replace("\n", " ") for text in input]
        response = self.client.embeddings.create(
            input=input_text, model=self.model_name
        )
        return [data.embedding for data in response.data]


class VectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        self.embedding_fn = AliyunEmbeddingFunction(
            api_key=API_KEY, base_url=API_BASE_URL, model_name=MODEL_EMBEDDING
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=self.embedding_fn
        )

    def add_entry(self, entry: ProcessedEntry):
        """
        Adds a processed paper entry to ChromaDB.
        Strategies:
        1. Embed the natural description.
        2. Embed the LP representation (textual).
        3. Store the rest as metadata.
        """

        # We create a composite document for embedding to capture both semantic and structural meaning
        # "Problem: <desc> \n Model: <lp_model>"
        doc_text = f"Title: {entry['paper_info']['title']}\n"
        doc_text += f"Problem Description: {entry['paper_info']['problem_description_natural']}\n"
        doc_text += f"Problem Type: {entry['paper_info']['problem_type']}\n"
        doc_text += f"LP Model: {json.dumps(entry['paper_info']['lp_model'])}"

        # Metadata must be flat primitives (str, int, float, bool)
        # Complex objects like lists need to be JSON stringified
        metadata = {
            "paper_id": entry["paper_info"]["paper_id"],
            "title": entry["paper_info"]["title"],
            "problem_type": entry["paper_info"]["problem_type"],
            "keywords": ",".join(entry["keywords"]),
            "algo_desc": entry["paper_info"]["algorithm_description"][
                :1000
            ],  # Trucate if necessary
            "repo_path": entry["repo_path"],
            "code_snippet": entry["code_implementation"]["code_snippet"][
                :2000
            ],  # Limit size
        }

        self.collection.add(
            documents=[doc_text],
            metadatas=[metadata],
            ids=[entry["paper_info"]["paper_id"]],
        )
        print(f"Added paper {entry['paper_info']['paper_id']} to ChromaDB.")

    def search(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        return results
