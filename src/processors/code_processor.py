import os
import json
from typing import Optional
from ..clients import LLMClients
from .grounding import CodeGrounder
from ..types import PaperStructure, CodeMapping


class CodeProcessor:
    """
    Facade class that handles the code processing pipeline:
    1. Retrieval (Finding relevant files)
    2. Grounding (Mapping paper concepts to code variables)
    """

    def __init__(self, clients: LLMClients):
        self.grounder = CodeGrounder(clients)

    def process(
        self,
        repo_path: str,
        paper_data: PaperStructure,
        output_dir: Optional[str] = None,
    ) -> CodeMapping:
        """
        Runs the full code association workflow.
        If output_dir is provided, saves result immediately.
        """
        print(f"[CodeProcessor] Starting Code Grounding in: {repo_path}")
        print(
            f"[CodeProcessor] Target Algo: {paper_data['algorithm_description'][:50]}..."
        )
        mapping = self.grounder.find_code(repo_path, paper_data)

        if mapping["file_path"]:
            print(
                f"[CodeProcessor] SUCCESS: Found implementation in {mapping['file_path']}"
            )
        else:
            print(
                f"[CodeProcessor] WARNING: Could not lock onto specific code snippet."
            )

        # IMMEDIATE SAVE: Code Mapping
        if output_dir and mapping:
            json_path = os.path.join(output_dir, "code_mapping.json")
            print(f"[CodeProcessor] Saving intermediate Code Mapping to {json_path}")
            os.makedirs(output_dir, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)

        return mapping
