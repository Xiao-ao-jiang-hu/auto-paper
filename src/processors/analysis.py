import json
import re
from ..clients import LLMClients
from ..types import PaperStructure, LPFormulation


class PaperAnalyzer:
    def __init__(self, clients: LLMClients):
        self.clients = clients

    def analyze_markdown(self, paper_markdown: str, paper_id: str) -> PaperStructure:
        print(f"[{paper_id}] Starting Analysis Phase...")

        # --- Step 0: Preprocessing (Remove noise) ---
        print(
            f"  > [Step 0/3] Preprocessing Markdown (Removing References/Appendices)..."
        )

        # Using full context for high accuracy cleaning as requested
        cleanup_prompt = f"""
        You are a research paper preprocessor.
        Input is the raw OCR markdown of a paper.
        
        Task: 
        1. Identify the main body of the paper (Abstract, Intro, Method, Experiments, Conclusion).
        2. REMOVE the "References" section entirely.
        3. REMOVE "Appendix" sections ONLY IF they contain purely supplementary proofs or raw tables. If the Appendix contains core algorithmic details or pseudo-code, KEEP IT.
        
        Return the FULL CLEANED MARKDOWN text. Do not summarize. Maintain original structure.
        
        Input Text:
        {paper_markdown}
        """

        try:
            # Use Instruction Model for large context processing
            processed_text = self.clients.get_instruction_completion(
                "You are a text editor. Output only the cleaned markdown.",
                cleanup_prompt,
            )

            if processed_text and len(processed_text) > 100:  # Basic validation
                paper_markdown = processed_text
                print(
                    f"    > Successfully cleaned text. New length: {len(paper_markdown)}"
                )
            else:
                print(
                    "    ! Cleanup returned empty or too short text. Reverting to original."
                )

        except Exception as e:
            print(f"    ! Preprocessing warning: {e}")

        # --- Step 1: Basic Info Extraction ---
        print(f"  > [Step 1/3] Extracting Basic Info & Datasets...")

        basic_info_prompt = f"""
        Paper Text:
        {paper_markdown}
        
        Task: Extract valid JSON with:
        - title
        - abstract (summary)
        - problem_description_natural (What is the optimization problem?)
        - problem_type (MILP, LP, Graph Matching, etc.)
        - datasets (List of strings, e.g. ["PascalVOC", "Willow Object"])
        - performance_metrics (List of strings, e.g. ["Accuracy", "F1-Score"])
        
        Output JSON Only.
        """

        response_1 = self.clients.get_instruction_completion(
            "You are a weak listener. Output JSON.", basic_info_prompt
        )
        basic_data = self._parse_with_repair(response_1) or {}

        # --- Step 2: Mathematical Model Extraction ---
        print(f"  > [Step 2/3] Extracting Mathematical Model & Algorithm...")
        math_prompt = f"""
        Paper Text:
        {paper_markdown} 
        
        Task: Extract the Mathematical Formulation (Subject To, Objective Function).
        Output valid JSON:
        {{
            "lp_model": {{
                "objective": "Latex String (e.g. \\min x^T K x)",
                "constraints": ["Latex String 1", ...],
                "variables": ["Latex String - Description"]
            }},
            "raw_latex_model": "Original latex block",
            "algorithm_description": "Step-by-step text description of the algorithm"
        }}
        """

        # Use Reasoning model for Math
        response_math = self.clients.get_reasoning_completion(
            "You are an OR Expert.", math_prompt
        )
        math_data = self._parse_with_repair(response_math) or {}

        # --- Combine & Return ---
        print(f"  > Analysis Complete. Merging results.")

        # Fallback values
        lp_model_data = math_data.get("lp_model", {})
        if not isinstance(lp_model_data, dict):
            lp_model_data = {}

        return PaperStructure(
            paper_id=paper_id,
            title=basic_data.get("title", "Unknown"),
            abstract=basic_data.get("abstract", ""),
            problem_description_natural=basic_data.get(
                "problem_description_natural", ""
            ),
            problem_type=basic_data.get("problem_type", "Unknown"),
            datasets=basic_data.get("datasets", []),
            performance_metrics=basic_data.get("performance_metrics", []),
            lp_model=LPFormulation(**lp_model_data),
            raw_latex_model=math_data.get("raw_latex_model", ""),
            algorithm_description=math_data.get("algorithm_description", ""),
        )

    def _parse_with_repair(self, response_text: str) -> dict | None:
        """Tries to parse JSON, uses Instruction model to repair if needed."""
        data = self._try_parse_json(response_text)
        if not data:
            # print("    ! JSON parse failed. Repairing...")
            repair_prompt = f"Fix this broken JSON:\n{response_text[:2000]}"
            repaired = self.clients.get_instruction_completion(
                "Fix JSON. Output only JSON.", repair_prompt
            )
            data = self._try_parse_json(repaired)
        return data

    def _try_parse_json(self, text: str) -> dict | None:
        """Helper to extract and parse JSON from text"""
        json_str = ""
        # 1. Clean markdown code blocks ```json ... ```
        if "```json" in text:
            parts = text.split("```json")
            if len(parts) > 1:
                content = parts[1]
                if "```" in content:
                    json_str = content.split("```")[0].strip()

        # 2. Generic code blocks ``` ... ```
        if not json_str and "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                candidate = parts[1].strip()
                if candidate.startswith("{"):
                    json_str = candidate

        # 3. Fallback: Find outer braces
        if not json_str:
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx : end_idx + 1]
            else:
                json_str = text.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
