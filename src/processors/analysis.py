import json
import re
from ..clients import LLMClients
from ..types import PaperStructure, LPFormulation


class PaperAnalyzer:
    def __init__(self, clients: LLMClients):
        self.clients = clients

    def _extract_basic_info(self, paper_markdown: str) -> dict:
        prompt = f"""
        Paper Text:
        {paper_markdown[:20000]} 
        
        Task: Extract valid JSON with:
        - title
        - abstract (summary)
        - problem_description_natural (What is the optimization problem?)
        - problem_type (MILP, LP, Graph Matching, combinatorial optimization, etc.)
        
        Output valid JSON:
        {{
            "title": "Paper Title",
            "abstract": "Summary...",
            "problem_description_natural": "Detailed description...",
            "problem_type": "Type..."
        }}
        """
        response = self.clients.get_instruction_completion(
            "You are a research assistant. Output JSON.", prompt
        )
        return self._parse_with_repair(response) or {}

    def _extract_datasets(self, paper_markdown: str) -> dict:
        prompt = f"""
        Paper Text:
        {paper_markdown}
        
        Task: Identify and extract specific details about the Datasets and Experiments.
        
        Requirements:
        1. "datasets": List of EXACT, specific dataset names used. 
           - Good: "TSPLIB", "MIPLIB 2017", "PascalVOC 2012", "Cora", "Citeseer".
           - Avoid generic terms if possible (e.g. use "Generated ER Graphs" instead of just "Synthethic").
        2. "performance_metrics": List of metrics (e.g. "Optimality Gap", "Top-1 Accuracy").
        
        Output valid JSON:
        {{
            "datasets": ["Name1", "Name2"],
            "performance_metrics": ["Metric1", "Metric2"]
        }}
        """
        # Using Reasoning model or high-quality instruction model for precision
        response = self.clients.get_instruction_completion(
            "You are a Data Scientist. Be precise with dataset names.", prompt
        )
        return self._parse_with_repair(response) or {}

    def _extract_math_model(self, paper_markdown: str) -> dict:
        prompt = f"""
        Paper Text:
        {paper_markdown} 
        
        Task: Extract the Mathematical Formulation of the **UNDERLYING OPTIMIZATION PROBLEM** being solved.
        
        CRITICAL INSTRUCTION - READ CAREFULLY:
        1. **Target:** Extract the mathematical definition of the Operations Research / Combinatorial Optimization problem (e.g., TSP, VRP, Bin Packing, Scheduling, SAT, MILP). 
           - Look for "Objective Function", "Constraints", "Subject to", "Maximize/Minimize".
           - Even if the paper solves it using Deep Learning, Reinforcement Learning, or GNNs, we want the MODEL of the **PROBLEM** (e.g. min sum distances), NOT the model of the neural network.

        2. **Specificity:** Extract the **MOST SPECIFIC** problem formulation.
           - If the paper discusses a General Problem (e.g. general Lot Sizing) but solves a SPECIFIC variant (e.g. Capacitated Lot Sizing with Setup Times), extract the **SPECIFIC VARIANT**.
           - We need the exact constraints and objective used in the experiments/contribution, not the textbook definition.
        
        3. **Exclude:** Do NOT extract the equations of the Machine Learning method as the "lp_model". 
           - Do NOT extract Loss Functions, Backpropagation equations, GNN aggregation formulas, attention weights, or generic MDP definitions unless they define the physical problem constraints.
           - Those belong in "algorithm_description", not "lp_model".
        
        4. **Form:**
           - Objective: The cost/reward function of the physical/logical problem.
           - Constraints: The physical/logical constraints (capacity, time windows, non-overlapping).
           - Variables: The decision variables (x_ij = 1 if edge is used).
           - **Latex Formatting**: ALWAYS wrap Latex formulas in single dollar signs like $...$ for inline math or double $$...$$ for block math. Ensure all backslashes are properly escaped in JSON.

        Output valid JSON:
        {{
            "lp_model": {{
                "objective": "Latex String wrapped in $...$ (e.g. $\\min \\sum c_{{ij}} x_{{ij}}$)",
                "constraints": ["Latex String 1 wrapped in $...$", "Latex String 2 wrapped in $...$", "... (Include ALL specific constraints defined in the problem formulation)"],
                "variables": ["Latex String - Description (e.g. $x_{{ij}}$: binary decision variable)", ... (Include ALL key decision variables)"]
            }},
            "raw_latex_model": "The latex block defining the OPTIMIZATION PROBLEM formulation (wrapped in $$...$$ if multi-line).",
            "algorithm_description": "Brief description of the Machine Learning / Heuristic method used to solve this problem."
        }}
        """
        # Use Reasoning model for Math
        response = self.clients.get_reasoning_completion(
            "You are an OR Expert. Focus on extracting the Optimization Problem Formulation, not the AI Model.",
            prompt,
        )
        return self._parse_with_repair(response) or {}

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
        3. REMOVE "Appendix" sections ONLY IF they contain purely supplementary proofs or raw tables. 
        
        CRITICAL EXCEPTIONS - DO NOT REMOVE:
        - If the Appendix contains core algorithmic details or pseudo-code, KEEP IT.
        - If the Appendix contains details about DATASETS, EXPERIMENTAL SETUP, or BENCHMARKS, KEEP IT.
        
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

        # --- Step 1: Parallel Extraction ---
        print(f"  > [Step 1] Extracting Info in Parallel (Basic, Datasets, Math)...")

        import concurrent.futures

        results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            try:
                future_basic = executor.submit(self._extract_basic_info, paper_markdown)
                future_data = executor.submit(self._extract_datasets, paper_markdown)
                future_math = executor.submit(self._extract_math_model, paper_markdown)

                # Wait for all with explicit timeout check to allow interrupt
                futures = [future_basic, future_data, future_math]
                for f in concurrent.futures.as_completed(futures):
                    # Blocking call but allows signal interrupt if handled at OS level,
                    # but Python's thread pool wait masks SIGINT.
                    # We rely on parent process handling, but let's wrap results carefully.
                    pass

                basic_data = future_basic.result()
                dataset_data = future_data.result()
                math_data = future_math.result()
            except KeyboardInterrupt:
                print(
                    "    !!! KeyboardInterrupt in Analysis Phase. Shutting down pool... !!!"
                )
                executor.shutdown(wait=False, cancel_futures=True)
                raise

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
            datasets=dataset_data.get("datasets", []),
            performance_metrics=dataset_data.get("performance_metrics", []),
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
