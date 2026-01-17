import os
import json
from ..clients import LLMClients
from ..types import PaperStructure, CodeMapping


class CodeGrounder:
    def __init__(self, clients: LLMClients):
        self.clients = clients

    def _get_repo_structure(self, repo_path: str) -> str:
        """Generates a file tree string."""
        tree = []
        for root, dirs, files in os.walk(repo_path):
            level = root.replace(repo_path, "").count(os.sep)
            indent = " " * 4 * (level)
            tree.append(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 4 * (level + 1)
            for f in files:
                if f.endswith((".py", ".cpp", ".h", ".java")):  # User can filter
                    tree.append(f"{subindent}{f}")
        return "\n".join(tree)

    def _read_file_content(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except:
            return "Error reading file."

    def find_code(self, repo_path: str, paper_data: PaperStructure) -> CodeMapping:
        repo_tree = self._get_repo_structure(repo_path)

        # Safeguard: Handle missing/empty LP model gracefully
        lp_vars = paper_data.get("lp_model", {}).get("variables", [])
        lp_obj = paper_data.get("lp_model", {}).get("objective", "Not specific")
        lp_constr = paper_data.get("lp_model", {}).get("constraints", [])

        # Step 1: Filter candidates (File Level)
        print("  Scanning repository structure...")
        filter_prompt = f"""
        Paper Algorithm: {paper_data.get('algorithm_description', 'Unknown')}
        Paper LP Model Variables: {lp_vars}
        
        Repository Structure:
        {repo_tree}
        
        Task: Identify the TOP 7 file paths in this repo that most likely contain the implementation of the core algorithm or the LP model construction.
        Return ONLY a JSON list of strings. Example: ["src/solver.py", "include/model.h"]
        """

        response_1 = self.clients.get_instruction_completion(
            "You are a Senior Software Engineer acting as a Code Navigator. Select the most relevant files.",
            filter_prompt,
        )

        try:
            import re

            match = re.search(r"\[.*\]", response_1, re.DOTALL)
            files = json.loads(match.group(0)) if match else []
        except:
            files = []

        print(f"  Candidate files: {files}")

        # Step 2: Extract in Parallel
        from ..config import CODE_GROUNDING_PARALLELISM
        import concurrent.futures

        # We will process each file INDIVIDUALLY to find the best match
        # This avoids context length limits and allows parallel processing

        print(
            f"  > Analyzing {len(files)} files concurrently (parallelism={CODE_GROUNDING_PARALLELISM})..."
        )

        best_match = None

        def analyze_single_file(rel_path):
            full_path = os.path.join(repo_path, rel_path.strip().strip('"').strip("'"))
            if not os.path.exists(full_path):
                return None

            content = self._read_file_content(full_path)[:15000]  # 15k chars per file

            grounding_prompt = f"""
            File: {rel_path}
            Content:
            {content}
            
            ---
            TARGET: Does this file implement the following Core Logic?
            
            Paper Title: {paper_data['title']}
            Problem Description: {paper_data['problem_description_natural']}
            
            Optimization Model:
            Objective: {lp_obj}
            Constraints: {lp_constr}
            Algorithm Steps: {paper_data.get('algorithm_description', 'Unknown')}
            
            Your Goal:
            1. Analyze if this specific file implements the core optimization logic or mathematical model.
            2. If YES, extract the snippet and explain the mapping.
            3. If NO (e.g. it is just utility code, config, or unrelated), return empty values.
            
            Output JSON:
            {{
                "is_match": true/false,
                "file_path": "{rel_path}",
                "function_name": "The containing function (or empty)",
                "code_snippet": "The actual lines of code (approx 10-50 lines). PRESERVE INDENTATION. (or empty)",
                "description": "Analysis of compatibility."
            }}
            """

            try:
                # Use Instruction model here for speed/cost, or Reasoning if high accuracy needed.
                # Since we parallelize, let's use Instruction model first or a lighter Reasoning call.
                # Based on previous context, user prefers Reasoning for "deep alignment".
                resp = self.clients.get_reasoning_completion(
                    "You are an expert at mapping Mathematical Optimization Models to Code implementations.",
                    grounding_prompt,
                )

                # Parse JSON (Simplified)
                json_str = ""
                if "```json" in resp:
                    parts = resp.split("```json")
                    if len(parts) > 1:
                        json_str = parts[1].split("```")[0].strip()
                elif "```" in resp:
                    # Check other generic blocks
                    parts = resp.split("```")
                    for p in parts:
                        if p.strip().startswith("{"):
                            json_str = p.strip()
                            break

                if not json_str and "{" in resp:
                    # Fallback find braces
                    start = resp.find("{")
                    end = resp.rfind("}")
                    if start != -1 and end != -1:
                        json_str = resp[start : end + 1]
                elif "{" in resp:
                    start = resp.find("{")
                    end = resp.rfind("}")
                    json_str = resp[start : end + 1]

                if json_str:
                    data = json.loads(json_str)
                    if data.get("is_match") is True and data.get("code_snippet"):
                        return CodeMapping(
                            file_path=data["file_path"],
                            function_name=data.get("function_name", ""),
                            code_snippet=data.get("code_snippet", ""),
                            description=data.get("description", ""),
                        )
            except Exception as e:
                # print(f"Error processing {rel_path}: {e}")
                pass
            return None

        # Execute Parallel Analysis
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=CODE_GROUNDING_PARALLELISM
        ) as executor:
            futures = [executor.submit(analyze_single_file, f) for f in files]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    # Heuristic: Prefer the one with longest description or just first match
                    # For now, return the first valid match found
                    print(f"  > Match found in: {result['file_path']}")
                    return result

        # Fallback if no match found in individual files
        return CodeMapping(
            file_path="",
            function_name="",
            code_snippet="",
            description="No exact match found in candidate files.",
        )
