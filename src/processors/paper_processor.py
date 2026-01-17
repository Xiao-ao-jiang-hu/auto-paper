import os
import json
from typing import List, Tuple, Optional
from ..clients import LLMClients
from .ingestion import PaperIngestion
from .analysis import PaperAnalyzer
from ..types import PaperStructure


class PaperProcessor:
    """
    Facade class that handles the entire paper processing pipeline:
    1. Ingestion (Images -> Markdown)
    2. Analysis (Markdown -> Structured Knowledge)
    """

    def __init__(self, clients: LLMClients):
        self.ingestor = PaperIngestion(clients)
        self.analyzer = PaperAnalyzer(clients)

    def process(
        self, image_paths: List[str], paper_id: str, output_dir: Optional[str] = None
    ) -> Tuple[PaperStructure, str]:
        """
        Runs the full paper processing workflow.
        Returns (StructuredData, RawMarkdown)

        If output_dir is provided, saves intermediate results immediately.
        """
        print(f"[{paper_id}] Step 1/2: Digitizing {len(image_paths)} pages...")
        try:
            paper_markdown = self.ingestor.process_paper_images(image_paths)

            # IMMEDIATE SAVE: Markdown
            if output_dir and paper_markdown:
                md_path = os.path.join(output_dir, "paper_content.md")
                print(f"[{paper_id}] Saving intermediate OCR result to {md_path}")
                os.makedirs(output_dir, exist_ok=True)

                # Cleanup potential ```markdown wrappers and other stray markers
                clean_md = paper_markdown
                if clean_md.startswith("```markdown"):
                    clean_md = clean_md[11:]
                elif clean_md.startswith("```"):
                    clean_md = clean_md[3:]

                if clean_md.strip().endswith("```"):
                    clean_md = clean_md.strip()[:-3]

                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(clean_md)

        except Exception as e:
            print(f"[{paper_id}] Error during digitization: {e}")
            paper_markdown = ""

        if not paper_markdown:
            print(
                f"[{paper_id}] Digitization failed (empty output). Skipping analysis."
            )
            # Return a dummy error structure to prevent crash
            dummy = {
                "paper_id": paper_id,
                "title": "Error",
                "problem_type": "Error",
                "abstract": "",
                "problem_description_natural": "",
                "lp_model": {"objective": "", "constraints": [], "variables": []},
                "algorithm_description": "",
                "raw_latex_model": "",
            }
            return dummy, ""

        print(
            f"[{paper_id}] Digitization complete. Markdown length: {len(paper_markdown)} chars."
        )

        print(f"[{paper_id}] Step 2/2: Analyzing content & extracting LP Model...")
        paper_structure = self.analyzer.analyze_markdown(paper_markdown, paper_id)

        # IMMEDIATE SAVE: Analysis Structure
        if output_dir and paper_structure:
            json_path = os.path.join(output_dir, "analysis_result.json")
            print(f"[{paper_id}] Saving intermediate Analysis result to {json_path}")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(paper_structure, f, indent=2, ensure_ascii=False)

        return paper_structure, paper_markdown
