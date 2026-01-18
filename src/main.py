import os
import argparse
from .clients import LLMClients
from .processors.paper_processor import PaperProcessor
from .processors.code_processor import CodeProcessor
from .database import VectorDB
from .types import ProcessedEntry
from datetime import datetime


import json


def run_workflow(paper_images, repo_path, paper_id):
    """
    Executes the AI Paper Workflow programmatically.
    """
    # Create output directory
    output_dir = os.path.join("output", paper_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Saving intermediate results to: {output_dir} ---")

    # Initialize Clients
    try:
        clients = LLMClients()
    except ValueError as e:
        print(e)
        return

    # Group 1: Paper Processing
    print("--- Group 1: Paper Processing ---")
    paper_structure = None
    if paper_images:
        paper_processor = PaperProcessor(clients)

        # Check if analysis result is already cached
        cached_analysis_path = os.path.join(output_dir, "analysis_result.json")
        cached_md_path = os.path.join(output_dir, "paper_content.md")

        if (
            os.path.exists(cached_analysis_path)
            and os.path.getsize(cached_analysis_path) > 100
        ):
            print(f"[{paper_id}] Found cached Analysis result. Loading...")
            with open(cached_analysis_path, "r", encoding="utf-8") as f:
                paper_structure = json.load(f)
        elif os.path.exists(cached_md_path):
            print(f"[{paper_id}] Found cached OCR result. Loading...")
            with open(cached_md_path, "r", encoding="utf-8") as f:
                paper_markdown = f.read()
            # Still need to analyze logic
            paper_structure = paper_processor.analyzer.analyze_markdown(
                paper_markdown, paper_id
            )
            # Save analysis result to cache future runs
            with open(cached_analysis_path, "w", encoding="utf-8") as f:
                json.dump(paper_structure, f, indent=2, ensure_ascii=False)
        else:
            paper_structure, paper_markdown = paper_processor.process(
                paper_images, paper_id, output_dir=output_dir
            )

        print(f"Paper Analyzed: {paper_structure.get('title', 'Unknown')}")
        print(f"Problem Type: {paper_structure.get('problem_type', 'Unknown')}")
    else:
        print("No paper images provided.")
        return

    # Group 2: Code Processing
    print("--- Group 2: Code Processing ---")
    code_mapping = None
    if repo_path:
        # Check if code mapping is already cached
        cached_code_path = os.path.join(output_dir, "code_mapping.json")

        if os.path.exists(cached_code_path) and os.path.getsize(cached_code_path) > 100:
            print(f"[{paper_id}] Found cached Code Mapping result. Loading...")
            with open(cached_code_path, "r", encoding="utf-8") as f:
                code_mapping = json.load(f)
        else:
            code_processor = CodeProcessor(clients)
            code_mapping = code_processor.process(
                repo_path, paper_structure, output_dir=output_dir
            )

        print(f"Code Found in: {code_mapping.get('file_path', 'N/A')}")
        print(f"Function: {code_mapping.get('function_name', 'N/A')}")
        print(f"Snippet Preview: {code_mapping.get('code_snippet', '')[:50]}...")
    else:
        print("No repo path provided, skipping code grounding.")

    # Storage (ChromaDB)
    print("--- Phase 3: Storage (ChromaDB) ---")
    db = VectorDB()

    if code_mapping:
        entry: ProcessedEntry = {
            "paper_info": paper_structure,
            "code_implementation": code_mapping,
            "keywords": [paper_structure["problem_type"], "optimization"],
            "repo_path": repo_path,
        }
        db.add_entry(entry)
        print("Entry added to database.")
    else:
        print("Skipping database entry due to missing code mapping.")

    print("Workflow Complete.")


def main():
    parser = argparse.ArgumentParser(description="AI Paper Workflow")
    parser.add_argument(
        "--paper_images", nargs="+", help="List of image paths for the paper pages"
    )
    parser.add_argument("--repo_path", help="Path to the local code repository")
    parser.add_argument(
        "--paper_id",
        default=datetime.now().strftime("%Y%m%d%H%M"),
        help="Unique ID for the paper",
    )

    args = parser.parse_args()

    run_workflow(args.paper_images, args.repo_path, args.paper_id)


if __name__ == "__main__":
    main()
