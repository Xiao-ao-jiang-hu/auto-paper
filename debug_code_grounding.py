import os
import json
import argparse
from src.clients import LLMClients
from src.processors.code_processor import CodeProcessor
from src.types import PaperStructure


def main():
    parser = argparse.ArgumentParser(description="Debug Stage 2: Code Grounding")
    parser.add_argument(
        "--paper_dir",
        required=True,
        help="Path to the output directory of the paper (containing analysis_result.json)",
    )
    parser.add_argument(
        "--repo_root",
        default="scripts/data",
        help="Root directory where repos are downloaded",
    )

    args = parser.parse_args()

    # Paths
    paper_dir = args.paper_dir
    analysis_path = os.path.join(paper_dir, "analysis_result.json")

    if not os.path.exists(analysis_path):
        print(f"Error: {analysis_path} not found.")
        return

    # Load Analysis Result
    with open(analysis_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Ensure LP Formualtion is correctly typed if needed, but dict usually works for simple passing
        # However, CodeProcessor expects PaperStructure
        paper_struct = data  # Load as dict

    print(f"Loaded analysis for: {data.get('title', 'Unknown')}")
    paper_id = data.get("paper_id")

    # Locate Repo
    # Assuming repo is in scripts/data/{paper_id}/repo
    # or user might provide a specific repo path.
    # Let's try to infer it from the output folder name if it matches the data folder name
    folder_name = os.path.basename(paper_dir.rstrip("\\/"))
    repo_path = os.path.join(args.repo_root, folder_name, "repo")

    if not os.path.exists(repo_path):
        print(f"Repo not found at expected path: {repo_path}")
        # Try finding it by ID?
        # Fallback manual check
        return

    print(f"Using Repo Path: {repo_path}")

    # Init Clients
    clients = LLMClients()

    # Run Code Processor
    processor = CodeProcessor(clients)

    print("\n--- Starting Code Grounding (Debug Mode) ---")
    mapping = processor.process(repo_path, paper_struct, output_dir=paper_dir)

    print("\n--- Result ---")
    print(json.dumps(mapping, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
