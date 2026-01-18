import os
import sys
import json
import argparse
from tqdm import tqdm
import concurrent.futures

# Add parent directory to path to allow importing src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from src.clients import LLMClients
    from src.processors.analysis import PaperAnalyzer
except ImportError as e:
    print(f"Error importing src modules: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


def process_single_paper(d, output_dir, analyzer):
    paper_path = os.path.join(output_dir, d)
    analysis_path = os.path.join(paper_path, "analysis_result.json")
    md_path = os.path.join(paper_path, "paper_content.md")

    if not os.path.exists(analysis_path) or not os.path.exists(md_path):
        return None

    try:
        # Load Markdown
        with open(md_path, "r", encoding="utf-8") as f:
            paper_markdown = f.read()

        # Load Existing Analysis
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)

        # Re-run Math Extraction
        # Accessing protected method _extract_math_model strictly for this script
        new_math_data = analyzer._extract_math_model(paper_markdown)

        # Update specific fields
        if new_math_data:
            analysis_data["lp_model"] = new_math_data.get("lp_model", {})
            analysis_data["raw_latex_model"] = new_math_data.get("raw_latex_model", "")
            analysis_data["algorithm_description"] = new_math_data.get(
                "algorithm_description", ""
            )

            # Save immediately
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            return "Updated"
        else:
            return f"Failed extraction"

    except Exception as e:
        return f"Error: {e}"


def regenerate_models(output_dir):
    print("Initializing LLM Clients...")
    clients = LLMClients()
    analyzer = PaperAnalyzer(clients)

    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return

    paper_dirs = [
        d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))
    ]

    print(f"Found {len(paper_dirs)} papers in {output_dir}")

    # Parallel Processing
    max_workers = 8
    print(f"Starting regeneration with {max_workers} threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_paper, d, output_dir, analyzer): d
            for d in paper_dirs
        }

        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(paper_dirs),
            desc="Regenerating Models",
        ):
            d = futures[future]
            try:
                result = future.result()
                if result and result != "Updated":
                    tqdm.write(f"[{d}] {result}")
            except Exception as e:
                tqdm.write(f"[{d}] Exception: {e}")


if __name__ == "__main__":
    output_dir = os.path.join(parent_dir, "output")
    regenerate_models(output_dir)
