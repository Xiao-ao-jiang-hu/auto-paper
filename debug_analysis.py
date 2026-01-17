import os
import glob
import sys
import argparse
import json

# Ensure src is in path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.clients import LLMClients
from src.processors.paper_processor import PaperProcessor


def run_debug_analysis(paper_id):
    # Locate paper directory
    data_dir = os.path.join(project_root, "scripts", "data")
    paper_dir = os.path.join(data_dir, paper_id)
    images_dir = os.path.join(paper_dir, "paper_images")
    output_dir = os.path.join(project_root, "output", paper_id)

    if not os.path.exists(images_dir):
        print(f"Error: Images dir not found at {images_dir}")
        return

    # Get all jpg images
    image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
    try:
        image_paths.sort(
            key=lambda x: (
                int(os.path.splitext(os.path.basename(x))[0].split("_")[1])
                if "_" in os.path.basename(x)
                else os.path.basename(x)
            )
        )
    except:
        image_paths.sort()

    print(f"Loaded {len(image_paths)} images for {paper_id}")

    clients = LLMClients()
    processor = PaperProcessor(clients)

    # We only want to test analysis, but PaperProcessor usually does both.
    # However, if we skip ingestion (OCR), we need the markdown.
    # Let's see if we have cached markdown.

    cached_md_path = os.path.join(output_dir, "paper_content.md")
    paper_markdown = ""

    if os.path.exists(cached_md_path):
        print(f"Found cached Markdown at {cached_md_path}")
        with open(cached_md_path, "r", encoding="utf-8") as f:
            paper_markdown = f.read()
    else:
        print("No cached Markdown found. Running OCR (Ingestion)...")
        # Run standard process which includes ingestion
        # But we only want to debug analysis.
        # For now, let's just run the full processor but we know OCR works.
        pass

    if paper_markdown:
        # If we have markdown, we can call analyzer directly
        print("Running Analyzer directly on cached Markdown...")
        try:
            structure = processor.analyzer.analyze_markdown(paper_markdown, paper_id)

            # Save result
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "analysis_result.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(structure, f, indent=2, ensure_ascii=False)

            print(f"Analysis complete. Structure saved to {out_path}")
            print(json.dumps(structure, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"Analysis Failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        # Run full pipeline with PaperProcessor (which handles ingestion + analysis)
        print("Running full Paper Processing...")
        processor.process(image_paths, paper_id, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_id", required=True)
    args = parser.parse_args()

    run_debug_analysis(args.paper_id)
