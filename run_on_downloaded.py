import os
import random
import sys
import glob

# Constants
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "scripts", "data")
BATCH_SIZE = 2

# Ensure project root is in python path to allow imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.main import run_workflow as run_pipeline


def get_valid_papers(data_dir):
    valid_papers = []
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return []

    for item in os.listdir(data_dir)[1:3]:
        paper_path = os.path.join(data_dir, item)
        if not os.path.isdir(paper_path):
            continue

        # Check required subfolders
        repo_path = os.path.join(paper_path, "repo")
        images_path = os.path.join(paper_path, "paper_images")

        has_repo = os.path.exists(repo_path) and len(os.listdir(repo_path)) > 0

        # Get image files
        image_files = glob.glob(os.path.join(images_path, "*.jpg"))
        has_images = len(image_files) > 0

        if has_repo and has_images:
            valid_papers.append(
                {"id": item, "repo_path": repo_path, "images": image_files}
            )

    return valid_papers


def process_paper(paper):
    print(f"\n{'='*50}")
    print(f"Running workflow for: {paper['id']}")
    print(f"{'='*50}")

    # Sort images to ensure correct order
    paper["images"].sort(
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1])
    )

    try:
        run_pipeline(
            paper_images=paper["images"],
            repo_path=paper["repo_path"],
            paper_id=paper["id"],
        )
    except Exception as e:
        print(f"Exception running workflow: {e}")


def main():
    print(f"Searching for papers in {DATA_DIR}...")
    valid_papers = get_valid_papers(DATA_DIR)

    if not valid_papers:
        print("No valid papers found (must have 'repo' and 'paper_images' with JPGs).")
        return

    print(f"Found {len(valid_papers)} valid papers.")

    # Select random batch
    selected_papers = random.sample(valid_papers, min(BATCH_SIZE, len(valid_papers)))

    print(f"Selected {len(selected_papers)} papers for processing:")
    for p in selected_papers:
        print(f" - {p['id']}")

    for paper in selected_papers:
        process_paper(paper)


if __name__ == "__main__":
    main()
