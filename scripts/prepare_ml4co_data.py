import os
import requests
import fitz  # PyMuPDF
from git import Repo
import shutil
import csv
import re

# Adjust paths as necessary
CSV_PATH = os.path.join(
    "e:\\",
    "senior_fall",
    "research",
    "auto-paper",
    "awesome-ml4co",
    "data",
    "papers.csv",
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")  # Output in scripts/data/


def clean_filename(s):
    # Remove invalid characters and shorten
    s = re.sub(r'[\\/*?:"<>|]', "", s)
    s = s.replace(" ", "_")
    return s[:60]


def get_pdf_url(link):
    if not link:
        return None
    link = link.strip()

    # 1. Arxiv
    if "arxiv.org/abs/" in link:
        return link.replace("arxiv.org/abs/", "arxiv.org/pdf/") + ".pdf"
    if "arxiv.org/pdf/" in link and not link.endswith(".pdf"):
        return link + ".pdf"

    # 2. OpenReview (e.g. forum?id=... -> pdf?id=...)
    if "openreview.net/forum" in link:
        return link.replace("/forum", "/pdf")

    # 3. NeurIPS / NIPS (papers.nips.cc)
    # e.g. .../hash/.....-Abstract.html -> .../hash/.....-Paper.pdf
    if "papers.nips.cc/paper/" in link and link.endswith("-Abstract.html"):
        return link.replace("-Abstract.html", "-Paper.pdf")

    # 4. PMLR
    # e.g. http://proceedings.mlr.press/v162/author22a.html -> http://proceedings.mlr.press/v162/author22a/author22a.pdf
    # This is tricky for generic handling without scraping, but some links might be direct.
    if "proceedings.mlr.press" in link and link.endswith(".html"):
        # Heuristic: try replacing .html with .pdf (often works for v1 and some layouts)
        return link.replace(".html", ".pdf")

    # 5. ACL Anthology
    if "aclanthology.org" in link and not link.endswith(".pdf"):
        if link.endswith("/"):
            return link[:-1] + ".pdf"
        return link + ".pdf"

    # 6. CVF (CVPR/ICCV)
    # openaccess/content_cvpr_2016/html/Author_Title_CVPR_2016_paper.html -> ..._paper.pdf
    if "thecvf.com" in link and link.endswith(".html"):
        return link.replace(".html", ".pdf")

    # 7. Generic fallback: if it ends in .pdf, verify logic in download
    if link.lower().endswith(".pdf"):
        return link

    return link


def download_pdf(url, output_path):
    if not url:
        return False

    print(f"  Downloading PDF from {url}...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200 and "application/pdf" in response.headers.get(
            "Content-Type", ""
        ):
            with open(output_path, "wb") as f:
                f.write(response.content)
            print("  PDF downloaded success.")
            return True
        elif response.status_code == 200 and url.endswith(".pdf"):
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(
                f"  Failed to download PDF (Status: {response.status_code}, Type: {response.headers.get('Content-Type')})"
            )
            return False
    except Exception as e:
        print(f"  Exception downloading PDF: {e}")
        return False


def pdf_to_images(pdf_path, output_dir):
    try:
        print(f"  Converting {os.path.basename(pdf_path)} to images...")
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            print("  PDF is empty or invalid.")
            return

        for i in range(len(doc)):
            page = doc.load_page(i)
            # Zoom x2 for better OCR quality
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            image_path = os.path.join(output_dir, f"page_{i+1}.jpg")
            pix.save(image_path)
        print(f"  Created {len(doc)} images.")
    except Exception as e:
        print(f"  Error converting PDF to images: {e}")


def clone_repo(repo_url, output_dir):
    if not repo_url or repo_url.strip() == "":
        return False

    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"  Repo dir already exists (skipping clone): {output_dir}")
        return True

    print(f"  Cloning {repo_url}...")
    try:
        # Handle github links that are not .git ending
        if "github.com" in repo_url and not repo_url.endswith(".git"):
            repo_url += ".git"

        Repo.clone_from(repo_url, output_dir, depth=1)
        print("  Repo cloned success.")
        return True
    except Exception as e:
        print(f"  Error cloning repo: {e}")
        return False


def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV file not found at: {CSV_PATH}")
        return

    print(f"Reading papers from: {CSV_PATH}")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    count = 0
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("title", "Untitled").strip()
            link = row.get("link", "").strip()
            code = row.get("code", "").strip()

            # Filter: Must have code
            if not code:
                continue

            print(f"\n[{count+1}] Processing: {title}")
            print(f"  Code: {code}")
            print(f"  Link: {link}")

            # Prepare folders
            safe_id = clean_filename(title)
            paper_dir = os.path.join(DATA_DIR, safe_id)
            repo_dir = os.path.join(paper_dir, "repo")
            paper_img_dir = os.path.join(paper_dir, "paper_images")

            os.makedirs(repo_dir, exist_ok=True)
            os.makedirs(paper_img_dir, exist_ok=True)

            # 1. Clone Repo
            repo_success = clone_repo(code, repo_dir)

            # 2. Download and Convert PDF
            pdf_url = get_pdf_url(link)
            pdf_path = os.path.join(paper_dir, "paper.pdf")

            # Allow attempting download if we successfully resolved a URL
            if pdf_url:
                if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) < 1000:
                    description = f"Downloading PDF from {pdf_url}"
                    if download_pdf(pdf_url, pdf_path):
                        pdf_to_images(pdf_path, paper_img_dir)
                elif not os.listdir(paper_img_dir):
                    pdf_to_images(pdf_path, paper_img_dir)
                else:
                    print(f"  PDF/Images already ready for {safe_id}.")
            else:
                print(f"  Could not determine PDF URL from: {link}")

            count += 1
            if count >= 10:
                print("Limit of 10 papers reached for this run.")
                break


if __name__ == "__main__":
    main()
