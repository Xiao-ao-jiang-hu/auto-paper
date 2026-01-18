import base64
import os
from ..clients import LLMClients


class PaperIngestion:
    def __init__(self, clients: LLMClients):
        self.clients = clients

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def process_paper_images(self, image_paths: list[str]) -> str:
        """
        Takes a list of paths to images (pages of the paper).
        Returns the markdown text of the full paper.
        """
        full_markdown = []

        prompt = """
        You are a highly accurate academic paper digitizer.
        Convert this SINGLE PAGE of a research paper into Markdown.
        
        CRITICAL INSTRUCTIONS:
        1. Mathematics: You MUST transcribe all equations into LaTeX format.
           - Inline math: $...$
           - Block math: $$...$$
        2. Structure: Preserve hierarchy using #, ##, ###.
        3. Layout: Ignore page headers, footers, and page numbers. 
        4. Content: Do not summarize. Transcribe full text exactly.
        """

        from ..config import OCR_PARALLELISM
        import concurrent.futures

        print(
            f"  > Processing {len(image_paths)} pages concurrently (parallelism={OCR_PARALLELISM})..."
        )

        # Placeholder for ordered results
        ordered_pages = [None] * len(image_paths)

        def process_single_page(idx, img_path):
            print(
                f"    > Digitizing Page {idx+1}/{len(image_paths)}: {os.path.basename(img_path)}..."
            )
            try:
                b64_img = self.encode_image(img_path)
                # Send single page
                page_content = self.clients.get_vision_completion([b64_img], prompt)
                if page_content:
                    # Cleanup specific to per-page output
                    cleaned = page_content.strip()
                    if cleaned.startswith("```markdown"):
                        cleaned = cleaned[11:]
                    elif cleaned.startswith("```"):
                        cleaned = cleaned[3:]

                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]

                    return idx, f"<!-- Page {idx+1} -->\n{cleaned.strip()}"
            except Exception as e:
                print(f"    ! Error on Page {idx+1}: {e}")
            return idx, None

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=OCR_PARALLELISM
            ) as executor:
                # Submit all tasks
                futures = [
                    executor.submit(process_single_page, idx, img_path)
                    for idx, img_path in enumerate(image_paths)
                ]

                # Wait for completion and gather results
                # Using wait allows better interrupt handling than simple iteration if structured correctly,
                # but as_completed loop is standard.
                # To handle KeyboardInterrupt immediately, we need to catch it outside.
                for future in concurrent.futures.as_completed(futures):
                    idx, content = future.result()
                    if content:
                        ordered_pages[idx] = content
        except KeyboardInterrupt:
            print(
                "    !!! KeyboardInterrupt in Ingestion Phase. Shutting down pool... !!!"
            )
            executor.shutdown(wait=False, cancel_futures=True)
            raise

        full_markdown = [p for p in ordered_pages if p is not None]
        return "\n\n".join(full_markdown)
