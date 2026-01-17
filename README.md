# AI Automated Paper-to-Code Workflow

This project implements an automated pipeline to convert research papers (specifically in the domain of Planning and Combinatorial Optimization) into structured data and code mappings. It uses a multi-agent LLM approach to digitize, analyze, and ground the papers into their corresponding code repositories.

## 🚀 Features

- **Multi-Modal Ingestion**: Converts PDF page images into structured Markdown using `qwen-vl-plus`. 
  - *New*: Supports multi-threaded OCR (parallelism = 3 by default) for faster processing.
- **Deep Reasoning**: Uses `kimi-k2-thinking` to extract mathematical formulations, constraints, and objective functions into standard LP (Linear Programming) formats.
- **Code Grounding**: Maps mathematical symbols and logic to actual source code implementations using `qwen-max`.
- **Structured Storage**: Embeds knowledge using `text-embedding-v4` and stores it in a local ChromaDB vector database.

## 🏗 Architecture

The workflow consists of three main processors managed by a facade:
1.  **IngestionProcessor**: Handles OCR and layout analysis.
2.  **AnalysisProcessor**: Extracts semantic logic and mathematical models.
3.  **GroundingProcessor**: Binds the abstract logic to concrete code implementation.

## 🛠 Setup

1.  **Prerequisites**:
    *   Python 3.10+
    *   DashScope API Key (Alibaba Cloud) for Qwen models.
    *   Moonshot API Key for Kimi models.

2.  **External Data Source**:
    Clone the `awesome-ml4co` repository parallel to this project folder:
    ```bash
    cd ..
    git clone https://github.com/Thinklab-SJTU/awesome-ml4co.git
    cd ai_paper_workflow
    ```

3.  **Installation**:
    ```bash
    conda env create -f environment.yml
    ```

4.  **Environment Variables**:
    Set your API keys in the environment:
    ```bash
    export DASHSCOPE_API_KEY="your_dashscope_key"
    ```

## 🏃 Usage

### 1. Data Preparation
To download sample papers from the ML4CO dataset:
```bash
python scripts/prepare_ml4co_data.py
```
This will download papers and repos into the `scripts/data` directory.

### 2. Run Single Pipeline
To run the workflow on a specific paper and repo:
```bash
python -m src.main --repo_path "path/to/repo" --paper_images "page1.jpg" "page2.jpg" ...
```

### 3. Batch Processing
To run the workflow on randomly selected downloaded papers:
```bash
python run_on_downloaded.py
```


## 📂 Output

Intermediate results are saved in `output/{paper_id}/`:
*   `full_paper.md`: The digitized markdown text.
*   `analysis.json`: Extracted LP formulations.
*   `grounding.json`: Symbol-to-Code equivalence mappings.

## 🤖 Models Used

*   **Vision**: `qwen-vl-max` / `qwen-vl-plus`
*   **Reasoning**: `kimi-k2-thinking`
*   **Code Understanding**: `qwen-max`
*   **Embedding**: `text-embedding-v4`
