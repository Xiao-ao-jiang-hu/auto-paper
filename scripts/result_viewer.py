import os
import sys
import subprocess

# Helper to run via python directly: python scripts/result_viewer.py
if __name__ == "__main__" and "STREAMLIT_RUN" not in os.environ:
    # Set flag to prevent recursion when streamlit executes this script again
    os.environ["STREAMLIT_RUN"] = "1"
    cmd = [sys.executable, "-m", "streamlit", "run", __file__]
    print(f"Launching Streamlit... {' '.join(cmd)}")
    sys.exit(subprocess.run(cmd).returncode)

import streamlit as st
import json

# Page Config
st.set_page_config(
    page_title="AI Paper Analysis Viewer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
output_dir = os.path.join(project_root, "output")

st.title("📄 Auto-Paper Result Viewer")

# Sidebar: Paper Selection
st.sidebar.header("Paper Selection")

if not os.path.exists(output_dir):
    st.error(f"Output directory not found: {output_dir}")
    st.stop()

# Get list of paper directories
paper_dirs = [
    d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))
]
paper_dirs.sort()

search_query = st.sidebar.text_input("Search Papers", "")
if search_query:
    paper_dirs = [d for d in paper_dirs if search_query.lower() in d.lower()]

selected_paper_dir_name = st.sidebar.selectbox(
    "Select a Paper", paper_dirs, index=0 if paper_dirs else None
)

if not selected_paper_dir_name:
    st.info("No papers found in output directory.")
    st.stop()

selected_paper_path = os.path.join(output_dir, selected_paper_dir_name)

# Status Indicators
analysis_path = os.path.join(selected_paper_path, "analysis_result.json")
code_path = os.path.join(selected_paper_path, "code_mapping.json")
md_path = os.path.join(selected_paper_path, "paper_content.md")

has_analysis = os.path.exists(analysis_path)
has_code = os.path.exists(code_path)
has_md = os.path.exists(md_path)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Status:**")
st.sidebar.markdown(f"{'✅' if has_md else '❌'} Markdown Content")
st.sidebar.markdown(f"{'✅' if has_analysis else '❌'} Analysis Result")
st.sidebar.markdown(f"{'✅' if has_code else '❌'} Code Mapping")


# Load Data
analysis_data = {}
code_data = {}
md_content = ""

if has_analysis:
    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis_data = json.load(f)

if has_code:
    with open(code_path, "r", encoding="utf-8") as f:
        code_data = json.load(f)

if has_md:
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()


# Main Content Area
if analysis_data:
    st.header(analysis_data.get("title", selected_paper_dir_name))
else:
    st.header(selected_paper_dir_name)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Analysis & Model", "📝 Paper Content", "💻 Code Mapping", "📂 Raw Data"]
)

with tab1:
    if has_analysis:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Basic Info")
            st.markdown(f"**Problem Type:** {analysis_data.get('problem_type', 'N/A')}")
            st.markdown("**Abstract:**")
            st.info(analysis_data.get("abstract", "N/A"))

            st.markdown("**Algorithm Description:**")
            st.write(analysis_data.get("algorithm_description", "N/A"))

        with col2:
            st.subheader("Mathematical Model")
            lp_model = analysis_data.get("lp_model", {})

            if not lp_model and not analysis_data.get("raw_latex_model"):
                st.warning("No explicit mathematical model extracted.")

            if analysis_data.get("raw_latex_model"):
                st.markdown("### Raw LaTeX")
                st.latex(analysis_data.get("raw_latex_model"))

            if lp_model:
                st.markdown("### Structured Model")

                # Objective
                obj = lp_model.get("objective", "")
                if obj:
                    st.markdown("**Objective Function:**")
                    st.latex(obj)

                # Constraints
                constraints = lp_model.get("constraints", [])
                if constraints:
                    st.markdown("**Constraints:**")
                    for c in constraints:
                        # Clean up text for latex rendering if needed
                        st.latex(c)

                # Variables
                variables = lp_model.get("variables", [])
                if variables:
                    st.markdown("**Variables:**")
                    for v in variables:
                        st.write(f"- {v}")
    else:
        st.warning("Analysis result (analysis_result.json) not found.")

with tab2:
    if has_md:
        st.markdown(md_content)
    else:
        st.warning("Markdown content (paper_content.md) not found.")

with tab3:
    if has_code:
        st.subheader("Code Grounding Result")
        st.success(f"File Found: `{code_data.get('file_path', 'N/A')}`")
        st.info(f"Target Function/Class: `{code_data.get('function_name', 'N/A')}`")

        st.markdown("**Relevant Code Snippet:**")
        snippet = code_data.get("code_snippet", "")
        if snippet:
            # Guess language or default to python
            file_ext = os.path.splitext(code_data.get("file_path", ""))[-1]
            lang = (
                "python"
                if file_ext in [".py"]
                else "cpp" if file_ext in [".cpp", ".h", ".cc"] else "text"
            )
            st.code(snippet, language=lang)
    else:
        st.warning("Code mapping (code_mapping.json) not found.")

with tab4:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Analysis JSON")
        if has_analysis:
            st.json(analysis_data)
        else:
            st.text("Not available")

    with col_b:
        st.markdown("### Code JSON")
        if has_code:
            st.json(code_data)
        else:
            st.text("Not available")

st.markdown("---")
st.markdown("Generated by Auto-Paper Analysis Workflow")
