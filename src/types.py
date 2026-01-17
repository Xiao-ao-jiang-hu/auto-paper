from typing import List, Dict, Optional, TypedDict


class LPFormulation(TypedDict):
    objective: str  # LaTeX string
    constraints: List[str]  # List of LaTeX strings
    variables: List[str]  # List of variable definitions


class PaperStructure(TypedDict):
    paper_id: str
    title: str
    abstract: str
    problem_description_natural: str  # Natural language summary of the problem
    lp_model: LPFormulation  # Structured LP format
    raw_latex_model: str  # The original model extracted from paper
    algorithm_description: str
    problem_type: str  # e.g. MILP, LP, IP, MINLP
    datasets: List[str]  # Experimental datasets used
    performance_metrics: List[str]  # Metrics used for evaluation


class CodeMapping(TypedDict):
    file_path: str
    function_name: str
    code_snippet: str
    description: str  # Why this code is relevant to the LP model
    dependencies: List[str]  # List of internal/external dependencies found


class ProcessedEntry(TypedDict):
    paper_info: PaperStructure
    code_implementation: CodeMapping
    keywords: List[str]
    repo_path: str
