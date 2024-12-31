from pathlib import Path
from typing import Dict, TypeAlias, Union

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Tree

# Custom types
# ============================================================
FilepathType: TypeAlias = Union[str, Path]
ASTMapping: TypeAlias = Dict[FilepathType, Tree]

# Tree-sitter
# ============================================================
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

# Project Paths
# ============================================================
SRC_ROOT = Path(__file__).parent


class ProjectPaths:
    SRC = Path(__file__).parent
    QUERIES = SRC / "queries"
    VOCAB_FILE = SRC / "llm" / "spiece.model"


project_paths = ProjectPaths()
