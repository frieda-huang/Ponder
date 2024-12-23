from pathlib import Path
from typing import Dict, TypeAlias, Union

import tree_sitter_python as tspython
from tree_sitter import Language, Tree

# Custom types
# ============================================================
FilepathType: TypeAlias = Union[str, Path]
ASTMapping: TypeAlias = Dict[FilepathType, Tree]

# Tree-sitter
# ============================================================
PY_LANGUAGE = Language(tspython.language())

# Project Paths
# ============================================================
SRC_ROOT = Path(__file__).parent


class ProjectPaths:
    SRC = Path(__file__).parent
    QUERIES = SRC / "queries"


project_paths = ProjectPaths()
