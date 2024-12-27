from pathlib import Path
from typing import Generator, List

import networkx as nx
from codebase_parser.ast_graph_builder import ASTGraphBuilder
from rich import print
from src.commons import PY_LANGUAGE, ASTMapping
from tree_sitter import Node, Parser, Tree


class CodebaseParser:
    def __init__(self, directory: str):
        self.directory = directory
        self.parser = Parser(PY_LANGUAGE)
        self.ast_tree: ASTMapping = {}

    def get_python_filepaths(self) -> List[Path]:
        """Get all Python filepaths in the codebase."""
        return [
            filepath
            for filepath in Path(self.directory).rglob("*.py")
            if self.is_valid_python_file(filepath)
        ]

    def parse(self) -> dict:
        """Parse all relevant files and store the results in a dictionary."""
        for filepath in self.get_python_filepaths():
            tree = self._parse_file(filepath)
            self.ast_tree[str(filepath)] = tree
        return self.ast_tree

    def is_valid_python_file(self, filepath: Path) -> bool:
        """Comprehensive validation for Python files."""
        # Exclude virtual environment directories
        excluded_dirs = [
            ".venv",
            "venv",
            "env",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".git",
            ".github",
            "node_modules",
            "build",
            "dist",
        ]

        # Check against excluded directories
        path_parts = filepath.parts
        if any(excluded in path_parts for excluded in excluded_dirs):
            return False

        # Ignore hidden files/directories
        if any(part.startswith(".") for part in path_parts):
            return False

        # Additional file validation
        if not filepath.is_file():
            return False

        # Ensure it's a Python file
        if filepath.suffix != ".py":
            return False

        # Skip empty files
        if filepath.stat().st_size == 0:
            return False

        return True

    def print_tree(self, tree: Tree) -> Generator[Node, None, None]:
        """
        Based on https://github.com/tree-sitter/py-tree-sitter/blob/master/examples/walk_tree.py

        Traverse the tree and print each node.
        """
        cursor = tree.walk()
        visited_children = False

        while True:
            print(cursor.node)
            if not visited_children:
                yield cursor.node
                if not cursor.goto_first_child():
                    visited_children = True
            elif cursor.goto_next_sibling():
                visited_children = False
            elif not cursor.goto_parent():
                break

    def build_graph(
        self, ast_tree: ASTMapping, ast_graph_builder: ASTGraphBuilder
    ) -> nx.MultiDiGraph:
        """Build a graph from the AST."""
        return ast_graph_builder.build_graph_from_ast(ast_tree)

    def _read_file(self, filepath: Path) -> str:
        """Read a file and return its contents."""
        return filepath.read_text()

    def _parse_file(self, filepath: Path) -> Tree:
        """Parse a single file and store the results in a dictionary."""
        code = self._read_file(filepath)
        return self.parser.parse(bytes(code, "utf-8"))
