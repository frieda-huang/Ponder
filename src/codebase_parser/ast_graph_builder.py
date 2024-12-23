import os
from typing import Generator, List, Optional

import networkx as nx
from codebase_parser.schemas import Tag, TagKind, TagRole
from commons import PY_LANGUAGE, ASTMapping, FilepathType, project_paths
from loguru import logger
from tree_sitter import Tree


class ASTGraphBuilder:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def build_graph_from_ast(self, ast_tree: ASTMapping) -> nx.MultiDiGraph:
        tags = self._build_tags(ast_tree)
        return self.build_graph_from_tags(tags)

    def build_graph_from_tags(self, tags: List[Tag]) -> nx.MultiDiGraph:
        pass

    def _build_tags(self, ast_tree: ASTMapping) -> List[Tag]:
        """Build tags for all ASTs in the codebase."""
        all_tags: List[Tag] = []
        for filepath, tree in ast_tree.items():
            logger.info(f"Building tags for {filepath}")
            tags = list(self._build_tag(filepath, tree))
            all_tags.extend(tags)
        return all_tags

    def _get_role(self, capture_name: str) -> Optional[TagRole]:
        """Get the role of an entity in the code.

        role: definition or reference
        """
        capture_roles = {
            "name.definition.": TagRole.DEFINITION,
            "name.reference.": TagRole.REFERENCE,
        }
        return next(
            (
                role
                for prefix, role in capture_roles.items()
                if capture_name.startswith(prefix)
            ),
            None,
        )

    def _get_kind(self, capture_parts: List[str]) -> TagKind:
        """Return the kind of an entity in the code.

        kind: class, function, or call
        """
        return next(
            part
            for part in capture_parts
            if part in [TagKind.CLASS, TagKind.FUNCTION, TagKind.CALL]
        )

    def _build_tag(
        self, filepath: FilepathType, tree: Tree
    ) -> Generator[Tag, None, None]:
        """Build a tag based on a filepath and its corresponding AST."""
        query_string = self.read_query_file()
        query = PY_LANGUAGE.query(query_string)
        captures = query.captures(tree.root_node)

        for capture_name, nodes in captures.items():
            for node in nodes:
                role = self._get_role(capture_name)
                if role is None:
                    continue

                # Convert capture_name to hierarchical parts
                # e.g., "name.definition.class" -> ['name', 'definition', 'class']
                capture_parts = str(capture_name).split(".")
                kind = self._get_kind(capture_parts)
                rel_filename = os.path.relpath(filepath)
                name = node.text.decode("utf-8")

                logger.debug(name)

                yield Tag(
                    name=name,
                    kind=kind,
                    role=role,
                    start_line=node.start_point[0],
                    rel_filename=rel_filename,
                    abs_filename=filepath,
                )

    @staticmethod
    def read_query_file() -> str:
        """Load and parse a Tree-sitter query specifically for Python code."""
        query_path = project_paths.QUERIES / "tree-sitter-python-tags.scm"
        return query_path.read_text()
