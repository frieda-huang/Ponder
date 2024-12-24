import os
from collections import defaultdict
from typing import Generator, List, Optional, Tuple

import networkx as nx
from codebase_parser.schemas import Define, Reference, Tag, TagKind, TagRole
from commons import PY_LANGUAGE, ASTMapping, FilepathType, project_paths
from loguru import logger
from tree_sitter import Tree


class ASTGraphBuilder:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.tags: List[Tag] = []

    def build_graph_from_ast(self, ast_tree: ASTMapping) -> nx.MultiDiGraph:
        self._build_tags(ast_tree)
        return self.build_graph_from_tags()

    def build_graph_from_tags(self) -> nx.MultiDiGraph:
        defines, references = self._process_tags()

    def _build_defines(self) -> List[Define]:
        """Build defines from the list of tags."""
        return [
            Define(identifier=tag.name, filepath=tag.rel_filename)
            for tag in self.tags
            if tag.role == TagRole.DEFINITION
        ]

    def _build_references(self) -> List[Reference]:
        """
        Build references from the list of tags.

        Example:
            >>> references = _build_references(tags)
            >>> print(references)
            [Reference(identifier='name', filepaths=['file1.py', 'file2.py'])]
        """
        references_tracker = defaultdict(list)

        for tag in (t for t in self.tags if t.role == TagRole.REFERENCE):
            references_tracker[tag.name].append(tag.rel_filename)

        return [
            Reference(identifier=identifier, filepaths=filepaths)
            for identifier, filepaths in references_tracker.items()
        ]

    def _process_tags(self) -> Tuple[List[Define], List[Reference]]:
        """Separate defines and references from the list of tags."""
        return self._build_defines(), self._build_references()

    def _build_tags(self, ast_tree: ASTMapping) -> List[Tag]:
        """Build tags for all ASTs in the codebase."""
        for filepath, tree in ast_tree.items():
            logger.info(f"Building tags for {filepath}")
            tags = list(self._build_tag(filepath, tree))
            self.tags.extend(tags)
        return self.tags

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
