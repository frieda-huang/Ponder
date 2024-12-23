import networkx as nx


class ASTGraphBuilder:
    def __init__(self) -> None:
        self.tags = []

    def build_tags(self, ast_tree: dict) -> list:
        for filepath, parsed_tree in list(ast_tree.items()):
            captures = query.captures(parsed_tree.root_node)
            for node, tag in captures.items():
                for t in tag:
                    capture_parts = str(node).split(".")

                    role = (
                        TagRole.DEFINITION
                        if "definition" in capture_parts
                        else TagRole.REFERENCE
                    )
                    kind = next(
                        (
                            part
                            for part in capture_parts
                            if part in ["class", "function", "call"]
                        )
                    )

                    rel_filename = os.path.relpath(filepath)
                    tag = Tag(
                        name=t.text.decode("utf-8"),
                        kind=kind,
                        role=role,
                        line=t.start_point[0],
                        rel_filename=rel_filename,
                        abs_filename=filepath,
                    )
                    self.tags.append(tag)

    def build_graph_from_ast(self, ast_tree: dict) -> nx.MultiDiGraph:
        self.build_tags(ast_tree)
