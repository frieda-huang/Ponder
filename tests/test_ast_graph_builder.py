import pytest
from codebase_parser.ast_graph_builder import ASTGraphBuilder
from commons import parser
from tree_sitter import Tree


@pytest.fixture
def tree() -> Tree:
    code_string = """
    class Calculator:
        def __init__(self, numbers):
            self.numbers = numbers
        
        def sum(self):
            return sum(self.numbers)
        
        def multiply(self):
            result = 1
            for n in self.numbers:
                result *= n
            return result

    calc = Calculator([1, 2, 3, 4, 5])
    print(calc.sum(), calc.multiply())
    """

    return parser.parse(bytes(code_string, "utf8"))


def test_build_tag(tree):
    ast_graph_builder = ASTGraphBuilder()
    filepath = "/Users/friedahuang/Documents/Ponder/src/codebase_parser/parser.py"
    tags = list(ast_graph_builder._build_tag(filepath, tree))
    names = [tag.name for tag in tags]

    assert tags[0].abs_filename == filepath
    assert tags[0].rel_filename == "src/codebase_parser/parser.py"
    assert len(set(tag.abs_filename for tag in tags)) == 1
    assert len(names) == 9
