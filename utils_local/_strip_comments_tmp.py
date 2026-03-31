import ast
import io
import tokenize
import sys

TARGET = r"e:\instance segmentation\mlmp_testrealse\utils_local\logits_sam.py"


def strip_hash_comments(source: str) -> str:
    tokens = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                continue
            tokens.append(tok)
    except tokenize.TokenError:
        return source
    try:
        return tokenize.untokenize(tokens)
    except Exception:
        return source


class DocstringStripper(ast.NodeTransformer):
    def _body_without_docstring(self, body):
        if not body:
            return body
        first = body[0]
        if isinstance(first, ast.Expr):
            v = first.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                rest = body[1:]
                return rest if rest else [ast.Pass()]
        return body

    def visit_Module(self, node):
        self.generic_visit(node)
        node.body = self._body_without_docstring(node.body)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.body = self._body_without_docstring(node.body)
        if not node.body:
            node.body = [ast.Pass()]
        return node

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        node.body = self._body_without_docstring(node.body)
        if not node.body:
            node.body = [ast.Pass()]
        return node


def strip_docstrings(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    tree = DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def main():
    path = TARGET
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    out = strip_hash_comments(src)
    out = strip_docstrings(out)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    if not out.endswith("\n"):
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n")


if __name__ == "__main__":
    main()
