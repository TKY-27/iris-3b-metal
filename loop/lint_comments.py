"""Comments of eight or more words added since a commit.

python loop/lint_comments.py <base-commit> [tree]
"""

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

from common import git

LIMIT = 8
HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@")
METAL_COMMENT = re.compile(r"//\s*(.+)")


def added_lines(tree: Path, base: str) -> dict[str, set[int]]:
    found, path = {}, None
    for line in git(tree, "diff", "-U0", base, "HEAD", "--", "*.py").splitlines():
        if line.startswith("+++ "):
            path = None if line == "+++ /dev/null" else line[6:]
        elif path and (match := HUNK.match(line)):
            start, count = int(match[1]), int(match[2] or 1)
            found.setdefault(path, set()).update(range(start, start + count))
    return found


def long_texts(source: str):
    """Yields (first line, last line, text) per comment."""
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            yield token.start[0], token.start[0], token.string.lstrip("#").strip()
        elif token.type == tokenize.STRING:
            for offset, text in enumerate(token.string.splitlines()):
                if match := METAL_COMMENT.search(text):
                    line = token.start[0] + offset
                    yield line, line, match[1].strip()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            doc = ast.get_docstring(node)
            if doc:
                first = node.body[0]
                yield first.lineno, first.end_lineno, " ".join(doc.split())


def flag(tree: Path, base: str) -> list[str]:
    flagged = []
    for path, lines in sorted(added_lines(tree, base).items()):
        source = (tree / path).read_text()
        for first, last, text in long_texts(source):
            words = len(text.split())
            if words >= LIMIT and lines & set(range(first, last + 1)):
                flagged.append(f"{path}:{first}: ({words} words) {text[:120]}")
    return flagged


if __name__ == "__main__":
    for item in flag(Path(sys.argv[2] if len(sys.argv) > 2 else ".").resolve(), sys.argv[1]):
        print(item)
