"""
AT: Verify pydub is completely removed from production code.
"""

import ast
import pathlib


def test_no_pydub_import_in_production_code():
    """No production module imports pydub."""
    src = pathlib.Path("speechlib")
    for py_file in src.glob("*.py"):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                assert not any("pydub" in n for n in names), (
                    f"pydub still imported in {py_file.name}"
                )
