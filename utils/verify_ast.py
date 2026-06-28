"""
Post-Backprop 4B — AST Verification
=====================================
Static analysis to prove that NO autograd primitives are used anywhere
in the training codebase.

Scans all .py files (excluding this file and data preparation scripts)
for forbidden patterns:
  - .backward()
  - requires_grad
  - torch.autograd
  - .grad attribute access
  - retain_graph
  - create_graph
  - grad_fn
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple


# Forbidden attribute names (accessed on any object)
FORBIDDEN_ATTRS = {
    "backward",
    "grad",
    "grad_fn",
    "requires_grad",
    "requires_grad_",
    "retain_graph",
    "create_graph",
}

# Forbidden module-level references
FORBIDDEN_MODULES = {
    "autograd",
}

# Files / directories to skip
SKIP_PATTERNS = {
    "verify_ast.py",       # this file
    "__pycache__",
    ".git",
    "cache",
    "checkpoints",
}


class AutogradViolation:
    """Record of a single autograd usage violation."""
    def __init__(self, filepath: str, lineno: int, col: int,
                 kind: str, detail: str):
        self.filepath = filepath
        self.lineno = lineno
        self.col = col
        self.kind = kind
        self.detail = detail

    def __str__(self):
        return (f"  {self.filepath}:{self.lineno}:{self.col} "
                f"[{self.kind}] {self.detail}")


def check_file(filepath: str) -> List[AutogradViolation]:
    """Scan a single Python file for autograd violations."""
    violations = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, UnicodeDecodeError) as e:
        violations.append(AutogradViolation(
            filepath, 0, 0, "PARSE_ERROR", str(e)))
        return violations

    for node in ast.walk(tree):
        # Check attribute access:  x.backward, x.grad, etc.
        if isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                violations.append(AutogradViolation(
                    filepath, node.lineno, node.col_offset,
                    "FORBIDDEN_ATTR",
                    f"Access to '.{node.attr}'"
                ))

        # Check names: 'autograd' as a standalone name
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_MODULES:
                violations.append(AutogradViolation(
                    filepath, node.lineno, node.col_offset,
                    "FORBIDDEN_MODULE",
                    f"Reference to '{node.id}'"
                ))

        # Check imports: from torch import autograd / import torch.autograd
        if isinstance(node, ast.ImportFrom):
            if node.module and "autograd" in node.module:
                violations.append(AutogradViolation(
                    filepath, node.lineno, node.col_offset,
                    "FORBIDDEN_IMPORT",
                    f"Import from '{node.module}'"
                ))
            if node.names:
                for alias in node.names:
                    if alias.name in FORBIDDEN_MODULES:
                        violations.append(AutogradViolation(
                            filepath, node.lineno, node.col_offset,
                            "FORBIDDEN_IMPORT",
                            f"Import of '{alias.name}'"
                        ))

        if isinstance(node, ast.Import):
            for alias in node.names:
                if "autograd" in alias.name:
                    violations.append(AutogradViolation(
                        filepath, node.lineno, node.col_offset,
                        "FORBIDDEN_IMPORT",
                        f"Import of '{alias.name}'"
                    ))

        # Check string literals mentioning 'requires_grad=True'
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "requires_grad=True" in node.value:
                violations.append(AutogradViolation(
                    filepath, node.lineno, node.col_offset,
                    "SUSPICIOUS_STRING",
                    "String contains 'requires_grad=True'"
                ))

    return violations


def scan_directory(root: str) -> Tuple[int, List[AutogradViolation]]:
    """Recursively scan a directory for autograd violations.

    Returns:
        (n_files_scanned, list_of_violations)
    """
    all_violations = []
    n_files = 0
    root_path = Path(root)

    for py_file in root_path.rglob("*.py"):
        # Skip excluded patterns
        parts = py_file.parts
        if any(skip in parts for skip in SKIP_PATTERNS):
            continue
        if py_file.name in SKIP_PATTERNS:
            continue

        n_files += 1
        violations = check_file(str(py_file))
        all_violations.extend(violations)

    return n_files, all_violations


def main():
    """Entry point: scan the project and report violations."""
    source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("Post-Backprop 4B — Autograd Verification")
    print("=" * 60)
    print(f"Scanning: {source_dir}")
    print()

    n_files, violations = scan_directory(source_dir)

    print(f"Files scanned: {n_files}")
    print(f"Violations found: {len(violations)}")
    print()

    if violations:
        print("VIOLATIONS:")
        for v in violations:
            print(v)
        print()
        print("RESULT: ❌ FAILED — autograd primitives detected")
        sys.exit(1)
    else:
        print("RESULT: ✅ PASSED — zero autograd usage confirmed")
        sys.exit(0)


if __name__ == "__main__":
    main()
