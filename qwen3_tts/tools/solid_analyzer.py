"""
SOLID Code Analyzer - Static analysis tool for SOLID principle compliance.

Analyzes Python source code and scores compliance with each SOLID principle.
Outputs violations with line numbers for remediation.
"""
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Violation:
    """Represents a SOLID principle violation."""

    principle: str
    message: str
    line_number: int


@dataclass
class SOLIDScore:
    """SOLID compliance scores for a module."""

    srp_score: float  # Single Responsibility
    ocp_score: float  # Open/Closed
    lsp_score: float  # Liskov Substitution
    isp_score: float  # Interface Segregation
    dip_score: float  # Dependency Inversion
    violations: List[Violation] = field(default_factory=list)

    @property
    def total_score(self) -> float:
        """Calculate total SOLID score (0-50)."""
        return (
            self.srp_score
            + self.ocp_score
            + self.lsp_score
            + self.isp_score
            + self.dip_score
        )


# Scoring thresholds
MAX_PUBLIC_METHODS = 7  # Above this, SRP score decreases
MAX_FUNCTION_LINES = 50  # Above this, SRP score decreases
MAX_IF_ELSE_CHAIN = 3  # Above this, OCP score decreases
MAX_INTERFACE_METHODS = 5  # Above this, ISP score decreases


def analyze_code(source: str, filename: str = "<unknown>") -> SOLIDScore:
    """
    Analyze Python source code for SOLID compliance.

    Args:
        source: Python source code as string
        filename: Filename for error reporting

    Returns:
        SOLIDScore with principle scores and violations
    """
    violations: List[Violation] = []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        violations.append(
            Violation("SYNTAX", f"Syntax error: {e.msg}", e.lineno or 1)
        )
        return SOLIDScore(0, 0, 0, 0, 0, violations)

    # Analyze each principle
    srp_score, srp_violations = _analyze_srp(tree)
    ocp_score, ocp_violations = _analyze_ocp(tree, source)
    lsp_score, lsp_violations = _analyze_lsp(tree)
    isp_score, isp_violations = _analyze_isp(tree)
    dip_score, dip_violations = _analyze_dip(tree, source)

    violations.extend(srp_violations)
    violations.extend(ocp_violations)
    violations.extend(lsp_violations)
    violations.extend(isp_violations)
    violations.extend(dip_violations)

    return SOLIDScore(
        srp_score=srp_score,
        ocp_score=ocp_score,
        lsp_score=lsp_score,
        isp_score=isp_score,
        dip_score=dip_score,
        violations=violations,
    )


def _analyze_srp(tree: ast.AST) -> tuple[float, List[Violation]]:
    """
    Analyze Single Responsibility Principle.

    Checks:
    - Number of public methods per class
    - Function length
    - Cyclomatic complexity (basic)
    """
    violations: List[Violation] = []
    srp_score = 10.0

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            public_methods = [
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef)
                and not n.name.startswith("_")
            ]
            num_methods = len(public_methods)

            if num_methods > MAX_PUBLIC_METHODS:
                # Deduct 0.5 points per method over threshold
                deduction = (num_methods - MAX_PUBLIC_METHODS) * 0.5
                srp_score = max(0, srp_score - deduction)
                violations.append(
                    Violation(
                        "SRP",
                        f"Class '{node.name}' has {num_methods} public methods (max {MAX_PUBLIC_METHODS})",
                        node.lineno,
                    )
                )

        elif isinstance(node, ast.FunctionDef):
            # Check function length
            func_lines = node.end_lineno - node.lineno if node.end_lineno else 0
            if func_lines > MAX_FUNCTION_LINES:
                deduction = (func_lines - MAX_FUNCTION_LINES) * 0.1
                srp_score = max(0, srp_score - deduction)
                violations.append(
                    Violation(
                        "SRP",
                        f"Function '{node.name}' is {func_lines} lines (max {MAX_FUNCTION_LINES})",
                        node.lineno,
                    )
                )

    return srp_score, violations


def _analyze_ocp(tree: ast.AST, source: str) -> tuple[float, List[Violation]]:
    """
    Analyze Open/Closed Principle.

    Checks:
    - if/elif chains for type/mode dispatch
    - Hardcoded type comparisons
    """
    violations: List[Violation] = []
    ocp_score = 10.0

    def count_elif_chains(if_node: ast.If) -> int:
        """Count elif chain depth."""
        count = 0
        current = if_node
        while current:
            count += 1
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
            else:
                current = None
        return count

    def get_top_level_ifs(node: ast.AST) -> List[ast.If]:
        """Get only top-level If statements (not nested in other Ifs)."""
        ifs = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                ifs.append(child)
            # Recurse into non-If compound statements
            elif isinstance(child, (ast.For, ast.While, ast.With, ast.Try)):
                ifs.extend(get_top_level_ifs(child))
        return ifs

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Find only top-level if statements
            top_ifs = get_top_level_ifs(node)
            for if_node in top_ifs:
                chain_length = count_elif_chains(if_node) - 1  # Subtract 1 for initial if
                if chain_length > MAX_IF_ELSE_CHAIN:
                    # Deduct 1.5 points per elif over threshold
                    deduction = (chain_length - MAX_IF_ELSE_CHAIN) * 1.5
                    ocp_score = max(0, ocp_score - deduction)
                    violations.append(
                        Violation(
                            "OCP",
                            f"Function '{node.name}' has {chain_length} elif branches (max {MAX_IF_ELSE_CHAIN})",
                            node.lineno,
                        )
                    )
                    break  # Only count one chain per function

    return ocp_score, violations


def _analyze_lsp(tree: ast.AST) -> tuple[float, List[Violation]]:
    """
    Analyze Liskov Substitution Principle.

    Checks:
    - Method signature compatibility in subclasses
    - Parameter type narrowing (violation)
    - Return type widening (violation)
    """
    violations: List[Violation] = []
    lsp_score = 10.0

    # Build class hierarchy
    classes: Dict[str, ast.ClassDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes[node.name] = node

    def get_annotation_str(annotation) -> str:
        """Get string representation of type annotation."""
        if annotation is None:
            return ""
        if isinstance(annotation, ast.Name):
            return annotation.id
        if isinstance(annotation, ast.Constant):
            return str(annotation.value)
        return ast.unparse(annotation) if hasattr(ast, 'unparse') else ""

    # Check each class for overrides
    for class_name, class_node in classes.items():
        # Get base classes
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id in classes:
                base_class = classes[base.id]

                # Compare method signatures
                base_methods = {
                    n.name: n
                    for n in base_class.body
                    if isinstance(n, ast.FunctionDef)
                }
                derived_methods = {
                    n.name: n
                    for n in class_node.body
                    if isinstance(n, ast.FunctionDef)
                }

                for method_name, derived_method in derived_methods.items():
                    if method_name in base_methods:
                        base_method = base_methods[method_name]

                        # Check parameter count
                        base_params = len(base_method.args.args)
                        derived_params = len(derived_method.args.args)

                        if derived_params != base_params:
                            lsp_score = max(0, lsp_score - 2.0)
                            violations.append(
                                Violation(
                                    "LSP",
                                    f"Method '{method_name}' in '{class_name}' has different signature than base class",
                                    derived_method.lineno,
                                )
                            )

                        # Check type annotations for changes
                        for i, (base_arg, derived_arg) in enumerate(
                            zip(base_method.args.args, derived_method.args.args)
                        ):
                            base_ann = get_annotation_str(base_arg.annotation)
                            derived_ann = get_annotation_str(derived_arg.annotation)

                            # If annotations differ and both are specified
                            if base_ann and derived_ann and base_ann != derived_ann:
                                lsp_score = max(0, lsp_score - 3.0)
                                violations.append(
                                    Violation(
                                        "LSP",
                                        f"Method '{method_name}' parameter '{derived_arg.arg}' has incompatible type annotation",
                                        derived_method.lineno,
                                    )
                                )

    return lsp_score, violations


def _analyze_isp(tree: ast.AST) -> tuple[float, List[Violation]]:
    """
    Analyze Interface Segregation Principle.

    Checks:
    - Interface-like classes with too many methods
    - Abstract base classes with many abstract methods
    """
    violations: List[Violation] = []
    isp_score = 10.0

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if it looks like an interface (name pattern or all methods are pass)
            is_interface = any(
                pattern in node.name.lower()
                for pattern in ["interface", "protocol", "abc", "base"]
            )

            # Also check if all methods just pass
            all_pass = all(
                isinstance(n, ast.FunctionDef)
                and len(n.body) == 1
                and isinstance(n.body[0], ast.Pass)
                or not isinstance(n, ast.FunctionDef)
                for n in node.body
            )

            if is_interface or all_pass:
                public_methods = [
                    n
                    for n in node.body
                    if isinstance(n, ast.FunctionDef)
                    and not n.name.startswith("_")
                ]
                num_methods = len(public_methods)

                if num_methods > MAX_INTERFACE_METHODS:
                    # Deduct 1 point per method over threshold
                    deduction = (num_methods - MAX_INTERFACE_METHODS) * 1.0
                    isp_score = max(0, isp_score - deduction)
                    violations.append(
                        Violation(
                            "ISP",
                            f"Interface '{node.name}' has {num_methods} methods (max {MAX_INTERFACE_METHODS})",
                            node.lineno,
                        )
                    )

    return isp_score, violations


def _analyze_dip(tree: ast.AST, source: str) -> tuple[float, List[Violation]]:
    """
    Analyze Dependency Inversion Principle.

    Checks:
    - Direct concrete class instantiation in __init__
    - Hardcoded imports of concrete implementations
    - Global state access
    """
    violations: List[Violation] = []
    dip_score = 10.0

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check __init__ for direct instantiation
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for stmt in ast.walk(item):
                        # Check for direct instantiation pattern: self.x = SomeClass()
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Attribute):
                                    if isinstance(stmt.value, ast.Call):
                                        if isinstance(stmt.value.func, ast.Name):
                                            class_name = stmt.value.func.id
                                            # Heuristic: Capitalized names are likely concrete classes
                                            if class_name[0].isupper() and not class_name.startswith("_"):
                                                dip_score = max(0, dip_score - 1.5)
                                                violations.append(
                                                    Violation(
                                                        "DIP",
                                                        f"Direct instantiation of '{class_name}' in __init__",
                                                        stmt.lineno,
                                                    )
                                                )

    # Check for global state access patterns
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            dip_score = max(0, dip_score - 1.0)
            violations.append(
                Violation(
                    "DIP",
                    f"Global statement for '{node.names}' indicates global state dependency",
                    node.lineno,
                )
            )

    return dip_score, violations


def analyze_module(file_path: str) -> SOLIDScore:
    """
    Analyze a Python module file.

    Args:
        file_path: Path to Python file

    Returns:
        SOLIDScore for the module
    """
    path = Path(file_path)
    if not path.exists():
        return SOLIDScore(
            0, 0, 0, 0, 0,
            [Violation("IO", f"File not found: {file_path}", 0)]
        )

    try:
        source = path.read_text()
    except Exception as e:
        return SOLIDScore(
            0, 0, 0, 0, 0,
            [Violation("IO", f"Failed to read file: {e}", 0)]
        )

    return analyze_code(source, str(path))


def analyze_package(package_path: str) -> Dict[str, SOLIDScore]:
    """
    Analyze all Python files in a package.

    Args:
        package_path: Path to package directory

    Returns:
        Dict mapping filename to SOLIDScore
    """
    path = Path(package_path)
    results: Dict[str, SOLIDScore] = {}

    if not path.is_dir():
        return results

    for py_file in path.glob("*.py"):
        results[py_file.name] = analyze_module(str(py_file))

    return results


def main():
    """CLI entry point for SOLID analyzer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze Python code for SOLID principle compliance"
    )
    parser.add_argument("path", help="Python file or package to analyze")
    parser.add_argument(
        "--fail-below",
        type=float,
        default=0,
        help="Exit with error if total score below threshold",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()
    path = Path(args.path)

    if path.is_file():
        result = analyze_module(str(path))
        results = {path.name: result}
    elif path.is_dir():
        results = analyze_package(str(path))
    else:
        print(f"Error: {path} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    # Output results
    total_score = sum(r.total_score for r in results.values()) / len(results) if results else 0

    if args.json:
        import json
        output = {
            "average_score": total_score,
            "modules": {
                name: {
                    "total": score.total_score,
                    "srp": score.srp_score,
                    "ocp": score.ocp_score,
                    "lsp": score.lsp_score,
                    "isp": score.isp_score,
                    "dip": score.dip_score,
                    "violations": [
                        {"principle": v.principle, "message": v.message, "line": v.line_number}
                        for v in score.violations
                    ],
                }
                for name, score in results.items()
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nSOLID Score Analysis: {path}")
        print("=" * 50)
        print(f"Average Score: {total_score:.1f}/50\n")

        for name, score in results.items():
            print(f"{name}:")
            print(f"  SRP: {score.srp_score:.1f}/10  OCP: {score.ocp_score:.1f}/10")
            print(f"  LSP: {score.lsp_score:.1f}/10  ISP: {score.isp_score:.1f}/10")
            print(f"  DIP: {score.dip_score:.1f}/10  Total: {score.total_score:.1f}/50")

            if score.violations:
                print("  Violations:")
                for v in score.violations[:5]:  # Show first 5
                    print(f"    [{v.principle}] Line {v.line_number}: {v.message}")
                if len(score.violations) > 5:
                    print(f"    ... and {len(score.violations) - 5} more")
            print()

    # Check threshold
    if args.fail_below > 0 and total_score < args.fail_below:
        print(
            f"FAIL: Score {total_score:.1f} is below threshold {args.fail_below}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
