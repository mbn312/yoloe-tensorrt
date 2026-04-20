from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "yoloe_tensorrt"
NATIVE_ROOT = REPO_ROOT / "src" / "native"

OWNED_MODULES = (
    "yoloe_tensorrt.engine",
    "yoloe_tensorrt.prompts",
    "yoloe_tensorrt.preprocess",
    "yoloe_tensorrt.gstreamer",
    "yoloe_tensorrt.native_backend",
    "yoloe_tensorrt.tracking",
)


class _ImportCollector(ast.NodeVisitor):
    def __init__(self, module_name: str, module_names: set[str]) -> None:
        self._module_names = module_names
        self._package_parts = module_name.split(".")
        self._type_checking_depth = 0
        self._deferred_runtime_depth = 0
        self.runtime_imports: set[str] = set()
        self.all_imports: set[str] = set()

    def visit_Module(self, node: ast.Module) -> None:
        for statement in node.body:
            self.visit(statement)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_deferred_runtime_body(node.body)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_deferred_runtime_body(node.body)

    def _visit_deferred_runtime_body(self, body: list[ast.stmt]) -> None:
        self._deferred_runtime_depth += 1
        for statement in body:
            self.visit(statement)
        self._deferred_runtime_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        is_type_checking = _is_type_checking_guard(node.test)
        if is_type_checking:
            self._type_checking_depth += 1
            for statement in node.body:
                self.visit(statement)
            self._type_checking_depth -= 1
            for statement in node.orelse:
                self.visit(statement)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            base = self._package_parts[: -node.level]
            target = ".".join(base + (node.module.split(".") if node.module else []))
        else:
            if not node.module:
                return
            target = node.module
        self._record(target)
        for alias in node.names:
            self._record(f"{target}.{alias.name}" if target else alias.name)

    def _record(self, target: str) -> None:
        candidate = target
        while candidate:
            if candidate in self._module_names:
                self.all_imports.add(candidate)
                if self._type_checking_depth == 0 and self._deferred_runtime_depth == 0:
                    self.runtime_imports.add(candidate)
                return
            if "." not in candidate:
                return
            candidate = candidate.rsplit(".", 1)[0]


def _is_type_checking_guard(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "TYPE_CHECKING"
    return isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING" and isinstance(node.value, ast.Name)


def _module_paths() -> dict[str, Path]:
    return {".".join(path.relative_to(REPO_ROOT).with_suffix("").parts): path for path in PACKAGE_ROOT.rglob("*.py")}


def _collect_imports() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    module_paths = _module_paths()
    runtime_imports: dict[str, set[str]] = {}
    all_imports: dict[str, set[str]] = {}
    for module_name, path in module_paths.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        collector = _ImportCollector(module_name, set(module_paths))
        collector.visit(tree)
        runtime_imports[module_name] = collector.runtime_imports
        all_imports[module_name] = collector.all_imports
    return runtime_imports, all_imports


def _native_include_graph() -> dict[str, set[str]]:
    include_pattern = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
    paths = sorted(NATIVE_ROOT.glob("*.cpp")) + sorted(NATIVE_ROOT.glob("*.h"))
    path_by_name = {path.name: path for path in paths}
    graph: dict[str, set[str]] = {}

    for path in paths:
        graph[path.name] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            match = include_pattern.match(line)
            if not match:
                continue
            include_name = Path(match.group(1)).name
            if include_name in path_by_name:
                graph[path.name].add(include_name)

    return graph


def _strongly_connected_components(graph: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbor in graph.get(node, ()):
            if neighbor not in indices:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])
        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                current = stack.pop()
                on_stack.remove(current)
                component.append(current)
                if current == node:
                    break
            if len(component) > 1:
                components.append(sorted(component))

    for node in graph:
        if node not in indices:
            visit(node)
    return sorted(components)


def test_package_modules_have_no_static_cycles() -> None:
    _, all_imports = _collect_imports()

    assert _strongly_connected_components(all_imports) == []


def test_owned_modules_have_no_static_cycles() -> None:
    _, all_imports = _collect_imports()
    graph = {
        module_name: {dep for dep in all_imports[module_name] if dep in OWNED_MODULES} for module_name in OWNED_MODULES
    }

    assert _strongly_connected_components(graph) == []


def test_native_sources_have_no_local_include_cycles() -> None:
    assert _strongly_connected_components(_native_include_graph()) == []


def test_engine_defers_tracking_export_and_prompt_dependencies() -> None:
    runtime_imports, _ = _collect_imports()

    assert "yoloe_tensorrt.tracking" not in runtime_imports["yoloe_tensorrt.engine"]
    assert "yoloe_tensorrt.export" not in runtime_imports["yoloe_tensorrt.engine"]
    assert "yoloe_tensorrt.prompts" not in runtime_imports["yoloe_tensorrt.engine"]


def test_tracking_avoids_eager_engine_and_inputs_imports() -> None:
    runtime_imports, _ = _collect_imports()

    assert "yoloe_tensorrt.engine" not in runtime_imports["yoloe_tensorrt.tracking"]
    assert "yoloe_tensorrt.inputs" not in runtime_imports["yoloe_tensorrt.tracking"]
