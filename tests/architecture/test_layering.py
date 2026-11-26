"""
Layering rule (CLAUDE.md rule 1, ADR 0001): modules import only from layers below them.

    cli > experiments > (paper, evaluation, training) > (execution, backtest, models)
        > (datasets, features) > data > core

Packages in the same layer may not import each other either: whatever two peers share
belongs in a lower layer (usually core/types). atu2 also may not import the legacy
top-level modules of the repository (orb_eval, alpaca_data, src, ...): code is ported,
not imported.
"""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

LAYERS: dict[str, int] = {
    "core": 0,
    "data": 1,
    "datasets": 2,
    "features": 2,
    "execution": 3,
    "backtest": 3,
    "models": 3,
    "paper": 4,
    "evaluation": 4,
    "training": 4,
    "experiments": 5,
    "cli": 6,
}
ROOT_PACKAGE = "atu2"


def legacy_modules(repo: Path) -> set[str]:
    """Top-level first-party modules and packages outside atu2/ (the legacy code)."""
    names = {p.stem for p in repo.glob("*.py")}
    names |= {p.parent.name for p in repo.glob("*/__init__.py")}
    return names - {ROOT_PACKAGE, "tests", "scripts", "setup", "conftest"}


def _module_name(path: Path, pkg_root: Path) -> str:
    rel = path.relative_to(pkg_root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _component(module: str) -> str | None:
    """'atu2.backtest.engine' -> 'backtest'; 'atu2.cli' -> 'cli'; 'atu2' -> None (package root)."""
    parts = module.split(".")
    if parts[0] != ROOT_PACKAGE or len(parts) == 1:
        return None
    return parts[1]


def _imports(path: Path, module: str, is_package: bool) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = module.split(".") if is_package else module.split(".")[:-1]
                base = base[: len(base) - (node.level - 1)]
                target = ".".join(base + ([node.module] if node.module else []))
                if node.module is None:  # from . import x  -> each x may be a module
                    out += [(node.lineno, f"{target}.{a.name}") for a in node.names]
                    continue
            else:
                target = node.module or ""
            if target == ROOT_PACKAGE:  # from atu2 import cli -> atu2.cli
                out += [(node.lineno, f"{target}.{a.name}") for a in node.names]
                continue
            out.append((node.lineno, target))
    return out


def find_violations(pkg_root: Path, legacy: set[str] | None = None) -> list[str]:
    """Returns one message per import that breaks the layering rule under `pkg_root`."""
    legacy = legacy or set()
    problems: list[str] = []
    for path in sorted(pkg_root.rglob("*.py")):
        module = _module_name(path, pkg_root)
        src = _component(module)
        for lineno, target in _imports(path, module, path.name == "__init__.py"):
            top = target.split(".")[0]
            where = f"{path.relative_to(pkg_root.parent)}:{lineno}"
            if top in legacy:
                problems.append(f"{where}: imports legacy module '{target}'")
                continue
            if top != ROOT_PACKAGE:
                continue
            dst = _component(target)
            if dst is None or dst.startswith("__"):  # atu2 itself or atu2.__version__
                problems.append(f"{where}: imports the package root '{target}'")
                continue
            if dst not in LAYERS:
                problems.append(f"{where}: '{dst}' is not a known layer (add it to LAYERS and ADR 0001)")
                continue
            if src is None:
                problems.append(f"{where}: the package root may not import '{target}'")
                continue
            if src == dst:
                continue
            if src not in LAYERS:
                problems.append(f"{where}: '{src}' is not a known layer")
            elif LAYERS[dst] >= LAYERS[src]:
                kind = "peer" if LAYERS[dst] == LAYERS[src] else "upward"
                problems.append(f"{where}: {kind} import {src} -> {dst} ('{target}')")
    return problems


def test_atu2_respects_layering():
    assert find_violations(REPO / ROOT_PACKAGE, legacy_modules(REPO)) == []


def test_every_atu2_subpackage_is_a_known_layer():
    found = {p.name for p in (REPO / ROOT_PACKAGE).iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    found |= {"cli"} if (REPO / ROOT_PACKAGE / "cli.py").exists() else set()
    assert found == set(LAYERS)


def test_legacy_modules_are_detected():
    legacy = legacy_modules(REPO)
    assert {"orb_eval", "alpaca_data", "src"} <= legacy
    assert ROOT_PACKAGE not in legacy


# --- negative tests: the checker must catch deliberate violations ---------------------
def _make_pkg(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / ROOT_PACKAGE
    for rel, text in {"__init__.py": "", **files}.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    for d in [p for p in root.rglob("*") if p.is_dir()]:
        (d / "__init__.py").touch()
    return root


@pytest.mark.parametrize(
    "files,expected",
    [
        ({"core/x.py": "from atu2.data import cache\n"}, "upward import core -> data"),
        ({"data/x.py": "import atu2.backtest.engine\n"}, "upward import data -> backtest"),
        ({"backtest/x.py": "from ..evaluation import metrics\n"}, "upward import backtest -> evaluation"),
        ({"backtest/x.py": "from atu2.models.losses import f\n"}, "peer import backtest -> models"),
        ({"training/x.py": "from atu2 import cli\n"}, "upward import training -> cli"),
        ({"training/x.py": "from atu2.evaluation.metrics import sharpe\n"}, "peer import training -> evaluation"),
        ({"features/x.py": "import orb_eval\n"}, "legacy module"),
        ({"core/x.py": "from atu2 import __version__\n"}, "package root"),
        ({"__init__.py": "from atu2.core import errors\n"}, "package root may not import"),
        ({"core/x.py": "from atu2.unknown import y\n"}, "not a known layer"),
    ],
)
def test_checker_catches_violations(tmp_path, files, expected):
    root = _make_pkg(tmp_path, files)
    problems = find_violations(root, {"orb_eval"})
    assert any(expected in p for p in problems), problems


def test_checker_allows_downward_and_internal_imports(tmp_path):
    root = _make_pkg(
        tmp_path,
        {
            "cli.py": "from atu2.experiments import runner\nimport atu2.core.config\n",
            "experiments/runner.py": "from atu2.evaluation.metrics import m\nfrom atu2.backtest import engine\n",
            "evaluation/metrics.py": "from atu2.backtest.engine import run\nfrom .bootstrap import ci\n",
            "evaluation/bootstrap.py": "import numpy as np\n",
            "backtest/engine.py": "from atu2.datasets.splits import s\nfrom atu2.core import types\n",
            "datasets/splits.py": "from atu2.data.universe import u\n",
            "data/universe.py": "from atu2.core.errors import DataError\nfrom . import cache\n",
        },
    )
    assert find_violations(root, {"orb_eval"}) == []
