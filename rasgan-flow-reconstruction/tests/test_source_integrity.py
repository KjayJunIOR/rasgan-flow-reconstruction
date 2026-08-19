from __future__ import annotations

import ast
import compileall
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "rasgan"


def test_all_python_sources_compile():
    assert compileall.compile_dir(str(ROOT / "src"), quiet=1)
    assert compileall.compile_dir(str(ROOT / "scripts"), quiet=1)
    assert compileall.compile_dir(str(ROOT / "examples"), quiet=1)


def test_no_private_absolute_paths_in_python_source():
    forbidden = ("/home/", "/home-old/", "/scratch/", "/workspace/")
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path


def test_relative_import_targets_exist():
    """A small static check that package-relative imports resolve to modules/packages."""
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        module_parts = list(rel.with_suffix("").parts)
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level == 0:
                continue
            base = module_parts[:-1] if rel.name != "__init__.py" else module_parts
            if node.level > 1:
                base = base[: -(node.level - 1)]
            target_parts = base + ((node.module or "").split(".") if node.module else [])
            target_file = SRC.joinpath(*target_parts).with_suffix(".py")
            target_pkg = SRC.joinpath(*target_parts) / "__init__.py"
            assert target_file.exists() or target_pkg.exists(), f"{path}: unresolved {node.module}"


def test_pixelshuffle_icnr_reaches_the_wrapped_kernel():
    text = (SRC / "tf_layers.py").read_text(encoding="utf-8")
    assert "self.conv.conv.kernel" in text
    assert "self.conv.layer.kernel" not in text


def test_bce_helpers_pass_logits_explicitly():
    text = (SRC / "losses" / "gan.py").read_text(encoding="utf-8")
    assert "logits=r" in text
    assert text.count("logits=f") >= 2
