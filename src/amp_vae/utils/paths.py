"""Path helpers that work from notebooks, scripts, and Streamlit."""

from __future__ import annotations

from pathlib import Path


def project_root(start: str | Path | None = None) -> Path:
    path = Path(start or __file__).resolve()
    if path.is_file():
        path = path.parent

    for candidate in [path, *path.parents]:
        if (candidate / "README.md").exists() and (candidate / "data").exists() and (candidate / "models").exists():
            return candidate

    return path


def repo_path(*parts: str | Path, start: str | Path | None = None) -> Path:
    return project_root(start=start).joinpath(*parts)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

