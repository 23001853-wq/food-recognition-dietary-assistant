from pathlib import Path


def confined_path(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Path escapes its allowed root")
    return target
