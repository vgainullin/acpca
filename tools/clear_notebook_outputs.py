#!/usr/bin/env python3
"""Strip outputs and execution counts from Jupyter notebooks in-place."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _clear_cell_outputs(cell: dict) -> None:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def _clear_notebook(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as fh:
        nb = json.load(fh)

    if "cells" not in nb:
        return False

    for cell in nb["cells"]:
        _clear_cell_outputs(cell)

    nb.setdefault("metadata", {}).pop("widgets", None)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
        fh.write("\n")

    return True


def main(argv: list[str]) -> int:
    if len(argv) == 0:
        return 0

    failed = []
    for name in argv:
        path = Path(name)
        try:
            if not _clear_notebook(path):
                failed.append(name)
        except Exception as exc:  # pragma: no cover - defensive: surface failure
            print(f"[clear-notebook] failed to process {name}: {exc}", file=sys.stderr)
            failed.append(name)

    if failed:
        print(
            "[clear-notebook] unable to strip outputs from: " + ", ".join(failed),
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
