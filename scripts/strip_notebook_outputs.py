#!/usr/bin/env python3
"""Strip cell outputs and widget state from a Jupyter notebook (stdlib only).

Used as a git clean filter (see .gitattributes) so notebook results never
get committed, only code. Reads a notebook from stdin and writes the
stripped notebook to stdout; also usable directly on a file path.
"""
import json
import sys


def strip(nb: dict) -> dict:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            cell.get("metadata", {}).pop("widgets", None)
    nb.get("metadata", {}).pop("widgets", None)
    return nb


def main() -> None:
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r+") as f:
            nb = json.load(f)
            nb = strip(nb)
            f.seek(0)
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
            f.truncate()
    else:
        nb = json.load(sys.stdin)
        nb = strip(nb)
        json.dump(nb, sys.stdout, indent=1, ensure_ascii=False)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
