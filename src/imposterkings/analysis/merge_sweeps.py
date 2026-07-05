"""Merge budget_scaling run directories into combined CSVs, de-duped, without clobbering the source.

    python -m imposterkings.analysis.merge_sweeps results/budget_scaling results/budget_scaling/ext_k70k100 \
        --out-dir results/budget_scaling --prefix merged

Reads the three base tables (``winrate.csv``, ``deal_outcomes.csv``, ``evals.csv``) from each input
dir, concatenates, and drops duplicate rows by a natural key (so re-running or overlapping specs --
e.g. the hybrid-k50 baseline appearing in two runs -- collapse to one). Writes ``<prefix>_winrate.csv``
etc. to ``--out-dir``. It only ever READS the base filenames and only WRITES the prefixed ones, so it
can point at the same directory as a source without eating its own output or overwriting raw data.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Callable, Dict, List

# (base filename, sort key, dedup key) for each table.
_TABLES = {
    "winrate.csv": (lambda r: (r["baseline"], float(r["k"])),
                    lambda r: (r["challenger"], r["baseline"])),
    "deal_outcomes.csv": (lambda r: (r["baseline"], float(r["k"]), int(r["seed"])),
                          lambda r: (r["challenger"], r["baseline"], r["seed"])),
    "evals.csv": (lambda r: (r["spec"], int(r["seed"])),
                  lambda r: (r["spec"], r["seed"])),
}


def _read(path: str):
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


def merge_table(inputs: List[str], base: str, sort_key: Callable, dedup_key: Callable):
    """Return (header, deduped+sorted rows, n_dupes) across every input dir that has ``base``."""
    header, seen, rows, dupes = None, {}, [], 0
    for d in inputs:
        p = os.path.join(d, base)
        if not os.path.exists(p):
            continue
        cols, part = _read(p)
        header = header or cols
        for r in part:
            key = dedup_key(r)
            if key in seen:
                dupes += 1
                continue
            seen[key] = True
            rows.append(r)
    rows.sort(key=sort_key)
    return header, rows, dupes


def _write(path: str, header: List[str], rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Merge budget_scaling run dirs (de-duped, non-clobbering).")
    p.add_argument("inputs", nargs="+", help="run directories to merge")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", default="merged")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    for base, (sort_key, dedup_key) in _TABLES.items():
        header, rows, dupes = merge_table(args.inputs, base, sort_key, dedup_key)
        if not rows:
            print(f"  {base}: (none found)")
            continue
        out = os.path.join(args.out_dir, f"{args.prefix}_{base}")
        assert os.path.abspath(out) != os.path.abspath(os.path.join(args.out_dir, base)), \
            "refusing to overwrite a base table"
        _write(out, header, rows)
        print(f"  {base}: {len(rows)} rows ({dupes} dupes dropped) -> {out}")


if __name__ == "__main__":
    main()
