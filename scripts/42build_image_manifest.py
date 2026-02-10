#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import pandas as pd


DATA_REL = Path("data") / "kelly_auranofin"
OUT_REL = Path("results") / "01_manifest"


@dataclass(frozen=True)
class ParsedName:
    filename: str
    stem: str
    genotype: str
    time_h: int
    treatment: str          # CTRL | AURA
    concentration_uM: float # 0.0, 0.5, 1.0, 2.0


_TIME_RE = re.compile(r"(\d+)\s*h", flags=re.IGNORECASE)
_CONC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*u\s*m", flags=re.IGNORECASE)


def parse_kelly_filename(name: str) -> ParsedName:
    """
    Expected examples:
      '01) Kelly ctrl 2h.jpg'
      '02) Kelly Aura 0.5uM 2h.jpg'
      '12) Kelly Aura 2uM 24h.jpg'
    """
    p = Path(name)
    stem = p.stem

    low = stem.lower()

    # Genotype фиксирован под ваш набор (Kelly).
    genotype = "KELLY"

    # Time
    tm = _TIME_RE.search(low)
    if not tm:
        raise ValueError(f"Cannot parse time like '2h/6h/24h' from: {name}")
    time_h = int(tm.group(1))

    # Treatment + concentration
    if "aura" in low:
        treatment = "AURA"
        cm = _CONC_RE.search(low.replace(",", "."))
        if not cm:
            raise ValueError(f"Cannot parse concentration like '0.5uM/1uM/2uM' from: {name}")
        concentration_uM = float(cm.group(1))
    else:
        # ctrl
        treatment = "CTRL"
        concentration_uM = 0.0

    return ParsedName(
        filename=p.name,
        stem=stem,
        genotype=genotype,
        time_h=time_h,
        treatment=treatment,
        concentration_uM=concentration_uM,
    )


def expected_conditions() -> List[Tuple[int, str, float]]:
    times = [2, 6, 24]
    concs = [0.0, 0.5, 1.0, 2.0]
    out = []
    for t in times:
        for c in concs:
            tr = "CTRL" if c == 0.0 else "AURA"
            out.append((t, tr, c))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="Project root (default: auto from scripts/)")
    ap.add_argument("--data_dir", type=str, default=None, help="Override data dir (default: <root>/data/kelly_auranofin)")
    ap.add_argument("--out_dir", type=str, default=None, help="Override out dir (default: <root>/results/01_manifest)")
    ap.add_argument("--strict", type=int, default=1, help="1: require all 12 expected conditions, 0: allow partial")
    args = ap.parse_args()

    # root autodetect: .../Analysis_of_phases.../scripts/01_build_image_manifest.py -> project root
    if args.root:
        root = Path(args.root).resolve()
    else:
        root = Path(__file__).resolve().parents[1]

    data_dir = Path(args.data_dir).resolve() if args.data_dir else (root / DATA_REL)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (root / OUT_REL)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load images
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    images = sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    if not images:
        raise FileNotFoundError(f"No images found in: {data_dir}")

    rows: List[Dict] = []
    errors: List[str] = []

    for p in images:
        try:
            parsed = parse_kelly_filename(p.name)
            row = asdict(parsed)
            row["abs_path"] = str(p.resolve())
            row["rel_path"] = str(p.resolve().relative_to(root)) if p.resolve().is_relative_to(root) else str(p.resolve())
            rows.append(row)
        except Exception as e:
            errors.append(f"{p.name}: {e}")

    df = pd.DataFrame(rows)

    # Save parse errors
    err_path = out_dir / "parse_errors.txt"
    err_path.write_text("\n".join(errors), encoding="utf-8")

    if df.empty:
        raise RuntimeError(f"All files failed to parse. See: {err_path}")

    # Basic sanity checks
    df["time_h"] = df["time_h"].astype(int)
    df["concentration_uM"] = df["concentration_uM"].astype(float)
    df["treatment"] = df["treatment"].astype(str).str.upper()
    df["genotype"] = df["genotype"].astype(str).str.upper()

    # Uniqueness: one image per condition
    key_cols = ["genotype", "time_h", "treatment", "concentration_uM"]
    dup = df[df.duplicated(subset=key_cols, keep=False)].sort_values(key_cols)
    dup_path = out_dir / "duplicates.csv"
    dup.to_csv(dup_path, index=False)

    if len(dup) > 0:
        raise RuntimeError(
            "Found duplicate condition keys (same genotype/time/treatment/concentration). "
            f"See: {dup_path}"
        )

    # Strict check: must match 12 expected conditions
    if int(args.strict) == 1:
        exp = expected_conditions()
        got = sorted([(int(r.time_h), str(r.treatment), float(r.concentration_uM)) for r in df.itertuples(index=False)])
        missing = [x for x in exp if x not in got]
        extra = [x for x in got if x not in exp]

        miss_path = out_dir / "missing_conditions.txt"
        extra_path = out_dir / "extra_conditions.txt"
        miss_path.write_text("\n".join(map(str, missing)), encoding="utf-8")
        extra_path.write_text("\n".join(map(str, extra)), encoding="utf-8")

        if missing or extra:
            raise RuntimeError(
                "Condition set mismatch. "
                f"Missing: {len(missing)} (see {miss_path}); Extra: {len(extra)} (see {extra_path})."
            )

    # Order for readability
    df = df.sort_values(["time_h", "treatment", "concentration_uM", "filename"]).reset_index(drop=True)

    out_csv = out_dir / "kelly_auranofin_manifest.csv"
    df.to_csv(out_csv, index=False)

    # Small console report
    print(f"Data dir: {data_dir}")
    print(f"Images parsed: {len(df)}")
    print(f"Manifest saved: {out_csv}")
    if errors:
        print(f"Warnings: {len(errors)} files could not be parsed (see {err_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
