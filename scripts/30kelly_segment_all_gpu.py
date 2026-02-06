# scripts/11_segment_failed_cpu.py
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd


def project_root_from_script() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def autodetect_latest_workdir(results_root: Path) -> Optional[Path]:
    """
    Pick the newest results/kelly_auranofin_* directory that contains segment_manifest_images.csv
    """
    cands = []
    for d in results_root.glob("kelly_auranofin_*"):
        if d.is_dir() and (d / "segment_manifest_images.csv").exists():
            cands.append(d)
    if not cands:
        return None
    cands.sort(key=lambda p: (p / "segment_manifest_images.csv").stat().st_mtime, reverse=True)
    return cands[0]


def load_seg_cfg(work_dir: Path) -> dict:
    cfg_path = work_dir / "seg_config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def extract_last_json(stdout_text: str) -> Optional[dict]:
    lines = [ln.strip() for ln in (stdout_text or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return json.loads(ln)
            except Exception:
                continue
    return None


def run_worker(script_1: Path, image_path: Path, work_dir: Path, cfg: dict) -> Dict[str, Any]:
    cmd = [
        sys.executable, str(script_1),
        "--worker",
        "--device", "cpu",
        "--image", str(image_path),
        "--work_dir", str(work_dir),
        "--cellpose_model", cfg.get("cellpose_model", "cyto2"),
        "--flow_threshold", str(cfg.get("flow_threshold", 0.4)),
        "--cellprob_threshold", str(cfg.get("cellprob_threshold", 0.0)),
        "--timeout_sec", str(int(cfg.get("timeout_sec", 15 * 60))),
        "--clahe", "1" if bool(cfg.get("use_clahe", True)) else "0",
    ]
    if cfg.get("diameter") is not None:
        cmd += ["--diameter", str(cfg["diameter"])]

    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=int(cfg.get("timeout_sec", 15 * 60)))
    except subprocess.TimeoutExpired:
        return {"status": "fail", "error": f"timeout>{cfg.get('timeout_sec')}s", "device_used": "cpu", "elapsed_sec": time.time() - t0}

    rec = extract_last_json(p.stdout or "")
    if rec is None:
        tail = (p.stderr or p.stdout or "")[-4000:]
        return {"status": "fail", "error": f"no_json_from_worker; rc={p.returncode}; tail={tail}", "device_used": "cpu", "elapsed_sec": time.time() - t0}

    rec["elapsed_sec"] = time.time() - t0
    rec["returncode"] = int(p.returncode)
    return rec


def latest_status_per_image(df: pd.DataFrame) -> pd.DataFrame:
    """
    Our manifest is append-only (each attempt becomes a new row).
    We need the last attempt per image_path.
    """
    df2 = df.copy()
    if "image_path" not in df2.columns:
        raise ValueError("segment_manifest_images.csv missing 'image_path'")
    # preserve original order => last row in file is the last attempt
    df2["_row"] = range(len(df2))
    df2 = df2.sort_values("_row").groupby("image_path", as_index=False).tail(1).drop(columns=["_row"])
    return df2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default=None, help="Path to results/kelly_auranofin_<run_id>. If omitted, auto-detect latest.")
    args = ap.parse_args()

    root = project_root_from_script()
    results_root = root / "results"

    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        work_dir = autodetect_latest_workdir(results_root)
        if work_dir is None:
            raise FileNotFoundError(f"Could not auto-detect work_dir in {results_root}")

    manifest_path = work_dir / "segment_manifest_images.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")

    print(f"Using work_dir: {work_dir}")
    cfg = load_seg_cfg(work_dir)

    df = pd.read_csv(manifest_path)
    if "status" not in df.columns:
        raise ValueError("segment_manifest_images.csv missing 'status' column")

    df_last = latest_status_per_image(df)
    todo = df_last[df_last["status"].astype(str) != "ok"]["image_path"].astype(str).tolist()

    if not todo:
        print("Nothing to do: all images are already OK (latest status).")
        return

    script_1 = Path(__file__).resolve().parent / "30kelly_segment_all_gpu.py"
    if not script_1.exists():
        # fallback name if you kept original 10_*.py
        script_1_alt = Path(__file__).resolve().parent / "10_segment_all_gpu.py"
        if script_1_alt.exists():
            script_1 = script_1_alt
        else:
            raise FileNotFoundError(f"Cannot find script 1 worker: {script_1}")

    total = len(todo)
    failed_after = []

    for i, ip_str in enumerate(todo, start=1):
        ip = Path(ip_str)
        print(f"[{i:02d}/{total:02d}] CPU segment: {ip.name} ...")
        rec = {"image_path": str(ip), "image_name": ip.name}
        rec.update(run_worker(script_1, ip, work_dir, cfg))

        status = rec.get("status", "fail")
        if status == "ok":
            print(f"          OK  cells={rec.get('n_cells', 'NA')}  time={rec.get('elapsed_sec', 0):.1f}s")
        else:
            print(f"          FAIL time={rec.get('elapsed_sec', 0):.1f}s  error={str(rec.get('error',''))[:160]}")
            failed_after.append(ip.name)

        # append attempt
        df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
        df.to_csv(manifest_path, index=False)

    (work_dir / "failed_after_cpu.txt").write_text("\n".join(failed_after) + ("\n" if failed_after else ""), encoding="utf-8")
    print(f"\nDone. CPU processed: {total}; still failed: {len(failed_after)}")
    print(f"Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
