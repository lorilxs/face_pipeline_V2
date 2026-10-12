#!/usr/bin/env python3
"""Stage 0-11 batch driver.

Runs the existing single-video harness (tests/stage0_11_integration_test.py)
once per video in a directory, saves each video's JSON result under
tests/reports/batch/<video_id>.json, then aggregates everything into a
statistics table (markdown + CSV + JSON).

Usage (from the project root, inside the container):

    python tests/stage0_11_batch_test.py \\
        --input-dir test_data/chinese_talking_videos \\
        --config configs/config.yaml \\
        --test-base outputs/stage0_11_batch \\
        --results-dir tests/reports/batch \\
        [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _harness_json_path(root: Path) -> Path:
    # The harness always writes PROJECT_ROOT/tests/reports/stage0_11_results.json
    return root / "tests" / "reports" / "stage0_11_results.json"


def _render_markdown(summary: Dict[str, Any]) -> str:
    rows = summary["results"]
    # ordered stage names as first seen
    stage_names: List[str] = []
    for r in rows:
        for s in r.get("stages", []):
            if s["stage"] not in stage_names:
                stage_names.append(s["stage"])

    lines: List[str] = []
    add = lines.append
    add("# Face Pipeline V2 — Stage 0-11 Batch Test Summary")
    add("")
    add(f"- **Generated at (UTC)**: {summary['generated_at']}")
    add(f"- **Python**: {summary['python']}")
    add(f"- **Videos**: {len(rows)}")
    add(f"- **Command**: `python tests/stage0_11_batch_test.py {summary.get('cmdline', '')}`")
    add("")

    # ---- per-video status table -------------------------------------------
    add("## 1. Per-video Final Status")
    add("")
    add("| Video | Final | Total elapsed (s) | Stages PASS / FAIL / SKIP |")
    add("| --- | --- | ---: | --- |")
    for r in rows:
        st = r.get("stages", [])
        n_pass = sum(1 for s in st if s["status"] == "PASS")
        n_fail = sum(1 for s in st if s["status"] == "FAIL")
        n_skip = sum(1 for s in st if s["status"] == "SKIPPED")
        add(f"| {r['video']} | {r.get('final_status', '-')} | {r.get('elapsed_total_s', 0)} "
            f"| {n_pass} / {n_fail} / {n_skip} |")
    add("")

    # ---- per-stage aggregate ---------------------------------------------
    add("## 2. Per-stage Aggregate (across videos)")
    add("")
    add("| Stage | PASS | FAIL | SKIPPED | PASS rate | Avg elapsed (s) | Max elapsed (s) |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in stage_names:
        statuses = [s["status"] for r in rows for s in r.get("stages", []) if s["stage"] == name]
        if not statuses:
            continue
        n_pass = statuses.count("PASS")
        n_fail = statuses.count("FAIL")
        n_skip = statuses.count("SKIPPED")
        elapsed = [float(s["elapsed_s"] or 0)
                   for r in rows for s in r.get("stages", [])
                   if s["stage"] == name and s.get("elapsed_s") is not None]
        avg = (sum(elapsed) / len(elapsed)) if elapsed else 0.0
        mx = max(elapsed) if elapsed else 0.0
        rate = (n_pass / len(statuses) * 100) if statuses else 0.0
        add(f"| {name} | {n_pass} | {n_fail} | {n_skip} | {rate:.1f}% | {avg:.2f} | {mx:.2f} |")
    add("")

    # ---- per-stage per-video matrix ---------------------------------------
    add("## 3. Stage × Video Status Matrix")
    add("")
    header = "| Video | " + " | ".join(name.replace("|", "\\|") for name in stage_names) + " |"
    add(header)
    add("| --- | " + " | ".join(["---"] * len(stage_names)) + " |")
    for r in rows:
        by_stage = {s["stage"]: s["status"] for s in r.get("stages", [])}
        cells = [by_stage.get(n, "-") for n in stage_names]
        add(f"| {r['video']} | " + " | ".join(cells) + " |")
    add("")

    # ---- errors ------------------------------------------------------------
    add("## 4. Errors")
    add("")
    any_err = False
    for r in rows:
        for s in r.get("stages", []):
            if s.get("error"):
                any_err = True
                add(f"- `{r['video']}` / {s['stage']}: `{s['error']}`")
    if not any_err:
        add("No stage raised an exception in any video.")
    add("")
    return "\n".join(lines) + "\n"


def _write_csv(summary: Dict[str, Any], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "final_status", "total_elapsed_s", "stage", "status",
                    "elapsed_s", "output_summary", "error"])
        for r in summary["results"]:
            for s in r.get("stages", []):
                w.writerow([r["video"], r.get("final_status"), r.get("elapsed_total_s"),
                            s["stage"], s["status"], s.get("elapsed_s"),
                            (s.get("output_summary") or "").replace("\n", " "),
                            (s.get("error") or "")])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True, help="Directory of input videos")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--test-base", default="outputs/stage0_11_batch")
    ap.add_argument("--harness", default="tests/stage0_11_integration_test.py")
    ap.add_argument("--results-dir", default="tests/reports/batch")
    ap.add_argument("--limit", type=int, default=0, help="0 = all videos")
    args = ap.parse_args()

    root = Path.cwd()
    input_dir = Path(args.input_dir)
    videos = sorted(p for p in input_dir.glob("*.mp4") if p.is_file())
    if args.limit and args.limit > 0:
        videos = videos[: args.limit]

    if not videos:
        print(f"No .mp4 files found in {input_dir}")
        return 1

    resdir = Path(args.results_dir)
    resdir.mkdir(parents=True, exist_ok=True)
    harness = Path(args.harness)
    config = Path(args.config)

    summary: Dict[str, Any] = {
        "generated_at": _now_utc(),
        "python": sys.version.split()[0],
        "cmdline": " ".join(sys.argv[1:]),
        "input_dir": str(input_dir),
        "results": [],
    }

    for i, v in enumerate(videos, 1):
        vid = v.stem
        tb = f"{args.test_base}/{vid}"
        print(f"[{i}/{len(videos)}] {v.name} -> {tb}", flush=True)
        cmd = [sys.executable, str(harness),
               "--video", str(v), "--config", str(config), "--test-base", tb]
        t0 = time.monotonic()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            rc = proc.returncode
        except Exception as exc:  # noqa: BLE001
            rc = -1
            print(f"  subprocess error: {exc}")
        elapsed = round(time.monotonic() - t0, 2)

        entry: Dict[str, Any] = {"video": v.name, "rc": rc, "elapsed_total_s": elapsed}
        json_path = _harness_json_path(root)
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                data = {}
                print(f"  json parse error: {exc}")
            entry["final_status"] = data.get("final_status", "NO_REPORT")
            entry["stages"] = data.get("stages", [])
            (resdir / f"{vid}.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            entry["final_status"] = "NO_REPORT"
            entry["stages"] = []
        summary["results"].append(entry)
        print(f"  -> status={entry['final_status']} rc={rc} {elapsed}s", flush=True)

    # ---- write aggregates ---------------------------------------------------
    agg_path = resdir / "stage0_11_batch_summary.json"
    agg_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = resdir / "stage0_11_batch_summary.md"
    md_path.write_text(_render_markdown(summary), encoding="utf-8")
    csv_path = resdir / "stage0_11_batch_summary.csv"
    _write_csv(summary, csv_path)

    print(f"\n===== BATCH DONE: {len(summary['results'])} videos =====")
    for r in summary["results"]:
        print(f"  [{r.get('final_status', '-'):8s}] {r['video']:45s} {r['elapsed_total_s']}s")
    print(f"Summary JSON -> {agg_path}")
    print(f"Summary MD   -> {md_path}")
    print(f"Summary CSV  -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

