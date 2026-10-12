# `pipeline_manifest.json` — Per-Run Production Statistics Manifest

**Module:** `face_pipeline_v2/utils/pipeline_manifest.py` (v1.1)
**Files written:** `<run_root>/pipeline_manifest.json` (atomic: tmp + `os.replace`)

## 1. Motivation

The open-domain pipeline writes many intermediate stage outputs under the
Stage0-11 test base (`clips/`, `audio/`, `faces/`, `scenes/`, `videos/`,
`stage_inputs/`, `stage12_flame/`, `stage13_gaze/`, `stage14_audio/`,
`stage17_fusion/`, ...). These intermediates are regenerable and are routinely
deleted to free disk — but that also erased the **per-run production
statistics** (how many clips were created, how many were rejected at Stage 5.2
/ 5.3 and why, per-clip lineage, fusion counts, final deliverable inventory).

`pipeline_manifest.json` is a **single self-contained JSON document** that
captures those statistics *before* the intermediates are deleted, so the
statistics survive. It is the only record that does.

## 2. Two-phase lifecycle

| Phase | Writer | When | Data source |
|---|---|---|---|
| Phase 1 | `scripts/stage0_17_e2e_validate.py::_finalize` | immediately after `stage0_11_results.json` | **in-memory** `env` / `context` — no on-disk intermediate required |
| Phase 2 | `scripts/export_production_multimodal.py::main` | after the Stage17 multimodal export loop | Stage12/13/14 official-runner metadata JSONs + Stage17 export metadata + export dir tree |

Phase 2 is idempotent and re-runnable: it loads the existing manifest (or
creates a phase2-only doc if no phase-1 manifest exists), merges feature
stages, and re-saves.

## 3. Path layout

```
<run_root>/pipeline_manifest.json              # <- the manifest
<run_root>/stage0_11/                          # <- the Stage0-11 test base
    clips/  audio/  faces/  scenes/  videos/
    stage12_flame/<clip_id>/flame_metadata.json
    stage13_gaze/<clip_id>/gaze_metadata.json
    stage14_audio/<clip_id>/audio_metadata.json
    stage17_fusion/...
<run_root>/stage17_multimodal_export/          # <- Stage17 deliverable
    flame/<clip_id>.npy  audio/<clip_id>.npy
    timestamp/<clip_id>.npz  metadata/<clip_id>.json  manifest.csv
```

`run_root` is always `test_base.parent`; both scripts derive it the same way,
so phase 1 and phase 2 agree on the manifest location without configuration.

## 4. Document structure (v1.1)

```
manifest_version  generated_at  updated_at
run               run_root, test_base, input_video, config_path,
                  pipeline_mode, final_status, window_seconds, note
source_video      video_id, codec, width/height/fps, duration,
                  total_frames, has_audio
config_snapshot   stage_00…stage_11 + enhancements + stage_04
                  (mode/window_size/stride/target_fps verbatim)
stages            stage_id -> {status, phase, note}  (phase1 + phase2)
stage_details     stage_id -> {clip_count, reason_counts, frames_total,
                  mean_score, ...}  (per-stage aggregates)
clips             per-clip records:
                    clip_id, scene_id, window_seconds,
                    stage_03, stage_05, stage_05_1, stage_05_2,
                    stage_05_3, stage_06, stage_07, stage_07_1,
                    stage_08, stage_09, stage_09_5, stage_10, stage_11
                  + (phase2) stage_12, stage_13, stage_14, stage_17
final_assets      partial flag, asset_count, export_dir,
                  assets[]: clip_id + flame/audio/timestamp/metadata
                  paths (relative to run_root) + num_frames/valid_length/
                  visual_dim/audio_dim/duration
audit             stage_05_2_single_face_filter / stage_05_3_temporal_stability
                  -> rejected[]: {clip_id, reasons, main_track_id,
                                   anomaly_frames, anomaly_ratio, ...}
retention         required paths + note
_phase1_complete  _phase2_complete
```

## 5. Stage 5.2 / 5.3 reject audit (the deletion-proof part)

Rejected clips never appear in any later output, so their reject reasons are
otherwise unrecoverable after deletion. The audit is collected **live** at
filter time:

- `PipelineContext.stage_audit` (new field in `pipeline/context.py`) holds
  `{stage_id: {"rejected": [...]}}`. `add_stage_audit()` appends entries.
- `stage_05_2_single_face_filter.stage` and
  `stage_05_3_temporal_stability.stage` call `context.add_stage_audit()` in
  their reject branch, recording `reasons` plus the identity/stability metrics
  that caused the rejection.
- The audit append is wrapped in `try/except`: an audit failure can never
  break the filter itself.
- `_build_audit()` copies `context.stage_audit` into the manifest verbatim.

## 6. Stage 4 window semantics

When `stage_04.mode == "sliding_window"`, every downstream stage operates on
fixed-length windows (e.g. 8 s with a 4 s stride) and `window_seconds` is
recorded on the run object and each clip record. When `mode == "segment"`,
clips are scene-derived 1:1 and `window_seconds` is `None`. The manifest
records the mode/window_size/stride verbatim in `config_snapshot.stage_04` so
downstream statistics stay interpretable after the fact.

## 7. Non-fatal design

Every public function is wrapped so a manifest failure can never fail the
pipeline:

| Function | On failure |
|---|---|
| `load(path)` | returns `None` + warning |
| `save(doc, path)` | returns `False` + warning |
| `build_phase1(...)` | caller `_finalize` wraps in try/except → warning only |
| `merge_phase2(base, export_dir, logger)` | logs + returns partial doc; caller wraps → warning only |

The module never deletes anything.

## 8. Integration points

1. **`face_pipeline_v2/pipeline/context.py`** — added `stage_audit` field and
   `add_stage_audit()` (frozen dataclass permits in-place dict mutation).
2. **`stage_05_2_single_face_filter/stage.py`** / **`stage_05_3_temporal_stability/stage.py`**
   — audit appends in the reject branches.
3. **`scripts/stage0_17_e2e_validate.py`** — `_finalize` now takes `context`
   and writes phase 1 right after `stage0_11_results.json` (best-effort).
4. **`scripts/export_production_multimodal.py`** — `main()` calls
   `merge_phase2(base, export_dir, logger)` before returning (best-effort).

## 9. Verification recipe

```bash
# Phase 1
python scripts/stage0_17_e2e_validate.py \
  --video 'videos/chinese_talking_videos_bilibili/中文访谈授课视频单人脸/CNFACE_00001_BV18Qg96YE1W_000000.mp4' \
  --config configs/config.yaml \
  --test-base outputs/CNFACE_00001_BV18Qg96YE1W_000000_open_domain/stage0_11
# -> outputs/CNFACE_00001_BV18Qg96YE1W_000000_open_domain/pipeline_manifest.json

# Phase 2 (after Stage12/13/14 + Stage17 runs)
python scripts/export_production_multimodal.py \
  --base outputs/CNFACE_00001_BV18Qg96YE1W_000000_open_domain/stage0_11 \
  --export-dir outputs/CNFACE_00001_BV18Qg96YE1W_000000_open_domain/stage17_multimodal_export

# Survival check: move the test base aside and re-read
mv outputs/CNFACE_00001_BV18Qg96YE1W_000000_open_domain/stage0_11 /tmp/backup_stage0_11
python -c "import json; d=json.load(open('outputs/CNFACE_00001_BV18Qg96YE1W_000000_open_domain/pipeline_manifest.json')); print(len(d['clips']), d['_phase1_complete'], d['_phase2_complete'])"
```

