# Dataset Mode Implementation Report

Date: 2026-08-29
Scope: Add an optional **dataset** pipeline mode that skips the quality-filtering
stages (Stage 6 / 7 / 7.1 / 8 / 9) for existing high-quality datasets, while
keeping the original **open_domain** mode (Stage 0-17) byte-for-byte unchanged.

Reference audit: `docs/Stage6_9_Optional_Mode_Audit.md`.

---

## 1. Architecture Change

A single orchestration-level switch, `pipeline_mode`, selects between two
end-to-end chains in the driver `scripts/stage0_17_e2e_validate.py`:

| mode          | chain                                                              |
|---------------|--------------------------------------------------------------------|
| `open_domain` | Stage0 → 5 → 6 → 7 → 7.1 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 17 (default, unchanged) |
| `dataset`     | Stage0 → 5 → 10 → 11 → 12 → 13 → 14 → 17                          |

No stage implementation was modified (Stage 6/7/7.1/8/9 and Stage 10-17
algorithms are untouched). The change is **wiring + adapter** only:

1. **Stage10 input adapter** — in dataset mode, `accepted_clip_ids` is computed
   as *all valid Stage 5 clips* (the set of `FaceSequence.clip_id`), because
   Stage 10 internally filters `face_sequences` by `accepted_clip_ids`.
2. **Stage14 audio adapter** — Stage 14 requires an `AVAlignmentInfo` whose
   `aligned_audio_path` points to a real, clip-level 16 kHz wav. In dataset
   mode Stage 7.1 is skipped, so the adapter synthesizes an
   `already_aligned` `AVAlignmentInfo` for every accepted clip whose aligned
   audio is the **original clip wav** cut from the Stage 1 standardized audio
   (`offset = 0`, no SyncNet alignment). It reuses Stage 7.1's own
   `extract_clip_audio` helper.

### Why not make Stage 6-9 individually skippable?

Each quality stage's `process()` returns `[]` when disabled, which fails the
downstream `validate_input` contract (audit §2). Skipping therefore happens as
one block at the wiring layer.

---

## 2. Files Modified / Added

| File | Change |
|------|--------|
| `configs/config.yaml` | Added top-level `pipeline_mode: open_domain` (default) with doc comment. |
| `face_pipeline_v2/config/config_loader.py` | Added optional `pipeline_mode` validation (only when the key is present; `open_domain`/`dataset` allowed). No required-key change → backward compatible. |
| `face_pipeline_v2/pipeline/dataset_adapter.py` | **New** adapter module (see §3). |
| `scripts/stage0_17_e2e_validate.py` | Added mode branching; extracted the verbatim open-domain Stage 6-9.5 block into `_run_open_domain_quality_stages()`; added `skip_stage()` helper and `pipeline_mode` in the results JSON. |
| `tests/test_dataset_mode.py` | **New** test suite (14 tests, see §5). |
| `docs/Dataset_Mode_Implementation_Report.md` | This document. |

Stage 6 / 7 / 7.1 / 8 / 9 / 10 / 11 / 12 / 13 / 14 / 17 implementations:
**not modified**.

---

## 3. New Adapter Module — `face_pipeline_v2/pipeline/dataset_adapter.py`

Public API:

- `resolve_pipeline_mode(config) -> str` — resolves and validates the
  `pipeline_mode` config key (`open_domain` default / `dataset`), raising
  `ValueError` on an invalid value.
- `collect_accepted_clip_ids(face_sequences) -> Set[str]` — Stage10 adapter:
  returns all valid Stage 5 clip ids (no quality filtering).
- `build_dataset_av_alignments(clips, face_sequences, std_info, output_dir)
  -> List[AVAlignmentInfo]` — Stage14 adapter: builds one `already_aligned`
  `AVAlignmentInfo` per accepted clip; `aligned_audio_path` = original clip
  wav extracted from the Stage 1 standardized audio (`offset = 0`).
- `resolve_stage14_alignment(existing, clip, std_info, output_dir) ->
  AVAlignmentInfo` — the generic Stage14 audio rule: **if a Stage 7.1
  alignment with a usable `aligned_audio_path` exists, use it; otherwise fall
  back to the original clip wav.**

The adapter imports `extract_clip_audio` from
`stage_07_1_av_alignment/processor.py` (pure logic, no model) without
modifying it. The HuBERT feature-extraction algorithm in


---

## 4. Data Flow — Dataset Mode

```
Stage 0  (FFprobe integrity)
Stage 0.5(resolution filter)
Stage 1  (FFmpeg standardization)  ──> std_info (video) + audio_info (audio_path)
Stage 2  (scene segmentation)
Stage 3  (coarse face screening)
Stage 4  (sliding-window clips)    ──> clips[]
Stage 5  (SCRFD detect/crop)       ──> face_sequences[]
5.1/5.2/5.3 (enhancements, passthrough when disabled)

   ┌─ dataset branch ────────────────────────────────────────────────┐
   │ Stage 6/7/7.1/8/9/9.5 SKIPPED (recorded as SKIPPED in results)  │
   │                                                                 │
   │ accepted_clip_ids = collect_accepted_clip_ids(face_sequences)   │
   │   = {seq.clip_id for all Stage5 sequences}   (no filtering)     │
   │                                                                 │
   │ av_alignments = build_dataset_av_alignments(clips,              │
   │                  face_sequences, std_info, output_dir)          │
   │   → per accepted clip: already_aligned AVAlignmentInfo,         │
   │     aligned_audio_path = <output_dir>/<clip>/<clip>.wav         │
   │     (original clip wav cut from Stage1 audio, offset=0)         │
   └─────────────────────────────────────────────────────────────────┘

Stage 10 (InsightFace landmark)   input: {face_sequences, accepted_clip_ids}
Stage 11 (affine alignment)       input: {landmarks, face_sequences}
  └─ metadata export (per accepted clip):
       <test_base>/stage_inputs/<clip>_aligned_faces.json  (Stage 12/13)
       <test_base>/stage_inputs/<clip>_av_alignment.json   (Stage 14)
Stage 12 (FLAME)  — official runner (subprocess) via aligned_faces.json
Stage 13 (Gaze)   — official runner via aligned_faces.json
Stage 14 (HuBERT) — official runner via av_alignment.json
       reads AVAlignmentInfo.aligned_audio_path (the original clip wav)
Stage 17 (Fusion) — consumes Stage 12/13/14 outputs
```

The `av_alignment.json` written for dataset mode carries
`alignment_status="already_aligned"`, `offset_frames=0`, and
`aligned_audio_path` pointing at the existing clip wav, so the Stage 14 runner
(unchanged) resolves audio exactly as it does for the Stage 7.1
`already_aligned` case in open-domain mode.

---

## 5. Backward Compatibility / open_domain unchanged

- `pipeline_mode` is an **optional** key; `ConfigLoader.validate()` only checks
  it when present. Configs without the key keep full open-domain behavior.
- `configs/config.yaml` ships `pipeline_mode: open_domain` — the default.
- The open-domain Stage 6-9.5 block was moved verbatim into
  `_run_open_domain_quality_stages()` (same calls, same order, same early
  returns). The driver returns the same exit codes and writes the same results
  JSON (plus one additive `pipeline_mode` field).
- Results JSON now includes `"pipeline_mode"` for both modes, and dataset mode
  records each skipped quality stage as `"status": "SKIPPED"`.

## 6. Dataset Mode Behavior

- Accepts **every** Stage 5 clip (no pose / SyncNet / HyperIQA filtering).
- Stage 14 runs on the **original clip wav** — no SyncNet offset correction.
  For high-quality datasets whose audio is already in sync this is identical
  to the Stage 7.1 `already_aligned` output.
- Stage 9.5 (clip export) is also skipped; the metadata export that feeds the
  Stage 12/13/14 official runners still runs.
- If a dataset actually needs SyncNet correction, run it in `open_domain`
  mode (or run Stage 7/7.1 on the dataset and reuse its `av_alignment.json`).

## 7. Testing

`tests/test_dataset_mode.py` (14 tests, all passing):

1. `pipeline_mode` resolution/validation (default, dataset, invalid, shipped
   config, ConfigLoader rejection).
2. Stage10 adapter — `accepted_clip_ids` = all valid Stage 5 clips.
3. Stage14 adapter — original-clip-wav fallback, existing-alignment reuse,
   missing-audio fallback (with `extract_clip_audio` mocked, no ffmpeg).
4. **E2E dataset mode** — all stages mocked: verifies Stage5 → Stage10 →
   Stage11 execute, **no quality stage (6/7/7.1/8/9/9.5) is invoked**, Stage10
   receives `accepted_clip_ids == {all clips}`, Stage14 metadata JSON points at
   the original clip wav, and the results JSON records `pipeline_mode=dataset`
   with SKIPPED quality stages.
5. **E2E open_domain mode** (regression) — verifies quality stages are still
   executed and Stage10 receives the Stage9-filtered accepted ids.

Existing test suite: 649 passed / 15 failed — the 15 failures are all
pre-existing issues in untouched modules (e.g. `GazeFeature` constructor
signature in `test_datamodel`, `crop_face` return type in
`test_stage_05_face_detection`, Stage9 hard-filter thresholds in
`test_stage_09_quality_fusion`, Stage9.1 passthrough in
`test_stage_09_1_quality_ranking`). None are caused by this change.

## 8. Known Limitations

- Dataset mode assumes the input audio is already aligned (offset=0). Clips
  with genuine A/V desync will not be corrected.
- Skipping is implemented at the wiring layer (e2e driver). Other callers that
  want dataset mode must route through `pipeline_runner` / the e2e driver, or
  reuse `dataset_adapter.build_dataset_av_alignments` +
  `collect_accepted_clip_ids` directly.
- `pipeline_runner.py` remains a stub (0 bytes); the real wiring lives in
  `scripts/stage0_17_e2e_validate.py`. Migrating shared wiring into
  `pipeline_runner.py` is an optional future step.

`stage_14_audio_feature/processor.py` is untouched.
