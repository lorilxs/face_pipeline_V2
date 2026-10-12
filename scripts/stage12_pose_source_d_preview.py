#!/usr/bin/env python3
"""Stage12 pose-source scheme-D preview -- small-scope validation.

Goal (AISDD): formally evaluate replacing Stage12 jaw fusion with pure
SPECTRE jaw (scheme D: head=EMICA global, jaw=SPECTRE jaw, expr=EMICA)
BEFORE touching production code.

Method (read-only w.r.t. production code):
  1. Re-encode the SAME 50 frames covered by the Stage12 npz
     (frame_index 100-149) with the corrected /255.0 normalization,
     using EMICAWrapper + SPECTREWrapper (identical methodology to
     scripts/stage12_pose_ablation_analyze.py).
  2. Build the scheme-D output array EXACTLY as production runner.py
     would for jaw_fusion_lambda = 1.0 (jaw_pose = SPECTRE_jaw,
     head_pose = EMICA global, expression = EMICA), plus the current
     scheme-A output (lambda = 0.6) for comparison.
  3. Write a scheme-D flame_features.npz (production schema) to
     outputs/stage12_pose_validation/ -- a *preview* artifact, NOT a
     production Stage12 output.
  4. Anomaly checks on the D output: NaN/Inf, |jaw| < 1 rad,
     temporal continuity, frame-count consistency.
  5. Interface checks: read_stage12_outputs() on the D preview
     (FLAMEFeature dims) and Stage17 build_training_sample()
     end-to-end with the existing Stage13 gaze + Stage14 audio.
  6. Persist everything to outputs/stage12_pose_validation/
     stage12_pose_source_d_preview.json.

No Stage12/13/14/17 production code is modified.
"""
import sys, os, json, glob, re, types, traceback
from pathlib import Path
import numpy as np, torch, cv2

P = Path("/workspace/Face_Pipeline_V2"); sys.path.insert(0, str(P))
for _p in [P / "third_party", P / "third_party/inferno/inferno-master",
           P / "third_party/inferno/spectre"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EMICA_CKPT = str(P / "models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/model-epoch=38-val/loss_total=3.88806534.ckpt")
SPECTRE_TAR = str(P / "third_party/inferno/spectre/spectre/data/deca_model.tar")
CLIP_ID = "CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001"
SRC_NPZ = str(P / f"outputs/stage12_flame/{CLIP_ID}/flame_features.npz")
ALIGN_DIR = str(P / f"outputs/stage0_11_test_verify/aligned_faces/{CLIP_ID}")
STAGE13_DIR = P / "outputs/stage13_gaze"
STAGE14_DIR = P / "outputs/stage14_audio"
VAL_DIR = P / "outputs/stage12_pose_validation"
JAW_LAMBDA = 0.6          # current fusion SPECTRE weight
DEV = "cpu"

# ---------------------------------------------------------------------------
# pytorch_lightning / chumpy stubs (identical to stage12_pose_ablation_analyze.py)
# ---------------------------------------------------------------------------
class _SM(type):
    def __getattr__(cls, n): return _Stub

class _Stub(metaclass=_SM):
    def __init__(self, *a, **k): pass
    def __setstate__(self, s):
        try:
            if isinstance(s, dict): self.__dict__.update(s)
        except Exception: pass
    def __getattr__(self, n): return _Stub()
    def __reduce__(self): return (_Stub, ())

import importlib.util, importlib.abc

def _stub_getattr(name):
    if name.startswith("__") and name.endswith("__"): raise AttributeError(name)
    return _Stub

class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        m = types.ModuleType(spec.name); m.__getattr__ = _stub_getattr
        m.__path__ = []; m.__all__ = []
        return m
    def exec_module(self, module): pass

class _Finder(importlib.abc.MetaPathFinder):
    _P = ("pytorch_lightning", "torchmetrics", "lightning", "lightning_fabric",
          "lightning_utilities", "chumpy")
    def find_spec(self, fn, path, target=None):
        if any(fn == p or fn.startswith(p + ".") for p in self._P):
            return importlib.util.spec_from_loader(fn, _Loader(), origin="stub")
        return None

class _Ch(np.ndarray):
    def __new__(cls, *a, **k): return super().__new__(cls, (0,))
    def __array_finalize__(self, obj): pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            x = state.get('x')
            if x is not None:
                if 'scipy.sparse' in str(type(x)): x = x.todense()
                x = np.asarray(x)
                if x.size > 0:
                    self.resize(x.shape, refcheck=False)
                    self[...] = x
            for k, v in state.items():
                if k != 'x':
                    try: object.__setattr__(self, k, v)
                    except Exception: pass
        else:
            super().__setstate__(state)
    def __array__(self, dtype=None):
        x = getattr(self, '_xdata', None)
        if x is None: return np.asarray(self)
        if 'scipy.sparse' in str(type(x)): x = x.todense()
        return np.asarray(x, dtype=dtype) if dtype else np.asarray(x)

def install_stubs():
    sys.meta_path.insert(0, _Finder())
    for nm in ("chumpy", "chumpy.ch"):
        m = types.ModuleType(nm); m.Ch = _Ch; m.__path__ = []
        m.__getattr__ = _stub_getattr
        sys.modules[nm] = m
# ---------------------------------------------------------------------------
# Frame loading / preprocessing
# ---------------------------------------------------------------------------
def prep(im):
    x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (224, 224)).astype(np.float32)
    x = x / 255.0          # corrected normalization [0,1] (fixed Stage12 behavior)
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

def load_frames(frame_indices):
    index_map = {}
    for p in sorted(glob.glob(os.path.join(ALIGN_DIR, "aligned_*.jpg"))):
        base = os.path.basename(p)
        m = re.match(r"aligned_(\d{6})", base)
        if not m:
            continue
        index_map[int(m.group(1))] = p
    imgs, used = [], []
    for idx in frame_indices:
        if idx in index_map:
            im = cv2.imread(index_map[idx])
            if im is not None:
                imgs.append(im); used.append(idx)
    return imgs, used

# ---------------------------------------------------------------------------
# Statistics (identical to stage12_pose_ablation_analyze.py)
# ---------------------------------------------------------------------------
def _deg(x): return float(np.degrees(x))

def rot_stats(a, key):
    ang = np.linalg.norm(a, axis=1)
    return {
        f"{key}_angle_mean_deg": _deg(ang.mean()),
        f"{key}_angle_std_deg": _deg(ang.std()),
        f"{key}_angle_max_deg": _deg(ang.max()),
        f"{key}_angle_mean_rad": float(ang.mean()),
        f"{key}_angle_std_rad": float(ang.std()),
        f"{key}_angle_max_rad": float(ang.max()),
    }

def temporal_stats(a, key):
    d = np.diff(a, axis=0)
    vel = np.linalg.norm(d, axis=1)
    return {
        f"{key}_mean_vel_deg": _deg(vel.mean()),
        f"{key}_max_vel_deg": _deg(vel.max()),
        f"{key}_p95_vel_deg": _deg(np.percentile(vel, 95)),
        f"{key}_mean_vel_rad": float(vel.mean()),
        f"{key}_max_vel_rad": float(vel.max()),
        f"{key}_p95_vel_rad": float(np.percentile(vel, 95)),
    }

def neutral_bias(jaw, key):
    ang = np.linalg.norm(jaw, axis=1)
    vel = np.linalg.norm(np.diff(jaw, axis=0), axis=1)
    lo = np.percentile(vel, 25) if vel.size else 0.0
    low_ang = ang[1:][vel < lo] if vel.size else np.zeros(0)
    return {
        f"{key}_first_frame_norm_deg": _deg(ang[0]),
        f"{key}_first_frame_norm_rad": float(ang[0]),
        f"{key}_low_motion_mean_deg": _deg(low_ang.mean()) if low_ang.size else float("nan"),
        f"{key}_low_motion_median_deg": _deg(np.median(low_ang)) if low_ang.size else float("nan"),
        f"{key}_low_motion_n": int(low_ang.size),
        f"{key}_all_frames_mean_deg": _deg(ang.mean()),
        f"{key}_all_frames_mean_rad": float(ang.mean()),
    }

def pearson(x, y):
    x = np.asarray(x, float) - float(np.mean(x))
    y = np.asarray(y, float) - float(np.mean(y))
    den = np.sqrt(float(np.sum(x * x)) * float(np.sum(y * y)))
    return float(np.sum(x * y) / den) if den > 0 else float("nan")

def metrics(head, jaw):
    """7-field AISDD metric block (identical mapping to the ablation json)."""
    s = {}
    s.update(rot_stats(head, "head"))
    s.update(rot_stats(jaw, "jaw"))
    s.update(temporal_stats(head, "head"))
    s.update(temporal_stats(jaw, "jaw"))
    s.update(neutral_bias(jaw, "jaw"))
    h = np.linalg.norm(head, axis=1); j = np.linalg.norm(jaw, axis=1)
    s["pearson_head_jaw_angle"] = pearson(h, j)
    m = {
        "head_angle_mean": s["head_angle_mean_deg"],
        "head_angle_std": s["head_angle_std_deg"],
        "jaw_angle_mean": s["jaw_angle_mean_deg"],
        "jaw_angle_std": s["jaw_angle_std_deg"],
        "jaw_rest_bias": s["jaw_low_motion_mean_deg"],
        "jaw_velocity": s["jaw_mean_vel_deg"],
        "head_jaw_correlation": s["pearson_head_jaw_angle"],
    }
    return m, s

def anomaly_checks(eg, D_jaw, ex, frame_indices, timestamps, T_src):
    """Anomaly checks required by the task spec."""
    jaw_norm = np.linalg.norm(D_jaw, axis=1)
    head_norm = np.linalg.norm(eg, axis=1)
    fi = np.asarray(frame_indices, int)
    checks = {
        "nan_count_total": int(np.isnan(eg).sum() + np.isnan(D_jaw).sum()
                               + np.isnan(ex).sum()),
        "inf_count_total": int(np.isinf(eg).sum() + np.isinf(D_jaw).sum()
                               + np.isinf(ex).sum()),
        "no_nan": bool(np.isfinite(eg).all() and np.isfinite(D_jaw).all()
                       and np.isfinite(ex).all()),
        "jaw_angle_max_deg": _deg(jaw_norm.max()),
        "jaw_angle_max_rad": float(jaw_norm.max()),
        "jaw_lt_1_rad": bool(jaw_norm.max() < 1.0),
        "head_angle_max_deg": _deg(head_norm.max()),
        "frame_count": int(T_src),
        "frame_index_monotonic": bool(np.all(np.diff(fi) > 0)),
        "frame_index_consecutive": bool(np.all(np.diff(fi) == 1)),
        "temporal_continuous": bool(np.all(np.diff(fi) == 1)),
        "timestamp_monotonic": bool(np.all(np.diff(timestamps) >= 0)),
    }
    return checks
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    install_stubs()

    # ---- Phase 0: load source npz metadata --------------------------------
    d = np.load(SRC_NPZ, allow_pickle=True)
    frame_indices = [int(x) for x in np.asarray(d["frame_index"]).reshape(-1)]
    timestamps = np.asarray(d["timestamp"], np.float64)
    npz_fields = {k: (list(np.asarray(d[k]).shape), str(np.asarray(d[k]).dtype))
                  for k in d.files}
    T_src = len(frame_indices)
    print(f"Phase0: source npz T={T_src} frames={frame_indices[0]}..{frame_indices[-1]}",
          flush=True)

    # ---- Phase 1: encode the exact npz frames, corrected /255.0 ------------
    imgs, used = load_frames(frame_indices)
    print(f"Phase1: npz frames={T_src} aligned_loaded={len(imgs)}", flush=True)
    emica = sp = None
    try:
        from face_pipeline_v2.models.wrappers.emica import EMICAWrapper
        emica = EMICAWrapper(Path(EMICA_CKPT), device=DEV); emica.load()
        print("EMICA loaded", flush=True)
    except Exception as e:
        print("EMICA FAIL", repr(e), flush=True); traceback.print_exc()
    try:
        from face_pipeline_v2.models.wrappers.spectre import SPECTREWrapper
        sp = SPECTREWrapper(Path(SPECTRE_TAR), device=DEV); sp.load()
        print("SPECTRE loaded", flush=True)
    except Exception as e:
        print("SPECTRE FAIL", repr(e), flush=True); traceback.print_exc()
    if emica is None or sp is None or not imgs:
        print("ABORT: encoder load or frames missing", flush=True); sys.exit(2)

    b = torch.cat([prep(im) for im in imgs], 0).to(DEV)
    with torch.no_grad():
        eo = emica.encode(b)
        eg = np.asarray(eo["globalpose"].cpu().numpy(), np.float64)  # EMICA global
        ej = np.asarray(eo["jawpose"].cpu().numpy(), np.float64)     # EMICA jaw
        ex = np.asarray(eo["expcode"].cpu().numpy(), np.float64)     # EMICA expr
        so = sp.encode(b)
        sj = np.asarray(so["jaw_pose"].cpu().numpy(), np.float64)    # SPECTRE jaw
    T = eg.shape[0]
    assert T == T_src, f"frame mismatch: encoded {T} vs npz {T_src}"
    print(f"Phase1: encoded T={T} eg={eg.shape} ej={ej.shape} ex={ex.shape} sj={sj.shape}",
          flush=True)

    # ---- Phase 2: build scheme outputs as production would -----------------
    A_jaw = (1.0 - JAW_LAMBDA) * ej + JAW_LAMBDA * sj   # current fusion (A)
    D_jaw = sj                                         # scheme D (== lambda=1.0)

    # identity check: D == pure SPECTRE jaw == stored spectre_jawpose relation
    npz_spectre = np.asarray(d["spectre_jawpose"], np.float64)
    d_vs_stored_max = float(np.abs(D_jaw - npz_spectre).max())

    A_metrics, A_verb = metrics(eg, A_jaw)
    D_metrics, D_verb = metrics(eg, D_jaw)
    print("\n=== A (current fusion) ===", flush=True)
    for k, v in A_metrics.items(): print(f"  {k:24s} {v:.6f}", flush=True)
    print("=== D (EMICA head + SPECTRE jaw) ===", flush=True)
    for k, v in D_metrics.items(): print(f"  {k:24s} {v:.6f}", flush=True)
    print(f"D vs stored npz spectre_jawpose max |diff| = {d_vs_stored_max:.6e} rad",
          flush=True)

    # ---- Phase 3: write scheme-D preview npz (production schema) -----------
    clip_out = VAL_DIR / CLIP_ID
    clip_out.mkdir(parents=True, exist_ok=True)
    d_preview = {
        "expression_code": ex.astype(np.float32),
        "jaw_pose": D_jaw.astype(np.float32),
        "head_pose": eg.astype(np.float32),
        "eye_pose": np.zeros((T, 6), dtype=np.float32),
        "spectre_jawpose": sj.astype(np.float32),
        "shapecode": np.zeros((T, 300), dtype=np.float32),
        "frame_index": np.asarray(frame_indices, dtype=np.int32),
        "timestamp": timestamps.astype(np.float64),
    }
    np.savez(str(clip_out / "flame_features.npz"), **d_preview)
    # also write scheme-A preview for npz-level A-vs-D value comparison
    a_preview = dict(d_preview)
    a_preview["jaw_pose"] = A_jaw.astype(np.float32)
    np.savez(str(VAL_DIR / f"{CLIP_ID}_A_preview.npz"), **a_preview)
    np.save(str(clip_out / "D_jaw_pose.npy"), D_jaw.astype(np.float32))
    np.save(str(clip_out / "D_head_pose.npy"), eg.astype(np.float32))
    print(f"WROTE D-preview npz -> {clip_out / 'flame_features.npz'}", flush=True)

    # ---- Phase 4: anomaly checks on the D output ---------------------------
    checks = anomaly_checks(eg, D_jaw, ex, frame_indices, timestamps, T_src)
    print("\n=== Anomaly checks (D) ===", flush=True)
    for k, v in checks.items(): print(f"  {k:28s} {v}", flush=True)
    # ---- Phase 5: interface check -- Stage12 read path + Stage17 ------------
    interface = {}
    try:
        from face_pipeline_v2.stages.stage_12_flame.integration import (
            read_stage12_outputs,
        )
        feats = read_stage12_outputs(VAL_DIR, CLIP_ID)
        n_feat = len(feats)
        dims_ok = all(
            len(f.expcode) == 100 and len(f.posecode) == 6
            and len(f.jawpose) == 3 and len(f.globalpose) == 3
            and len(f.spectre_jawpose) == 3
            for f in feats
        )
        jaw0 = np.asarray(feats[0].jawpose)
        interface["read_stage12_outputs"] = {
            "n_features": n_feat,
            "expcode_dim": 100, "posecode_dim": 6,
            "jawpose_dim": 3, "globalpose_dim": 3,
            "spectre_jawpose_dim": 3,
            "dims_ok": bool(dims_ok),
            "sample_jawpose_0": [float(x) for x in jaw0],
        }
        print(f"\n=== read_stage12_outputs on D-preview ===", flush=True)
        print(f"  n_features={n_feat} dims_ok={dims_ok}", flush=True)
        if n_feat != T_src or not dims_ok:
            print("INTERFACE FAIL: stage12 read path rejected D output", flush=True)
    except Exception as e:
        print("read_stage12_outputs FAIL", repr(e), flush=True)
        traceback.print_exc()
        feats = None
        interface["read_stage12_outputs"] = {"error": repr(e)}
    stage17 = {}
    if feats is not None:
        try:
            from face_pipeline_v2.stages.stage_13_gaze.integration import (
                read_gaze_features,
            )
            from face_pipeline_v2.stages.stage_14_audio_feature.integration import (
                read_audio_features,
            )
            from face_pipeline_v2.stages.stage_17_fusion.processor import (
                build_training_sample,
            )
            gaze = read_gaze_features(STAGE13_DIR, CLIP_ID)
            audio = read_audio_features(STAGE14_DIR, CLIP_ID)
            print(f"  gaze={len(gaze)} audio={len(audio)}", flush=True)
            sample = build_training_sample(
                clip_id=CLIP_ID,
                flame_features=feats,
                gaze_features=gaze,
                audio_features=audio,
                fps=25.0,
                visual_dim=112,
                audio_dim=768,
                audio_tolerance=0.04,
                audio_match="nearest_timestamp",
                metadata={"note": "scheme-D preview validation"},
            )
            vf = np.asarray(sample.visual_feature)
            stage17 = {
                "T": int(sample.frame_index.size),
                "valid_length": int(sample.valid_length),
                "visual_feature_shape": list(vf.shape),
                "visual_dim_ok": bool(vf.shape[1] == 112),
                "dropout_count": int(np.asarray(sample.dropout_mask).sum()),
                "pose_block": {
                    "expression": [0, 100],
                    "pose": [100, 106],
                    "gaze": [106, 112],
                },
            }
            # pose block = global[3] + jaw[3] per FLAMEFeature.posecode order
            pose_block = vf[:, 100:106]
            stage17["pose_block_first_frame"] = [float(x) for x in pose_block[0]]
            print(f"  Stage17 T={stage17['T']} valid_length={stage17['valid_length']} "
                  f"vf={list(vf.shape)}", flush=True)
        except Exception as e:
            print("Stage17 build_training_sample FAIL", repr(e), flush=True)
            traceback.print_exc()
            stage17 = {"error": repr(e)}

    # ---- Phase 6: compare with the existing ablation json -------------------
    ablation_cmp = {}
    try:
        abl = json.load(open(str(P / "outputs/stage12_pose_ablation.json")))
        for name, cur in (("A_current", A_metrics),
                          ("D_emica_head_spectre_jaw", D_metrics)):
            ref = abl["schemes"][name]["metrics"]
            diff = {k: round(cur[k] - ref[k], 6) for k in ref}
            ablation_cmp[name] = {"recomputed": cur, "ablation_ref": ref,
                                  "diff": diff,
                                  "max_abs_diff": max(abs(v) for v in diff.values())}
        print("\n=== ablation comparison ===", flush=True)
        for name, c in ablation_cmp.items():
            print(f"  {name}: max_abs_diff={c['max_abs_diff']}", flush=True)
    except Exception as e:
        print("ablation comparison FAIL", repr(e), flush=True)
        ablation_cmp = {"error": repr(e)}
    out = {
        "purpose": ("scheme-D (EMICA head + SPECTRE jaw) preview -- small-scope "
                    "validation; no production code modified"),
        "config": {
            "clip_id": CLIP_ID,
            "n_frames": T_src,
            "frames": [frame_indices[0], frame_indices[-1]],
            "normalization": "0_1 (/255.0, corrected)",
            "scheme_A": "head=EMICA_global; jaw=0.4*EMICA_jaw+0.6*SPECTRE_jaw",
            "scheme_D": "head=EMICA_global; jaw=SPECTRE_jaw; expr=EMICA  (== lambda=1.0)",
            "production_code_modified": False,
            "emica_loaded": emica is not None,
            "spectre_loaded": sp is not None,
        },
        "source_npz_fields": npz_fields,
        "d_vs_stored_spectre_jawpose_max_abs_diff_rad": d_vs_stored_max,
        "scheme_A_metrics": A_metrics,
        "scheme_D_metrics": D_metrics,
        "scheme_D_verbose": D_verb,
        "anomaly_checks": checks,
        "interface": interface,
        "stage17": stage17,
        "ablation_comparison": ablation_cmp,
        "preview_outputs": {
            "D_npz": str(clip_out / "flame_features.npz"),
            "A_npz": str(VAL_DIR / f"{CLIP_ID}_A_preview.npz"),
            "D_jaw_pose_npy": str(clip_out / "D_jaw_pose.npy"),
            "D_head_pose_npy": str(clip_out / "D_head_pose.npy"),
        },
    }
    with open(str(VAL_DIR / "stage12_pose_source_d_preview.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("\nWROTE", str(VAL_DIR / "stage12_pose_source_d_preview.json"), flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()




