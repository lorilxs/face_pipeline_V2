#!/usr/bin/env python3
"""Stage12 FLAME pose-source ablation - parameter-level comparison only.

Compares three head/jaw pose-source schemes on the identical frames the
Stage12 npz (frame_index 100-149) covers:

  A (current) : head = EMICA global;      jaw = 0.4*EMICA_jaw + 0.6*SPECTRE_jaw
  B (SPECTRE) : head = SPECTRE global;    jaw = SPECTRE_jaw
  C (EMICA)   : head = EMICA global;      jaw = EMICA_jaw
  D (candidate): head = EMICA global;     jaw = SPECTRE_jaw  (expr = EMICA)

Runs EMICA + SPECTRE (E_flame encoder only) on the aligned faces with the
CORRECTED /255.0 normalization. The on-disk flame_features.npz is a pre-fix
(/127.5-1.0) artifact and does NOT store global_pose / emica_jaw_pose, so the
ablation recomputes every pose component on the exact same frames (see Phase 1
data-check in the report). No FLAME decoder, no mesh, no pytorch3d.

Stubs pytorch_lightning / torchmetrics / lightning / chumpy so the EMICA
Lightning ckpt can be unpickled for its encoder state_dict.

Outputs: outputs/stage12_pose_ablation.json + printed tables.
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
NPZ = str(P / f"outputs/stage12_flame/{CLIP_ID}/flame_features.npz")
ALIGN_DIR = str(P / f"outputs/stage0_11_test_verify/aligned_faces/{CLIP_ID}")
JAW_LAMBDA = 0.6          # SPECTRE weight in the current jaw fusion
DEV = "cpu"

# ---------------------------------------------------------------------------
# pytorch_lightning / chumpy stubs (identical to stage12_revalidate.py)
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
# Image prep / frame loading
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
        m = re.match(r"aligned_(\d{6})", base)   # 6-digit leading frame index
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
# Statistics
# ---------------------------------------------------------------------------
def _deg(x): return float(np.degrees(x))

def rot_stats(a, key):
    """Rotation magnitude: axis-angle norm (radians) per frame."""
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
    """Temporal stability: frame_delta = pose[t]-pose[t-1]; speed = ||delta||."""
    d = np.diff(a, axis=0)                     # (T-1, 3)
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
    """Neutral bias: first-frame + low-mouth-motion frames (bottom quartile of
    jaw speed). A jaw at rest should sit near 0; a large low-motion mean
    indicates a baseline offset."""
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    install_stubs()

    # ---- Phase 1: data check of the Stage12 npz ---------------------------
    d = np.load(NPZ, allow_pickle=True)
    npz_fields = {k: (list(np.asarray(d[k]).shape), str(np.asarray(d[k]).dtype))
                  for k in d.files}
    present = {
        "expression": "expression_code" in d.files,
        "global_pose": "global_pose" in d.files,          # NOT stored by Stage12
        "head_pose": "head_pose" in d.files,              # == EMICA global (stored)
        "jaw_pose": "jaw_pose" in d.files,                # == fused (stored)
        "spectre_jaw_pose": "spectre_jawpose" in d.files, # stored
        "emica_jaw_pose": "emica_jawpose" in d.files,     # NOT stored by Stage12
    }
    missing = [k for k, v in present.items() if not v]
    frame_indices = [int(x) for x in np.asarray(d["frame_index"]).reshape(-1)]
    print("Phase1 npz fields:", json.dumps(npz_fields, indent=2), flush=True)
    print("Phase1 present:", json.dumps(present), "missing:", missing, flush=True)

    # reference: Case A as-stored in the (stale, pre-fix) npz
    npz_head = np.asarray(d["head_pose"], np.float64)
    npz_jaw = np.asarray(d["jaw_pose"], np.float64)
    refA = {}
    refA.update(rot_stats(npz_head, "head"))
    refA.update(rot_stats(npz_jaw, "jaw"))
    refA["note"] = "stale npz as-stored (pre-fix /127.5-1.0); reference only"

    # ---- Phase 2: encode on the exact npz frames, corrected /255.0 --------
    imgs, used = load_frames(frame_indices)
    print(f"Phase2: npz frames={len(frame_indices)} aligned_loaded={len(imgs)}",
          flush=True)
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
        eg = np.asarray(eo["globalpose"].cpu().numpy(), np.float64)  # [T,3] EMICA global
        ej = np.asarray(eo["jawpose"].cpu().numpy(), np.float64)     # [T,3] EMICA jaw
        so = sp.encode(b)
        sg = np.asarray(so["head_pose"].cpu().numpy(), np.float64)   # [T,3] SPECTRE global
        sj = np.asarray(so["jaw_pose"].cpu().numpy(), np.float64)    # [T,3] SPECTRE jaw
    T = eg.shape[0]
    print(f"Phase2: encoded T={T} eg={eg.shape} ej={ej.shape} sg={sg.shape} sj={sj.shape}",
          flush=True)

    schemes = {
        "A_current": {"head": eg, "jaw": (1.0 - JAW_LAMBDA) * ej + JAW_LAMBDA * sj},
        "B_spectre": {"head": sg, "jaw": sj},
        "C_emica":   {"head": eg, "jaw": ej},
        "D_emica_head_spectre_jaw": {"head": eg, "jaw": sj},
    }

    out = {
        "phase1_data_check": {
            "npz": NPZ,
            "npz_fields": npz_fields,
            "present": present,
            "missing": missing,
            "note": ("global_pose and emica_jaw_pose are not stored by the Stage12 "
                     "output; pipeline left unmodified, gap recorded only."),
        },
        "config": {
            "clip_id": CLIP_ID,
            "n_frames": T,
            "frames": used,
            "normalization": "0_1 (/255.0, corrected)",
            "jaw_fusion_lambda": JAW_LAMBDA,
            "schemes": {
                "A_current": "head=EMICA_global; jaw=0.4*EMICA_jaw+0.6*SPECTRE_jaw",
                "B_spectre": "head=SPECTRE_global; jaw=SPECTRE_jaw",
                "C_emica": "head=EMICA_global; jaw=EMICA_jaw",
                "D_emica_head_spectre_jaw": "head=EMICA_global; jaw=SPECTRE_jaw",
            },
            "expression_source": "EMICA (shared by all schemes; not pose-dependent)",
            "emica_loaded": emica is not None,
            "spectre_loaded": sp is not None,
            "flame_decoder": False,   # explicitly NOT used (parameter-only)
        },
        "schemes": {},
    }
    for name, sc in schemes.items():
        s = {}
        s.update(rot_stats(sc["head"], "head"))
        s.update(rot_stats(sc["jaw"], "jaw"))
        s.update(temporal_stats(sc["head"], "head"))
        s.update(temporal_stats(sc["jaw"], "jaw"))
        s.update(neutral_bias(sc["jaw"], "jaw"))
        h_ang = np.linalg.norm(sc["head"], axis=1)
        j_ang = np.linalg.norm(sc["jaw"], axis=1)
        s["pearson_head_jaw_angle"] = pearson(h_ang, j_ang)
        s["pearson_head_jaw_angle_note"] = (
            "Pearson(head_angle, jaw_angle); |r| near 0 = de-coupled")
        # AISDD-spec 7-field metric block
        s["metrics"] = {
            "head_angle_mean": s["head_angle_mean_deg"],
            "head_angle_std": s["head_angle_std_deg"],
            "jaw_angle_mean": s["jaw_angle_mean_deg"],
            "jaw_angle_std": s["jaw_angle_std_deg"],
            "jaw_rest_bias": s["jaw_low_motion_mean_deg"],
            "jaw_velocity": s["jaw_mean_vel_deg"],
            "head_jaw_correlation": s["pearson_head_jaw_angle"],
        }
        out["schemes"][name] = s
        print(f"\n===== {name} =====", flush=True)
        for k, v in s["metrics"].items():
            print(f"  {k:24s} {v:.6f}", flush=True)
        print("  --- verbose ---", flush=True)
        for k, v in s.items():
            if k == "metrics":
                continue
            if isinstance(v, float):
                print(f"  {k:34s} {v:.6f}", flush=True)
            else:
                print(f"  {k:34s} {v}", flush=True)
    out["reference_npz_as_stored"] = refA

    with open(str(P / "outputs/stage12_pose_ablation.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("\nWROTE", str(P / "outputs/stage12_pose_ablation.json"), flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()

