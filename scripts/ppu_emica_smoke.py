"""PPU EMICA/SPECTRE smoke test — runs the real Stage12 EMICA path on PPU (cuda).

Mirrors /tmp/emica_smoke3.py but drives the models onto the PPU device via
device="cuda" and the new py3.12 PPU torch venv (emica_ppu_env).

Usage (inside face_pipeline_lxs container):
    /workspace/emica_ppu_env/bin/python \
        /workspace/Face_Pipeline_V2/scripts/ppu_emica_smoke.py
"""
import sys
import time
from pathlib import Path

P = Path("/workspace/Face_Pipeline_V2")
sys.path.insert(0, str(P))
for _p in (
    P / "third_party",
    P / "third_party/inferno/inferno-master",
    P / "third_party/inferno/spectre",
):
    sys.path.insert(0, str(_p))

import importlib.util

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "r12", str(P / "face_pipeline_v2/stages/stage_12_flame/runner.py")
)
r12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r12)

emica_ckpt = str(
    P
    / "models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/"
    "model-epoch=38-val/loss_total=3.88806534.ckpt"
)
spectre_ckpt = str(P / "third_party/inferno/spectre/spectre/data/deca_model.tar")

print(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()} "
      f"| device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}",
      flush=True)
assert torch.cuda.is_available(), "PPU CUDA not available!"

print("loading EMICA...", flush=True)
t0 = time.time()
em = r12._load_emica(emica_ckpt, "cuda")
print(f"EMICA loaded in {time.time() - t0:.1f}s", flush=True)

print("loading SPECTRE...", flush=True)
sp = r12._load_spectre(spectre_ckpt, "cuda")
print("SPECTRE loaded", flush=True)

img = np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8)
t = r12.preprocess_image(img, (224, 224))
bt = torch.cat([t, t, t], dim=0).to("cuda")
with torch.no_grad():
    eo = r12.emica_encode(em, bt)
    so = r12.spectre_encode(sp, bt)

print("emica keys", {k: tuple(v.shape) for k, v in eo.items()}, flush=True)
print("spectre keys", {k: tuple(v.shape) for k, v in so.items()}, flush=True)
print("SMOKE TEST OK", flush=True)
