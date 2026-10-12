"""PPU EMICA encode throughput benchmark (realistic single-frame batches)."""
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

em = r12._load_emica(
    str(P / "models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/"
        "model-epoch=38-val/loss_total=3.88806534.ckpt"),
    "cuda",
)

img = np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8)
t = r12.preprocess_image(img, (224, 224))

# warmup (first call triggers PPU kernel compile / runtime init)
_bt = t.to("cuda")
with torch.no_grad():
    r12.emica_encode(em, _bt)

# single-frame batches (realistic stage12 usage)
N = 100
t0 = time.time()
with torch.no_grad():
    for _ in range(N):
        r12.emica_encode(em, _bt)
dt1 = time.time() - t0
print(f"PPU EMICA encode batch=1: {N} frames in {dt1:.2f}s -> {N / dt1:.2f} fps "
      f"({dt1 / N * 1000:.1f} ms/frame)", flush=True)

# batch=8
_bt8 = torch.cat([t] * 8, dim=0).to("cuda")
with torch.no_grad():
    r12.emica_encode(em, _bt8)
M = 25
t0 = time.time()
with torch.no_grad():
    for _ in range(M):
        r12.emica_encode(em, _bt8)
dt8 = time.time() - t0
f8 = M * 8
print(f"PPU EMICA encode batch=8: {f8} frames in {dt8:.2f}s -> {f8 / dt8:.2f} fps "
      f"({dt8 / f8 * 1000:.1f} ms/frame)", flush=True)
print("BENCH DONE", flush=True)
