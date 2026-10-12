import os, glob, re
import numpy as np

BASE = "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/chinese_videos_flame_audio_new_pipeline_0915"
FPS = 25

# Talkvid filename: {id}_{idx}_{start}_{end}_scene_0000_clip_0000
pat = re.compile(r"_(scene_\d+_clip_\d+)\.npy$")

def parse_ts(name):
    # name like: CPlu9SzgyQ0_9_447.93_455.48_scene_0000_clip_0000
    m = re.search(r"_(\d+\.\d+)_(\d+\.\d+)_scene", name)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))

rows = []
for folder in sorted(os.listdir(BASE)):
    fp = os.path.join(BASE, folder)
    if not os.path.isdir(fp):
        continue
    for npy in glob.glob(os.path.join(fp, "flame", "*.npy")):
        name = os.path.splitext(os.path.basename(npy))[0]
        s, e = parse_ts(name)
        if s is None:
            continue
        flame = np.load(npy, mmap_mode="r").shape[0]
        # candidate frame counts from time boundaries
        cand = {
            "floor_floor": int(np.floor(e*FPS)) - int(np.floor(s*FPS)),
            "round_round": int(round(e*FPS)) - int(round(s*FPS)),
            "floor_ceil_excl": int(np.ceil(e*FPS)) - int(np.floor(s*FPS)),
            "ceil_floor": int(np.floor(e*FPS)) - int(np.ceil(s*FPS)),
            "floor_incl": int(np.floor(e*FPS)) - int(np.floor(s*FPS)) + 1,
            "ceil_incl": int(np.ceil(e*FPS)) - int(np.floor(s*FPS)) + 1,
            "dur25": round((e-s)*FPS),
        }
        rows.append((folder, name, s, e, flame, cand))

# For each candidate, count how many it exactly matches flame count
from collections import Counter
match_counts = Counter()
mismatch_examples = []
for folder, name, s, e, flame, cand in rows:
    hit = [k for k,v in cand.items() if v == flame]
    for k in hit:
        match_counts[k] += 1
    if not hit:
        mismatch_examples.append((name, s, e, flame, cand))

print("total talkvid clips with time bounds:", len(rows))
print("\n==== how many clips each formula EXACTLY matches flame count ====")
for k, v in match_counts.most_common():
    print(f"  {k:16s}: {v}  ({v*100/len(rows):.1f}%)")

print("\n==== examples NOT matched by any candidate (up to 15) ====")
for name, s, e, flame, cand in mismatch_examples[:15]:
    print(f"  {name}: start={s} end={e} flame={flame} cand={cand}")