# Stage12 Real Data Validation Report

## 1. Input

- Input directory: `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/stage0_11_test_verify/aligned_faces/CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001`
- Total frames available: 200
- Frames tested: 50

## 2. Model

- EMICA checkpoint: `models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/model-epoch=38-val/loss_total=3.88806534.ckpt`
- SPECTRE checkpoint: `third_party/inferno/spectre/spectre/data/deca_model.tar`
- jaw fusion lambda: 0.6

## 3. Expression

- Dimension: 100
- mean: -0.086734
- std: 0.675291
- min: -2.399608
- max: 2.893075
- NaN: False
- Inf: False
- All zero: False

## 4. Head Pose

- Source: EMICA globalpose
- Dimension: 3
- mean: -0.089257
- std: 0.754977
- min: -0.807508
- max: 0.976821

## 5. Jaw Pose

### EMICA jaw

- Dimension: 3
- mean: -0.079871
- std: 0.321477

### SPECTRE jaw

- Dimension: 3
- mean: -2.472445
- std: 6.273046

### fused jaw

- Dimension: 3
- mean: -1.515415
- std: 3.738011

## 6. Fusion Verification

Randomly selected 5 frames:

| frame_id | emica_jaw | spectre_jaw | fused_jaw | expected | match |
|----------|-----------|-------------|-----------|----------|-------|
| 13 | [-0.4165, 0.3531, -0.1755] | [-6.9769, -6.9660, 6.4237] | [-4.3527, -4.0384, 3.7840] | [-4.3527, -4.0384, 3.7840] | ✓ |
| 17 | [-0.4142, 0.3508, -0.1740] | [-6.8554, -6.8693, 6.3569] | [-4.2789, -3.9812, 3.7446] | [-4.2789, -3.9812, 3.7446] | ✓ |
| 30 | [-0.4233, 0.3619, -0.1851] | [-7.1389, -7.0735, 6.5436] | [-4.4527, -4.0993, 3.8522] | [-4.4527, -4.0993, 3.8522] | ✓ |
| 39 | [-0.4163, 0.3583, -0.1859] | [-6.7875, -6.8929, 6.3936] | [-4.2390, -3.9924, 3.7618] | [-4.2390, -3.9924, 3.7618] | ✓ |
| 45 | [-0.4134, 0.3529, -0.1754] | [-6.8114, -6.8856, 6.3594] | [-4.2522, -3.9902, 3.7455] | [-4.2522, -3.9902, 3.7455] | ✓ |

## 7. FLAMEFeature

- Total features: 50
- Dimension errors: 0
- expression_code: 100
- jaw_pose: 3
- head_pose (globalpose): 3
- eye_pose: 6
- spectre_jawpose: 3
- shapecode: 300

## 8. Failures

No failures encountered.
