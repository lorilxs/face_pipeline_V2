# Stage12 Minimal Inference Check Report

**Date**: 2026-08-25
**Status**: ✅ ALL PASS

---

## 1. EMICA Checkpoint Loading

**Result**: ✅ PASS

EMICA checkpoint successfully loaded from:
`models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/model-epoch=38-val/loss_total=3.88806534.ckpt`

**Loading strategy**: Directly instantiated `DecaEncoder` and `ExpressionEncoder` from config, extracted weights from checkpoint by prefix-matching:
- `face_encoder.mica_deca_encoder.deca_encoder.*` → DecaEncoder (322 keys)
- `face_encoder.expression_encoder.*` → ExpressionEncoder (322 keys)

This bypasses the `MicaEncoder` (which requires `yacs` and MICA model weights), enabling minimal loading.

---

## 2. ExpressionEncoder Output (expcode)

**Expected**: shape [1, 100]

**Result**: ✅ PASS

```
expcode: shape=[1, 100], all_zero=False, mean=-0.034801, std=0.504037
```

---

## 3. EMICA DECA Pose Outputs

**jawpose expected**: shape [1, 3]
**globalpose expected**: shape [1, 3]

**Result**: ✅ PASS

```
jawpose:    shape=[1, 3], all_zero=False, mean=-0.004131, std=0.107528
globalpose: shape=[1, 3], all_zero=False, mean=0.001204,  std=0.441799
```

---

## 4. SPECTRE Pose Output

**Expected**: shape [1, 6]

**Result**: ✅ PASS

```
pose:              shape=[1, 6], all_zero=False, mean=-1.286708, std=1.812726
head_pose ([:3]):  shape=[1, 3], all_zero=False, mean=-1.146430, std=0.915791
jaw_pose  ([3:]):  shape=[1, 3], all_zero=False, mean=-1.426986, std=2.386194
```

**Loading strategy**: Used `ResnetEncoder` from SPECTRE directly, loading `E_flame` weights from `deca_model.tar`. Bypassed `pytorch3d` renderer dependency (not needed for encoder-only inference).

**Note**: SPECTRE's default `PerceptualEncoder` backbone (`mobilenetv2`) requires `torch.hub.load` (GitHub API access). Switched to `resnet50` backbone which uses cached `torchvision` weights.

---

## 5. Additional EMICA Outputs

Other parameters decoded by `DecaEncoder`:

```
cam:       shape=[1, 3],  all_zero=False, mean=6.903592,  std=9.698939
texcode:   shape=[1, 50], all_zero=False, mean=0.303411,  std=11.130993
lightcode: shape=[1, 27], all_zero=False, mean=1.142617,  std=2.431077
```

---

## 6. Conclusion: Ready for Stage12 Wrapper Development

| Check | Status |
|-------|--------|
| EMICA checkpoint loading | ✅ PASS |
| ExpressionEncoder expcode [100] | ✅ PASS |
| EMICA jawpose [3] | ✅ PASS |
| EMICA globalpose [3] | ✅ PASS |
| SPECTRE pose [6] | ✅ PASS |
| SPECTRE head_pose [3] | ✅ PASS |
| SPECTRE jaw_pose [3] | ✅ PASS |

**Verdict**: ✅ **YES — ready to proceed with Stage12 wrapper development.**

### Key Implementation Notes for Stage12 Wrapper

1. **EMICA minimal loading**: Directly instantiate `DecaEncoder` and `ExpressionEncoder` from config, extract weights by checkpoint prefix. This avoids the `MicaEncoder` dependency chain (`yacs`, MICA model weights).

2. **SPECTRE minimal loading**: Use `ResnetEncoder` directly with `resnet50` backbone. Load `E_flame` weights from `deca_model.tar`. The `PerceptualEncoder` and renderer (`pytorch3d`) are not needed for pose-only extraction.

3. **Dependencies required**:
   - `omegaconf` (for EMICA config)
   - `yacs` (for SPECTRE config)
   - `kornia` (for SPECTRE tensor_cropper import, even if unused)
   - `torchvision` (ResNet50 backbone, cached)

4. **No third-party source modifications needed**.