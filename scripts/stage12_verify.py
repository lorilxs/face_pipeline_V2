"""Stage12 Minimal Inference Verification."""
import sys, os, numpy as np, torch, cv2

EMICA_CKPT = "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/model-epoch=38-val/loss_total=3.88806534.ckpt"
FLAME_PATH = "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/models/FLAME/geometry/generic_model.pkl"
SPECTRE_CKPT = "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/third_party/inferno/spectre/spectre/data/deca_model.tar"
TEST_IMAGE = "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/stage0_11_hdtf_rd_radio1/faces/RD_Radio1_000/RD_Radio1_000_scene_0000_clip_0000/crops/frame_000000_face_000.jpg"
PROJECT_ROOT = "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2"
INFERNO_ROOT = os.path.join(PROJECT_ROOT, "third_party/inferno/inferno-master")
SPECTRE_ROOT = os.path.join(PROJECT_ROOT, "third_party/inferno/spectre")
sys.path.insert(0, INFERNO_ROOT)
sys.path.insert(0, SPECTRE_ROOT)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}, PyTorch: {torch.__version__}")

def load_image(path, target_size=(224,224)):
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size[::-1])
    t = torch.from_numpy(img.astype(np.float32)).permute(2,0,1) / 255.0
    return t.unsqueeze(0)

def pti(name, tensor):
    arr = tensor.detach().cpu().numpy()
    az = bool(np.allclose(arr, 0.0))
    print(f"  {name}: shape={list(tensor.shape)}, all_zero={az}, mean={arr.mean():.6f}, std={arr.std():.6f}")

# ======================================================================
# 1. EMICA Verification
# ======================================================================
print("\n" + "="*70)
print("EMICA VERIFICATION")
print("="*70)
emica_ok = False

try:
    from omegaconf import OmegaConf
    from inferno.models.FaceReconstruction.FaceRecBase import FaceReconstructionBase

    cfg = OmegaConf.create({
        "model": {
            "face_encoder": {
                "type": "EmicaEncoder",
                "trainable": False,
                "predicts": {},
                "encoders": {
                    "expression_encoder": {
                        "type": "ExpressionEncoder",
                        "backbone": "ResNet50",
                        "input_size": 224,
                        "predicts": {"expcode": 100},
                        "trainable": False,
                        "last_layer_init_zero": True,
                    },
                    "mica_deca_encoder": {
                        "type": "MicaDecaEncoder",
                        "trainable": False,
                        "predicts": {},
                        "encoders": {
                            "deca_encoder": {
                                "type": "DecaEncoder",
                                "backbone": "ResNet50",
                                "input_size": 224,
                                "predicts": {
                                    "texcode": 50,
                                    "jawpose": 3,
                                    "globalpose": 3,
                                    "cam": 3,
                                    "lightcode": 27,
                                },
                                "trainable": False,
                                "last_layer_init_zero": True,
                            },
                            "mica_encoder": {
                                "type": "MicaEncoder",
                                "backbone": "ResNet50",
                                "mica_model_path": "MICA/model/mica.tar",
                                "mica_preprocessing": "fan",
                                "input_size": 224,
                                "predicts": {"shape": 300},
                                "trainable": False,
                            },
                        },
                    },
                },
            },
            "shape_model": {
                "type": "FlameShapeModel",
                "flame": {
                    "type": "flame_mediapipe",
                    "flame_model_path": FLAME_PATH,
                    "n_shape": 300,
                    "n_exp": 100,
                    "n_tex": 50,
                },
                "use_texture": False,
            },
            "renderer": None,
        },
    })

    print("Loading EMICA from checkpoint...")
    model = FaceReconstructionBase.instantiate(
        cfg=cfg, checkpoint=EMICA_CKPT, strict=False,
        checkpoint_kwargs={"map_location": DEVICE},
    )
    model = model.to(DEVICE)
    model.eval()
    print("EMICA loaded successfully.")

    img_tensor = load_image(TEST_IMAGE).to(DEVICE)
    print(f"Input image tensor: {img_tensor.shape}")

    with torch.no_grad():
        batch = {"image": img_tensor}
        batch = model.face_encoder.encode(batch, return_features=False)

    print("\nEMICA Encoder outputs:")
    for key in sorted(batch.keys()):
        if isinstance(batch[key], torch.Tensor):
            pti(key, batch[key])
        elif isinstance(batch[key], dict):
            print(f"  {key}: dict with keys {list(batch[key].keys())}")
        else:
            print(f"  {key}: {type(batch[key])}")

    checks = []
    if "expcode" in batch:
        checks.append(("expcode shape [1,100]", batch["expcode"].shape == (1, 100)))
    if "jawpose" in batch:
        checks.append(("jawpose shape [1,3]", batch["jawpose"].shape == (1, 3)))
    if "globalpose" in batch:
        checks.append(("globalpose shape [1,3]", batch["globalpose"].shape == (1, 3)))

    print("\nEMICA shape checks:")
    all_ok = True
    for check_name, result in checks:
        status = "PASS" if result else "FAIL"
        if not result: all_ok = False
        print(f"  [{status}] {check_name}")
    emica_ok = all_ok

    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    print("\nEMICA verification: PASS" if emica_ok else "\nEMICA verification: FAIL")

except Exception as e:
    print(f"EMICA verification FAILED: {e}")
    import traceback; traceback.print_exc()

# ======================================================================
# 2. SPECTRE Verification
# ======================================================================
print("\n" + "="*70)
print("SPECTRE VERIFICATION")
print("="*70)
spectre_ok = False

try:
    from spectre.src.spectre import SPECTRE
    from spectre.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.pretrained_modelpath = SPECTRE_CKPT
    cfg.model.topology_path = os.path.join(SPECTRE_ROOT, "spectre/data/head_template.obj")
    cfg.model.flame_model_path = os.path.join(SPECTRE_ROOT, "spectre/data/FLAME2020/generic_model.pkl")
    cfg.model.flame_lmk_embedding_path = os.path.join(SPECTRE_ROOT, "spectre/data/landmark_embedding.npy")
    cfg.model.face_mask_path = os.path.join(SPECTRE_ROOT, "spectre/data/uv_face_mask.png")
    cfg.model.face_eye_mask_path = os.path.join(SPECTRE_ROOT, "spectre/data/uv_face_eye_mask.png")
    cfg.model.mean_tex_path = os.path.join(SPECTRE_ROOT, "spectre/data/mean_texture.jpg")
    cfg.model.fixed_displacement_path = os.path.join(SPECTRE_ROOT, "spectre/data/fixed_displacement_256.npy")
    cfg.model.dense_template_path = os.path.join(SPECTRE_ROOT, "spectre/data/texture_data_256.npy")

    print("Loading SPECTRE from checkpoint...")
    spectre = SPECTRE(config=cfg, device=DEVICE)
    spectre.eval()

    checkpoint = torch.load(SPECTRE_CKPT, map_location=DEVICE)
    if "E_flame" in checkpoint:
        spectre.E_flame.load_state_dict(checkpoint["E_flame"], strict=True)
        print("SPECTRE E_flame loaded successfully.")
    else:
        print("WARNING: No E_flame key in checkpoint. Keys:", list(checkpoint.keys())[:5])

    img_tensor = load_image(TEST_IMAGE).to(DEVICE)
    print(f"Input image tensor: {img_tensor.shape}")

    with torch.no_grad():
        codedict = spectre.encode({"image": img_tensor})

    print("\nSPECTRE Encoder outputs:")
    for key in sorted(codedict.keys()):
        if isinstance(codedict[key], torch.Tensor):
            pti(key, codedict[key])
        elif isinstance(codedict[key], list):
            print(f"  {key}: list of {len(codedict[key])} items")
        else:
            print(f"  {key}: {type(codedict[key])}")

    if "pose" in codedict:
        pose = codedict["pose"]
        pti("pose", pose)
        if pose.shape[-1] == 6:
            head_pose = pose[..., :3]
            jaw_pose = pose[..., 3:]
            pti("head_pose (pose[:3])", head_pose)
            pti("jaw_pose (pose[3:])", jaw_pose)
            print(f"\n  [PASS] pose shape = {list(pose.shape)}, expected [...6]")
            spectre_ok = True
        else:
            print(f"\n  [FAIL] pose shape = {list(pose.shape)}, expected [...6]")
    else:
        print("\n  [FAIL] 'pose' not found in codedict")

    del spectre
    if DEVICE == "cuda": torch.cuda.empty_cache()
    print("\nSPECTRE verification: PASS" if spectre_ok else "\nSPECTRE verification: FAIL")

except Exception as e:
    print(f"SPECTRE verification FAILED: {e}")
    import traceback; traceback.print_exc()


# ======================================================================
# Summary
# ======================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  EMICA  : {'PASS' if emica_ok else 'FAIL'}")
print(f"  SPECTRE: {'PASS' if spectre_ok else 'FAIL'}")
print(f"  Ready for Stage12: {'YES' if emica_ok and spectre_ok else 'NO'}")
