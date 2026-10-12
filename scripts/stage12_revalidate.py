#!/usr/bin/env python3
"""Stage12 pose re-validation (emica_env w/o pytorch_lightning/pytorch3d).
Stubs pytorch_lightning to unpickle the EMICA Lightning ckpt -> extract encoder
state_dict; uses SPECTREWrapper (E_flame, renderer-free); runs neg1_1 + 0_1
norms on real aligned faces; forwards FLAME meshes (Cases 1-4)."""
import sys, os, json, glob, types, traceback
from pathlib import Path
import numpy as np, torch, cv2
from types import SimpleNamespace
P=Path("/workspace/Face_Pipeline_V2"); sys.path.insert(0,str(P))
for _p in [P/"third_party",P/"third_party/inferno/inferno-master",P/"third_party/inferno/spectre"]:
    if str(_p) not in sys.path: sys.path.insert(0,str(_p))
EMICA=str(P/"models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/model-epoch=38-val/loss_total=3.88806534.ckpt")
SPTAR=str(P/"third_party/inferno/spectre/spectre/data/deca_model.tar")
FDIR=str(P/"outputs/stage0_11_test_verify/aligned_faces/CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001")
DEV="cpu"
class _SM(type):
    def __getattr__(cls,n): return _Stub
class _Stub(metaclass=_SM):
    def __init__(self,*a,**k): pass
    def __setstate__(self,s):
        try:
            if isinstance(s,dict): self.__dict__.update(s)
        except Exception: pass
    def __getattr__(self,n): return _Stub()
    def __reduce__(self): return (_Stub,())
import importlib.util, importlib.abc
def _stub_getattr(name):
    if name.startswith("__") and name.endswith("__"): raise AttributeError(name)
    return _Stub
class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        m=types.ModuleType(spec.name); m.__getattr__=_stub_getattr; m.__path__=[]; m.__all__=[]
        return m
    def exec_module(self, module): pass
class _Finder(importlib.abc.MetaPathFinder):
    _P=("pytorch_lightning","torchmetrics","lightning","lightning_fabric","lightning_utilities","chumpy")
    def find_spec(self, fn, path, target=None):
        if any(fn==p or fn.startswith(p+".") for p in self._P):
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
            for k,v in state.items():
                if k!='x':
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
    for nm in ("chumpy","chumpy.ch"):
        m=types.ModuleType(nm); m.Ch=_Ch; m.__path__=[]; m.__getattr__=_stub_getattr
        sys.modules[nm]=m
def prep(im,norm):
    x=cv2.cvtColor(im,cv2.COLOR_BGR2RGB); x=cv2.resize(x,(224,224)).astype(np.float32)
    x=x/255.0 if norm=="0_1" else x/127.5-1.0
    return torch.from_numpy(x).permute(2,0,1).unsqueeze(0)
def st(a):
    a=np.asarray(a).reshape(-1).astype(np.float64)
    return {"mean":float(a.mean()),"std":float(a.std()),"min":float(a.min()),"max":float(a.max()),"norm":float(np.linalg.norm(a)),"n":int(a.size)}
def main():
    files=sorted(glob.glob(os.path.join(FDIR,"aligned_*.jpg")))[:6]
    imgs=[i for i in (cv2.imread(p) for p in files) if i is not None]
    print(f"frames={len(imgs)} dev={DEV}",flush=True)
    install_stubs()
    emica=sp=None
    try:
        from face_pipeline_v2.models.wrappers.emica import EMICAWrapper
        emica=EMICAWrapper(Path(EMICA),device=DEV); emica.load(); print("EMICA loaded",flush=True)
    except Exception as e:
        print("EMICA FAIL",repr(e),flush=True); traceback.print_exc(); emica=None
    try:
        from face_pipeline_v2.models.wrappers.spectre import SPECTREWrapper
        sp=SPECTREWrapper(Path(SPTAR),device=DEV); sp.load(); print("SPECTRE loaded",flush=True)
    except Exception as e:
        print("SPECTRE FAIL",repr(e),flush=True); traceback.print_exc(); sp=None
    res={}; cache={}
    for norm in ["neg1_1","0_1"]:
        b=torch.cat([prep(im,norm) for im in imgs],0).to(DEV); r={"norm":norm}
        if emica:
            with torch.no_grad(): o=emica.encode(b)
            eg=o["globalpose"].cpu().numpy(); ej=o["jawpose"].cpu().numpy(); ex=o["expcode"].cpu().numpy()
            r["emica_globalpose"]=st(eg); r["emica_jawpose"]=st(ej); r["emica_expcode"]=st(ex)
        if sp:
            with torch.no_grad(): so=sp.encode(b)
            sj=so["jaw_pose"].cpu().numpy(); r["spectre_jawpose"]=st(sj)
        if emica and sp:
            fused=0.4*ej+0.6*sj; r["fused_jawpose"]=st(fused); cache[norm]=dict(eg=eg,ej=ej,ex=ex,sj=sj,fused=fused)
        for k,v in list(r.items()):
            if isinstance(v,dict) and "std" in v: print(f"[{norm}] {k}: std={v['std']:.4f} norm={v['norm']:.4f} min={v['min']:.4f} max={v['max']:.4f}",flush=True)
        res[norm]=r
    flame=None
    try:
        from src.models.FLAME import FLAME as FM
        cfg=SimpleNamespace(flame_model_path=str(P/"third_party/inferno/spectre/spectre/data/FLAME2020/generic_model.pkl"),flame_lmk_embedding_path=str(P/"third_party/inferno/spectre/spectre/data/landmark_embedding.npy"),n_shape=300,n_exp=100)
        flame=FM(cfg).to(DEV).eval(); print("FLAME loaded",flush=True)
    except Exception as e:
        print("FLAME FAIL",repr(e),flush=True); traceback.print_exc()
    mesh={}
    if flame and "0_1" in cache and "neg1_1" in cache:
        good=cache["0_1"]; bad=cache["neg1_1"]
        def fwd(name,ex_,gp_,jw_):
            shp=torch.zeros(1,300,device=DEV); exp=torch.from_numpy(ex_[0:1]).float().to(DEV)
            pose=torch.from_numpy(np.concatenate([gp_[0],jw_[0]]).reshape(1,6)).float().to(DEV); ey=torch.zeros(1,6,device=DEV)
            with torch.no_grad(): out=flame(shp,exp,pose,ey)
            v=out[0][0].cpu().numpy()
            m={"jaw_rot_deg":float(np.degrees(np.linalg.norm(jw_[0]))),"bbox_x":float(v[:,0].max()-v[:,0].min()),"bbox_y":float(v[:,1].max()-v[:,1].min()),"bbox_z":float(v[:,2].max()-v[:,2].min()),"y_min":float(v[:,1].min()),"y_max":float(v[:,1].max())}
            op=str(P/f"outputs/stage12_mesh_{name}.obj"); fc=flame.faces_tensor.cpu().numpy()
            with open(op,"w") as fh:
                for p_ in v: fh.write(f"v {p_[0]:.5f} {p_[1]:.5f} {p_[2]:.5f}\n")
                for c in fc: fh.write(f"f {c[0]+1} {c[1]+1} {c[2]+1}\n")
            m["obj"]=op; return m
        Z=np.zeros_like(good["fused"])
        mesh["case1_exp_only"]=fwd("case1_exp_only",good["ex"],Z,Z)
        mesh["case2_exp_headpose"]=fwd("case2_exp_headpose",good["ex"],good["eg"],Z)
        mesh["case3_exp_jawpose"]=fwd("case3_exp_jawpose",good["ex"],Z,good["fused"])
        mesh["case4_full"]=fwd("case4_full",good["ex"],good["eg"],good["fused"])
        mesh["case_BAD_neg1_1_jaw"]=fwd("case_BAD_neg1_1_jaw",good["ex"],good["eg"],bad["fused"])
        for k,v in mesh.items(): print(k,{kk:vv for kk,vv in v.items() if kk!="obj"},flush=True)
    out={"pose_stats":res,"mesh":mesh,"emica_loaded":emica is not None,"spectre_loaded":sp is not None,"flame_loaded":flame is not None}
    with open(str(P/"outputs/stage12_revalidate.json"),"w") as fh: json.dump(out,fh,indent=2)
    print("WROTE",str(P/"outputs/stage12_revalidate.json"),flush=True); print("DONE",flush=True)
if __name__=="__main__": main()
