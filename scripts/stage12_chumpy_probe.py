import sys, types, numpy as np, pickle
class Ch(np.ndarray):
    def __new__(cls,*a,**k): return super().__new__(cls,(0,))
    def __array_finalize__(self,obj): pass
    def __setstate__(self,state):
        print("STATE type",type(state).__name__,"len",len(state) if hasattr(state,'__len__') else "NA",flush=True)
        if isinstance(state,dict): print("  dict keys",list(state.keys())[:30],flush=True)
        elif isinstance(state,tuple): print("  tuple item types",[type(x).__name__ for x in state],flush=True)
m=types.ModuleType("chumpy"); m.Ch=Ch; m.__path__=[]; sys.modules["chumpy"]=m
m2=types.ModuleType("chumpy.ch"); m2.Ch=Ch; m2.__path__=[]; sys.modules["chumpy.ch"]=m2
p="/workspace/Face_Pipeline_V2/third_party/inferno/spectre/spectre/data/FLAME2020/generic_model.pkl"
with open(p,"rb") as f: ss=pickle.load(f,encoding="latin1")
print("TOP",type(ss).__name__, list(ss.keys()) if isinstance(ss,dict) else "n/a",flush=True)
print("DONE",flush=True)
