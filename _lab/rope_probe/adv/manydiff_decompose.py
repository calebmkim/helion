"""Attribute the manydiff_chains (many-live-temporaries) win: how much comes from the register-aware
TILE SHRINK ([1,2048]->[1,512]) vs the NUM_WARPS ramp (4->8/16)? Matched-lever A/B."""
from __future__ import annotations
import sys
_WT="/home/calebkim/helion-new-heuristics/helion-pointwise"; sys.path.insert(0,_WT)
sys.path.insert(0,"/home/calebkim/helion-new-heuristics/local/rope_probe/adv")
import torch, helion
from oracle_vs_tc import load, tc_ref, bench

def run(M,N):
    fn,mk=load("manydiff_chains"); args=tuple(mk((M,N)))
    bk=fn.bind(args)
    dref=bk.compile_config(bk.config_spec.default_config())(*args)
    d_ms=bench(bk.compile_config(bk.config_spec.default_config()),args)
    torch._dynamo.reset()
    cf=torch.compile(tc_ref("manydiff_chains"),mode="max-autotune-no-cudagraphs"); cf(*args); torch.cuda.synchronize()
    tc_ms=bench(cf,args)
    print(f"\n### manydiff_chains [{M},{N}]   default[32,32]={d_ms*1000:.1f}us  tc={tc_ms*1000:.1f}us")
    print(f"    {'config':16s} {'us':>8s} {'G=tc/x':>8s} {'vs default':>11s}")
    grid={}
    for bs in [[1,2048],[1,1024],[1,512],[1,256]]:
        for w in [4,8,16]:
            try:
                r=bk.compile_config(helion.Config(block_sizes=bs,num_warps=w)); o=r(*args)
                ok=torch.allclose(o.float(),dref.float(),atol=2e-2,rtol=2e-2)
                ms=bench(r,args,n=7); grid[(tuple(bs),w)]=ms
                print(f"    {str(bs)+' w'+str(w):16s} {ms*1000:8.1f} {tc_ms/ms:8.2f} {d_ms/ms:10.2f}x  ok={ok}")
            except Exception as e:
                print(f"    {str(bs)+' w'+str(w):16s}  FAIL {type(e).__name__}")
    s=grid.get(((1,2048),4));
    if s:
        wonly=grid.get(((1,2048),8)); shonly=grid.get(((1,512),4)); both=grid.get(((1,512),8))
        print(f"    -- seed[1,2048]w4={s*1000:.1f}us --")
        if wonly: print(f"    WARPS-only  [1,2048]w4->w8 : {s/wonly:.2f}x faster")
        if shonly:print(f"    SHRINK-only [1,2048]->[1,512] w4 : {s/shonly:.2f}x faster")
        if both:  print(f"    BOTH        [1,512]w8 : {s/both:.2f}x faster (oracle)")

for M,N in [(4096,4096),(8192,8192)]:
    run(M,N)
