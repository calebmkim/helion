"""Verify the high-fan-in regression: is my budget-capped floor ([1,16]) worse than the
original hardcoded floor ([1,256]) and the default ([32,32])? Sweep tiles for fanin64."""
from __future__ import annotations
import sys
_WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"; sys.path.insert(0, _WT)
import torch, helion
import helion.language as hl
from helion.autotuner.benchmarking import do_bench

N_IN = 64
src = "@helion.kernel()\ndef fanin(" + ",".join(f"x{i}" for i in range(N_IN)) + "):\n"
src += "    out=torch.empty_like(x0)\n    for tm,tn in hl.tile(x0.size()):\n"
src += "        acc=" + "+".join(f"x{i}[tm,tn].to(torch.float32)" for i in range(N_IN)) + "\n"
src += "        out[tm,tn]=acc.to(x0.dtype)\n    return out\n"
ns={"torch":torch,"helion":helion,"hl":hl}
import os
os.makedirs("/tmp/fanin_gen",exist_ok=True); open("/tmp/fanin_gen/fk.py","w").write(
    "from __future__ import annotations\nimport torch, helion\nimport helion.language as hl\n"+src)
import importlib.util
sp=importlib.util.spec_from_file_location("fk","/tmp/fanin_gen/fk.py"); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
fn=m.fanin

M,N=4096,4096
args=tuple(torch.randn(M,N,device="cuda",dtype=torch.bfloat16) for _ in range(N_IN))
bk=fn.bind(args)
ref=bk.compile_config(helion.Config(block_sizes=[32,32]))(*args)
for label,bs in [("default",[32,32]),("mine[1,16]",[1,16]),("[1,64]",[1,64]),
                 ("[1,128]",[1,128]),("orig[1,256]",[1,256]),("[1,512]",[1,512]),
                 ("[16,64]",[16,64]),("[8,128]",[8,128])]:
    try:
        run=bk.compile_config(helion.Config(block_sizes=bs)); o=run(*args)
        ok=torch.allclose(o.float(),ref.float(),atol=2e-2,rtol=2e-2)
        ms=do_bench(lambda: run(*args), return_mode="median")
        print(f"  {label:12s} {str(bs):10s} {ms*1000:7.1f}us  correct={ok}")
    except Exception as e:
        print(f"  {label:12s} {str(bs):10s} FAIL {type(e).__name__}: {str(e)[:80]}")
