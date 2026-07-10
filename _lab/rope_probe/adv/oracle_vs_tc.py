"""For one adversarial kernel+shape: measure helion default / seed / ORACLE (targeted config
sweep) vs torch.compile(max-autotune). Answers: can the Helion oracle reach tc, or is tc a
codegen ceiling (weird shape -> retarget floor to 0.75x oracle)?

argv: kernel_name  M N
"""
from __future__ import annotations
import sys, json, importlib.util, os
_WT="/home/calebkim/helion-new-heuristics/helion-pointwise"; sys.path.insert(0,_WT)
import torch, helion
import helion.language as hl
assert helion.__file__.startswith(_WT), helion.__file__
from helion._compiler.autotuner_heuristics.triton import TritonPointwiseSeedHeuristic
from helion.autotuner.benchmarking import do_bench

SHARDS="/home/calebkim/helion-new-heuristics/local/rope_probe/adv/shards/"
KMAP={"transposed_both_scale":"shard_transpose.json","transposed_in_relu2":"shard_transpose.json",
      "transposed_out_add":"shard_transpose.json",
      "manydiff_chains":"shard_temporaries.json","wide_row_trig_chain":"shard_warps.json",
      "wide_row_moderate_chain":"shard_warps.json","skinny_m_wide_n_chain":"shard_warps.json",
      "fanin64_2d_add":"shard_fanin.json","fanin32_2d_silu":"shard_fanin.json",
      "fanin96_bf16_2d":"shard_fanin.json","bcast_rowvec_transcend":"shard_broadcast.json"}

_S2PI=0.7978845608028654
def tc_ref(name):
    if name=="transposed_both_scale":
        return lambda xT,s: xT*s
    if name=="transposed_in_relu2":
        def f(xT):
            v=torch.relu(xT.to(torch.float32)); return (v*v).to(xT.dtype)
        return f
    if name=="transposed_out_add":
        return lambda x,y: (x.to(torch.float32)+y.to(torch.float32)).to(x.dtype)
    if name=="manydiff_chains":
        def f(x):
            v=x.to(torch.float32)
            c1=torch.sin(v*1.1);c2=torch.cos(v*1.2);c3=torch.tanh(v*1.3);c4=torch.sigmoid(v*1.4)
            c5=torch.exp(v*0.15);c6=torch.log(torch.abs(v)*1.6+1.0);c7=torch.rsqrt(v*v+1.7)
            c8=torch.sin(v*1.8+c1);c9=torch.cos(v*1.9+c2);c10=torch.tanh(v*2.0+c3);c11=torch.sigmoid(v*2.1+c4)
            c12=torch.exp(v*0.22+c5*0.1);c13=torch.log(torch.abs(v*2.3+c6)+1.0);c14=torch.rsqrt(c7*c7+2.4)
            c15=torch.sin(c8*2.5+c9);c16=torch.cos(c10*2.6+c11);c17=torch.tanh(c12*0.27+c13)
            c18=torch.sigmoid(c14*2.8+c15);c19=torch.exp(c16*0.29+c17*0.1);c20=torch.log(torch.abs(c18*3.0+c19)+1.0)
            acc=c1+c2+c3+c4+c5+c6+c7+c8+c9+c10+c11+c12+c13+c14+c15+c16+c17+c18+c19+c20
            return (0.05*acc).to(x.dtype)
        return f
    if name=="wide_row_trig_chain":
        def f(x):
            v=x.to(torch.float32)
            v=torch.sin(v)+torch.cos(v*1.1); v=torch.tanh(v*0.9)+torch.exp(-torch.abs(v)*0.3)
            v=torch.sin(v*1.2)+torch.cos(v); v=torch.tanh(v)+torch.exp(-torch.abs(v)*0.4)
            v=torch.sin(v)*torch.cos(v*0.7)+v; v=torch.log(1.0+torch.abs(v))+torch.sqrt(torch.abs(v)+1e-3)
            return v.to(x.dtype)
        return f
    if name=="fanin64_2d_add":
        return lambda *xs: (sum(x.to(torch.float32) for x in xs)*0.015625).to(xs[0].dtype)
    if name=="fanin96_bf16_2d":
        return lambda *xs: (sum(x.to(torch.float32) for x in xs)*0.010416666666666666).to(xs[0].dtype)
    if name=="fanin32_2d_silu":
        def f(*xs):
            s=sum(x.to(torch.float32)*(0.50+0.01*i) for i,x in enumerate(xs))*0.031250
            return (s*torch.sigmoid(s)).to(xs[0].dtype)
        return f
    if name=="wide_row_moderate_chain":
        def f(x):
            v=x.to(torch.float32); v=torch.sin(v)+torch.cos(v*1.1); v=torch.tanh(v*0.9)+torch.exp(-torch.abs(v)*0.3)
            return v.to(x.dtype)
        return f
    if name=="skinny_m_wide_n_chain":
        def f(x):
            v=x.to(torch.float32)
            v=torch.sin(v*1.3)+torch.cos(v); v=torch.exp(-torch.abs(v)*0.35)+torch.tanh(v*0.8); v=torch.sin(v)+torch.cos(v*1.15)
            v=torch.tanh(v*1.1)+torch.exp(-v*v*0.2); v=torch.log(1.0+torch.abs(v))*torch.sin(v); v=torch.sqrt(torch.abs(v)+1e-3)+torch.cos(v*0.9)
            return v.to(x.dtype)
        return f
    if name=="bcast_rowvec_transcend":
        def f(x,r0,r1,r2,r3):
            v=x.to(torch.float32)
            a0=r0[:,None].to(torch.float32); a1=r1[:,None].to(torch.float32); a2=r2[:,None].to(torch.float32); a3=r3[:,None].to(torch.float32)
            v=torch.sin(v*a0)+torch.cos(v*a1)+torch.exp(-(v*a2)*(v*a2))+torch.tanh(v*a3)
            return v.to(x.dtype)
        return f
    raise KeyError(name)

def load(name):
    e=[x for x in json.load(open(SHARDS+KMAP[name])) if x["name"]==name][0]
    os.makedirs("/tmp/ovt",exist_ok=True); p=f"/tmp/ovt/{name}.py"
    open(p,"w").write("from __future__ import annotations\nimport torch,helion\nimport helion.language as hl\n"+e["code"])
    sp=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return getattr(m,name), m.make_inputs

def bench(fn,args,n=7):
    return sorted(do_bench(lambda: fn(*args), return_mode="median") for _ in range(n))[n//2]

def main():
    name=sys.argv[1]; M,N=int(sys.argv[2]),int(sys.argv[3])
    fn,mk=load(name); args=tuple(mk((M,N)))
    bk=fn.bind(args)
    seed=TritonPointwiseSeedHeuristic.get_seed_config(bk.env,bk.host_function.device_ir)
    dflt=bk.config_spec.default_config()
    dref=bk.compile_config(dflt)(*args)
    def agree(o):
        return bool(torch.allclose(o.float(),dref.float(),atol=2e-2,rtol=2e-2))
    # tc arm
    torch._dynamo.reset()
    ref=tc_ref(name); cf=torch.compile(ref,mode="max-autotune-no-cudagraphs")
    tcout=cf(*args); torch.cuda.synchronize()
    tc_ms=bench(cf,args)
    # helion default + seed
    d_ms=bench(bk.compile_config(dflt),args)
    s_run=bk.compile_config(seed); s_ms=bench(s_run,args); s_ok=agree(s_run(*args))
    # oracle sweep (moderate tiles, no ptxas-hang risk) x num_warps
    blocks=[[32,32],[64,64],[32,128],[128,32],[16,64],[64,16],[8,128],[128,8],[256,16],[16,256],
            [1,256],[1,512],[1,1024],[1,2048],[256,1],[1024,1],[64,4],[512,8]]
    warps=[4,8,16]
    best=(1e9,None)
    for b in blocks:
        for w in warps:
            try:
                r=bk.compile_config(helion.Config(block_sizes=b,num_warps=w))
                o=r(*args)
                if not agree(o): continue
                ms=bench(r,args,n=5)
                if ms<best[0]: best=(ms,(b,w))
            except Exception:
                continue
    o_ms,o_cfg=best
    print("RESULT "+json.dumps({
        "kernel":name,"shape":[M,N],
        "tc_ms":tc_ms,"default_ms":d_ms,"seed_ms":s_ms,"seed_cfg":seed.config["block_sizes"],
        "seed_correct":s_ok,"oracle_ms":o_ms,"oracle_cfg":o_cfg,
        "G_seed=tc/seed":tc_ms/s_ms,"G_oracle=tc/oracle":tc_ms/o_ms,
        "seed_vs_default":d_ms/s_ms,"oracle_vs_default":d_ms/o_ms,
    }))

if __name__=="__main__":
    main()
