"""Compare the mmap NVFP4 gather+dequant against a safetensors reference."""
import numpy as np, torch, json, glob, struct, sys
sys.path.insert(0,"/workspace/vllmenv/lib/python3.12/site-packages")
from vllm.models.qwen4_exp.nvidia import ple_mmap
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import dequantize_to_dtype
from safetensors import safe_open

MP="/workspace/model"
shards = ple_mmap.discover_shards(MP)
layer_idx = sorted(shards)[0]
ls = shards[layer_idx]
print("layer",layer_idx,"nvfp4:",ple_mmap.is_nvfp4(ls),"cols",ls.cols,"scale_cols",ls.scale_cols,
      "shards",len(ls.shards))

HEAD_DIM = ls.cols*2
vocab = sum(r for _,_,r in ls.shards.values())
shard_size = ls.shards[0][2]
print("head_dim",HEAD_DIM,"vocab",vocab,"shard_size",shard_size)

wt = ple_mmap.MmapPleTable(ls.shards, shard_size, ls.cols, torch.uint8,
        workers=8, chunk=512, model_path=MP)
st = ple_mmap.MmapPleTable(ls.scale_shards, shard_size, ls.scale_cols, torch.float8_e4m3fn,
        workers=8, chunk=512, model_path=MP)
gs = ple_mmap._read_scale(ls.global_scale_entry).reshape(()).to(torch.float32)
print("global_scale (weight_scale_2):", float(gs))

rng = np.random.default_rng(0)
ids = rng.integers(0, vocab, size=512).astype(np.int64)
rows = wt.gather(ids); scl = st.gather(ids)
print("gathered", rows.shape, rows.dtype, scl.shape)

# reference: read the same rows straight out of the checkpoint
files = {}
def ref_row(gid):
    si = gid // shard_size; off = gid % shard_size
    path,_,_ = ls.shards[si]
    if path not in files: files[path]=safe_open(path,framework="pt")
    f=files[path]
    pre=f"model.language_model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding.shard_{si}"
    w=f.get_slice(pre+".weight")[off:off+1]
    s=f.get_slice(pre+".weight_scale")[off:off+1]
    return w,s

mism_w=mism_s=0
for i,gid in enumerate(ids[:64]):
    w,s = ref_row(int(gid))
    if not torch.equal(w.cpu().view(torch.uint8).flatten(), torch.from_numpy(rows[i]).flatten()): mism_w+=1
    if not torch.equal(s.cpu().view(torch.uint8).flatten(), torch.from_numpy(scl[i]).flatten()): mism_s+=1
print("RAW BYTE MISMATCHES: weight",mism_w,"/64  scale",mism_s,"/64")

# dequant on GPU exactly as forward does
packed=torch.from_numpy(rows).cuda()
bs=torch.from_numpy(scl).view(torch.float8_e4m3fn).cuda()
out=dequantize_to_dtype(packed,bs,gs.cuda(),torch.bfloat16,block_size=16,swizzle=False)
print("dequant out",out.shape,out.dtype,"finite:",bool(torch.isfinite(out).all()))
print("  min/max/absmean: %.4f %.4f %.4f"%(out.min(),out.max(),out.abs().float().mean()))
# id bounds sanity
print("id range used:",ids.min(),ids.max(),"vocab",vocab)
