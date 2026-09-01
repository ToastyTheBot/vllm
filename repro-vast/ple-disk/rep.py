import json,sys
d=json.load(open(sys.argv[1])); s=d["selected_summary"]
print("===",sys.argv[2],"===")
for k in ("completed","correct","fail","errors","correct_rate"): print("  %-12s %s"%(k,s.get(k)))
ct=[r["completion_tokens"] for r in d["runs"] if r.get("ok") and r.get("completion_tokens")]
if ct:
    ct.sort(); print("  tokens p50/p90/max: %d/%d/%d"%(ct[len(ct)//2],ct[int(len(ct)*0.9)-1],ct[-1]))
bad=[r for r in d["runs"] if r.get("correct") is False]
for r in bad[:6]: print("   WRONG:",repr((r.get("final_answer") or "")[:60]))
err=[r for r in d["runs"] if not r.get("ok")]
if err: print("  first error:",(err[0].get("error") or "none")[:200])
