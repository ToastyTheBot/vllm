import json,sys
d=json.load(open(sys.argv[1]))
s=d["selected_summary"]
print("===",sys.argv[2],"===")
for k in ("completed","correct","fail","errors","correct_rate"):
    print("  %-12s %s"%(k,s.get(k)))
for r in d["runs"]:
    print("   ok=%s tok=%s correct=%s ans=%r"%(r.get("ok"),r.get("completion_tokens"),r.get("correct"),(r.get("final_answer") or "")[:45]))
