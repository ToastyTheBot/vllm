import json,urllib.request,sys
p=open("/workspace/estonia_prompt.txt").read()
body=json.dumps({"model":"q38fn","messages":[{"role":"user","content":p}],
                 "max_tokens":int(sys.argv[1]),"temperature":0}).encode()
r=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=body,
                         headers={"Content-Type":"application/json"})
d=json.load(urllib.request.urlopen(r,timeout=3600))
c=d["choices"][0]["message"]["content"] or ""
print("finish:",d["choices"][0].get("finish_reason"),"usage:",d.get("usage"))
print("--- last 400 chars ---"); print(repr(c[-400:]))
print("--- contains estonia:", "estonia" in c.lower())
