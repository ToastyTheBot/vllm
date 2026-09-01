"""Reproduce upstream #54173: prompts of DIFFERING LENGTHS that SHARE A PREFIX.

Sends a growing workload over one shared long prefix, which is the trigger the
issue describes ("prompts of differing lengths that share a prefix"). Much
cheaper than the estonia benchmark.
"""
import json, sys, threading, urllib.request, time

BASE = open("/workspace/estonia_prompt.txt").read()
URL = "http://localhost:8000/v1/completions"


def send(prompt, max_tokens, tag):
    body = json.dumps({"model": "q38fn", "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=1800))
        u = d["usage"]
        tail = d["choices"][0]["text"][-40:]
        print("  [%s] ok comp=%s prompt=%s tail=%r"
              % (tag, u["completion_tokens"], u["prompt_tokens"], tail), flush=True)
    except Exception as e:
        blob = b""
        if hasattr(e, "read"):
            try:
                blob = e.read()[:200]
            except Exception:
                pass
        print("  [%s] FAIL %s %s %s"
              % (tag, type(e).__name__, str(e)[:120], blob), flush=True)


rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
step = len(BASE) // (rounds + 4)
for r in range(rounds):
    n = step * (r + 4)
    prompt = BASE[:n]
    print("round %d: prefix chars=%d" % (r, n), flush=True)
    ts = [threading.Thread(target=send, args=(prompt, 64, "r%d.%d" % (r, k)))
          for k in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    time.sleep(1)
print("REPRO DRIVER DONE", flush=True)
