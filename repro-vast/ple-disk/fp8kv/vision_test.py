import base64, json, urllib.request

URL = "http://localhost:8000/v1/chat/completions"


def ask(path, prompt, max_tokens=200):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    body = json.dumps({
        "model": "q38fn",
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + b64}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens, "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    return (d["choices"][0]["message"]["content"] or "").strip(), d.get("usage", {})


tests = [
    ("/workspace/shapes.png",
     "Describe exactly what shapes you see, their colours and positions. Be brief.",
     ["red", "blue"]),
    ("/workspace/digit.png",
     "What single digit is written in this image? Answer with just the digit.",
     ["7"]),
]
for path, prompt, expect in tests:
    txt, usage = ask(path, prompt)
    hit = [e for e in expect if e.lower() in txt.lower()]
    verdict = "PASS" if len(hit) == len(expect) else "FAIL"
    print("--- %s" % path)
    print("    prompt_tokens=%s completion=%s" % (usage.get("prompt_tokens"),
                                                  usage.get("completion_tokens")))
    print("    expected %s -> found %s  %s" % (expect, hit, verdict))
    print("    answer: %r" % txt[-320:])
