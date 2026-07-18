# Self-contained: run the codex->OpenAI shim on a thread, then Mem0 (openai provider
# pointed at it) so Mem0's fact-extraction runs on gpt-5.6-sol via the codex OAuth login.
import json, os, re, subprocess, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
os.environ["MEM0_TELEMETRY"] = "False"
os.environ["OPENAI_API_KEY"] = "sk-codex-dummy"

def parse_codex(out):
    lines = out.splitlines(); start = 0
    for i, l in enumerate(lines):
        t = l.strip()
        if t == "codex" or t.endswith("] codex"): start = i + 1
    def banner(t):
        return t.startswith(("OpenAI Codex", "--------", "workdir:", "model:", "provider:",
            "approval:", "sandbox:", "reasoning ", "session id:", "tokens used")) or "tokens used" in t
    return "\n".join(l for l in lines[start:] if not banner(l.strip())).strip()

def codex(prompt, attempts=4):
    for a in range(attempts):
        r = subprocess.run(["timeout", "150", "codex", "exec", "--skip-git-repo-check", prompt],
                           capture_output=True, text=True)
        c = (r.stdout + r.stderr).lower()
        if r.returncode != 0:
            if any(x in c for x in ("at capacity", "try a different model", "rate limit", "429", "503", "overloaded")) and a + 1 < attempts:
                time.sleep(2 << a); continue
            raise RuntimeError(r.stderr[-200:])
        return parse_codex(r.stdout)
    raise RuntimeError("exhausted")

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(n) or b'{}')
        parts = [f"[{m.get('role')}]\n{m.get('content')}" for m in body.get("messages", [])]
        want_json = (body.get("response_format", {}) or {}).get("type") == "json_object"
        prompt = "\n\n".join(parts) + ("\n\nRespond with ONLY valid minified JSON, no prose, no code fences." if want_json else "")
        try:
            txt = codex(prompt)
            if want_json:
                mo = re.search(r'\{.*\}', txt, re.S)
                if mo: txt = mo.group(0)
        except Exception as e:
            txt = ('{"facts":[]}' if want_json else f"error {e}")
        out = json.dumps({"id": "c", "object": "chat.completion", "model": "codex",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": txt}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

srv = ThreadingHTTPServer(("127.0.0.1", 8899), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
time.sleep(1)

from mem0 import Memory

DATA = "/home/n/Code/rust-synaptic/tools/eval/data/locomo10.json"
K = 10
CATS = {"1":"MultiHop","2":"Temporal","3":"OpenDomain","4":"SingleHop","5":"Abstention"}
ABSTAIN = ["i don't know","i do not know","no information","not mentioned","not available",
           "cannot determine","can't determine","no answer","not enough information",
           "not specified","no relevant","isn't mentioned","doesn't say","does not say"]
ABSTAIN_WORDS = {"unknown","none"}

def is_abstention(a):
    l=a.lower()
    if any(p in l for p in ABSTAIN): return True
    return any(w in ABSTAIN_WORDS for w in re.split(r'[^a-z0-9]+', l))

def parse_codex(out):
    lines=out.splitlines()
    start=0
    for i,l in enumerate(lines):
        t=l.strip()
        if t=="codex" or t.endswith("] codex"): start=i+1
    def banner(t):
        return (t.startswith("OpenAI Codex") or t.startswith("--------") or t.startswith("workdir:")
                or t.startswith("model:") or t.startswith("provider:") or t.startswith("approval:")
                or t.startswith("sandbox:") or t.startswith("reasoning ") or t.startswith("session id:")
                or "tokens used" in t)
    return "\n".join(l for l in lines[start:] if not banner(l.strip())).strip()

def codex(prompt, attempts=4):
    for a in range(attempts):
        try:
            r=subprocess.run(["timeout","120","codex","exec","--skip-git-repo-check",prompt],
                             capture_output=True,text=True,stdin=subprocess.DEVNULL)
        except Exception as e:
            if a+1==attempts: raise
            time.sleep(2<<a); continue
        combined=(r.stdout+r.stderr).lower()
        if r.returncode!=0:
            if ("at capacity" in combined or "try a different model" in combined or "rate limit" in combined
                or "429" in combined or "503" in combined or "overloaded" in combined) and a+1<attempts:
                time.sleep(2<<a); continue
            raise RuntimeError("codex failed: "+r.stderr[-300:])
        return parse_codex(r.stdout)
    raise RuntimeError("codex exhausted")

def answer(q, mems):
    ctx="\n".join(f"[{i+1}] {m}" for i,m in enumerate(mems)) if mems else "(no memories recalled)"
    return codex(f"Answer concisely using ONLY the provided memories; if not answerable from them, "
                 f"say 'I don't know'. Do not use any other knowledge or tools.\n\nMemories:\n{ctx}\n\n"
                 f"Question: {q}\nAnswer:")

def grade(q, gold, pred):
    v=codex(f"Question: {q}\nGold answer: {gold}\nPredicted answer: {pred}\n\n"
            f"Does the predicted answer match the gold answer in meaning? Reply exactly YES or NO.")
    return v.strip().upper().startswith("YES") or "YES" in v.strip().upper()[:6]

def mem_config():
    return {"llm":{"provider":"openai","config":{"model":"gpt-4o-mini","openai_base_url":"http://127.0.0.1:8899/v1","api_key":"sk-codex-dummy","temperature":0.0}},
            "embedder":{"provider":"ollama","config":{"model":"nomic-embed-text:latest","ollama_base_url":"http://localhost:11434","embedding_dims":768}},
            "vector_store":{"provider":"qdrant","config":{"embedding_model_dims":768,"path":"/tmp/mem0cxq/locomo","collection_name":"locomo"}}}

def main():
    n_conv=int(sys.argv[1]) if len(sys.argv)>1 else 3
    per=int(sys.argv[2]) if len(sys.argv)>2 else 4
    qfilter=sys.argv[3] if len(sys.argv)>3 else None  # e.g. "abstention"
    data=json.load(open(DATA))[:n_conv]
    os.environ["MEM0_TELEMETRY"]="False"

    recs=[]  # (category, correct, abstained)
    m=Memory.from_config(mem_config())
    for ci,c in enumerate(data):
        coll=f"conv{ci}"
        conv=c["conversation"]
        # ingest sessions
        si=1
        while f"session_{si}" in conv:
            sm=f"/tmp/mem0cxq/.done_{coll}_s{si}"
            if not os.path.exists(sm):
                turns=conv[f"session_{si}"]
                msgs=[{"role":"user" if t["speaker"]==conv["speaker_a"] else "assistant",
                       "content":f'{t["speaker"]}: {t["text"]}'} for t in turns]
                m.add(msgs, user_id=coll, infer=True)
                open(sm,"w").write("1")
                print(f"conv{ci} s{si}: ingested", flush=True)
            si+=1
        qa=c.get("qa") or []
        qs=[q for q in qa if (qfilter is None or CATS.get(str(q.get("category")),"").lower()==qfilter)][:per]
        def do(q):
            mems=[x.get("memory") for x in m.search(q["question"], filters={"user_id":coll}, limit=K).get("results",[])]
            pred=answer(q["question"], mems)
            ab=is_abstention(pred)
            corr=grade(q["question"], str(q.get("answer")), pred)
            return (CATS.get(str(q.get("category")),"?"), corr, ab)
        with ThreadPoolExecutor(max_workers=3) as ex:
            for f in as_completed([ex.submit(do,q) for q in qs]):
                recs.append(f.result())
        print(f"conv{ci}: {len([r for r in recs])} done", flush=True)

    tot=len(recs); corr=sum(1 for _,c,_ in recs if c)
    print(f"\n=== MEM0 RESULT: graded={tot} correct={corr} accuracy={corr/tot:.4f} ===")
    bycat={}
    for cat,c,ab in recs: bycat.setdefault(cat,[0,0,0]); bycat[cat][0]+=1; bycat[cat][1]+=int(c); bycat[cat][2]+=int(ab)
    for cat,(t,cc,ab) in sorted(bycat.items()):
        print(f"  {cat}: n={t} acc={cc/t:.4f} abstained={ab}")

if __name__=="__main__": main()
