import json, os, re, subprocess, time, threading, asyncio, datetime, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
os.environ["OPENAI_API_KEY"]="sk-x"
DATA="/home/n/Code/rust-synaptic/tools/eval/data/locomo10.json"
CATS={"1":"MultiHop","2":"Temporal","3":"OpenDomain","4":"SingleHop","5":"Abstention"}

def parse_codex(out):
    lines=out.splitlines(); start=0
    for i,l in enumerate(lines):
        t=l.strip()
        if t=="codex" or t.endswith("] codex"): start=i+1
    def banner(t): return t.startswith(("OpenAI Codex","--------","workdir:","model:","provider:","approval:","sandbox:","reasoning ","session id:","tokens used")) or "tokens used" in t
    return "\n".join(l for l in lines[start:] if not banner(l.strip())).strip()
def codex_raw(prompt, attempts=4):
    for a in range(attempts):
        r=subprocess.run(["timeout","150","codex","exec","--skip-git-repo-check",prompt],capture_output=True,text=True,stdin=subprocess.DEVNULL)
        c=(r.stdout+r.stderr).lower()
        if r.returncode!=0:
            if any(x in c for x in ("at capacity","try a different model","rate limit","429","503","overloaded")) and a+1<attempts:
                time.sleep(2<<a); continue
            raise RuntimeError((r.stderr or r.stdout)[-150:])
        return parse_codex(r.stdout)
    raise RuntimeError("exhausted")
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_POST(self):
        n=int(self.headers.get('Content-Length',0)); body=json.loads(self.rfile.read(n) or b'{}')
        parts=[f"[{m.get('role')}]\n{m.get('content')}" for m in body.get("messages",[])]
        rf=body.get("response_format") or {}
        want=rf.get("type") in ("json_object","json_schema")
        prompt="\n\n".join(parts)+("\n\nRespond with ONLY valid minified JSON, no prose, no fences." if want else "")
        try:
            txt=codex_raw(prompt)
            if want:
                mo=re.search(r'\{.*\}',txt,re.S); txt=mo.group(0) if mo else txt
        except Exception as e: txt=('{}' if want else f"err {e}")
        out=json.dumps({"id":"c","object":"chat.completion","model":"codex","choices":[{"index":0,"message":{"role":"assistant","content":txt},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}).encode()
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(out))); self.end_headers(); self.wfile.write(out)
threading.Thread(target=ThreadingHTTPServer(("127.0.0.1",8899),H).serve_forever,daemon=True).start(); time.sleep(1)

def is_abstention(pred):
    p=pred.strip().lower()
    return any(k in p for k in ["i don't know","i do not know","not answerable","cannot be answered","no information","not enough information","unable to answer","don't have"])
def answer(ctx,q):
    return codex_raw(f"Answer concisely using ONLY the provided memories; if not answerable from them, say \"I don't know\". Do not use any other knowledge or tools.\n\nMemories:\n{ctx}\n\nQuestion: {q}\nAnswer:")
def grade(q,gold,pred):
    v=codex_raw(f"Question: {q}\nGold answer: {gold}\nPredicted answer: {pred}\n\nDoes the predicted answer match the gold answer in meaning? Reply exactly YES or NO.")
    return v.strip().upper().startswith("YES") or "YES" in v.strip().upper()[:6]

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.nodes import EpisodeType

MARK="/tmp/graphiti_marks"; os.makedirs(MARK, exist_ok=True)

async def main():
    n_conv=int(sys.argv[1]) if len(sys.argv)>1 else 3
    per=int(sys.argv[2]) if len(sys.argv)>2 else 4
    llmcfg=LLMConfig(api_key="sk-x", model="gpt-5.6-sol", base_url="http://127.0.0.1:8899/v1", small_model="gpt-5.6-sol")
    llm=OpenAIGenericClient(config=llmcfg, structured_output_mode="json_object")
    emb=OpenAIEmbedder(config=OpenAIEmbedderConfig(embedding_dim=768, embedding_model="nomic-embed-text:latest", api_key="sk-x", base_url="http://localhost:11434/v1"))
    rer=OpenAIRerankerClient(config=llmcfg, client=llm)
    driver=FalkorDriver(host="localhost", port=6379)
    g=Graphiti(graph_driver=driver, llm_client=llm, embedder=emb, cross_encoder=rer)
    await g.build_indices_and_constraints()
    data=json.load(open(DATA))[:n_conv]
    def parse_dt(s):
        for fmt in ("%I:%M %p on %d %B, %Y","%I:%M %p on %d %b, %Y"):
            try: return datetime.datetime.strptime(s.strip(), fmt)
            except: pass
        return datetime.datetime(2023,1,1)
    # ingest
    for ci,c in enumerate(data):
        conv=c["conversation"]; gid=f"conv{ci}"
        si=1
        while f"session_{si}" in conv:
            mk=f"{MARK}/g_{gid}_s{si}"
            if not os.path.exists(mk):
                turns=conv[f"session_{si}"]; date=conv.get(f"session_{si}_date_time","")
                body="\n".join(f'{t["speaker"]}: {t["text"]}' for t in turns)
                try:
                    await g.add_episode(name=f"{gid}_s{si}", episode_body=body[:8000], source_description="conversation",
                                        reference_time=parse_dt(date), source=EpisodeType.message, group_id=gid)
                    open(mk,"w").write("1")
                    print(f"{gid} s{si}: episode added", flush=True)
                except Exception as e:
                    print(f"{gid} s{si} ERR: {str(e)[:120]}", flush=True)
            si+=1
    # QA
    graded=correct=0; bycat={}
    for ci,c in enumerate(data):
        gid=f"conv{ci}"; qa=c.get("qa") or []
        qs=qa[:per]
        for q in qs:
            qt=CATS.get(str(q.get("category")),"?"); question=q.get("question","")
            gold=str(q.get("answer", q.get("adversarial_answer","")))
            try:
                res=await g.search(question, group_ids=[gid], num_results=10)
                ctx="\n".join(f"- {e.fact}" for e in res if hasattr(e,"fact"))
            except Exception as e:
                ctx=""; print(f"search err: {str(e)[:80]}")
            pred=answer(ctx or "(no memories)", question)
            ab=is_abstention(pred); ok=grade(question,gold,pred)
            graded+=1; correct+=int(ok)
            b=bycat.setdefault(qt,[0,0,0]); b[0]+=1; b[1]+=int(ok); b[2]+=int(ab)
        print(f"{gid}: done", flush=True)
    print(f"\n=== GRAPHITI RESULT: graded={graded} correct={correct} accuracy={correct/max(graded,1):.4f} ===")
    for k,b in sorted(bycat.items()):
        print(f"  {k}: n={b[0]} acc={b[1]/max(b[0],1):.4f} abstained={b[2]}")
    await g.close()

asyncio.run(main())
