# Minimal OpenAI-compatible /v1/chat/completions shim backed by `codex exec`.
import json, subprocess, time, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def parse_codex(out):
    lines=out.splitlines(); start=0
    for i,l in enumerate(lines):
        t=l.strip()
        if t=="codex" or t.endswith("] codex"): start=i+1
    def banner(t):
        return (t.startswith(("OpenAI Codex","--------","workdir:","model:","provider:",
                "approval:","sandbox:","reasoning ","session id:","tokens used")) or "tokens used" in t)
    return "\n".join(l for l in lines[start:] if not banner(l.strip())).strip()

def codex(prompt, attempts=4):
    for a in range(attempts):
        r=subprocess.run(["timeout","150","codex","exec","--skip-git-repo-check",prompt],
                         capture_output=True,text=True)
        c=(r.stdout+r.stderr).lower()
        if r.returncode!=0:
            if any(x in c for x in ("at capacity","try a different model","rate limit","429","503","overloaded")) and a+1<attempts:
                time.sleep(2<<a); continue
            raise RuntimeError(r.stderr[-300:])
        return parse_codex(r.stdout)
    raise RuntimeError("exhausted")

class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_POST(self):
        n=int(self.headers.get('Content-Length',0)); body=json.loads(self.rfile.read(n) or b'{}')
        msgs=body.get("messages",[])
        # flatten to a single prompt; if JSON output is requested, tell the model to emit ONLY JSON
        parts=[f"[{m.get('role')}]\n{m.get('content')}" for m in msgs]
        want_json = (body.get("response_format",{}) or {}).get("type")=="json_object"
        prompt="\n\n".join(parts)
        if want_json:
            prompt+="\n\nRespond with ONLY valid minified JSON, no prose, no code fences."
        try:
            txt=codex(prompt)
            if want_json:
                mobj=re.search(r'\{.*\}', txt, re.S)  # extract the JSON object
                if mobj: txt=mobj.group(0)
        except Exception as e:
            txt=f'{{"error":"{str(e)[:100]}"}}'
        resp={"id":"chatcmpl-codex","object":"chat.completion","model":body.get("model","codex"),
              "choices":[{"index":0,"message":{"role":"assistant","content":txt},"finish_reason":"stop"}],
              "usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
        out=json.dumps(resp).encode()
        self.send_response(200); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(out))); self.end_headers(); self.wfile.write(out)
    def do_GET(self):
        # /v1/models etc.
        out=json.dumps({"object":"list","data":[{"id":"codex","object":"model"}]}).encode()
        self.send_response(200); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(out))); self.end_headers(); self.wfile.write(out)

if __name__=="__main__":
    ThreadingHTTPServer(("127.0.0.1",8899),H).serve_forever()
