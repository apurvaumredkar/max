"""Max web console: static server with live reload. Stdlib only, no build step.

Everything in public/ is served as-is. A tiny script is injected into HTML pages;
it listens on /__reload (SSE) and refreshes the page the moment any file in public/
changes (CSS-only edits swap the stylesheet without a full reload).
"""
import json, os, socket, subprocess, threading, time, urllib.error, urllib.request
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).parent.resolve() / "public"
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

CLIENT = b"""<script>
new EventSource('/__reload').onmessage = e => {
  if (e.data === 'css') document.querySelectorAll('link[rel=stylesheet]')
    .forEach(l => l.href = l.href.split('?')[0] + '?' + Date.now());
  else location.reload();
};
</script>"""


STATS = {"cpu": 0.0, "cores": [], "mem_used": 0, "mem_total": 0}


def _cpu_times():
    out = []
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu"):
            f = [int(x) for x in line.split()[1:]]
            out.append((f[3] + f[4], sum(f)))  # idle+iowait, total
    return out  # [aggregate, core0, core1, ...]


def sample_stats():
    prev = _cpu_times()
    while True:
        time.sleep(1)
        cur = _cpu_times()
        pct = [100 * (1 - (ci - pi) / max(ct - pt, 1)) for (pi, pt), (ci, ct) in zip(prev, cur)]
        prev = cur
        mem = {k: int(v.split()[0]) for k, v in
               (l.split(":") for l in Path("/proc/meminfo").read_text().splitlines())}
        STATS.update(cpu=pct[0], cores=pct[1:], mem_total=mem["MemTotal"] * 1024,
                     mem_used=(mem["MemTotal"] - mem["MemAvailable"]) * 1024)


def load_env(path=Path(__file__).resolve().parent.parent / ".env"):
    """The workspace's shared max/.env (non-secret config + CLOUDFLARE_API_TOKEN for admin reads)."""
    try:
        for line in path.read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    except OSError:
        pass


TUNNELS = {"ok": False, "error": "loading", "tunnels": []}
CF_API = "https://api.cloudflare.com/client/v4"


def cf_get(path):
    req = urllib.request.Request(CF_API + path, headers={
        "Authorization": "Bearer " + os.environ["CLOUDFLARE_API_TOKEN"]})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)["result"]


def poll_tunnels():
    """All Cloudflare Tunnels on the account (from the CF API, not just this host), every 30s. GET only."""
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    while True:
        try:
            if not os.environ.get("CLOUDFLARE_API_TOKEN"):
                raise RuntimeError("CLOUDFLARE_API_TOKEN not set in max/.env")
            account = account or cf_get("/accounts")[0]["id"]
            out = []
            for t in cf_get(f"/accounts/{account}/cfd_tunnel?is_deleted=false&per_page=100"):
                hosts = []
                try:
                    cfg = cf_get(f"/accounts/{account}/cfd_tunnel/{t['id']}/configurations")
                    hosts = [r["hostname"] for r in (cfg.get("config") or {}).get("ingress", []) if r.get("hostname")]
                except Exception:
                    pass  # locally managed tunnels have no remote config
                conns = t.get("connections") or []
                out.append({"name": t["name"], "status": t.get("status", "unknown"), "conns": len(conns),
                            "colos": sorted({c.get("colo_name", "?") for c in conns}), "hosts": hosts})
            TUNNELS.update(ok=True, error=None, tunnels=sorted(out, key=lambda x: x["name"]))
        except Exception as e:
            TUNNELS.update(ok=False, error=str(e)[:120])
        time.sleep(30)


GDRIVE = {"ok": False, "error": "loading"}
NODE = Path.home() / ".local/node22/bin/node"
GDRIVE_CLI = Path(__file__).resolve().parent.parent / "edge-store/google-drive/dist/drive/cli.js"


def poll_gdrive():
    """Google Drive quota via edge-store's google-drive module (`cli.js quota`), every 5 min."""
    while True:
        try:
            r = subprocess.run([str(NODE), str(GDRIVE_CLI), "quota"], capture_output=True, text=True, timeout=60)
            if r.returncode:
                raise RuntimeError((r.stderr.strip() or r.stdout.strip()).removeprefix("Error: ")[:200])
            GDRIVE.clear()
            GDRIVE.update(ok=True, error=None, **json.loads(r.stdout))
        except Exception as e:
            GDRIVE.clear()
            GDRIVE.update(ok=False, error=str(e))
        time.sleep(300)


DOCKER = {"ok": False, "containers": []}


def poll_docker():
    while True:
        try:
            out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}"],
                                 capture_output=True, text=True, timeout=20, check=True).stdout
            rows = [json.loads(l) for l in out.splitlines() if l.strip()]
            DOCKER.update(ok=True, containers=sorted(
                ({"name": r["Name"], "cpu": float(r["CPUPerc"].rstrip("%")),
                  "mem": float(r["MemPerc"].rstrip("%")), "usage": r["MemUsage"].split(" / ")[0]}
                 for r in rows), key=lambda c: c["name"]))
        except (OSError, subprocess.SubprocessError, ValueError, KeyError):
            DOCKER.update(ok=False, containers=[])
        time.sleep(2)


INFERENCE = {"hosts": [], "openrouter": {"ok": False, "error": "loading"}}
def inference_hosts():
    """INFERENCE_HOSTS in max/.env: "label=tailscale-hostname,..." (kept out of the repo)."""
    pairs = (p.split("=", 1) for p in os.environ.get("INFERENCE_HOSTS", "").split(",") if "=" in p)
    return [(label.strip(), name.strip().lower()) for label, name in pairs]
CLOUDFLARED = os.environ.get("CLOUDFLARED") or str(Path.home() / ".local/bin/cloudflared")


def poll_tailnet_inference():
    """Tailscale peer status for the inference boxes plus whether Ollama answers on :11434, every 10s."""
    while True:
        try:
            st = json.loads(subprocess.run(["tailscale", "status", "--json"], capture_output=True,
                                           text=True, timeout=10, check=True).stdout)
            # match on the MagicDNS name: HostName is the OS display name (e.g. "Apurva’s MacBook Pro")
            peers = {p.get("DNSName", "").split(".")[0].lower(): p for p in (st.get("Peer") or {}).values()}
            hosts = []
            if not inference_hosts():
                raise RuntimeError("INFERENCE_HOSTS not set in max/.env")
            for label, name in inference_hosts():
                p = peers.get(name)
                h = {"label": label, "name": name, "found": bool(p)}
                if p:
                    ip = (p.get("TailscaleIPs") or [""])[0]
                    h.update(ip=ip, online=bool(p.get("Online")),
                             path="direct" if p.get("CurAddr") else ("relay " + p["Relay"] if p.get("Relay") else ""))
                    try:
                        t0 = time.monotonic()
                        with urllib.request.urlopen(f"http://{ip}:11434/api/version", timeout=3) as r:
                            h.update(ollama=json.load(r).get("version", "?"), ms=round(1000 * (time.monotonic() - t0)))
                    except Exception:
                        h["ollama"] = None
                hosts.append(h)
            INFERENCE.update(hosts=hosts, tailnet_error=None)
        except Exception as e:
            INFERENCE.update(hosts=[], tailnet_error=str(e)[:120])
        time.sleep(10)


def get_secret(name):
    """A credential from the Access-gated secrets Worker (needs `cloudflared access login $SECRETS_URL`)."""
    url = os.environ.get("SECRETS_URL")
    if not url:
        raise RuntimeError("SECRETS_URL not set in max/.env")
    tok = subprocess.run([CLOUDFLARED, "access", "token", f"-app={url}"], capture_output=True, text=True, timeout=20)
    if tok.returncode:
        raise RuntimeError(f"no Access login: run `cloudflared access login {url}`")
    req = urllib.request.Request(f"{url}/{name}", headers={"cf-access-token": tok.stdout.strip(),
                                                           "User-Agent": "max-web-console"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)["value"]


def poll_openrouter():
    """OpenRouter account credits (GET /api/v1/credits), every 5 min."""
    key = None
    while True:
        try:
            key = key or get_secret("OPENROUTER_API_KEY")
            req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                         headers={"Authorization": "Bearer " + key,
                                                  "User-Agent": "max-web-console"})  # urllib's default UA gets a 403
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.load(r)["data"]
            INFERENCE["openrouter"] = {"ok": True, "total": d["total_credits"], "used": d["total_usage"]}
        except Exception as e:
            if getattr(e, "code", None) in (401, 403):
                key = None  # re-fetch in case the key was rotated
            INFERENCE["openrouter"] = {"ok": False, "error": str(e)[:160]}
        time.sleep(300)


# Chat pane: forwarded to the Rust root agent (agents/root, user service max-root), which owns the model,
# persona and history. The console only relays.
ROOT_AGENT = "http://127.0.0.1:8007"


def root_agent(path, payload=None):
    req = urllib.request.Request(ROOT_AGENT + path, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=620) as r:
            return json.load(r), 200
    except urllib.error.HTTPError as e:
        # 502/504 would be swapped for Cloudflare's own error page, hiding the agent's message; send 500 instead
        return json.load(e), 500 if e.code in (502, 504) else e.code
    except Exception as e:
        return {"error": f"root agent unreachable: {e}"[:200]}, 500


AGENT_LOG = Path(__file__).resolve().parent.parent / "agents/logs/agents.jsonl"  # written by agents/log (agentlog)


def storage():
    """One row per physical disk: mounted partitions summed under their parent disk."""
    disks = {}
    for line in Path("/proc/mounts").read_text().splitlines():
        dev, mnt = line.split()[:2]
        sysdev = Path("/sys/class/block") / dev[5:]
        if not dev.startswith("/dev/") or not sysdev.exists():
            continue
        real = sysdev.resolve()
        disk = real.parent.name if (real / "partition").exists() else real.name
        if disk.startswith("loop"):
            continue
        try:
            st = os.statvfs(mnt.replace("\\040", " "))
        except OSError:
            continue
        d = disks.setdefault(disk, {"dev": disk, "mounts": [], "total": 0, "used": 0, "seen": set()})
        if dev in d["seen"]:
            continue
        d["seen"].add(dev)
        total = st.f_blocks * st.f_frsize
        d["mounts"].append(mnt)
        d["total"] += total
        d["used"] += total - st.f_bfree * st.f_frsize
    rows = sorted(({k: v for k, v in d.items() if k != "seen"} for d in disks.values()),
                  key=lambda r: ("/" not in r["mounts"], r["dev"]))
    return {"disks": rows, "total": sum(r["total"] for r in rows), "used": sum(r["used"] for r in rows)}


def snapshot():
    return {str(p): p.stat().st_mtime_ns for p in ROOT.rglob("*") if p.is_file()}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.split("?")[0] != "/api/chat":
            return self.send_error(404)
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except ValueError:
            return self.send_json({"error": "bad request"}, 400)
        self.send_json(*root_agent("/chat", payload))

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/__reload":
            return self.sse()
        if path == "/api/voice":
            return self.voice_relay()
        if path == "/api/logs":
            return self.logs_sse()
        if path == "/api/chat":
            return self.send_json(*root_agent("/history"))
        api = {"/api/stats": lambda: STATS, "/api/storage": storage, "/api/docker": lambda: DOCKER,
               "/api/tunnels": lambda: TUNNELS, "/api/gdrive": lambda: GDRIVE,
               "/api/inference": lambda: INFERENCE}
        if path in api:
            body = json.dumps(api[path]()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        fs = Path(self.translate_path(path))
        if fs.is_dir():
            fs = fs / "index.html"
        if fs.suffix == ".html" and fs.is_file():
            body = fs.read_bytes()
            body = body.replace(b"</body>", CLIENT + b"</body>", 1) if b"</body>" in body else body + CLIENT
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        seen = snapshot()
        beat = 0
        try:
            while True:
                time.sleep(0.2)
                now = snapshot()
                if now != seen:
                    changed = {k for k in now.keys() | seen.keys() if now.get(k) != seen.get(k)}
                    kind = "css" if all(k.endswith(".css") for k in changed) else "reload"
                    self.wfile.write(f"data: {kind}\n\n".encode())
                    self.wfile.flush()
                    seen = now
                beat += 1
                if beat % 75 == 0:  # keep the tunnel from idling the stream out
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def voice_relay(self):
        """Voice mode WebSocket: hand the connection to the root agent's GET /voice and pipe bytes both ways untouched
        (the agent does the WebSocket handshake and all voice work; the console just relays)."""
        try:
            up = socket.create_connection(("127.0.0.1", 8007), timeout=5)
        except OSError as e:
            return self.send_json({"error": f"root agent unreachable: {e}"}, 503)
        up.settimeout(None)
        head = "GET /voice HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in self.headers.items()) + "\r\n"
        up.sendall(head.encode())

        def pump(src, dst):
            try:
                while chunk := src.recv(65536):
                    dst.sendall(chunk)
            except OSError:
                pass
            for s in (src, dst):  # one side ending ends both
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        back = threading.Thread(target=pump, args=(up, self.connection), daemon=True)
        back.start()
        pump(self.connection, up)
        back.join()
        up.close()
        self.close_connection = True

    def logs_sse(self):
        """Agent logs, live: the last 200 lines, then each new line as it's appended (one JSON line per event)."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        send = lambda line: self.wfile.write(b"data: " + line + b"\n\n")
        pos, buf, beat = 0, b"", 0
        try:
            if AGENT_LOG.exists():
                data = AGENT_LOG.read_bytes()
                pos = len(data)
                for line in data.splitlines()[-200:]:
                    send(line)
                self.wfile.flush()
            while True:
                time.sleep(0.5)
                size = AGENT_LOG.stat().st_size if AGENT_LOG.exists() else 0
                if size < pos:  # file truncated or replaced: start over
                    pos, buf = 0, b""
                if size > pos:
                    with AGENT_LOG.open("rb") as f:
                        f.seek(pos)
                        buf += f.read(size - pos)
                    pos = size
                    *lines, buf = buf.split(b"\n")  # keep a partial last line for next time
                    for line in lines:
                        if line.strip():
                            send(line)
                    self.wfile.flush()
                beat += 1
                if beat % 30 == 0:  # keep the tunnel from idling the stream out
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        if not any(p in (args[0] if args else "") for p in ("/__reload", "/api/logs")):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    ThreadingHTTPServer.daemon_threads = True
    threading.Thread(target=sample_stats, daemon=True).start()
    threading.Thread(target=poll_docker, daemon=True).start()
    load_env()
    threading.Thread(target=poll_tunnels, daemon=True).start()
    threading.Thread(target=poll_gdrive, daemon=True).start()
    threading.Thread(target=poll_tailnet_inference, daemon=True).start()
    threading.Thread(target=poll_openrouter, daemon=True).start()
    print(f"web-console on http://{HOST}:{PORT} serving {ROOT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
