# max

Max is a self-hosted personal assistant that runs on an NVIDIA Jetson Orin Nano. This repo holds its agent code and the
web console used to talk to it and watch the box:

| Dir | What |
|---|---|
| [`agents/`](agents/) | Rust Cargo workspace: the root agent (`root/`) and the shared logging crate (`log/`) |
| [`web-console/`](web-console/) | Management UI: a stdlib Python server with live reload and a static page, no build step |

Speech runs on two companion services from separate repos: [canary-stt-server](https://github.com/apurvaumredkar/canary-stt-server)
(speech to text) and [tts-server](https://github.com/apurvaumredkar/tts-server) (text to speech, Phonon engine). The
language model is served by [Ollama](https://ollama.com), which can run on another machine on the same private network.

```
browser ──https──▶ web-console :8000 ──▶ root agent :8007 ──▶ Ollama          (chat turns)
   │  mic/speaker      (relays /api/chat,          ├──▶ canary-stt /ws   (voice: speech to text)
   └──── WebSocket ───▶ /api/voice)                └──▶ tts-server /stream (voice: text to speech)
```

Both services bind `127.0.0.1`; the console is meant to be published through something like a Cloudflare Tunnel with an
access policy in front of it.

## Root agent (`agents/root`)

One binary, two modes:

- `root` serves HTTP on `127.0.0.1:$ROOT_PORT` (default 8007):
  - `GET /history`: the whole conversation
  - `POST /chat` `{"text"}`: one turn, streamed as Server-Sent Events (thinking, text, tool calls and results as
    they happen, then the final reply) with a keep-alive every 15 s, so long tool runs survive proxy timeouts.
    Slash commands answer with plain JSON.
  - `GET /voice`: WebSocket for voice mode (protocol documented at the top of [`voice.rs`](agents/root/src/voice.rs))
- `root chat` is the same conversation as a terminal loop.

Each turn sends the persona, the stored conversation and the new message to Ollama. The persona is read from
`context/MAX.md` on every turn (kept out of the repo), so edits apply without a restart. History lives in SQLite at
`agents/agents.db` (table `root`, also gitignored). Each reply is stored with its steps in order (thinking, text said
before a tool call, tool calls with output) for the chat pane; only the final answer goes back into the model's
context in later turns. The context window is 131,072 tokens (`NUM_CTX`), shown by a meter above the chat input.

### Tools

The model can call tools ([`tools.rs`](agents/root/src/tools.rs)) through Ollama's native tool calling, over the same
plain HTTP as everything else (no SDK): each request carries the tool definitions, and when the reply contains
`tool_calls` the agent runs them, sends the results back as `tool` messages and asks again, until the model answers
without a call (at most 25 rounds per turn). Tool calls are stored with the reply and shown in the chat pane.

| Tool | What it does |
|---|---|
| `bash` | Runs a command with `bash -c` in the home directory, no stdin, 120 s timeout; returns exit code, stdout and stderr |

There is no sandbox yet: `bash` can do anything the agent's user can.

### Slash commands

| Command | Effect |
|---|---|
| `/clear <N>` | Delete the last N turns from the history, so they also leave the model's context |
| `/think [on\|off]` | Turn the model's thinking on or off (bare `/think` flips it; starts on) |
| `/voice [on\|off]` | Voice mode, in the web console only (it needs the browser's microphone) |

### Voice mode

The browser streams its microphone as 16 kHz PCM; the agent relays it to the STT server, whose voice activity detector
cuts utterances. Each utterance becomes a streamed chat turn, and finished sentences go to the TTS (Phonon, voice
"Russell") while the model is still writing, so Max starts talking about half a second after the first words of the
reply. Phonon generates one request at a time, so speech chunks are synthesized in order, each one taking every
sentence finished by then.

While Max has the floor the browser sends silence instead of the mic, so the STT never hears him. Space cuts him off:
generation and speech stop, and the partial reply is stored, so the context matches what was actually said.

## Logging (`agents/log`)

Every agent uses the `agentlog` crate: call `agentlog::init("<agent>", path, stderr)` once, then the `tracing` macros
(`info!`, `warn!`, ...). Events are written as JSON lines to `agents/logs/agents.jsonl`, one file for all agents:

```json
{"ts":"2026-11-11T21:06:02.941-05:00","agent":"root","level":"INFO","msg":"reply","fields":{"ms":18711,"reply_tokens":657}}
```

`RUST_LOG` sets the level (default `info`). The web console streams this file live.

## Web console (`web-console`)

`server.py` serves `public/` and reloads the page whenever a file there changes. The page has:

- a chat pane on the left (history, live voice turns, thoughts in grey italics), relayed to the root agent;
- System (CPU, memory), Storage (local disks, Google Drive quota), Docker (container CPU and memory), Inference (the
  Ollama machines on the tailnet, OpenRouter credits), Tunnels (Cloudflare tunnels and their routes) and a live Logs
  box.

Storage, Inference and Tunnels depend on local tools and credentials (Docker, `tailscale`, the Cloudflare API, a
Google Drive CLI); each box shows an error if its source isn't available.

## Setup

1. **Config.** Copy [`.env.example`](.env.example) to `.env` in the folder that contains `agents/` and `web-console/`, and fill
   it in. Host addresses live only there. Put the persona in `context/MAX.md` in that same folder.
2. **Build the agents.** Install Rust with [rustup](https://rustup.rs), then:
   ```bash
   cd agents && cargo build --release      # binaries in agents/target/release/
   ```
3. **Run as user services.** The unit files hardcode their paths; adjust `WorkingDirectory`/`ExecStart` first.
   ```bash
   ln -s "$PWD/agents/root/max-root.service" "$PWD/web-console/web-console.service" ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now max-root web-console
   ```
   Or run them directly: `agents/target/release/root` and `python3 web-console/server.py`.

After changing agent code, rebuild and `systemctl --user restart max-root`. Console edits under `public/` show up
immediately; changes to `server.py` need `systemctl --user restart web-console`.
