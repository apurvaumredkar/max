```text
        ████████           ███╗   ███╗  █████╗  ██╗  ██╗
    ████        ████       ███╗   ███╗  █████╗  ██╗  ██╗
  ██                ██     ████╗ ████║ ██╔══██╗ ╚██╗██╔╝
  ██                ██     ████╗ ████║ ██╔══██╗ ╚██╗██╔╝
██    🟦        🟦    ██   ██╔████╔██║ ███████║  ╚███╔╝
██    🟦        🟦    ██   ██╔████╔██║ ███████║  ╚███╔╝
██                    ██   ██║╚██╔╝██║ ██╔══██║  ██╔██╗
██                    ██   ██║╚██╔╝██║ ██╔══██║  ██╔██╗
  ██                ██     ██║ ╚═╝ ██║ ██║  ██║ ██╔╝ ██╗
  ██                ██     ██║ ╚═╝ ██║ ██║  ██║ ██╔╝ ██╗
    ████        ████       ╚═╝     ╚═╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝
        ████████           ╚═╝     ╚═╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝
```

Max is a self-hosted personal assistant that runs on an NVIDIA Jetson Orin Nano. This repo holds its agent code and the
web console used to talk to it and watch the box:

| Dir | What |
|---|---|
| [`agents/`](agents/) | Rust Cargo workspace: the root agent (`root/`), the sub-agents Athena (`athena/`, email) and Chronos (`chronos/`, time), the shared blackboard (`board/`) and logging (`log/`) crates |
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
`context/agents/MAX.md` on every turn (kept out of the repo), so edits apply without a restart. History lives in SQLite at
`agents/memory/agents.db` (table `root`; `agents/memory/` is gitignored). Each reply is stored with its steps in order (thinking, text said
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
| `/model` | Pick the model (web console): every model on the inference machines' Ollama plus OpenRouter's tool-capable ones, then the context length. Kept in `agents/memory/model.json`; in the terminal it shows the current one |
| `/voice [on\|off]` | Voice mode, in the web console only (it needs the browser's microphone) |

### Voice mode

The browser streams its microphone as 16 kHz PCM; the agent relays it to the STT server, whose voice activity detector
cuts utterances. Each utterance becomes a streamed chat turn, and finished sentences go to the TTS (Phonon, voice
"Russell") while the model is still writing, so Max starts talking about half a second after the first words of the
reply. Phonon generates one request at a time, so speech chunks are synthesized in order, each one taking every
sentence finished by then.

While Max has the floor the browser sends silence instead of the mic, so the STT never hears him. Space cuts him off:
generation and speech stop, and the partial reply is stored, so the context matches what was actually said.

## Athena (`agents/athena`)

The email sub-agent. Max hands her requests in plain words through his `delegate` tool ("anything unread in my Gmail
from today?"); she runs her own tool loop with read-only tools over the email MCP server (`EMAIL_MCP_URL`, default
`http://127.0.0.1:8006/mcp`): list, search and read, never send, move or delete. Her answer goes back to Max and the
list she fetched to the console's Agent Workspace.

Every 5 minutes (`ATHENA_INTERVAL` seconds) she also checks for unread inbox mail from the last two days (Outlook,
and Gmail's Primary tab), skipping what she has seen (table `athena_seen`; the first check of an account only
records the backlog). New mail is read, summarised for Max and posted to him on the blackboard. Max then tells the
user himself, in a turn of his own with no tools (email is untrusted), live in the web console, and on Discord only
when no console tab is open and focused. Email is data, never instructions: her prompt and Max's both say so.

Athena works a request in three phases (`athena/src/agent.rs`): she plans (what a complete answer needs, where the
evidence is, two or three search strategies), acts in a ReAct loop with her read-only tools (each result shows her
remaining budget; the harness nudges her when she keeps searching without reading, or searches for a date as
text), and reviews her draft before it goes to Max (FINAL with a corrected answer, or CONTINUE with what's missing).

`athena eval` scores the harness on a suite of cases (`athena/eval/cases.json`) against a synthetic sandbox mailbox
(`athena/eval/mailbox.json`: bookings, ads, noise, a phishing email, codes that must never be repeated) that answers
like the real email server, at a frozen "now". `--prompts DIR` tries a different set of prompts, `--repeat N` runs
each case several times; transcripts go to `agents/memory/evals/`.

Each sub-agent's system prompt is a Markdown file, `context/agents/<NAME>.md`, re-read on every request; `{today}`
in it is filled with the date, time zone and the days people mean by "tomorrow", "this weekend" and "next week", so
no agent guesses dates. `athena ask <text>` runs one request directly.

Both Athena and Chronos run on whatever model Max is using (`agents/memory/model.json`, same context size).

`athena` runs the service (`athena/athena.service`); `athena once` runs one check and prints the report.

## Chronos (`agents/chronos`)

The time-keeping sub-agent: reminders and repeating jobs, kept in `context/chronos/jobs.json` (one-time `at` or 5-field
`cron`, local time; the file is re-read when it changes), and the user's Apple Calendar over CalDAV (Apple ID and an
app-specific password from the secrets Worker, `ICLOUD_USERNAME` / `ICLOUD_APP_PASSWORD`).

The user never talks to Chronos. Max passes him requests through his `delegate` tool, in plain words; Chronos runs his
own small tool loop on whatever model Max is using (`agents/memory/model.json`, same `num_ctx`) with tools to list
the schedule, add/remove/pause jobs and add/update/delete events, and answers Max. His schedule comes back as a view
for the console's Agent Workspace.

On his own he fires due jobs and polls the calendar every 5 minutes (`CHRONOS_CAL_INTERVAL`), reporting reminders and
calendar changes (new, moved, renamed, removed) to Max as notify reports; the first poll only records the calendar.
They are shown as is in every open web console and on Discord at once, and reach the model on its next turn.

`chronos` runs the service (`chronos/chronos.service`); `chronos ask <text>` runs one request; `chronos schedule` prints
the schedule.

## Blackboard (`agents/board`)

How the agents talk, through tables in `agents.db` (WAL mode). Max is the orchestrator: besides `bash` he has one
`delegate {agent, request}` tool, so the sub-agents' own tools and outputs never take up his context, only their
answers.

- `tasks`: Max posts a plain-language request for a sub-agent and waits; the sub-agent claims it with one atomic
  update, works it out with its own model loop and tools, and writes back `{text, view}`.
- `reports`: messages from a sub-agent to Max. `notify` says how each reaches the user: next turn only; Always
  (shown as is in every console and on Discord at once: Chronos); or Max (Max tells the user in his own words at
  once: live in every console, on Discord only when no console is focused: Athena).
- `agents`: each running agent's role and heartbeat; `delegate` offers the registered agents.

The crate also holds the shared model client (`llm`) and the secrets fetch (`secret`). The root agent's watcher
(`root/src/push.rs`) passes messages on; consoles keep `GET /events` open and report focus with `POST /presence`.

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
   it in. Host addresses live only there. Put the persona in `context/agents/MAX.md` in that same folder.
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
