//! Root agent: chats with Ollama, using context/agents/MAX.md as the persona and keeping the history in
//! agents/memory/agents.db (table `root`).
//!
//!   root            HTTP service on 127.0.0.1:$ROOT_PORT (default 8007): GET /history, POST /chat {"text"}
//!   root chat       the same conversation as a terminal loop (/exit or Ctrl-D to quit)
//!
//! Tools (tools.rs): the model can call `bash`; a turn loops request -> tool calls -> results until it answers.
//! All model traffic is plain HTTP to Ollama's /api/chat (JSON built here, NDJSON stream parsed here).
//!
//! Slash commands (web and terminal): /clear <N> deletes the last N turns from the history and the context;
//! /think [on|off] toggles the model's thinking (on at start).
//! /voice (web console only) opens a voice session on GET /voice (WebSocket): see voice.rs.
//! /model (web console) picks the model: GET/POST /model, see model.rs; in the terminal /model shows the current one.
//! Discord (served mode only): DMs and @mentions from allowed users run the same turn; see discord.rs.
//!
//! Logs (via the shared `agentlog` crate) go to agents/logs/agents.jsonl, which the web console streams.

mod discord;
mod model;
mod push;
mod tools;
mod voice;

use std::{path::PathBuf, sync::{Arc, RwLock, atomic::{AtomicBool, AtomicU64, Ordering}}, time::Instant};

use agentlog::{error, info};

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, sse::{Event, KeepAlive, Sse}},
    routing::{get, post},
};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio_util::sync::CancellationToken;

// Default model (Ollama at OLLAMA_URL) until one is picked with /model; the pick is kept in agents/memory/model.json.
const MODEL: &str = "gemma4:e4b";
const NUM_CTX: u32 = 131072; // gemma4:e4b full context
static CTX_MAX: AtomicU64 = AtomicU64::new(NUM_CTX as u64); // the context meter's maximum: the current num_ctx

struct Agent {
    home: PathBuf,
    db: PathBuf,
    persona: PathBuf,
    sel: RwLock<model::Sel>,               // the model in use (see model.rs)
    sel_path: PathBuf,
    or_key: tokio::sync::Mutex<Option<String>>, // OpenRouter key, fetched from the secrets Worker on first use
    stt_url: String,
    tts_url: String,
    http: reqwest::Client,
    turn: tokio::sync::Mutex<()>, // one turn at a time so the history stays in order
    think: AtomicBool,            // model thinking on/off (/think); starts on, not kept across restarts
    events: tokio::sync::broadcast::Sender<Value>, // live events for open web consoles (GET /events, see push.rs)
    consoles: std::sync::Mutex<std::collections::HashMap<String, bool>>, // open console sessions -> focused
    discord: RwLock<Option<(String, String)>>, // (auth header, channel) for sending to the user's channel, once connected
}

const MAX_ROUNDS: usize = 25; // model calls per turn (each tool round is one more)

/// A piece of a streamed turn.
pub(crate) enum Delta<'a> {
    Thinking(&'a str),
    Content(&'a str),
    ToolCall { name: &'a str, args: &'a Value },
    ToolResult { output: &'a str },
    /// Structured data for the console's workspace panel (e.g. Athena's email list).
    View(&'a Value),
}

/// A finished (or interrupted) turn. `reply` is the final answer only: it is all that goes back into the model's
/// context in later turns. `steps` is everything before it, in order, kept for display:
///   {"type":"thinking","text"} | {"type":"text","text"} (said before a tool call) | {"type":"tool","name","command","output"}
#[derive(Default)]
pub(crate) struct Turn {
    pub reply: String,
    pub thinking: Option<String>, // all thinking, joined (logs; also in steps)
    pub steps: Vec<Value>,
    pub interrupted: bool,
    pub ctx: Option<u64>, // tokens in the context window at the end of the turn (last request: prompt + output)
}

/// Turn events for the chat pane (typed chat over SSE, and voice): the same JSON either way.
pub(crate) fn delta_event(d: &Delta) -> Value {
    match d {
        Delta::Thinking(t) => json!({"type": "thinking_delta", "text": t}),
        Delta::Content(c) => json!({"type": "reply_delta", "text": c}),
        Delta::ToolCall { name, args } => json!({"type": "tool_call", "name": name, "command": tools::label(name, args)}),
        Delta::ToolResult { output } => {
            json!({"type": "tool_result", "output": output.chars().take(4000).collect::<String>()}) // pane shows the top
        }
        Delta::View(v) => json!({"type": "view", "view": v}),
    }
}

pub(crate) fn reply_event(t: &Turn) -> Value {
    json!({"type": "reply", "text": t.reply, "thinking": t.thinking, "steps": t.steps, "interrupted": t.interrupted,
           "context": context(t.ctx)})
}

/// A stored message (see `open` for the table).
struct Row {
    msg: Msg,
    at: String,
    thinking: Option<String>,
    tools: Option<String>, // older rows: tool calls without order (before steps existed)
    steps: Option<String>,
    ctx: Option<i64>,
}

/// Context meter for the chat pane: tokens used (None: not known yet) out of the window.
pub(crate) fn context(used: Option<u64>) -> Value {
    json!({"used": used, "max": CTX_MAX.load(Ordering::Relaxed)})
}

#[derive(Serialize, Deserialize, Clone)]
struct Msg {
    role: String,
    content: String,
}

impl Agent {
    fn new() -> Self {
        // the workspace is two levels up from this crate (max/agents/root); MAX_HOME overrides it
        let home = std::env::var("MAX_HOME").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("workspace dir")
        });
        load_env(&home.join(".env"));
        // host-specific endpoints live only in the gitignored max/.env (see .env.example)
        let need = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} must be set in max/.env (see .env.example)"));
        let base = need("OLLAMA_URL");
        let base = base.trim_end_matches('/').trim_end_matches("/v1").to_string();
        let sel_path = home.join("agents/memory/model.json");
        let sel = model::load(&sel_path).unwrap_or(model::Sel {
            provider: "ollama".into(), host: "default".into(), base, model: MODEL.into(), num_ctx: NUM_CTX,
        });
        CTX_MAX.store(sel.num_ctx as u64, Ordering::Relaxed);
        Agent {
            db: home.join("agents/memory/agents.db"), // agent memory lives in agents/memory/ (with the knowledge graph)
            persona: home.join("context/agents/MAX.md"),
            sel: RwLock::new(sel),
            sel_path,
            or_key: tokio::sync::Mutex::new(None),
            stt_url: need("STT_URL"),
            tts_url: need("TTS_URL"),
            http: reqwest::Client::builder().timeout(std::time::Duration::from_secs(600)).build().unwrap(),
            turn: tokio::sync::Mutex::new(()),
            think: AtomicBool::new(true),
            events: tokio::sync::broadcast::channel(64).0,
            consoles: Default::default(),
            discord: RwLock::new(None),
            home,
        }
    }

    fn open(&self) -> rusqlite::Result<Connection> {
        let db = Connection::open(&self.db)?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS root (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                thinking TEXT,
                tools TEXT,
                ctx INTEGER,
                steps TEXT)",
        )?;
        // older tables lack these columns; add them (fails harmlessly once they exist)
        let _ = db.execute("ALTER TABLE root ADD COLUMN thinking TEXT", []);
        let _ = db.execute("ALTER TABLE root ADD COLUMN tools TEXT", []);
        let _ = db.execute("ALTER TABLE root ADD COLUMN ctx INTEGER", []);
        let _ = db.execute("ALTER TABLE root ADD COLUMN steps TEXT", []);
        // sub-agents (athena, ...) leave reports here for the next turn; see `pending_reports`
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                text TEXT NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                delivered_at TEXT)",
        )?;
        db.busy_timeout(std::time::Duration::from_secs(10))?;
        Ok(db)
    }

    /// Sub-agent reports not yet passed to the model: (id, agent, created_at, text), oldest first.
    fn pending_reports(&self) -> Result<Vec<(i64, String, String, String)>, String> {
        let db = self.open().map_err(|e| e.to_string())?;
        let mut st = db.prepare("SELECT id, agent, created_at, text FROM reports WHERE delivered_at IS NULL ORDER BY id")
            .map_err(|e| e.to_string())?;
        st.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
            .and_then(|rows| rows.collect())
            .map_err(|e| e.to_string())
    }

    fn mark_delivered(&self, ids: &[i64]) -> Result<(), String> {
        let db = self.open().map_err(|e| e.to_string())?;
        for id in ids {
            db.execute("UPDATE reports SET delivered_at = CURRENT_TIMESTAMP WHERE id = ?1", [id])
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Every stored message with its timestamp and, for replies, the model's thoughts and tool calls (JSON).
    fn history(&self) -> Result<Vec<Row>, String> {
        let db = self.open().map_err(|e| e.to_string())?;
        let mut st = db.prepare("SELECT role, content, created_at, thinking, tools, ctx, steps FROM root ORDER BY id")
            .map_err(|e| e.to_string())?;
        st.query_map([], |r| Ok(Row { msg: Msg { role: r.get(0)?, content: r.get(1)? }, at: r.get(2)?, thinking: r.get(3)?,
                                      tools: r.get(4)?, ctx: r.get(5)?, steps: r.get(6)? }))
            .and_then(|rows| rows.collect())
            .map_err(|e| e.to_string())
    }

    /// One exchange, not streamed (typed chat and the terminal): the same loop as `chat_stream`, nothing to show
    /// along the way and no interrupt.
    async fn chat(&self, text: &str) -> Result<Turn, String> {
        self.chat_stream(text, &CancellationToken::new(), |_| {}).await
    }

    /// A turn from another channel (Discord): `note` is a system message about where Max is, sent just before
    /// `text` in this turn only (not stored). `tools: false` is for messages that aren't from the user (another
    /// agent, e.g. a Discord bot): same conversation, but the model gets no tools, so nothing in it can make Max
    /// run a command.
    pub(crate) async fn chat_from(&self, text: &str, note: &str, tools: bool) -> Result<Turn, String> {
        self.chat_with(text, Some(note), tools, &CancellationToken::new(), |_| {}).await
    }

    /// What the model sees for a turn: persona, the stored conversation, any new sub-agent reports, then `text`.
    /// Past thoughts and tool calls are kept for display only and not sent back; a report is shown in one turn only.
    fn messages_for(&self, text: &str, reports: &[(i64, String, String, String)], note: Option<&str>)
        -> Result<Vec<Value>, String> {
        let persona = std::fs::read_to_string(&self.persona)
            .map_err(|e| format!("{}: {e}", self.persona.display()))?;
        let mut messages = vec![json!({"role": "system", "content": persona})];
        messages.extend(self.history()?.into_iter().map(|r| json!({"role": r.msg.role, "content": r.msg.content})));
        if !reports.is_empty() {
            let lines: Vec<String> = reports.iter().map(|(_, agent, at, t)| format!("[{agent}, {at} UTC] {t}")).collect();
            messages.push(json!({"role": "system", "content": format!(
                "Reports from your sub-agents since the user last spoke. Mention them briefly in your reply, before \
                 or after answering the user's message:\n{}", lines.join("\n"))}));
        }
        match (note, text.is_empty()) {
            // Max speaks on his own (push.rs): models answer a user turn, so the cue goes in one, marked as automatic
            (Some(n), true) => messages.push(json!({"role": "user",
                "content": format!("(Automatic message from Max's system, not typed by the user.) {n}")})),
            (Some(n), false) => messages.push(json!({"role": "system", "content": n})),
            (None, _) => {}
        }
        if !text.is_empty() {
            messages.push(json!({"role": "user", "content": text}));
        }
        Ok(messages)
    }

    fn store(&self, text: &str, turn: &Turn) -> Result<(), String> {
        let db = self.open().map_err(|e| e.to_string())?;
        let steps = (!turn.steps.is_empty()).then(|| Value::Array(turn.steps.clone()).to_string());
        if !text.is_empty() {
            db.execute("INSERT INTO root (role, content) VALUES ('user', ?1)", [text]).map_err(|e| e.to_string())?;
        }
        db.execute("INSERT INTO root (role, content, thinking, steps, ctx) VALUES ('assistant', ?1, ?2, ?3, ?4)",
                   rusqlite::params![turn.reply, turn.thinking, steps, turn.ctx.map(|c| c as i64)])
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// One turn, streamed: `on` gets thinking, reply text and tool activity as they happen. The model may call tools;
    /// each round of calls is run and sent back until it answers without one (at most MAX_ROUNDS requests).
    /// Cancelling `cancel` stops generation or the running tool; the partial turn is stored, so the context matches
    /// what was said.
    pub(crate) async fn chat_stream(&self, text: &str, cancel: &CancellationToken, on: impl FnMut(Delta))
        -> Result<Turn, String> {
        self.chat_with(text, None, true, cancel, on).await
    }

    async fn chat_with(&self, text: &str, note: Option<&str>, tools: bool, cancel: &CancellationToken, mut on: impl FnMut(Delta))
        -> Result<Turn, String> {
        let _turn = self.turn.lock().await;
        let started = Instant::now();
        info!(chars = text.chars().count(), tools, "user message");
        let result = self.turn_inner(text, note, tools, cancel, &mut on).await;
        let ms = started.elapsed().as_millis() as u64;
        match &result {
            Ok((t, stats)) => info!(ms, prompt_tokens = stats.0, reply_tokens = stats.1, chars = t.reply.chars().count(),
                thinking_chars = t.thinking.as_deref().map_or(0, |t| t.chars().count()), steps = t.steps.len(),
                interrupted = t.interrupted, "reply"),
            Err(e) => error!(ms, error = %e, "turn failed"),
        }
        result.map(|(t, _)| t)
    }

    async fn turn_inner(&self, text: &str, note: Option<&str>, tools: bool, cancel: &CancellationToken, on: &mut impl FnMut(Delta))
        -> Result<(Turn, (u64, u64)), String> {
        let reports = self.pending_reports()?;
        let mut messages = self.messages_for(text, &reports, note)?;
        let (mut turn, mut thinking, mut stats) = (Turn::default(), String::new(), (0, 0));
        for (_, agent, _, t) in &reports {
            turn.steps.push(json!({"type": "report", "agent": agent, "text": format!("{agent}: {t}")}));
        }
        // Within the turn the model sees its own tool calls and results (`messages` grows); none of it is stored
        // as context: only `text` and the final `turn.reply` are, while `turn.steps` keeps it all for display.
        // the model is fixed for the whole turn: a /model switch mid-turn applies from the next one
        let sel = self.sel.read().unwrap().clone();
        let openai = sel.provider == "openrouter"; // OpenAI-style tool messages instead of Ollama's
        let specs = if tools { tools::specs(&self.db) } else { json!([]) };
        for round in 0..MAX_ROUNDS {
            let think = self.think.load(Ordering::Relaxed);
            let r = if openai {
                let key = self.openrouter_key().await?;
                let r = model::openrouter(&self.http, &key, &sel, &messages, &specs, think, cancel, &mut *on).await;
                if r.as_ref().is_err_and(|e| e.contains("401") || e.contains("403")) {
                    *self.or_key.lock().await = None; // re-fetch next time, in case the key was rotated
                }
                r?
            } else {
                match model::ollama(&self.http, &sel, &messages, &specs, think, cancel, &mut *on).await {
                    // some models (e.g. llama3.1) can't think: ask again with thinking off rather than fail the turn
                    Err(e) if think && e.contains("does not support thinking") => {
                        model::ollama(&self.http, &sel, &messages, &specs, false, cancel, &mut *on).await?
                    }
                    r => r?,
                }
            };
            stats.0 += r.prompt;
            stats.1 += r.eval;
            if r.prompt + r.eval > 0 {
                turn.ctx = Some(r.prompt + r.eval); // last request wins: what the window held when the model finished
            }
            turn.interrupted |= r.interrupted;
            let (content, round_thinking, calls) = (r.content, r.thinking, r.calls);
            if !round_thinking.trim().is_empty() {
                turn.steps.push(json!({"type": "thinking", "text": round_thinking.trim()}));
                thinking.push_str(round_thinking.trim());
                thinking.push_str("\n\n");
            }
            // a tool-less turn never runs a call, even one the model makes up
            if turn.interrupted || calls.is_empty() || !tools {
                turn.reply = content; // the final answer (or what was said before an interrupt)
                break;
            }

            // run this round's tool calls and hand the results back for the next round
            if !content.trim().is_empty() {
                turn.steps.push(json!({"type": "text", "text": content.trim()}));
            }
            messages.push(json!({"role": "assistant", "content": content, "tool_calls": calls}));
            for call in &calls {
                let name = call["function"]["name"].as_str().unwrap_or("?");
                let args = &call["function"]["arguments"];
                on(Delta::ToolCall { name, args });
                let output = tokio::select! {
                    o = tools::run(name, args, &self.db) => o,
                    _ = cancel.cancelled() => { turn.interrupted = true; String::from("(interrupted)").into() }
                };
                on(Delta::ToolResult { output: &output.text });
                if let Some(v) = &output.view {
                    on(Delta::View(v));
                }
                turn.steps.push(json!({"type": "tool", "name": name, "command": tools::label(name, args),
                                       "output": output.text, "view": output.view}));
                messages.push(if openai {
                    json!({"role": "tool", "tool_call_id": call["id"], "content": output.text})
                } else {
                    json!({"role": "tool", "tool_name": name, "content": output.text})
                });
                if turn.interrupted {
                    break;
                }
            }
            if turn.interrupted {
                break;
            }
            if round + 1 == MAX_ROUNDS {
                turn.reply = "(stopped: too many tool rounds in one turn)".into();
            }
        }
        if turn.reply.trim().is_empty() && turn.interrupted {
            turn.reply = "…".into();
        }
        turn.thinking = Some(thinking.trim().to_string()).filter(|t| !t.is_empty());
        self.store(text, &turn)?;
        if !reports.is_empty() {
            self.mark_delivered(&reports.iter().map(|r| r.0).collect::<Vec<_>>())?;
            info!(reports = reports.len(), "sub-agent reports delivered");
        }
        Ok((turn, stats))
    }
}

impl Agent {
    /// The OpenRouter key, fetched once from the secrets Worker and cached.
    async fn openrouter_key(&self) -> Result<String, String> {
        let mut k = self.or_key.lock().await;
        if k.is_none() {
            *k = Some(model::openrouter_key(&self.http).await?);
        }
        Ok(k.clone().unwrap())
    }

    /// Switch the model (from POST /model). Waits for a running turn to finish; kept in agents/memory/model.json.
    async fn set_model(&self, sel: model::Sel) -> Result<model::Sel, String> {
        sel.validate()?;
        let _turn = self.turn.lock().await;
        model::save(&self.sel_path, &sel)?;
        CTX_MAX.store(sel.num_ctx as u64, Ordering::Relaxed);
        info!(provider = %sel.provider, host = %sel.host, model = %sel.model, num_ctx = sel.num_ctx, "model set");
        *self.sel.write().unwrap() = sel.clone();
        Ok(sel)
    }
}

impl Agent {
    /// Slash commands (text starting with `/`). Ok(note) is shown to the user and not stored; Err is a usage error.
    ///   /think [on|off]  toggle the model's thinking (bare /think flips it)
    ///   /clear <N>   delete the last N turns (each user message and everything after it) from agents.db, so they
    ///                leave the model's context too (the context is rebuilt from the table every turn)
    async fn command(&self, text: &str) -> Result<String, String> {
        let mut parts = text.split_whitespace();
        match (parts.next(), parts.next(), parts.next()) {
            (Some("/clear"), Some(n), None) => {
                let n: u32 = n.parse().ok().filter(|&n| n > 0).ok_or("usage: /clear <N>, N a whole number above 0")?;
                self.clear(n).await
            }
            (Some("/clear"), _, _) => Err("usage: /clear <N>".into()),
            // voice needs a microphone: the web console's chat pane handles /voice itself (GET /voice here)
            (Some("/think"), arg, None) => {
                let on = match arg {
                    None => !self.think.load(Ordering::Relaxed),
                    Some("on") => true,
                    Some("off") => false,
                    Some(_) => return Err("usage: /think [on|off]".into()),
                };
                self.think.store(on, Ordering::Relaxed);
                info!(on, "thinking toggled");
                Ok(format!("thinking {}", if on { "on" } else { "off" }))
            }
            (Some("/think"), _, _) => Err("usage: /think [on|off]".into()),
            (Some("/voice"), _, _) => Err("voice mode runs in the web console's chat pane; type /voice there".into()),
            (Some("/model"), None, None) => Ok(format!("model: {} (pick another with /model in the web console)",
                                                       self.sel.read().unwrap().label())),
            (Some(cmd), _, _) => Err(format!("unknown command {cmd}; available: /clear <N>, /think [on|off], /model, /voice")),
            _ => Err("empty command".into()),
        }
    }

    async fn clear(&self, n: u32) -> Result<String, String> {
        let _turn = self.turn.lock().await; // not mid-turn
        let db = self.open().map_err(|e| e.to_string())?;
        // the Nth-newest user message starts the oldest turn to remove (fewer turns than N: remove them all)
        let cutoff: i64 = db
            .query_row("SELECT id FROM root WHERE role = 'user' ORDER BY id DESC LIMIT 1 OFFSET ?1", [n - 1], |r| r.get(0))
            .or_else(|_| db.query_row("SELECT COALESCE(MIN(id), 0) FROM root", [], |r| r.get(0)))
            .map_err(|e| e.to_string())?;
        let turns: u32 = db
            .query_row("SELECT COUNT(*) FROM root WHERE role = 'user' AND id >= ?1", [cutoff], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        let rows = db.execute("DELETE FROM root WHERE id >= ?1", [cutoff]).map_err(|e| e.to_string())?;
        info!(requested = n, turns, rows, "cleared");
        Ok(match turns {
            0 => "nothing to clear".into(),
            1 => "cleared the last turn".into(),
            t => format!("cleared the last {t} turns"),
        })
    }
}

/// Fill unset env vars from the shared max/.env (KEY=VALUE lines).
fn load_env(path: &PathBuf) {
    let Ok(text) = std::fs::read_to_string(path) else { return };
    for line in text.lines() {
        if let Some((k, v)) = line.trim().split_once('=') {
            let k = k.trim();
            if !k.starts_with('#') && std::env::var_os(k).is_none() {
                unsafe { std::env::set_var(k, v.trim().trim_matches(|c| c == '"' || c == '\'')) };
            }
        }
    }
}

type Shared = State<Arc<Agent>>;

async fn get_history(State(agent): Shared) -> (StatusCode, Json<Value>) {
    match agent.history() {
        Ok(rows) => (StatusCode::OK, Json(json!({
            "context": context(rows.iter().rev().find_map(|r| r.ctx).map(|c| c as u64)),
            "messages": rows.into_iter()
            .map(|r| json!({"role": r.msg.role, "content": r.msg.content, "thinking": r.thinking, "at": r.at,
                "tools": r.tools.and_then(|t| serde_json::from_str::<Value>(&t).ok()),
                "steps": r.steps.and_then(|t| serde_json::from_str::<Value>(&t).ok())}))
            .collect::<Vec<_>>()}))),
        Err(e) => {
            error!(error = %e, "history read failed");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e})))
        }
    }
}

#[derive(Deserialize)]
struct ChatIn {
    text: String,
}

/// POST /chat {"text"}: always answered with Server-Sent Events, one JSON event per `data:` line, plus a keep-alive
/// comment every 15 s so a long turn never sits silent behind a proxy timeout. A message streams the turn (same
/// events as voice mode: thinking_delta, reply_delta, tool_call, tool_result, then reply or error); a slash command
/// sends one {"type":"note"} or error. A turn runs in its own task: if the client goes away it still finishes and is
/// stored.
async fn post_chat(State(agent): Shared, Json(req): Json<ChatIn>) -> Response {
    let text = req.text.trim().to_string();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
    if text.is_empty() {
        let _ = tx.send(json!({"type": "error", "error": "empty message"}));
    } else if text.starts_with('/') {
        let _ = tx.send(match agent.command(&text).await {
            Ok(note) => json!({"type": "note", "text": note}),
            Err(e) => json!({"type": "error", "error": e}),
        });
    } else {
        tokio::spawn(async move {
            let result = agent.chat_stream(&text, &CancellationToken::new(), |d| { let _ = tx.send(delta_event(&d)); }).await;
            let _ = tx.send(result.map_or_else(|e| json!({"type": "error", "error": e}), |t| reply_event(&t)));
        });
    }
    let events = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|v| (Ok::<_, std::convert::Infallible>(Event::default().data(v.to_string())), rx))
    });
    Sse::new(events).keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15))).into_response()
}

/// GET /agents: every agent and how much of its context window is in use, for the console's Agents tab. Max: his
/// conversation (tokens of his last turn, out of the window). Sub-agents (board registry): their latest model call,
/// since each request starts a fresh window.
async fn get_agents(State(agent): Shared) -> Json<Value> {
    let max_ctx = agent.open().ok().and_then(|db| db.query_row(
        "SELECT ctx, created_at FROM root WHERE ctx IS NOT NULL ORDER BY id DESC LIMIT 1", [],
        |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))).ok());
    let sel = agent.sel.read().unwrap().clone();
    let mut list = vec![json!({"name": "max", "role": "orchestrator: talks to you, hands tasks to his sub-agents",
        "online": true, "model": sel.model, "ctx_used": max_ctx.as_ref().map(|c| c.0), "ctx_max": sel.num_ctx,
        "ctx_at": max_ctx.map(|c| c.1), "window": "conversation"})];
    if let Ok(c) = board::open(&agent.db) {
        for mut a in board::agents_status(&c).unwrap_or_default() {
            a["window"] = json!("per request");
            list.push(a);
        }
    }
    Json(json!({"agents": list}))
}

/// GET /model: the model in use. POST /model {provider, host, base, model, num_ctx}: switch to another.
async fn get_model(State(agent): Shared) -> Json<Value> {
    Json(json!(*agent.sel.read().unwrap()))
}

async fn post_model(State(agent): Shared, Json(sel): Json<model::Sel>) -> (StatusCode, Json<Value>) {
    match agent.set_model(sel).await {
        Ok(s) => (StatusCode::OK, Json(json!({"ok": true, "current": s, "note": format!("model: {}", s.label())}))),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": e}))),
    }
}

async fn serve(agent: Arc<Agent>) {
    let current = agent.sel.read().unwrap().label();
    tokio::spawn(discord::run(agent.clone()));
    let port = std::env::var("ROOT_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8007u16);
    tokio::spawn(push::watch(agent.clone()));
    let app = Router::new()
        .route("/history", get(get_history))
        .route("/chat", post(post_chat))
        .route("/voice", get(voice::upgrade))
        .route("/model", get(get_model).post(post_model))
        .route("/agents", get(get_agents))
        .route("/events", get(push::events))
        .route("/presence", post(push::presence))
        .with_state(agent);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.expect("bind");
    info!(port, model = %current, "listening");
    axum::serve(listener, app).await.unwrap();
}

async fn terminal(agent: Arc<Agent>) {
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut out = tokio::io::stdout();
    loop {
        out.write_all(b"you> ").await.ok();
        out.flush().await.ok();
        let Ok(Some(line)) = lines.next_line().await else { break };
        let text = line.trim();
        match text {
            "" => continue,
            "/exit" | "/quit" => break,
            _ if text.starts_with('/') => match agent.command(text).await {
                Ok(note) => println!("\x1b[2m{note}\x1b[0m"),
                Err(e) => eprintln!("{e}"),
            },
            _ => match agent.chat(text).await {
                Ok(t) => {
                    for step in &t.steps {
                        let txt = |k: &str| step[k].as_str().unwrap_or("").to_string();
                        match step["type"].as_str() {
                            Some("thinking") => println!("\x1b[2;3m{}\x1b[0m", txt("text")), // dim italic
                            Some("tool") => println!("\x1b[2m$ {}\n{}\x1b[0m", txt("command"), txt("output")),
                            _ => println!("\x1b[2m{}\x1b[0m", txt("text")),
                        }
                    }
                    println!("max> {}", t.reply)
                }
                Err(e) => eprintln!("error: {e}"),
            },
        }
    }
    println!();
}

#[tokio::main]
async fn main() {
    let agent = Arc::new(Agent::new());
    let mode = std::env::args().nth(1);
    // logs go to the shared agents/logs/agents.jsonl; stderr too when serving (not in the interactive chat)
    agentlog::init("root", &agent.home.join("agents/logs/agents.jsonl"), mode.as_deref() != Some("chat"));
    match mode.as_deref() {
        None | Some("serve") => serve(agent).await,
        Some("chat") => terminal(agent).await,
        Some(other) => eprintln!("unknown command {other:?}; use `root` (serve) or `root chat`"),
    }
}
