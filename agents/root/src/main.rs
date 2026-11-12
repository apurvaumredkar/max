//! Root agent: chats with Ollama, using context/MAX.md as the persona and keeping the history in
//! agents/agents.db (table `root`).
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
//!
//! Logs (via the shared `agentlog` crate) go to agents/logs/agents.jsonl, which the web console streams.

mod tools;
mod voice;

use std::{path::PathBuf, sync::{Arc, atomic::{AtomicBool, Ordering}}, time::Instant};

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
use futures_util::StreamExt;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio_util::sync::CancellationToken;

const MODEL: &str = "gemma4:e4b";
const NUM_CTX: u32 = 131072; // gemma4:e4b full context

struct Agent {
    home: PathBuf,
    db: PathBuf,
    persona: PathBuf,
    chat_url: String,
    stt_url: String,
    tts_url: String,
    http: reqwest::Client,
    turn: tokio::sync::Mutex<()>, // one turn at a time so the history stays in order
    think: AtomicBool,            // model thinking on/off (/think); starts on, not kept across restarts
}

const MAX_ROUNDS: usize = 25; // model calls per turn (each tool round is one more)

/// A piece of a streamed turn.
pub(crate) enum Delta<'a> {
    Thinking(&'a str),
    Content(&'a str),
    ToolCall { name: &'a str, args: &'a Value },
    ToolResult { output: &'a str },
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
    json!({"used": used, "max": NUM_CTX})
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
        Agent {
            db: home.join("agents/agents.db"),
            persona: home.join("context/MAX.md"),
            chat_url: format!("{base}/api/chat"),
            stt_url: need("STT_URL"),
            tts_url: need("TTS_URL"),
            http: reqwest::Client::builder().timeout(std::time::Duration::from_secs(600)).build().unwrap(),
            turn: tokio::sync::Mutex::new(()),
            think: AtomicBool::new(true),
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
        Ok(db)
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

    /// What the model sees for a turn: persona, the stored conversation, then `text`. Past thoughts and tool calls
    /// are kept for display only and not sent back.
    fn messages_for(&self, text: &str) -> Result<Vec<Value>, String> {
        let persona = std::fs::read_to_string(&self.persona)
            .map_err(|e| format!("{}: {e}", self.persona.display()))?;
        let mut messages = vec![json!({"role": "system", "content": persona})];
        messages.extend(self.history()?.into_iter().map(|r| json!({"role": r.msg.role, "content": r.msg.content})));
        messages.push(json!({"role": "user", "content": text}));
        Ok(messages)
    }

    fn store(&self, text: &str, turn: &Turn) -> Result<(), String> {
        let db = self.open().map_err(|e| e.to_string())?;
        let steps = (!turn.steps.is_empty()).then(|| Value::Array(turn.steps.clone()).to_string());
        db.execute("INSERT INTO root (role, content) VALUES ('user', ?1)", [text]).map_err(|e| e.to_string())?;
        db.execute("INSERT INTO root (role, content, thinking, steps, ctx) VALUES ('assistant', ?1, ?2, ?3, ?4)",
                   rusqlite::params![turn.reply, turn.thinking, steps, turn.ctx.map(|c| c as i64)])
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// One turn, streamed: `on` gets thinking, reply text and tool activity as they happen. The model may call tools;
    /// each round of calls is run and sent back until it answers without one (at most MAX_ROUNDS requests).
    /// Cancelling `cancel` stops generation or the running tool; the partial turn is stored, so the context matches
    /// what was said.
    pub(crate) async fn chat_stream(&self, text: &str, cancel: &CancellationToken, mut on: impl FnMut(Delta))
        -> Result<Turn, String> {
        let _turn = self.turn.lock().await;
        let started = Instant::now();
        info!(chars = text.chars().count(), "user message");
        let result = self.turn_inner(text, cancel, &mut on).await;
        let ms = started.elapsed().as_millis() as u64;
        match &result {
            Ok((t, stats)) => info!(ms, prompt_tokens = stats.0, reply_tokens = stats.1, chars = t.reply.chars().count(),
                thinking_chars = t.thinking.as_deref().map_or(0, |t| t.chars().count()), steps = t.steps.len(),
                interrupted = t.interrupted, "reply"),
            Err(e) => error!(ms, error = %e, "turn failed"),
        }
        result.map(|(t, _)| t)
    }

    async fn turn_inner(&self, text: &str, cancel: &CancellationToken, on: &mut impl FnMut(Delta))
        -> Result<(Turn, (u64, u64)), String> {
        let mut messages = self.messages_for(text)?;
        let (mut turn, mut thinking, mut stats) = (Turn::default(), String::new(), (0, 0));
        // Within the turn the model sees its own tool calls and results (`messages` grows); none of it is stored
        // as context: only `text` and the final `turn.reply` are, while `turn.steps` keeps it all for display.
        for round in 0..MAX_ROUNDS {
            let req = self.http.post(&self.chat_url)
                .json(&json!({"model": MODEL, "messages": messages, "tools": tools::specs(), "stream": true,
                              "think": self.think.load(Ordering::Relaxed), "options": {"num_ctx": NUM_CTX}}))
                .send();
            let resp = tokio::select! {
                r = req => r.map_err(|e| format!("ollama: {e}"))?,
                _ = cancel.cancelled() => { turn.interrupted = true; break }
            };
            if !resp.status().is_success() {
                let status = resp.status();
                let body: Value = resp.json().await.unwrap_or_default();
                return Err(format!("ollama {status}: {}", body["error"].as_str().unwrap_or("?")));
            }

            // read the NDJSON stream: thinking/content pieces, tool calls, and the final stats line
            let (mut body, mut line) = (resp.bytes_stream(), Vec::<u8>::new());
            let (mut content, mut round_thinking, mut calls) = (String::new(), String::new(), Vec::<Value>::new());
            loop {
                let chunk = tokio::select! {
                    c = body.next() => c,
                    _ = cancel.cancelled() => { turn.interrupted = true; break } // dropping the stream stops Ollama
                };
                let Some(chunk) = chunk else { break };
                line.extend_from_slice(&chunk.map_err(|e| format!("ollama: {e}"))?);
                while let Some(end) = line.iter().position(|&b| b == b'\n') {
                    let Ok(v) = serde_json::from_slice::<Value>(&line.drain(..=end).collect::<Vec<_>>()) else { continue };
                    if let Some(e) = v["error"].as_str() {
                        return Err(format!("ollama: {e}"));
                    }
                    let m = &v["message"];
                    if let Some(t) = m["thinking"].as_str().filter(|t| !t.is_empty()) {
                        round_thinking.push_str(t);
                        on(Delta::Thinking(t));
                    }
                    if let Some(c) = m["content"].as_str().filter(|c| !c.is_empty()) {
                        content.push_str(c);
                        on(Delta::Content(c));
                    }
                    if let Some(tc) = m["tool_calls"].as_array() {
                        calls.extend(tc.iter().cloned());
                    }
                    if v["done"].as_bool() == Some(true) {
                        stats.0 += v["prompt_eval_count"].as_u64().unwrap_or(0);
                        stats.1 += v["eval_count"].as_u64().unwrap_or(0);
                        // last request wins: what the window held when the model finished
                        turn.ctx = Some(v["prompt_eval_count"].as_u64().unwrap_or(0) + v["eval_count"].as_u64().unwrap_or(0));
                    }
                }
            }
            if !round_thinking.trim().is_empty() {
                turn.steps.push(json!({"type": "thinking", "text": round_thinking.trim()}));
                thinking.push_str(round_thinking.trim());
                thinking.push_str("\n\n");
            }
            if turn.interrupted || calls.is_empty() {
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
                    o = tools::run(name, args) => o,
                    _ = cancel.cancelled() => { turn.interrupted = true; "(interrupted)".to_string() }
                };
                on(Delta::ToolResult { output: &output });
                turn.steps.push(json!({"type": "tool", "name": name, "command": tools::label(name, args), "output": output}));
                messages.push(json!({"role": "tool", "tool_name": name, "content": output}));
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
        Ok((turn, stats))
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
            (Some(cmd), _, _) => Err(format!("unknown command {cmd}; available: /clear <N>, /think [on|off], /voice")),
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

async fn serve(agent: Arc<Agent>) {
    let agent_url = agent.chat_url.clone();
    let port = std::env::var("ROOT_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8007u16);
    let app = Router::new()
        .route("/history", get(get_history))
        .route("/chat", post(post_chat))
        .route("/voice", get(voice::upgrade))
        .with_state(agent);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.expect("bind");
    info!(port, model = MODEL, ollama = %agent_url, "listening");
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
