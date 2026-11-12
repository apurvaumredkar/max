//! Root agent: chats with Ollama, using context/MAX.md as the persona and keeping the history in
//! agents/agents.db (table `root`).
//!
//!   root            HTTP service on 127.0.0.1:$ROOT_PORT (default 8007): GET /history, POST /chat {"text"}
//!   root chat       the same conversation as a terminal loop (/exit or Ctrl-D to quit)
//!
//! Slash commands (web and terminal): /clear <N> deletes the last N turns from the history and the context;
//! /think [on|off] toggles the model's thinking (on at start).
//! /voice (web console only) opens a voice session on GET /voice (WebSocket): see voice.rs.
//!
//! Logs (via the shared `agentlog` crate) go to agents/logs/agents.jsonl, which the web console streams.

mod voice;

use std::{path::PathBuf, sync::{Arc, atomic::{AtomicBool, Ordering}}, time::Instant};

use agentlog::{error, info};

use axum::{Json, Router, extract::State, http::StatusCode, routing::{get, post}};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use futures_util::StreamExt;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio_util::sync::CancellationToken;

const MODEL: &str = "ornith-1.5:35b";
const NUM_CTX: u32 = 65536;

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

/// A piece of a streamed turn.
pub(crate) enum Delta<'a> {
    Thinking(&'a str),
    Content(&'a str),
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
                thinking TEXT)",
        )?;
        // tables made before thoughts were kept lack the column; add it (fails harmlessly once it exists)
        let _ = db.execute("ALTER TABLE root ADD COLUMN thinking TEXT", []);
        Ok(db)
    }

    /// Every stored message with its timestamp and (for replies) the model's thoughts.
    fn history(&self) -> Result<Vec<(Msg, String, Option<String>)>, String> {
        let db = self.open().map_err(|e| e.to_string())?;
        let mut st = db.prepare("SELECT role, content, created_at, thinking FROM root ORDER BY id")
            .map_err(|e| e.to_string())?;
        st.query_map([], |r| Ok((Msg { role: r.get(0)?, content: r.get(1)? }, r.get(2)?, r.get(3)?)))
            .and_then(|rows| rows.collect())
            .map_err(|e| e.to_string())
    }

    /// One exchange: send persona + full history + `text`, store both sides once the reply arrives.
    /// Returns (reply, thoughts). Past thoughts are kept for display but not sent back to the model.
    async fn chat(&self, text: &str) -> Result<(String, Option<String>), String> {
        let _turn = self.turn.lock().await;
        let started = Instant::now();
        info!(chars = text.chars().count(), "user message");
        let result = self.turn_inner(text).await;
        let ms = started.elapsed().as_millis() as u64;
        match &result {
            Ok((reply, thinking, stats)) => info!(ms, prompt_tokens = stats.0, reply_tokens = stats.1,
                chars = reply.chars().count(), thinking_chars = thinking.as_deref().map_or(0, |t| t.chars().count()),
                "reply"),
            Err(e) => error!(ms, error = %e, "turn failed"),
        }
        result.map(|(reply, thinking, _)| (reply, thinking))
    }

    /// What the model sees for a turn: persona, the stored conversation (no thoughts), then `text`.
    fn messages_for(&self, text: &str) -> Result<Vec<Msg>, String> {
        let persona = std::fs::read_to_string(&self.persona)
            .map_err(|e| format!("{}: {e}", self.persona.display()))?;
        let mut messages = vec![Msg { role: "system".into(), content: persona }];
        messages.extend(self.history()?.into_iter().map(|(m, _, _)| m));
        messages.push(Msg { role: "user".into(), content: text.into() });
        Ok(messages)
    }

    fn store(&self, text: &str, reply: &str, thinking: Option<&str>) -> Result<(), String> {
        let db = self.open().map_err(|e| e.to_string())?;
        db.execute("INSERT INTO root (role, content) VALUES ('user', ?1)", [text]).map_err(|e| e.to_string())?;
        db.execute("INSERT INTO root (role, content, thinking) VALUES ('assistant', ?1, ?2)",
                   rusqlite::params![reply, thinking]).map_err(|e| e.to_string())?;
        Ok(())
    }

    async fn turn_inner(&self, text: &str) -> Result<(String, Option<String>, (u64, u64)), String> {
        let messages = self.messages_for(text)?;
        let resp = self
            .http
            .post(&self.chat_url)
            .json(&json!({"model": MODEL, "messages": messages, "stream": false, "think": self.think.load(Ordering::Relaxed),
                          "options": {"num_ctx": NUM_CTX}}))
            .send()
            .await
            .map_err(|e| format!("ollama: {e}"))?;
        let status = resp.status();
        let body: Value = resp.json().await.map_err(|e| format!("ollama: {e}"))?;
        if !status.is_success() {
            return Err(format!("ollama {status}: {}", body["error"].as_str().unwrap_or("?")));
        }
        let reply = body["message"]["content"].as_str().ok_or("ollama: no message in reply")?.to_string();
        let thinking = body["message"]["thinking"].as_str().map(str::trim).filter(|t| !t.is_empty()).map(String::from);
        let stats = (body["prompt_eval_count"].as_u64().unwrap_or(0), body["eval_count"].as_u64().unwrap_or(0));

        self.store(text, &reply, thinking.as_deref())?;
        Ok((reply, thinking, stats))
    }

    /// Like `chat`, but streams: `on` gets each piece of thinking and reply as the model writes it. Cancelling `cancel`
    /// stops generation; the partial reply is stored (so the context matches what was said). Returns
    /// (reply, thoughts, interrupted).
    pub(crate) async fn chat_stream(&self, text: &str, cancel: &CancellationToken, mut on: impl FnMut(Delta))
        -> Result<(String, Option<String>, bool), String> {
        let _turn = self.turn.lock().await;
        let started = Instant::now();
        info!(chars = text.chars().count(), stream = true, "user message");
        let result = self.stream_inner(text, cancel, &mut on).await;
        let ms = started.elapsed().as_millis() as u64;
        match &result {
            Ok((reply, thinking, interrupted, stats)) => info!(ms, prompt_tokens = stats.0, reply_tokens = stats.1,
                chars = reply.chars().count(), thinking_chars = thinking.as_deref().map_or(0, |t| t.chars().count()),
                interrupted, "reply"),
            Err(e) => error!(ms, error = %e, "turn failed"),
        }
        result.map(|(reply, thinking, interrupted, _)| (reply, thinking, interrupted))
    }

    async fn stream_inner(&self, text: &str, cancel: &CancellationToken, on: &mut impl FnMut(Delta))
        -> Result<(String, Option<String>, bool, (u64, u64)), String> {
        let messages = self.messages_for(text)?;
        let req = self.http.post(&self.chat_url)
            .json(&json!({"model": MODEL, "messages": messages, "stream": true, "think": self.think.load(Ordering::Relaxed),
                          "options": {"num_ctx": NUM_CTX}}))
            .send();
        let (mut reply, mut thinking, mut stats, mut interrupted) = (String::new(), String::new(), (0, 0), false);
        let resp = tokio::select! {
            r = req => Some(r.map_err(|e| format!("ollama: {e}"))?),
            _ = cancel.cancelled() => None,
        };
        if let Some(resp) = resp {
            if !resp.status().is_success() {
                let status = resp.status();
                let body: Value = resp.json().await.unwrap_or_default();
                return Err(format!("ollama {status}: {}", body["error"].as_str().unwrap_or("?")));
            }
            let (mut body, mut line) = (resp.bytes_stream(), Vec::<u8>::new());
            loop {
                let chunk = tokio::select! {
                    c = body.next() => c,
                    _ = cancel.cancelled() => { interrupted = true; break } // dropping the stream stops Ollama
                };
                let Some(chunk) = chunk else { break };
                line.extend_from_slice(&chunk.map_err(|e| format!("ollama: {e}"))?);
                while let Some(end) = line.iter().position(|&b| b == b'\n') {
                    let Ok(v) = serde_json::from_slice::<Value>(&line.drain(..=end).collect::<Vec<_>>()) else { continue };
                    if let Some(e) = v["error"].as_str() {
                        return Err(format!("ollama: {e}"));
                    }
                    if let Some(t) = v["message"]["thinking"].as_str().filter(|t| !t.is_empty()) {
                        thinking.push_str(t);
                        on(Delta::Thinking(t));
                    }
                    if let Some(c) = v["message"]["content"].as_str().filter(|c| !c.is_empty()) {
                        reply.push_str(c);
                        on(Delta::Content(c));
                    }
                    if v["done"].as_bool() == Some(true) {
                        stats = (v["prompt_eval_count"].as_u64().unwrap_or(0), v["eval_count"].as_u64().unwrap_or(0));
                    }
                }
            }
        } else {
            interrupted = true;
        }
        let reply = if reply.trim().is_empty() && interrupted { "…".to_string() } else { reply };
        let thinking = Some(thinking.trim().to_string()).filter(|t| !t.is_empty());
        self.store(text, &reply, thinking.as_deref())?;
        Ok((reply, thinking, interrupted, stats))
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
        Ok(rows) => (StatusCode::OK, Json(json!({"messages": rows.into_iter()
            .map(|(m, at, thinking)| json!({"role": m.role, "content": m.content, "thinking": thinking, "at": at}))
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

async fn post_chat(State(agent): Shared, Json(req): Json<ChatIn>) -> (StatusCode, Json<Value>) {
    let text = req.text.trim();
    if text.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "empty message"})));
    }
    if text.starts_with('/') {
        return match agent.command(text).await {
            Ok(note) => (StatusCode::OK, Json(json!({"reply": note, "command": true}))),
            Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": e, "command": true}))),
        };
    }
    match agent.chat(text).await {
        Ok((reply, thinking)) => (StatusCode::OK, Json(json!({"reply": reply, "thinking": thinking}))),
        Err(e) => (StatusCode::BAD_GATEWAY, Json(json!({"error": e}))),
    }
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
                Ok((reply, thinking)) => {
                    if let Some(t) = thinking {
                        println!("\x1b[2;3m{t}\x1b[0m"); // dim italic
                    }
                    println!("max> {reply}")
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
