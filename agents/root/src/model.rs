//! Which model Max runs on, and one streamed request to it.
//!
//! Providers:
//!   - `ollama`: any inference box's Ollama, native /api/chat (NDJSON stream); `num_ctx` is sent as an option.
//!   - `openrouter`: OpenAI-style /chat/completions (SSE stream); key OPENROUTER_API_KEY from the secrets Worker.
//!     The model runs at its own context size there; `num_ctx` only sets the context meter's maximum.
//! The choice is set with POST /model (the console's /model picker) and kept in agents/memory/model.json.

use std::path::Path;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::Delta;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sel {
    pub provider: String, // "ollama" | "openrouter"
    pub host: String,     // display label: "windows", "macOS", "openrouter"
    pub base: String,     // http://<host>:11434 for Ollama, https://openrouter.ai/api/v1 for OpenRouter
    pub model: String,
    pub num_ctx: u32,
}

impl Sel {
    pub fn label(&self) -> String {
        format!("{} on {}, context {}", self.model, self.host, self.num_ctx)
    }

    pub fn validate(&self) -> Result<(), String> {
        match self.provider.as_str() {
            "ollama" if self.base.starts_with("http://") || self.base.starts_with("https://") => {}
            "ollama" => return Err("ollama needs an http(s) base URL".into()),
            "openrouter" => {}
            p => return Err(format!("unknown provider {p:?}")),
        }
        if self.model.trim().is_empty() {
            return Err("no model".into());
        }
        if !(1024..=2_097_152).contains(&self.num_ctx) {
            return Err("num_ctx must be between 1024 and 2097152".into());
        }
        Ok(())
    }
}

pub fn load(path: &Path) -> Option<Sel> {
    let s: Sel = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    s.validate().ok().map(|_| s)
}

pub fn save(path: &Path, sel: &Sel) -> Result<(), String> {
    std::fs::write(path, serde_json::to_string_pretty(sel).unwrap()).map_err(|e| format!("{}: {e}", path.display()))
}

/// What one model request produced. `calls` are in the provider's own format (sent back as they are).
#[derive(Default)]
pub struct Round {
    pub content: String,
    pub thinking: String,
    pub calls: Vec<Value>,
    pub prompt: u64,
    pub eval: u64,
    pub interrupted: bool,
}

/// One request to Ollama's /api/chat, streamed (NDJSON: thinking/content pieces, tool calls, then a stats line).
pub async fn ollama(http: &reqwest::Client, sel: &Sel, messages: &[Value], tools: &Value, think: bool,
                    cancel: &CancellationToken, on: &mut impl FnMut(Delta)) -> Result<Round, String> {
    let mut r = Round::default();
    let req = http.post(format!("{}/api/chat", sel.base.trim_end_matches('/')))
        .json(&json!({"model": sel.model, "messages": messages, "tools": tools, "stream": true, "think": think,
                      "options": {"num_ctx": sel.num_ctx}}))
        .send();
    let resp = tokio::select! {
        x = req => x.map_err(|e| format!("ollama: {e}"))?,
        _ = cancel.cancelled() => { r.interrupted = true; return Ok(r) }
    };
    if !resp.status().is_success() {
        let status = resp.status();
        let body: Value = resp.json().await.unwrap_or_default();
        return Err(format!("ollama {status}: {}", body["error"].as_str().unwrap_or("?")));
    }
    let (mut body, mut line) = (resp.bytes_stream(), Vec::<u8>::new());
    loop {
        let chunk = tokio::select! {
            c = body.next() => c,
            _ = cancel.cancelled() => { r.interrupted = true; break } // dropping the stream stops Ollama
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
                r.thinking.push_str(t);
                on(Delta::Thinking(t));
            }
            if let Some(c) = m["content"].as_str().filter(|c| !c.is_empty()) {
                r.content.push_str(c);
                on(Delta::Content(c));
            }
            if let Some(tc) = m["tool_calls"].as_array() {
                r.calls.extend(tc.iter().cloned());
            }
            if v["done"].as_bool() == Some(true) {
                r.prompt = v["prompt_eval_count"].as_u64().unwrap_or(0);
                r.eval = v["eval_count"].as_u64().unwrap_or(0);
            }
        }
    }
    Ok(r)
}

/// One request to OpenRouter's /chat/completions, streamed (SSE `data:` chunks; tool calls arrive in pieces, keyed
/// by index, and are put back together here).
pub async fn openrouter(http: &reqwest::Client, key: &str, sel: &Sel, messages: &[Value], tools: &Value, think: bool,
                        cancel: &CancellationToken, on: &mut impl FnMut(Delta)) -> Result<Round, String> {
    let mut r = Round::default();
    let req = http.post(format!("{}/chat/completions", sel.base.trim_end_matches('/')))
        .bearer_auth(key)
        .header("X-Title", "Max")
        .json(&{
            let mut body = json!({"model": sel.model, "messages": messages, "tools": tools, "stream": true,
                                  "stream_options": {"include_usage": true}, "reasoning": {"enabled": think}});
            if tools.as_array().is_some_and(|t| t.is_empty()) {
                body.as_object_mut().unwrap().remove("tools"); // OpenAI-style APIs reject an empty list
            }
            body
        })
        .send();
    let resp = tokio::select! {
        x = req => x.map_err(|e| format!("openrouter: {e}"))?,
        _ = cancel.cancelled() => { r.interrupted = true; return Ok(r) }
    };
    if !resp.status().is_success() {
        let status = resp.status();
        let body: Value = resp.json().await.unwrap_or_default();
        return Err(format!("openrouter {status}: {}", body["error"]["message"].as_str().unwrap_or("?")));
    }
    let mut parts: Vec<(String, String, String)> = Vec::new(); // (id, name, arguments) per tool call index
    let (mut body, mut line) = (resp.bytes_stream(), Vec::<u8>::new());
    'read: loop {
        let chunk = tokio::select! {
            c = body.next() => c,
            _ = cancel.cancelled() => { r.interrupted = true; break }
        };
        let Some(chunk) = chunk else { break };
        line.extend_from_slice(&chunk.map_err(|e| format!("openrouter: {e}"))?);
        while let Some(end) = line.iter().position(|&b| b == b'\n') {
            let raw = String::from_utf8_lossy(&line.drain(..=end).collect::<Vec<_>>()).trim().to_string();
            let Some(data) = raw.strip_prefix("data:").map(str::trim) else { continue }; // ": OPENROUTER PROCESSING"
            if data == "[DONE]" {
                break 'read;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
            if let Some(e) = v["error"]["message"].as_str() {
                return Err(format!("openrouter: {e}"));
            }
            let d = &v["choices"][0]["delta"];
            if let Some(t) = d["reasoning"].as_str().filter(|t| !t.is_empty()) {
                r.thinking.push_str(t);
                on(Delta::Thinking(t));
            }
            if let Some(c) = d["content"].as_str().filter(|c| !c.is_empty()) {
                r.content.push_str(c);
                on(Delta::Content(c));
            }
            for tc in d["tool_calls"].as_array().into_iter().flatten() {
                let i = tc["index"].as_u64().unwrap_or(parts.len() as u64) as usize;
                while parts.len() <= i {
                    parts.push(Default::default());
                }
                if let Some(id) = tc["id"].as_str() {
                    parts[i].0 = id.to_string();
                }
                if let Some(n) = tc["function"]["name"].as_str() {
                    parts[i].1.push_str(n);
                }
                if let Some(a) = tc["function"]["arguments"].as_str() {
                    parts[i].2.push_str(a);
                }
            }
            if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
                r.prompt = u["prompt_tokens"].as_u64().unwrap_or(0);
                r.eval = u["completion_tokens"].as_u64().unwrap_or(0);
            }
        }
    }
    r.calls = parts
        .into_iter()
        .enumerate()
        .filter(|(_, (_, name, _))| !name.is_empty())
        .map(|(i, (id, name, args))| {
            let id = if id.is_empty() { format!("call_{i}") } else { id };
            json!({"id": id, "type": "function", "function": {"name": name,
                   "arguments": if args.trim().is_empty() { "{}".to_string() } else { args }}})
        })
        .collect();
    Ok(r)
}

/// OPENROUTER_API_KEY from the Access-gated secrets Worker.
pub async fn openrouter_key(http: &reqwest::Client) -> Result<String, String> {
    secret(http, "OPENROUTER_API_KEY").await
}

/// A secret by name from the Access-gated secrets Worker (same flow as the console: `cloudflared access token`).
pub async fn secret(http: &reqwest::Client, name: &str) -> Result<String, String> {
    let url = std::env::var("SECRETS_URL").map_err(|_| "SECRETS_URL not set in max/.env".to_string())?;
    let cf = std::env::var("CLOUDFLARED").unwrap_or_else(|_| {
        format!("{}/.local/bin/cloudflared", std::env::var("HOME").unwrap_or_default())
    });
    let out = tokio::process::Command::new(&cf).args(["access", "token", &format!("-app={url}")]).output().await
        .map_err(|e| format!("{cf}: {e}"))?;
    if !out.status.success() {
        return Err(format!("no Access login: run `cloudflared access login {url}`"));
    }
    let token = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let v: Value = http.get(format!("{url}/{name}"))
        .header("cf-access-token", token)
        .header("User-Agent", "max-root")
        .send().await.and_then(|r| r.error_for_status()).map_err(|e| format!("secrets: {e}"))?
        .json().await.map_err(|e| format!("secrets: {e}"))?;
    v["value"].as_str().map(String::from).ok_or_else(|| "secrets: no value".into())
}
