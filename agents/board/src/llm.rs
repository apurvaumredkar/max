//! The sub-agents' model: always the one Max is on (agents/memory/model.json, set with /model in the web console;
//! re-read for every request), with the same num_ctx, so Ollama never reloads the model to switch between agents.
//! Non-streaming: Ollama's native /api/chat, or OpenRouter's OpenAI-style /chat/completions.

use std::path::Path;

use serde_json::{Value, json};

pub struct Sel {
    pub provider: String,
    pub base: String,
    pub model: String,
    pub num_ctx: u64,
}

pub fn load(home: &Path) -> Sel {
    let v: Value = std::fs::read_to_string(home.join("agents/memory/model.json")).ok()
        .and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    let ollama = std::env::var("OLLAMA_URL").unwrap_or_default().trim_end_matches('/').trim_end_matches("/v1").to_string();
    Sel {
        provider: v["provider"].as_str().unwrap_or("ollama").into(),
        base: v["base"].as_str().map(String::from).unwrap_or(ollama),
        model: v["model"].as_str().unwrap_or("gemma4:e4b").into(),
        num_ctx: v["num_ctx"].as_u64().unwrap_or(131072),
    }
}

pub struct Call {
    pub id: Value,
    pub name: String,
    pub args: Value,
}

pub struct Reply {
    pub content: String,
    pub calls: Vec<Call>,
    pub message: Value, // the assistant message, to append to the conversation
    pub tokens: u64,    // what this call filled of the context window: prompt + output tokens (0 if not reported)
}

pub async fn chat(http: &reqwest::Client, sel: &Sel, key: Option<&str>, messages: &[Value], tools: &Value)
    -> Result<Reply, String> {
    if sel.provider == "openrouter" {
        let key = key.ok_or("no OpenRouter key")?;
        let v: Value = http.post(format!("{}/chat/completions", sel.base.trim_end_matches('/'))).bearer_auth(key)
            .header("X-Title", "Max (sub-agent)")
            .json(&{
                let mut body = json!({"model": sel.model, "messages": messages, "tools": tools});
                if tools.as_array().is_some_and(|t| t.is_empty()) {
                    body.as_object_mut().unwrap().remove("tools"); // OpenAI-style APIs reject an empty list
                }
                body
            })
            .send().await.and_then(|r| r.error_for_status()).map_err(|e| format!("openrouter: {e}"))?
            .json().await.map_err(|e| format!("openrouter: {e}"))?;
        let m = v["choices"][0]["message"].clone();
        let calls = m["tool_calls"].as_array().into_iter().flatten().map(|c| Call {
            id: c["id"].clone(),
            name: c["function"]["name"].as_str().unwrap_or("").into(),
            args: c["function"]["arguments"].as_str().and_then(|a| serde_json::from_str(a).ok()).unwrap_or(json!({})),
        }).collect();
        let tokens = v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) + v["usage"]["completion_tokens"].as_u64().unwrap_or(0);
        return Ok(Reply { content: m["content"].as_str().unwrap_or("").into(), calls, message: m, tokens });
    }
    let v: Value = http.post(format!("{}/api/chat", sel.base.trim_end_matches('/')))
        .json(&json!({"model": sel.model, "messages": messages, "tools": tools, "stream": false,
                      "options": {"num_ctx": sel.num_ctx}}))
        .send().await.and_then(|r| r.error_for_status()).map_err(|e| format!("ollama: {e}"))?
        .json().await.map_err(|e| format!("ollama: {e}"))?;
    let m = &v["message"];
    let calls = m["tool_calls"].as_array().into_iter().flatten().map(|c| Call {
        id: Value::Null,
        name: c["function"]["name"].as_str().unwrap_or("").into(),
        args: match &c["function"]["arguments"] {
            Value::String(s) => serde_json::from_str(s).unwrap_or(json!({})),
            a => a.clone(),
        },
    }).collect();
    let message = json!({"role": "assistant", "content": m["content"], "tool_calls": m["tool_calls"]});
    let tokens = v["prompt_eval_count"].as_u64().unwrap_or(0) + v["eval_count"].as_u64().unwrap_or(0);
    Ok(Reply { content: m["content"].as_str().unwrap_or("").into(), calls, message, tokens })
}

/// A tool result message in the provider's format.
pub fn tool_message(sel: &Sel, call: &Call, content: &str) -> Value {
    if sel.provider == "openrouter" {
        json!({"role": "tool", "tool_call_id": call.id, "content": content})
    } else {
        json!({"role": "tool", "tool_name": call.name, "content": content})
    }
}
