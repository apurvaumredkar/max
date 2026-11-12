//! The root agent's tools, offered to the model through Ollama's native tool calling (the `tools` field of
//! /api/chat). The model answers with `message.tool_calls`; the agent runs them here and sends each result back as a
//! `{"role":"tool","tool_name":..,"content":..}` message, looping until the model replies without calling a tool.
//!
//! No sandbox yet: `bash` runs anything, as the agent's user.

use std::{process::Stdio, time::{Duration, Instant}};

use agentlog::info;
use serde_json::{Value, json};

const BASH_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_OUTPUT: usize = 12_000; // chars per stream sent back to the model

/// Tool definitions for the request body.
pub fn specs() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command with bash on Max's own machine (Linux, aarch64) and get back its \
                exit code, stdout and stderr. Runs in the home directory with no stdin and a 120 second timeout. \
                Use it to look things up or get things done on the machine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "the command or script to run"}
                },
                "required": ["command"]
            }
        }
    }])
}

/// Arguments as an object (some models send them as a JSON string).
fn args_of(args: &Value) -> Value {
    match args {
        Value::String(s) => serde_json::from_str(s).unwrap_or_default(),
        v => v.clone(),
    }
}

/// A one-line description of a call, for display and logs.
pub fn label(name: &str, args: &Value) -> String {
    let args = args_of(args);
    match name {
        "bash" => args["command"].as_str().unwrap_or("").to_string(),
        _ => format!("{name} {args}"),
    }
}

/// Run one tool call; the result is always text for the model (errors included).
pub async fn run(name: &str, args: &Value) -> String {
    let args = args_of(args);
    match name {
        "bash" => match args["command"].as_str() {
            Some(cmd) if !cmd.trim().is_empty() => bash(cmd).await,
            _ => "error: bash needs a non-empty \"command\"".into(),
        },
        _ => format!("error: unknown tool {name:?}; available: bash"),
    }
}

async fn bash(cmd: &str) -> String {
    let started = Instant::now();
    let home = std::env::var("HOME").unwrap_or_else(|_| "/".into());
    let child = tokio::process::Command::new("bash")
        .arg("-c")
        .arg(cmd)
        .current_dir(home)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true) // timeout or interrupt drops the future, which kills bash
        .spawn();
    let result = match child {
        Err(e) => format!("error: could not start bash: {e}"),
        Ok(child) => match tokio::time::timeout(BASH_TIMEOUT, child.wait_with_output()).await {
            Err(_) => format!("error: timed out after {} s (killed)", BASH_TIMEOUT.as_secs()),
            Ok(Err(e)) => format!("error: {e}"),
            Ok(Ok(out)) => {
                let code = out.status.code().map_or("killed by signal".to_string(), |c| c.to_string());
                let mut s = format!("exit code: {code}");
                for (label, bytes) in [("stdout", &out.stdout), ("stderr", &out.stderr)] {
                    let text = String::from_utf8_lossy(bytes);
                    if !text.trim().is_empty() {
                        s.push_str(&format!("\n--- {label}\n{}", clip(text.trim_end())));
                    }
                }
                s
            }
        },
    };
    info!(command = %cmd, ms = started.elapsed().as_millis() as u64,
          result = %result.lines().next().unwrap_or(""), "tool: bash");
    result
}

/// Keep the head and tail of long output.
fn clip(s: &str) -> String {
    let n = s.chars().count();
    if n <= MAX_OUTPUT {
        return s.to_string();
    }
    let head: String = s.chars().take(MAX_OUTPUT * 2 / 3).collect();
    let tail: String = s.chars().skip(n - MAX_OUTPUT / 3).collect();
    format!("{head}\n[... {} chars cut ...]\n{tail}", n - MAX_OUTPUT)
}
