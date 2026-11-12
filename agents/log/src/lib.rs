//! Shared logging for Max's agents. Every agent calls `agentlog::init("<name>", ..)` once at startup and then uses the
//! `tracing` macros (`info!`, `warn!`, `error!`, ...), re-exported here as `agentlog::{info, ..}`.
//!
//! Each event becomes one JSON line in agents/logs/agents.jsonl, shared by all agents (the web console tails it):
//!   {"ts":"2026-11-11T21:04:05.123-05:00","agent":"root","level":"INFO","msg":"reply","fields":{"ms":5120}}
//! Level filter comes from RUST_LOG (default `info`).

use std::{fs::{File, OpenOptions}, io::Write, path::Path, sync::Mutex};

use serde_json::{Map, Value, json};
use tracing::{Event, Subscriber, field::{Field, Visit}};
use tracing_subscriber::{EnvFilter, Layer, layer::{Context, SubscriberExt}, util::SubscriberInitExt};

pub use tracing::{debug, error, info, trace, warn};

/// Start logging for `agent` into `file` (created, with its directory, if missing). With `stderr`, events are also
/// printed for humans (off for interactive modes, where they'd interleave with the conversation).
pub fn init(agent: &str, file: &Path, stderr: bool) {
    if let Some(dir) = file.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let out = OpenOptions::new().create(true).append(true).open(file)
        .unwrap_or_else(|e| panic!("open log {}: {e}", file.display()));
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let human = stderr.then(|| tracing_subscriber::fmt::layer().with_target(false).with_writer(std::io::stderr));
    tracing_subscriber::registry()
        .with(filter)
        .with(JsonFile { agent: agent.to_string(), out: Mutex::new(out) })
        .with(human)
        .init();
}

struct JsonFile {
    agent: String,
    out: Mutex<File>,
}

impl<S: Subscriber> Layer<S> for JsonFile {
    fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
        let mut fields = Fields(Map::new());
        event.record(&mut fields);
        let msg = match fields.0.remove("message") {
            Some(Value::String(s)) => s,
            _ => String::new(),
        };
        let line = json!({
            "ts": chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, false),
            "agent": self.agent,
            "level": event.metadata().level().as_str(),
            "msg": msg,
            "fields": fields.0,
        });
        if let Ok(mut out) = self.out.lock() {
            let _ = writeln!(out, "{line}"); // one write per line, so concurrent agents don't interleave mid-line
        }
    }
}

struct Fields(Map<String, Value>);

impl Visit for Fields {
    fn record_debug(&mut self, f: &Field, v: &dyn std::fmt::Debug) {
        self.0.insert(f.name().into(), Value::String(format!("{v:?}")));
    }
    fn record_str(&mut self, f: &Field, v: &str) {
        self.0.insert(f.name().into(), Value::String(v.into()));
    }
    fn record_i64(&mut self, f: &Field, v: i64) {
        self.0.insert(f.name().into(), v.into());
    }
    fn record_u64(&mut self, f: &Field, v: u64) {
        self.0.insert(f.name().into(), v.into());
    }
    fn record_f64(&mut self, f: &Field, v: f64) {
        self.0.insert(f.name().into(), v.into());
    }
    fn record_bool(&mut self, f: &Field, v: bool) {
        self.0.insert(f.name().into(), v.into());
    }
}
