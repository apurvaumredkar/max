//! The blackboard: how Max's agents talk to each other, through agents/memory/agents.db (WAL mode, so separate
//! processes can read and write at once). No agent calls another directly.
//!
//!   tasks    root -> sub-agent work. The root agent posts a task for an agent (`post`) and waits for it (`wait`);
//!            the sub-agent claims the oldest open one (`claim`, one atomic UPDATE, so a task never runs twice),
//!            does it and writes the result or error back (`finish`). Status: open -> claimed -> done | failed,
//!            or expired when the poster stopped waiting.
//!   reports  sub-agent -> Max messages nobody asked for (new mail, a reminder firing, a calendar change), in plain
//!            words. `notify` says how each reaches the user (see `Notify`); `delivered_at` is when the model saw it,
//!            `notified_at` when the user was told.
//!   agents   registry: each agent's role and a heartbeat, so the root and the console can see who is running, and
//!            its context use (`usage`): tokens its latest model call filled of its window, out of the window size.
//!
//! The user only ever talks to Max (the root agent); sub-agents only talk to the board.
//!
//! Also shared by the sub-agents: `llm` (the model client, on Max's model), `secret` (credentials from the
//! Access-gated secrets Worker), `prompt` (each agent's system prompt from context/agents/<NAME>.md, re-read for every
//! request so it can be tuned without a rebuild) and `today` (the dates an agent needs to place "this weekend").

pub mod llm;

use std::{path::Path, time::{Duration, Instant}};

use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;

/// Open agents.db with the board's tables (created if missing, older tables gain the newer columns).
pub fn open(db: &Path) -> rusqlite::Result<Connection> {
    let c = Connection::open(db)?;
    c.busy_timeout(Duration::from_secs(10))?;
    let _ = c.pragma_update(None, "journal_mode", "WAL");
    c.execute_batch(
        "CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent TEXT NOT NULL,
            kind TEXT NOT NULL,
            args TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            result TEXT,
            error TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT,
            poster TEXT NOT NULL DEFAULT 'root');
         CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent TEXT NOT NULL,
            text TEXT NOT NULL,
            detail TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            delivered_at TEXT,
            notify INTEGER NOT NULL DEFAULT 0,
            notified_at TEXT);
         CREATE TABLE IF NOT EXISTS agents (
            name TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            pid INTEGER,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            heartbeat_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)",
    )?;
    // columns added after the first version (fail harmlessly once they exist)
    let _ = c.execute("ALTER TABLE tasks ADD COLUMN poster TEXT NOT NULL DEFAULT 'root'", []);
    let _ = c.execute("ALTER TABLE reports ADD COLUMN notify INTEGER NOT NULL DEFAULT 0", []);
    let _ = c.execute("ALTER TABLE reports ADD COLUMN notified_at TEXT", []);
    // context use, per agent: what its latest model call filled of its window
    let _ = c.execute("ALTER TABLE agents ADD COLUMN ctx_used INTEGER", []);
    let _ = c.execute("ALTER TABLE agents ADD COLUMN ctx_max INTEGER", []);
    let _ = c.execute("ALTER TABLE agents ADD COLUMN ctx_at TEXT", []);
    let _ = c.execute("ALTER TABLE agents ADD COLUMN model TEXT", []);
    Ok(c)
}

/// A claimed task.
pub struct Task {
    pub id: i64,
    pub kind: String,
    pub args: Value,
}

/// Post a task for `agent`; returns its id.
pub fn post(db: &Path, agent: &str, kind: &str, args: &Value) -> Result<i64, String> {
    let c = open(db).map_err(|e| e.to_string())?;
    c.execute("INSERT INTO tasks (agent, kind, args) VALUES (?1, ?2, ?3)", params![agent, kind, args.to_string()])
        .map_err(|e| e.to_string())?;
    Ok(c.last_insert_rowid())
}

/// Wait for task `id` to finish: its result, or why not. If nobody claimed it within `timeout` it is withdrawn
/// (expired), so it isn't done late; `who` names the agent in the error.
pub async fn wait(db: &Path, id: i64, who: &str, timeout: Duration) -> Result<Value, String> {
    let deadline = Instant::now() + timeout;
    loop {
        tokio::time::sleep(Duration::from_millis(300)).await;
        let row = open(db).and_then(|c| {
            c.query_row("SELECT status, result, error FROM tasks WHERE id = ?1", [id],
                        |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?, r.get::<_, Option<String>>(2)?)))
                .optional()
        });
        match row {
            Ok(Some((s, result, _))) if s == "done" => {
                return Ok(result.and_then(|r| serde_json::from_str(&r).ok()).unwrap_or(Value::Null));
            }
            Ok(Some((s, _, err))) if s == "failed" => return Err(err.unwrap_or_else(|| "failed".into())),
            Ok(Some(_)) if Instant::now() < deadline => continue,
            Ok(Some((s, _, _))) => {
                let _ = open(db).and_then(|c| c.execute(
                    "UPDATE tasks SET status = 'expired', updated_at = CURRENT_TIMESTAMP WHERE id = ?1 AND status = 'open'",
                    [id]));
                return Err(if s == "open" {
                    format!("{who} didn't pick up the task (is the {who} service running?)")
                } else {
                    format!("{who} didn't finish within {} s", timeout.as_secs())
                });
            }
            Ok(None) => return Err("the task disappeared".into()),
            Err(e) => return Err(format!("task board: {e}")),
        }
    }
}

/// Claim the oldest open task for `agent`. Tasks older than `stale_secs` were given up on by their poster and are
/// expired first.
pub fn claim(c: &Connection, agent: &str, stale_secs: i64) -> Option<Task> {
    let _ = c.execute(
        "UPDATE tasks SET status = 'expired', updated_at = CURRENT_TIMESTAMP
         WHERE agent = ?1 AND status = 'open' AND created_at < datetime('now', ?2)",
        params![agent, format!("-{stale_secs} seconds")]);
    c.query_row(
        "UPDATE tasks SET status = 'claimed', updated_at = CURRENT_TIMESTAMP
         WHERE id = (SELECT id FROM tasks WHERE agent = ?1 AND status = 'open' ORDER BY id LIMIT 1) AND status = 'open'
         RETURNING id, kind, args",
        [agent], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)))
        .ok()
        .map(|(id, kind, args)| Task { id, kind, args: serde_json::from_str(&args).unwrap_or(Value::Null) })
}

/// Write a claimed task's outcome back.
pub fn finish(c: &Connection, id: i64, outcome: &Result<Value, String>) -> rusqlite::Result<()> {
    let (status, result, error) = match outcome {
        Ok(v) => ("done", Some(v.to_string()), None),
        Err(e) => ("failed", None, Some(e.clone())),
    };
    c.execute("UPDATE tasks SET status = ?1, result = ?2, error = ?3, updated_at = CURRENT_TIMESTAMP WHERE id = ?4",
              params![status, result, error, id])?;
    Ok(())
}

/// How a sub-agent's message reaches the user (column `notify`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Notify {
    /// 0: only through Max, when the user next talks to him (the model gets it with that turn).
    Next = 0,
    /// 1: relayed as is, right away: to every open web console, and to Discord always (Chronos' reminders and
    ///    calendar changes). The model still gets it on the next turn.
    Always = 1,
    /// 2: Max takes it right away in a turn of his own and tells the user in his own words: shown live in every open
    ///    web console, and sent to Discord only if no console is open and focused (Athena's new mail).
    Max = 2,
}

impl Notify {
    pub fn from_i64(n: i64) -> Notify {
        match n {
            1 => Notify::Always,
            2 => Notify::Max,
            _ => Notify::Next,
        }
    }
}

/// Leave a message for Max.
pub fn report(c: &Connection, agent: &str, text: &str, detail: Option<&Value>, notify: Notify) -> rusqlite::Result<i64> {
    c.execute("INSERT INTO reports (agent, text, detail, notify) VALUES (?1, ?2, ?3, ?4)",
              params![agent, text, detail.map(|d| d.to_string()), notify as i64])?;
    Ok(c.last_insert_rowid())
}

/// A message waiting to be passed to the user right away.
pub struct Pending {
    pub id: i64,
    pub agent: String,
    pub text: String,
    pub notify: Notify,
    pub at: String,
}

/// Messages (Always or Max) the user hasn't been told about yet, oldest first.
pub fn to_notify(c: &Connection) -> rusqlite::Result<Vec<Pending>> {
    let mut st = c.prepare("SELECT id, agent, text, notify, created_at FROM reports
                            WHERE notify IN (1, 2) AND notified_at IS NULL ORDER BY id")?;
    let rows = st.query_map([], |r| Ok(Pending { id: r.get(0)?, agent: r.get(1)?, text: r.get(2)?,
                                                 notify: Notify::from_i64(r.get(3)?), at: r.get(4)? }))?;
    rows.collect()
}

/// The user has been told (and, with `delivered`, the model has seen it too).
pub fn notified(c: &Connection, id: i64, delivered: bool) -> rusqlite::Result<()> {
    c.execute(if delivered {
        "UPDATE reports SET notified_at = CURRENT_TIMESTAMP, delivered_at = CURRENT_TIMESTAMP WHERE id = ?1"
    } else {
        "UPDATE reports SET notified_at = CURRENT_TIMESTAMP WHERE id = ?1"
    }, [id])?;
    Ok(())
}

/// Register `agent` (on start) or refresh its heartbeat.
pub fn heartbeat(c: &Connection, agent: &str, role: &str) -> rusqlite::Result<()> {
    c.execute(
        "INSERT INTO agents (name, role, pid) VALUES (?1, ?2, ?3)
         ON CONFLICT(name) DO UPDATE SET role = ?2, heartbeat_at = CURRENT_TIMESTAMP,
           pid = ?3, started_at = CASE WHEN pid IS ?3 THEN started_at ELSE CURRENT_TIMESTAMP END",
        params![agent, role, std::process::id()])?;
    Ok(())
}

/// A secret from the Access-gated secrets Worker (SECRETS_URL; `cloudflared access token`, as the root agent does).
pub async fn secret(http: &reqwest::Client, name: &str) -> Result<String, String> {
    let url = std::env::var("SECRETS_URL").map_err(|_| "SECRETS_URL not set in max/.env".to_string())?;
    let cf = std::env::var("CLOUDFLARED")
        .unwrap_or_else(|_| format!("{}/.local/bin/cloudflared", std::env::var("HOME").unwrap_or_default()));
    let out = tokio::process::Command::new(&cf).args(["access", "token", &format!("-app={url}")]).output().await
        .map_err(|e| format!("{cf}: {e}"))?;
    if !out.status.success() {
        return Err(format!("no Access login: run `cloudflared access login {url}`"));
    }
    let token = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let r = http.get(format!("{url}/{name}")).header("cf-access-token", token).header("User-Agent", "max-agent")
        .send().await.map_err(|e| format!("secrets: {e}"))?;
    if r.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(format!("secret {name} is not in the secrets store yet"));
    }
    let v: Value = r.error_for_status().map_err(|e| format!("secrets: {e}"))?.json().await
        .map_err(|e| format!("secrets: {e}"))?;
    v["value"].as_str().map(String::from).ok_or_else(|| "secrets: no value".into())
}

/// The registered agents: (name, role, heartbeat_at), by name.
pub fn agents(c: &Connection) -> rusqlite::Result<Vec<(String, String, String)>> {
    let mut st = c.prepare("SELECT name, role, heartbeat_at FROM agents ORDER BY name")?;
    let rows = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
    rows.collect()
}

/// An agent's system prompt: context/agents/<NAME>.md under `home` (`fallback` if the file is missing), with each
/// `{key}` replaced by its value.
pub fn prompt(home: &Path, agent: &str, fallback: &str, vars: &[(&str, String)]) -> String {
    prompt_in(&home.join("context/agents"), agent, fallback, vars)
}

/// The same, from any folder of prompt files (`dir`/<NAME>.md): lets an evaluation try a variant set of prompts.
pub fn prompt_in(dir: &Path, agent: &str, fallback: &str, vars: &[(&str, String)]) -> String {
    let text = std::fs::read_to_string(dir.join(format!("{}.md", agent.to_uppercase())))
        .unwrap_or_else(|_| fallback.to_string());
    vars.iter().fold(text, |t, (k, v)| t.replace(&format!("{{{k}}}"), v))
}

/// The local time zone's name (TZ or /etc/timezone), else its UTC offset.
pub fn timezone() -> String {
    use chrono::Local;
    std::env::var("TZ").ok().or_else(|| std::fs::read_to_string("/etc/timezone").ok())
        .map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
        .unwrap_or_else(|| Local::now().format("UTC%:z").to_string())
}

/// Today, worked out so a model never has to: the date and time, the time zone, and the days people talk about
/// ("tomorrow", "this weekend", "next week").
pub fn today() -> String {
    today_at(chrono::Local::now())
}

/// `today` for a given moment (an evaluation runs at a frozen "now").
pub fn today_at(now: chrono::DateTime<chrono::Local>) -> String {
    use chrono::{Datelike, Duration};
    let d = now.date_naive();
    let day = |x: chrono::NaiveDate| x.format("%a %Y-%m-%d").to_string();
    let to_sat = (6 - d.weekday().num_days_from_sunday() as i64).rem_euclid(7);
    // on a Sunday, "this weekend" is today (and yesterday)
    let (sat, sun) = if d.weekday() == chrono::Weekday::Sun { (d - Duration::days(1), d) }
                     else { (d + Duration::days(to_sat), d + Duration::days(to_sat + 1)) };
    let next_mon = d + Duration::days(7 - d.weekday().num_days_from_monday() as i64);
    format!("Now: {} ({}), time zone {} (UTC{}). Tomorrow: {}. This weekend: {} and {}. Next week: {} to {}.",
            now.format("%A, %B %-d, %Y, %-I:%M %p"), d.format("%Y-%m-%d"), timezone(), now.format("%:z"),
            day(d + Duration::days(1)), day(sat), day(sun), day(next_mon), day(next_mon + Duration::days(6)))
}

/// Record what `agent`'s latest model call filled of its context window (`used` tokens of `max`, on `model`).
/// A sub-agent starts a fresh window for every request, so this is the size of its current (or last) one.
pub fn usage(db: &Path, agent: &str, used: u64, max: u64, model: &str) {
    if used == 0 {
        return;
    }
    let _ = open(db).and_then(|c| c.execute(
        "UPDATE agents SET ctx_used = ?2, ctx_max = ?3, ctx_at = CURRENT_TIMESTAMP, model = ?4 WHERE name = ?1",
        params![agent, used as i64, max as i64, model]));
}

/// The registry with context use, for the console: one JSON object per agent.
pub fn agents_status(c: &Connection) -> rusqlite::Result<Vec<Value>> {
    let mut st = c.prepare("SELECT name, role, heartbeat_at, started_at, ctx_used, ctx_max, ctx_at, model,
                                   CAST(strftime('%s','now') - strftime('%s', heartbeat_at) AS INTEGER)
                            FROM agents ORDER BY name")?;
    let rows = st.query_map([], |r| Ok(serde_json::json!({
        "name": r.get::<_, String>(0)?, "role": r.get::<_, String>(1)?, "heartbeat_at": r.get::<_, String>(2)?,
        "started_at": r.get::<_, String>(3)?, "ctx_used": r.get::<_, Option<i64>>(4)?, "ctx_max": r.get::<_, Option<i64>>(5)?,
        "ctx_at": r.get::<_, Option<String>>(6)?, "model": r.get::<_, Option<String>>(7)?,
        "online": r.get::<_, Option<i64>>(8)?.is_some_and(|age| age < 90)})))?;
    rows.collect()
}
