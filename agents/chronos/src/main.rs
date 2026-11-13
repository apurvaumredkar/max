//! Chronos: Max's time-keeping sub-agent. A separate, always-running process that keeps reminders and repeating
//! jobs (context/chronos/jobs.json) and watches the user's Apple Calendar (CalDAV, caldav.rs).
//!
//!   chronos              service
//!   chronos ask "<text>" run one request as if Max had sent it, print the answer (testing)
//!   chronos schedule     print the jobs and the next 14 days of events
//!
//! Chronos only talks to Max, through the blackboard (the `board` crate, agents/memory/agents.db):
//!   - tasks: Max's `chronos` tool posts {"text": <the user's request in plain words>}; Chronos claims it and runs his
//!     own tool loop on the model Max uses (llm.rs) with tools for jobs and calendar events, then writes back
//!     {"text": his answer for Max, "view": the schedule for the console's Agent Workspace (when he used a tool)}.
//!   - reports (notify): a job firing ("Reminder: ...") and calendar changes found by the 5-minute poll
//!     (CHRONOS_CAL_INTERVAL). Max relays these to the user's #max Discord channel at once and to the model on its
//!     next turn. The first calendar poll only records what is there.
//! Logs go to agents/logs/agents.jsonl via `agentlog`.

mod caldav;
mod jobs;

use std::{collections::HashMap, path::PathBuf, sync::Mutex, time::Duration};

use agentlog::{error, info, warn};
use chrono::{DateTime, Local};
use serde_json::{Value, json};

use board::{llm, secret};
use caldav::{Cal, Event, When};
use jobs::{Job, Jobs, MINUTE};

const ROLE: &str = "time keeper: reminders, scheduled jobs, and what is on the Apple Calendar (bookings that only came by email are not there: ask athena)";
const WINDOW_DAYS: i64 = 60; // the calendar poll watches this far ahead
const SHOW_DAYS: i64 = 14; // the schedule view's default span
const MAX_ROUNDS: usize = 8; // model calls per request
const TASK_STALE_SECS: i64 = 300;

struct Chronos {
    home: PathBuf,
    db: PathBuf,
    http: reqwest::Client,
    jobs: Mutex<Jobs>,
    cal: tokio::sync::Mutex<Option<Cal>>,
    cal_state: Mutex<String>,    // "ok", or why the calendar can't be read
    events: Mutex<Vec<Event>>,   // the last fetched events, to find one by uid
    or_key: tokio::sync::Mutex<Option<String>>,
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

fn show(t: DateTime<Local>) -> String {
    t.format("%a %b %-d, %-I:%M %p").to_string()
}

fn timezone() -> String {
    std::env::var("TZ").ok().or_else(|| std::fs::read_to_string("/etc/timezone").ok())
        .map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
        .unwrap_or_else(|| Local::now().format("UTC%:z").to_string())
}

fn job_when(j: &Job) -> String {
    match (&j.cron, &j.at) {
        (Some(c), _) => format!("cron {c}"),
        (None, Some(a)) => format!("once at {a}"),
        _ => "no schedule".into(),
    }
}

impl Chronos {
    fn new() -> Chronos {
        let home = std::env::var("MAX_HOME").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("workspace dir")
        });
        load_env(&home.join(".env"));
        Chronos {
            db: home.join("agents/memory/agents.db"),
            http: reqwest::Client::builder().timeout(Duration::from_secs(300)).build().unwrap(),
            jobs: Mutex::new(Jobs::load(home.join("context/chronos/jobs.json"))),
            cal: tokio::sync::Mutex::new(None),
            cal_state: Mutex::new("not checked yet".into()),
            events: Mutex::new(Vec::new()),
            or_key: tokio::sync::Mutex::new(None),
            home,
        }
    }

    // ---------------------------------------------------------------- calendar

    /// Events starting between `from` and `to` (signs in on first use; the secrets are re-fetched after a failure).
    async fn fetch(&self, from: DateTime<Local>, to: DateTime<Local>) -> Result<Vec<Event>, String> {
        let mut g = self.cal.lock().await;
        if g.is_none() {
            let creds = async {
                Ok::<_, String>((secret(&self.http, "ICLOUD_USERNAME").await?, secret(&self.http, "ICLOUD_APP_PASSWORD").await?))
            }.await;
            match creds {
                Ok((user, pass)) => *g = Some(Cal::new(self.http.clone(), user, pass)),
                Err(e) => {
                    *self.cal_state.lock().unwrap() = format!("not connected: {e}");
                    return Err(e);
                }
            }
        }
        let result = g.as_mut().unwrap().events(from, to).await;
        match &result {
            Ok(ev) => {
                *self.cal_state.lock().unwrap() = "ok".into();
                *self.events.lock().unwrap() = ev.clone();
            }
            Err(e) => {
                *self.cal_state.lock().unwrap() = format!("error: {e}");
                if e.contains("refused the sign-in") {
                    *g = None; // fetch the secrets again next time
                }
            }
        }
        result
    }

    /// The event with this uid (the first instance), from the last fetch or a wider fresh one.
    async fn find(&self, uid: &str) -> Result<Event, String> {
        let hit = |evs: &[Event]| evs.iter().find(|e| e.uid == uid).cloned();
        if let Some(e) = hit(&self.events.lock().unwrap()) {
            return Ok(e);
        }
        let now = Local::now();
        let evs = self.fetch(now - chrono::Duration::days(30), now + chrono::Duration::days(365)).await?;
        hit(&evs).ok_or_else(|| format!("no event with uid {uid:?} (use list_schedule to find it)"))
    }

    /// Poll the calendar and compare with the last poll (table chronos_events, one row per event resource; a repeating
    /// event counts once). With `report`, changes go to Max as a notify report; Chronos' own edits sync silently.
    async fn sync(&self, report: bool) -> Result<(), String> {
        let now = Local::now();
        let events = self.fetch(now - chrono::Duration::days(1), now + chrono::Duration::days(WINDOW_DAYS)).await?;
        let mut cur: HashMap<String, &Event> = HashMap::new();
        for e in &events {
            cur.entry(e.href.clone()).or_insert(e); // sorted by start: the earliest instance wins
        }
        let c = board::open(&self.db).map_err(|e| e.to_string())?;
        c.execute_batch(
            "CREATE TABLE IF NOT EXISTS chronos_events (href TEXT PRIMARY KEY, etag TEXT, title TEXT, start TEXT, shown TEXT);
             CREATE TABLE IF NOT EXISTS chronos_state (k TEXT PRIMARY KEY, v TEXT)").map_err(|e| e.to_string())?;
        let first = c.query_row("SELECT v FROM chronos_state WHERE k = 'calendar'", [], |r| r.get::<_, String>(0)).is_err();
        let mut old: HashMap<String, (String, String, String, String)> = HashMap::new();
        {
            let mut st = c.prepare("SELECT href, etag, title, start, shown FROM chronos_events").map_err(|e| e.to_string())?;
            let rows = st.query_map([], |r| Ok((r.get::<_, String>(0)?, (r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))))
                .map_err(|e| e.to_string())?;
            for row in rows.flatten() {
                old.insert(row.0, row.1);
            }
        }
        let edge = now + chrono::Duration::days(WINDOW_DAYS - 1); // events entering the window's far end aren't news
        let (mut added, mut changed, mut removed) = (Vec::new(), Vec::new(), Vec::new());
        for (href, e) in &cur {
            let tag = if e.recurring { " (repeating)" } else { "" };
            match old.get(href) {
                None if e.start.instant() < edge => added.push(format!("\"{}\" {}{tag}", e.title, e.start.show())),
                Some((etag, title, _, shown)) if *etag != e.etag => {
                    let now_shown = e.start.show();
                    changed.push(if *shown != now_shown {
                        format!("\"{}\" moved from {shown} to {now_shown}", e.title)
                    } else if *title != e.title {
                        format!("\"{title}\" renamed to \"{}\" ({now_shown})", e.title)
                    } else {
                        format!("\"{}\" ({now_shown}) was edited", e.title)
                    });
                }
                _ => {}
            }
        }
        for (href, (_, title, start, shown)) in &old {
            let future = DateTime::parse_from_rfc3339(start).is_ok_and(|s| s > now && s < edge);
            if !cur.contains_key(href) && future {
                removed.push(format!("\"{title}\" (was {shown})"));
            }
        }
        c.execute("DELETE FROM chronos_events", []).map_err(|e| e.to_string())?;
        for (href, e) in &cur {
            c.execute("INSERT INTO chronos_events (href, etag, title, start, shown) VALUES (?1, ?2, ?3, ?4, ?5)",
                      rusqlite::params![href, e.etag, e.title, e.start.instant().to_rfc3339(), e.start.show()])
                .map_err(|e| e.to_string())?;
        }
        c.execute("INSERT OR REPLACE INTO chronos_state (k, v) VALUES ('calendar', ?1)", [now.to_rfc3339()])
            .map_err(|e| e.to_string())?;
        if first {
            info!(events = cur.len(), "first calendar check: recording the calendar, not reporting it");
            return Ok(());
        }
        if !report || added.len() + changed.len() + removed.len() == 0 {
            return Ok(());
        }
        let mut parts = Vec::new();
        if !added.is_empty() { parts.push(format!("new: {}", added.join("; "))) }
        if !changed.is_empty() { parts.push(format!("changed: {}", changed.join("; "))) }
        if !removed.is_empty() { parts.push(format!("removed: {}", removed.join("; "))) }
        let text = format!("Calendar update. {}.", parts.join(". "));
        let detail = json!({"added": added, "changed": changed, "removed": removed});
        board::report(&c, "chronos", &text, Some(&detail), board::Notify::Always).map_err(|e| e.to_string())?;
        info!(added = added.len(), changed = changed.len(), removed = removed.len(), "calendar changes reported");
        Ok(())
    }

    // ---------------------------------------------------------------- jobs

    /// Fire every due job: a notify report to Max, then record the run (a one-time job is then disabled). A job with a
    /// `prompt` gets its message written fresh by the model (in its own task, so the tick isn't held up).
    fn tick_jobs(self: &std::sync::Arc<Self>) {
        let now = Local::now();
        let mut j = self.jobs.lock().unwrap();
        if j.reload_if_changed() {
            match &j.error {
                Some(e) => error!(error = %e, "jobs.json changed but can't be read; keeping the jobs as they were"),
                None => info!(jobs = j.file.jobs.len(), "jobs.json changed on disk; reloaded"),
            }
        }
        let minute = now.format(MINUTE).to_string();
        let mut fired: Vec<Job> = Vec::new();
        for job in j.file.jobs.iter_mut().filter(|job| jobs::due(job, now)) {
            job.last_run = Some(minute.clone());
            job.runs += 1;
            if job.at.is_some() {
                job.enabled = false; // once: done
            }
            fired.push(job.clone());
        }
        if fired.is_empty() {
            return;
        }
        if let Err(e) = j.save() {
            error!(error = %e, "could not save jobs.json");
        }
        drop(j);
        for job in fired {
            let k = self.clone();
            tokio::spawn(async move {
                let fixed = if job.message.trim().is_empty() { job.name.clone() } else { job.message.clone() };
                let what = match &job.prompt {
                    Some(p) => match k.write_message(&job, p).await {
                        Ok(m) => {
                            k.remember_message(&job.id, &m);
                            m
                        }
                        Err(e) => {
                            warn!(job = %job.id, error = %e, "fresh message failed; sending the fixed one");
                            fixed
                        }
                    },
                    None => fixed,
                };
                let late = job.at.as_deref().and_then(jobs::parse_local)
                    .filter(|at| now - *at > chrono::Duration::minutes(2))
                    .map(|at| format!(" (was due {})", show(at))).unwrap_or_default();
                let text = format!("Reminder: {what}{late}");
                let Ok(c) = board::open(&k.db) else { return };
                match board::report(&c, "chronos", &text, Some(&json!({"job": job.id, "name": job.name})), board::Notify::Always) {
                    Ok(_) => info!(job = %job.id, name = %job.name, fresh = job.prompt.is_some(), "job fired"),
                    Err(e) => error!(job = %job.id, error = %e, "job fired but the report failed"),
                }
            });
        }
    }

    /// A new wording for a job's reminder, following its prompt and unlike its recent ones.
    async fn write_message(&self, job: &Job, prompt: &str) -> Result<String, String> {
        let sel = llm::load(&self.home);
        let key = if sel.provider == "openrouter" { Some(self.openrouter_key().await?) } else { None };
        let recent = if job.recent.is_empty() { "none yet".to_string() } else { job.recent.join("\n") };
        let messages = [
            json!({"role": "system", "content": "You write one reminder message that is sent to the user's phone. \
                Reply with the message only: one or two short sentences, no quotes, no markdown, no preamble."}),
            json!({"role": "user", "content": format!(
                "Reminder: {}\nWhat it is about: {}\nHow to write it: {prompt}\nTime now: {}\n\
                 Earlier messages (write something clearly different):\n{recent}",
                job.name, job.message, Local::now().format("%A %-I:%M %p"))}),
        ];
        let r = llm::chat(&self.http, &sel, key.as_deref(), &messages, &json!([])).await?;
        board::usage(&self.db, "chronos", r.tokens, sel.num_ctx, &sel.model);
        let m = r.content.trim().trim_matches('"').trim().to_string();
        if m.is_empty() { Err("empty message".into()) } else { Ok(m) }
    }

    /// Keep the last few generated messages on the job, so the next one can differ.
    fn remember_message(&self, id: &str, m: &str) {
        let mut j = self.jobs.lock().unwrap();
        let Some(job) = j.file.jobs.iter_mut().find(|x| x.id == id) else { return };
        job.recent.push(m.to_string());
        let n = job.recent.len();
        if n > 6 {
            job.recent.drain(..n - 6);
        }
        if let Err(e) = j.save() {
            error!(error = %e, "could not save jobs.json");
        }
    }

    // ---------------------------------------------------------------- the schedule view

    async fn schedule(&self, days: i64) -> Value {
        let now = Local::now();
        let (jobs, jobs_error) = {
            let j = self.jobs.lock().unwrap();
            (j.file.jobs.clone(), j.error.clone())
        };
        let jobs: Vec<Value> = jobs.iter().map(|j| json!({
            "id": j.id, "name": j.name, "message": j.message, "when": job_when(j), "enabled": j.enabled,
            "next": jobs::next_run(j, now).map(show), "last_run": j.last_run, "runs": j.runs, "prompt": j.prompt,
            "last_message": j.recent.last()})).collect();
        let today = now.date_naive().and_hms_opt(0, 0, 0).and_then(|t| t.and_local_timezone(Local).earliest()).unwrap_or(now);
        let events: Vec<Value> = match self.fetch(today, now + chrono::Duration::days(days)).await {
            Ok(evs) => evs.iter().map(|e| json!({
                "uid": e.uid, "title": e.title, "start": e.start.show(), "end": e.end.map(|w| w.show()),
                "all_day": matches!(e.start, When::Day(_)), "location": e.location, "calendar": e.calendar,
                "recurring": e.recurring, "past": e.start.instant() < now})).collect(),
            Err(_) => Vec::new(),
        };
        let calendar = self.cal_state.lock().unwrap().clone();
        json!({"kind": "schedule", "by": "chronos", "at": now.format(MINUTE).to_string(), "days": days,
               "jobs": jobs, "jobs_error": jobs_error, "events": events, "calendar": calendar})
    }

    /// The current jobs as text, for Chronos' instructions (so he never has to look them up).
    fn jobs_text(&self) -> String {
        let now = Local::now();
        let j = self.jobs.lock().unwrap();
        if j.file.jobs.is_empty() {
            return "- none".into();
        }
        j.file.jobs.iter().map(|x| format!("- id={} \"{}\" {}, {}{}: message \"{}\"{}", x.id, x.name, job_when(x),
            if x.enabled { "enabled" } else { "disabled/done" },
            jobs::next_run(x, now).map(|n| format!(", next {}", show(n))).unwrap_or_default(), x.message,
            x.prompt.as_deref().map(|p| format!(", written fresh each time: {p}")).unwrap_or_default()))
            .collect::<Vec<_>>().join("\n")
    }

    /// The schedule as text, for the model.
    fn schedule_text(view: &Value) -> String {
        let mut s = String::from("Jobs:\n");
        let jobs = view["jobs"].as_array().cloned().unwrap_or_default();
        if jobs.is_empty() {
            s.push_str("- none\n");
        }
        for j in &jobs {
            s.push_str(&format!("- id={} \"{}\" {}, {}{}: message \"{}\"{}\n", j["id"].as_str().unwrap_or(""),
                j["name"].as_str().unwrap_or(""), j["when"].as_str().unwrap_or(""),
                if j["enabled"] == true { "enabled" } else { "disabled/done" },
                j["next"].as_str().map(|n| format!(", next {n}")).unwrap_or_default(), j["message"].as_str().unwrap_or(""),
                j["prompt"].as_str().map(|p| format!(", written fresh each time: {p}")).unwrap_or_default()));
        }
        let cal = view["calendar"].as_str().unwrap_or("");
        s.push_str(&format!("Calendar events, today and the next {} days (calendar: {cal}):\n", view["days"]));
        let events = view["events"].as_array().cloned().unwrap_or_default();
        if cal != "ok" {
            s.push_str("- unknown: the calendar can't be read right now, so don't say whether there are events\n");
        } else if events.is_empty() {
            s.push_str("- none\n");
        }
        for e in &events {
            let text = |k: &str| e[k].as_str().unwrap_or("").to_string();
            s.push_str(&format!("- uid={} \"{}\" {}{} [{}]{}{}{}\n", text("uid"), text("title"), text("start"),
                e["end"].as_str().map(|x| format!(" to {x}")).unwrap_or_default(), text("calendar"),
                if text("location").is_empty() { String::new() } else { format!(" at {}", text("location")) },
                if e["recurring"] == true { " (repeating)" } else { "" }, if e["past"] == true { " (already over)" } else { "" }));
        }
        s
    }

    // ---------------------------------------------------------------- requests from Max

    async fn openrouter_key(&self) -> Result<String, String> {
        let mut k = self.or_key.lock().await;
        if k.is_none() {
            *k = Some(secret(&self.http, "OPENROUTER_API_KEY").await?);
        }
        Ok(k.clone().unwrap())
    }

    /// One request from Max: Chronos' own tool loop, then his answer (and the schedule view if he used a tool).
    async fn request(&self, text: &str) -> Result<Value, String> {
        let sel = llm::load(&self.home);
        let key = if sel.provider == "openrouter" { Some(self.openrouter_key().await?) } else { None };
        let now = Local::now();
        let names: Vec<String> = self.cal.lock().await.as_ref()
            .map(|c| c.calendars().iter().map(|c| c.name.clone()).collect()).unwrap_or_default();
        // context/agents/CHRONOS.md (re-read every request; PROMPT if it's missing)
        let system = board::prompt(&self.home, "chronos", PROMPT, &[
            ("today", board::today()),
            ("now", now.format("%A %Y-%m-%d %H:%M").to_string()),
            ("tz", timezone()),
            ("calendars", if names.is_empty() { "unknown until listed".into() } else { names.join(", ") }),
            ("jobs", self.jobs_text()),
        ]);
        let mut messages = vec![json!({"role": "system", "content": system}), json!({"role": "user", "content": text})];
        let specs = tool_specs();
        let (mut answer, mut used, mut changed) = (String::new(), false, false);
        for round in 0..MAX_ROUNDS {
            let r = llm::chat(&self.http, &sel, key.as_deref(), &messages, &specs).await?;
            board::usage(&self.db, "chronos", r.tokens, sel.num_ctx, &sel.model);
            if r.calls.is_empty() {
                answer = r.content.trim().to_string();
                break;
            }
            messages.push(r.message.clone());
            for call in &r.calls {
                used = true;
                let out = self.tool(&call.name, &call.args, &mut changed).await;
                info!(tool = %call.name, args = %call.args, result = %out.lines().next().unwrap_or(""), "chronos tool");
                messages.push(llm::tool_message(&sel, call, &out));
            }
            if round + 1 == MAX_ROUNDS {
                // out of steps: one last call without tools, so what was found still gets reported
                messages.push(json!({"role": "user", "content": "You are out of tool steps. Answer Max now with \
                    what you found so far, and say what you couldn't finish."}));
                answer = match llm::chat(&self.http, &sel, key.as_deref(), &messages, &json!([])).await {
                    Ok(r) if !r.content.trim().is_empty() => r.content.trim().to_string(),
                    _ => "I ran out of steps before finishing.".into(),
                };
            }
        }
        if answer.is_empty() {
            answer = "(no answer)".into();
        }
        if changed && self.cal_state.lock().unwrap().as_str() == "ok" {
            if let Err(e) = self.sync(false).await {
                warn!(error = %e, "calendar sync after an edit failed");
            }
        }
        let view = if used { Some(self.schedule(SHOW_DAYS).await) } else { None };
        Ok(json!({"text": answer, "view": view}))
    }

    async fn tool(&self, name: &str, a: &Value, changed: &mut bool) -> String {
        let s = |k: &str| a[k].as_str().map(str::trim).filter(|v| !v.is_empty());
        match name {
            "list_schedule" => {
                let days = a["days"].as_i64().unwrap_or(SHOW_DAYS).clamp(1, 90);
                Self::schedule_text(&self.schedule(days).await)
            }
            "add_job" => {
                let Some(title) = s("name") else { return "error: add_job needs a name".into() };
                let mut j = self.jobs.lock().unwrap();
                if let Some(e) = &j.error {
                    return format!("error: jobs.json can't be read, so nothing was added: {e}");
                }
                if let Some(dup) = j.file.jobs.iter().find(|x| x.enabled && x.name.eq_ignore_ascii_case(title)) {
                    return format!("error: {} \"{}\" already exists; change it with update_job instead of adding \
                                    another", dup.id, dup.name);
                }
                let job = Job { id: j.new_id(), name: title.into(), message: s("message").unwrap_or(title).into(),
                                cron: s("cron").map(String::from), at: s("at").map(String::from),
                                prompt: s("prompt").map(String::from), enabled: true,
                                created_at: Local::now().format(MINUTE).to_string(), last_run: None, runs: 0,
                                recent: vec![] };
                if let Err(e) = jobs::validate(&job) {
                    return format!("error: {e}");
                }
                let next = jobs::next_run(&job, Local::now());
                if job.at.is_some() && next.is_some_and(|n| n < Local::now()) {
                    return "error: that time has already passed".into();
                }
                let id = job.id.clone();
                j.file.jobs.push(job);
                match j.save() {
                    Ok(()) => {
                        *changed = true;
                        format!("added {id}; it fires next at {}", next.map(show).unwrap_or_else(|| "never (check the cron)".into()))
                    }
                    Err(e) => format!("error: could not save jobs.json: {e}"),
                }
            }
            "remove_job" | "update_job" => {
                let Some(id) = s("id") else { return format!("error: {name} needs an id") };
                let mut j = self.jobs.lock().unwrap();
                let Some(pos) = j.file.jobs.iter().position(|x| x.id == id) else {
                    return format!("error: no job {id:?} (use list_schedule)");
                };
                let msg = if name == "remove_job" {
                    let gone = j.file.jobs.remove(pos);
                    format!("removed {id} \"{}\"", gone.name)
                } else {
                    let mut job = j.file.jobs[pos].clone();
                    if let Some(v) = s("name") { job.name = v.into() }
                    if let Some(v) = s("message") { job.message = v.into() }
                    match a["prompt"].as_str().map(str::trim) {
                        Some("") | Some("none") => job.prompt = None, // back to the fixed message
                        Some(v) => job.prompt = Some(v.into()),
                        None => {}
                    }
                    let rescheduled = s("cron").is_some() || s("at").is_some();
                    if let Some(v) = s("cron") { (job.cron, job.at) = (Some(v.into()), None) }
                    if let Some(v) = s("at") { (job.at, job.cron) = (Some(v.into()), None) }
                    if let Some(on) = a["enabled"].as_bool() { job.enabled = on }
                    if job.at.is_some() && (rescheduled || a["enabled"] == true) {
                        job.last_run = None; // a one-time job set for a new time (or resumed) fires again
                        job.enabled = a["enabled"].as_bool().unwrap_or(true);
                    }
                    if let Err(e) = jobs::validate(&job) {
                        return format!("error: {e}");
                    }
                    let next = jobs::next_run(&job, Local::now());
                    if job.at.is_some() && job.enabled && next.is_some_and(|n| n < Local::now()) {
                        return "error: that time has already passed".into();
                    }
                    j.file.jobs[pos] = job;
                    format!("updated {id}; {}", match next {
                        Some(n) => format!("it fires next at {}", show(n)),
                        None => "it is paused".into(),
                    })
                };
                match j.save() {
                    Ok(()) => {
                        *changed = true;
                        msg
                    }
                    Err(e) => format!("error: could not save jobs.json: {e}"),
                }
            }
            "add_event" => {
                let (Some(title), Some(start)) = (s("title"), s("start")) else {
                    return "error: add_event needs a title and a start".into();
                };
                let Some(start) = When::parse(start) else { return format!("error: bad start {start:?}") };
                let end = match s("end").map(|e| (e, When::parse(e))) {
                    Some((e, None)) => return format!("error: bad end {e:?}"),
                    Some((_, w)) => w,
                    None => None,
                };
                if self.fetch(Local::now(), Local::now()).await.is_err() {
                    return format!("error: calendar {}", self.cal_state.lock().unwrap());
                }
                let mut g = self.cal.lock().await;
                let Some(cal) = g.as_mut() else { return "error: calendar not connected".into() };
                match cal.create(s("calendar"), title, start, end, s("location").unwrap_or("")).await {
                    Ok((uid, calname)) => {
                        *changed = true;
                        format!("added \"{title}\" on {} to the {calname} calendar (uid={uid})", start.show())
                    }
                    Err(e) => format!("error: {e}"),
                }
            }
            "update_event" | "delete_event" => {
                let Some(uid) = s("uid") else { return format!("error: {name} needs a uid (from list_schedule)") };
                let ev = match self.find(uid).await {
                    Ok(e) => e,
                    Err(e) => return format!("error: {e}"),
                };
                let parse = |k: &str| s(k).map(|v| When::parse(v).ok_or_else(|| format!("bad {k} {v:?}"))).transpose();
                let (start, end) = match (parse("start"), parse("end")) {
                    (Ok(a), Ok(b)) => (a, b),
                    (Err(e), _) | (_, Err(e)) => return format!("error: {e}"),
                };
                let mut g = self.cal.lock().await;
                let Some(cal) = g.as_mut() else { return "error: calendar not connected".into() };
                let r = if name == "delete_event" {
                    cal.delete(&ev).await.map(|_| format!("deleted \"{}\" ({})", ev.title, ev.start.show()))
                } else {
                    cal.update(&ev, s("title"), start, end, s("location")).await
                        .map(|_| format!("updated \"{}\"", s("title").unwrap_or(&ev.title)))
                };
                match r {
                    Ok(m) => {
                        *changed = true;
                        m
                    }
                    Err(e) => format!("error: {e}"),
                }
            }
            _ => format!("error: unknown tool {name:?}"),
        }
    }
}

const PROMPT: &str = "You are Chronos, the time-keeping sub-agent of Max, the user's assistant. Max passes you the \
user's requests about reminders, scheduled jobs and the user's Apple Calendar. Do the request with your tools, then \
answer Max.

Now: {now}, time zone {tz}. Work out relative times (today, tomorrow, next Monday, in two hours) from this; all times \
you read and write are this local time, as YYYY-MM-DD HH:MM (or YYYY-MM-DD for an all-day event).

Jobs are your own reminders (kept in jobs.json). When a job fires, its message goes to Max and to the user's phone, \
so write the message as the reminder itself, like \"Call the dentist\".
- Once at a set time: add_job with at.
- To change a job (its time, schedule, message or style, or to pause it), use update_job on the existing job; \
never add a second job for the same reminder.
- If the user wants the wording to change each time (goofy, motivational, varied), give the job a prompt saying \
how to write it; it is then written fresh every time. Otherwise the message is sent as is.
- Repeating: add_job with cron, 5 fields in local time (minute hour day-of-month month day-of-week). Examples: \
\"0 9 * * 1-5\" every weekday at 9:00, \"30 18 * * *\" every day at 18:30, \"0 10 1 * *\" the 1st of each month at 10:00.
Calendar events are entries in the user's Apple Calendar (calendars: {calendars}). Use an event when the user wants \
something on their calendar (a meeting, an appointment, a trip); use a job when they want to be reminded.
The jobs right now (use these ids with update_job and remove_job directly):
{jobs}
Calendar events are not listed here: use list_schedule to see them and find an event's uid. To answer questions \
about the schedule, use list_schedule. You always have the tools listed; never say you lack one of them.

Rules:
- Only say something was done if a tool result says so. If a tool returns an error, say what failed.
- When you tell Max when something happens, use the time the tool result gave (\"fires next at ...\"), not your \
own guess.
- If the request lacks something you need (a time, a date), don't guess: say what is missing.
- Answer in 1 to 3 plain sentences with the exact days and times. No markdown, no lists.";

fn tool_specs() -> Value {
    let f = |name: &str, desc: &str, props: Value, req: &[&str]| json!({"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": req}}});
    json!([
        f("list_schedule", "The jobs, and the calendar events from today on.",
          json!({"days": {"type": "integer", "description": "how many days of events (default 14, max 90)"}}), &[]),
        f("add_job", "Add a reminder job: give either at (once) or cron (repeating), not both.",
          json!({"name": {"type": "string", "description": "short name"},
                 "message": {"type": "string", "description": "what the reminder says when it fires"},
                 "at": {"type": "string", "description": "once: YYYY-MM-DD HH:MM local"},
                 "cron": {"type": "string", "description": "repeating: 5-field cron, local time"},
                 "prompt": {"type": "string", "description": "only if the user wants it worded differently each time \
                     (goofy, motivational, a fun fact...): how to write it. It is then written fresh every time it fires"}}),
          &["name", "message"]),
        f("remove_job", "Delete a job.", json!({"id": {"type": "string"}}), &["id"]),
        f("update_job", "Change a job: only the fields given change. Giving cron or at replaces the schedule. \
           enabled false pauses it, true resumes it. prompt \"none\" goes back to the fixed message.",
          json!({"id": {"type": "string"}, "name": {"type": "string"}, "message": {"type": "string"},
                 "at": {"type": "string"}, "cron": {"type": "string"}, "prompt": {"type": "string"},
                 "enabled": {"type": "boolean"}}), &["id"]),
        f("add_event", "Add an event to the user's Apple Calendar.",
          json!({"title": {"type": "string"},
                 "start": {"type": "string", "description": "YYYY-MM-DD HH:MM, or YYYY-MM-DD for all day"},
                 "end": {"type": "string", "description": "same form; default: one hour after start (all day: that day)"},
                 "location": {"type": "string"},
                 "calendar": {"type": "string", "description": "calendar name; default the main one"}}), &["title", "start"]),
        f("update_event", "Change a calendar event; only the fields given change. Moving the start keeps its length.",
          json!({"uid": {"type": "string", "description": "from list_schedule"}, "title": {"type": "string"},
                 "start": {"type": "string"}, "end": {"type": "string"}, "location": {"type": "string"}}), &["uid"]),
        f("delete_event", "Delete a calendar event.", json!({"uid": {"type": "string", "description": "from list_schedule"}}), &["uid"]),
    ])
}

#[tokio::main]
async fn main() {
    let k = std::sync::Arc::new(Chronos::new());
    let mut args = std::env::args().skip(1);
    let mode = args.next();
    agentlog::init("chronos", &k.home.join("agents/logs/agents.jsonl"), true);
    match mode.as_deref() {
        Some("ask") => {
            let text: Vec<String> = args.collect();
            match k.request(&text.join(" ")).await {
                Ok(v) => println!("{}", v["text"].as_str().unwrap_or("")),
                Err(e) => eprintln!("error: {e}"),
            }
        }
        Some("schedule") => print!("{}", Chronos::schedule_text(&k.schedule(SHOW_DAYS).await)),
        None | Some("serve") => serve(k).await,
        Some(other) => eprintln!("unknown command {other:?}; use `chronos`, `chronos ask <text>` or `chronos schedule`"),
    }
}

async fn serve(k: std::sync::Arc<Chronos>) {
    let every = std::env::var("CHRONOS_CAL_INTERVAL").ok().and_then(|s| s.parse().ok()).unwrap_or(300u64);
    info!(cal_every = every, jobs = k.jobs.lock().unwrap().file.jobs.len(), "keeping time");
    if let Ok(c) = board::open(&k.db) {
        let _ = board::heartbeat(&c, "chronos", ROLE);
    }
    // requests from Max: one at a time, in order
    let worker = k.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_millis(500));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            let Ok(c) = board::open(&worker.db) else { continue };
            while let Some(task) = board::claim(&c, "chronos", TASK_STALE_SECS) {
                let started = std::time::Instant::now();
                let outcome = match task.kind.as_str() {
                    "request" => worker.request(task.args["text"].as_str().unwrap_or("")).await,
                    other => Err(format!("Chronos can't do {other:?} tasks")),
                };
                let ms = started.elapsed().as_millis() as u64;
                match &outcome {
                    Ok(_) => info!(task = task.id, ms, "task done"),
                    Err(e) => warn!(task = task.id, ms, error = %e, "task failed"),
                }
                let _ = board::finish(&c, task.id, &outcome);
            }
        }
    });
    // jobs: every 15 s, so a job fires within its minute
    let ticker = k.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(15));
        loop {
            tick.tick().await;
            ticker.tick_jobs();
        }
    });
    // heartbeat
    let beat = k.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(30));
        loop {
            tick.tick().await;
            if let Ok(c) = board::open(&beat.db) {
                let _ = board::heartbeat(&c, "chronos", ROLE);
            }
        }
    });
    // calendar poll
    let mut tick = tokio::time::interval(Duration::from_secs(every));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut last_err = String::new();
    loop {
        tick.tick().await;
        match k.sync(true).await {
            Ok(()) => last_err.clear(),
            Err(e) if e != last_err => {
                warn!(error = %e, "calendar check failed");
                last_err = e;
            }
            Err(_) => {} // same error as last time: logged once
        }
    }
}
