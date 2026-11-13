//! Chronos' jobs: context/chronos/jobs.json, reminders that fire on a cron schedule or once at a set time.
//!
//!   {"jobs": [{"id": "job-1", "name": "Stand-up", "message": "Stand-up in 5 minutes",
//!              "cron": "55 9 * * 1-5",            <- or "at": "2026-11-13 15:00" (local time, fires once)
//!              "prompt": "goofy, different every time",   <- optional: written fresh by the model each time
//!              "enabled": true, "created_at": "...", "last_run": "2026-11-13 09:55", "runs": 3,
//!              "recent": ["the last few messages sent, so a fresh one doesn't repeat them"]}]}
//!
//! Times are the Jetson's local time. Cron is the usual 5 fields (minute hour day-of-month month day-of-week) with
//! `*`, lists, ranges, `/step`, names (mon, jan) and @hourly/@daily/@weekly/@monthly/@yearly. The file is re-read
//! whenever it changes on disk, so a hand edit applies within one tick. A one-time job is disabled once it fired.

use std::{path::PathBuf, time::SystemTime};

use chrono::{DateTime, Datelike, Duration, Local, NaiveDateTime, TimeZone, Timelike};
use serde::{Deserialize, Serialize};

pub const MINUTE: &str = "%Y-%m-%d %H:%M";

#[derive(Serialize, Deserialize, Clone)]
pub struct Job {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cron: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<String>,
    /// How to write the message each time it fires (it is then generated, `message` being the fallback).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default = "yes")]
    pub enabled: bool,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub last_run: Option<String>,
    #[serde(default)]
    pub runs: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recent: Vec<String>,
}

fn yes() -> bool {
    true
}

#[derive(Serialize, Deserialize, Default)]
pub struct File {
    #[serde(default)]
    pub jobs: Vec<Job>,
}

pub struct Jobs {
    path: PathBuf,
    mtime: Option<SystemTime>,
    pub file: File,
    pub error: Option<String>, // the file exists but couldn't be read: jobs are kept as they were, not wiped
}

impl Jobs {
    pub fn load(path: PathBuf) -> Jobs {
        let mut j = Jobs { path, mtime: None, file: File::default(), error: None };
        j.reload();
        j
    }

    fn reload(&mut self) {
        self.mtime = std::fs::metadata(&self.path).and_then(|m| m.modified()).ok();
        match std::fs::read_to_string(&self.path) {
            Err(_) => self.error = None, // no file yet: no jobs
            Ok(text) => match serde_json::from_str::<File>(&text) {
                Ok(f) => (self.file, self.error) = (f, None),
                Err(e) => self.error = Some(format!("{}: {e}", self.path.display())),
            },
        }
    }

    /// Re-read the file if it changed on disk since we last read or wrote it. True if it did.
    pub fn reload_if_changed(&mut self) -> bool {
        let now = std::fs::metadata(&self.path).and_then(|m| m.modified()).ok();
        if now != self.mtime {
            self.reload();
            return true;
        }
        false
    }

    /// Write the file (whole, via a temp file and rename, so a reader never sees half of it).
    pub fn save(&mut self) -> Result<(), String> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let tmp = self.path.with_extension("json.tmp");
        let text = serde_json::to_string_pretty(&self.file).map_err(|e| e.to_string())?;
        std::fs::write(&tmp, text + "\n").map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &self.path).map_err(|e| e.to_string())?;
        self.mtime = std::fs::metadata(&self.path).and_then(|m| m.modified()).ok();
        Ok(())
    }

    pub fn new_id(&self) -> String {
        let n = self.file.jobs.iter().filter_map(|j| j.id.strip_prefix("job-")?.parse::<u64>().ok()).max().unwrap_or(0);
        format!("job-{}", n + 1)
    }
}

/// Check a job's schedule: exactly one of cron / at, and it parses.
pub fn validate(job: &Job) -> Result<(), String> {
    match (&job.cron, &job.at) {
        (Some(c), None) => Cron::parse(c).map(|_| ()),
        (None, Some(a)) => parse_local(a).map(|_| ()).ok_or_else(|| format!("bad time {a:?}: use YYYY-MM-DD HH:MM")),
        _ => Err("give exactly one of cron (repeating) or at (once, YYYY-MM-DD HH:MM)".into()),
    }
}

/// "YYYY-MM-DD HH:MM" (or with T) in local time.
pub fn parse_local(s: &str) -> Option<DateTime<Local>> {
    let s = s.trim().replace('T', " ");
    let s = s.get(..16).unwrap_or(&s);
    let naive = NaiveDateTime::parse_from_str(s, MINUTE).ok()?;
    Local.from_local_datetime(&naive).earliest()
}

/// Should this job fire in the minute `now` falls in?
pub fn due(job: &Job, now: DateTime<Local>) -> bool {
    if !job.enabled {
        return false;
    }
    let minute = now.format(MINUTE).to_string();
    if job.last_run.as_deref() == Some(minute.as_str()) {
        return false; // already fired this minute
    }
    match (&job.cron, &job.at) {
        (Some(c), _) => Cron::parse(c).is_ok_and(|c| c.matches(now)),
        // once: due from its minute on (a late start still fires it, up to a day late), and only if never run
        (None, Some(a)) => job.last_run.is_none()
            && parse_local(a).is_some_and(|at| now >= at && now - at < Duration::hours(24)),
        _ => false,
    }
}

/// When the job fires next, after `after`.
pub fn next_run(job: &Job, after: DateTime<Local>) -> Option<DateTime<Local>> {
    if !job.enabled {
        return None;
    }
    match (&job.cron, &job.at) {
        (Some(c), _) => Cron::parse(c).ok()?.next(after),
        (None, Some(a)) if job.last_run.is_none() => parse_local(a),
        _ => None,
    }
}

/// A parsed 5-field cron expression.
pub struct Cron {
    min: Vec<bool>,
    hour: Vec<bool>,
    dom: Vec<bool>,
    mon: Vec<bool>,
    dow: Vec<bool>,
    dom_any: bool,
    dow_any: bool,
}

const MONTHS: [&str; 12] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
const DAYS: [&str; 7] = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"];

impl Cron {
    pub fn parse(expr: &str) -> Result<Cron, String> {
        let expr = match expr.trim() {
            "@hourly" => "0 * * * *",
            "@daily" | "@midnight" => "0 0 * * *",
            "@weekly" => "0 0 * * 0",
            "@monthly" => "0 0 1 * *",
            "@yearly" | "@annually" => "0 0 1 1 *",
            e => e,
        };
        let f: Vec<&str> = expr.split_whitespace().collect();
        if f.len() != 5 {
            return Err(format!("cron {expr:?} needs 5 fields: minute hour day-of-month month day-of-week"));
        }
        let mut dow = field(f[4], 0, 7, &DAYS)?;
        if dow[7] {
            dow[0] = true; // 7 is Sunday too
        }
        dow.truncate(7);
        Ok(Cron {
            min: field(f[0], 0, 59, &[])?,
            hour: field(f[1], 0, 23, &[])?,
            dom: field(f[2], 1, 31, &[])?,
            mon: field(f[3], 1, 12, &MONTHS)?,
            dow,
            dom_any: f[2] == "*",
            dow_any: f[4] == "*",
        })
    }

    pub fn matches(&self, t: DateTime<Local>) -> bool {
        let (dom, dow) = (self.dom[t.day() as usize], self.dow[t.weekday().num_days_from_sunday() as usize]);
        // standard cron: when both day fields are restricted, either one matching is enough
        let day = if self.dom_any || self.dow_any { dom && dow } else { dom || dow };
        self.min[t.minute() as usize] && self.hour[t.hour() as usize] && self.mon[t.month() as usize] && day
    }

    /// The first matching minute after `after` (searching up to a year and a bit ahead).
    pub fn next(&self, after: DateTime<Local>) -> Option<DateTime<Local>> {
        let start = after.with_second(0)?.with_nanosecond(0)? + Duration::minutes(1);
        let mut t = start;
        while t - start < Duration::days(370) {
            if !self.mon[t.month() as usize] {
                // skip to the 1st of next month
                let (y, m) = if t.month() == 12 { (t.year() + 1, 1) } else { (t.year(), t.month() + 1) };
                t = Local.with_ymd_and_hms(y, m, 1, 0, 0, 0).earliest()?;
                continue;
            }
            if self.matches(t) {
                return Some(t);
            }
            t += Duration::minutes(1);
        }
        None
    }
}

/// One cron field as a table of allowed values (index = value; indices below `lo` unused).
fn field(s: &str, lo: u32, hi: u32, names: &[&str]) -> Result<Vec<bool>, String> {
    let mut ok = vec![false; hi as usize + 1];
    let num = |p: &str| -> Result<u32, String> {
        let p = p.to_ascii_lowercase();
        if let Some(i) = names.iter().position(|n| *n == p) {
            return Ok(i as u32 + if lo == 1 { 1 } else { 0 });
        }
        p.parse::<u32>().ok().filter(|v| (lo..=hi).contains(v)).ok_or_else(|| format!("{p:?} is not in {lo}-{hi}"))
    };
    for part in s.split(',') {
        let (range, step) = match part.split_once('/') {
            Some((r, st)) => (r, st.parse::<u32>().ok().filter(|&v| v > 0).ok_or_else(|| format!("bad step in {part:?}"))?),
            None => (part, 1),
        };
        let (a, b) = match range {
            "*" => (lo, hi),
            r => match r.split_once('-') {
                Some((a, b)) => (num(a)?, num(b)?),
                None if step > 1 => (num(r)?, hi), // "5/15": from 5 on, every 15
                None => (num(r)?, num(r)?),
            },
        };
        if a > b {
            return Err(format!("range {part:?} runs backwards"));
        }
        let mut v = a;
        while v <= b {
            ok[v as usize] = true;
            v += step;
        }
    }
    Ok(ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(s: &str) -> DateTime<Local> {
        parse_local(s).unwrap()
    }

    #[test]
    fn cron_matches_and_next() {
        let c = Cron::parse("55 9 * * mon-fri").unwrap();
        assert!(c.matches(at("2026-11-13 09:55"))); // a Friday
        assert!(!c.matches(at("2026-11-14 09:55"))); // Saturday
        assert_eq!(c.next(at("2026-11-13 10:00")).unwrap().format(MINUTE).to_string(), "2026-11-16 09:55");
        let every = Cron::parse("*/15 * * * *").unwrap();
        assert_eq!(every.next(at("2026-11-13 10:01")).unwrap().format(MINUTE).to_string(), "2026-11-13 10:15");
        assert!(Cron::parse("0 9 * *").is_err());
        assert!(Cron::parse("0 25 * * *").is_err());
        assert!(Cron::parse("0 9 * * 7").unwrap().matches(at("2026-11-15 09:00"))); // Sunday
    }

    #[test]
    fn once_jobs() {
        let mut j = Job { id: "job-1".into(), name: "x".into(), message: String::new(), cron: None,
                          at: Some("2026-11-13 15:00".into()), prompt: None, enabled: true,
                          created_at: String::new(), last_run: None, runs: 0, recent: vec![] };
        assert!(!due(&j, at("2026-11-13 14:59")));
        assert!(due(&j, at("2026-11-13 15:00")));
        assert!(due(&j, at("2026-11-13 18:00"))); // started late: still fires
        j.last_run = Some("2026-11-13 15:00".into());
        assert!(!due(&j, at("2026-11-13 15:01")));
    }
}
