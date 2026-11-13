//! `athena eval`: run Athena's harness on a suite of cases and score the answers, to compare prompt and harness
//! changes on evidence instead of one try.
//!
//!   athena eval [--cases FILE] [--prompts DIR] [--tag NAME] [--only ID[,ID]] [--repeat N]
//!
//! A cases file (default agents/athena/eval/cases.json) names its mailbox, "fixture": "<file next to it>" (the
//! sandbox, at the fixture's frozen "now") or "live": true (the real mailboxes through email-mcp, read-only, at the
//! real now), and lists cases: {id, request, must: [regex...], must_not: [regex...]}, matched case-insensitively
//! against the answer. A run passes when every `must` matches and no `must_not` does; its score is the share of
//! `must` matched (0 if any `must_not` matched). The model samples, so --repeat N runs each case N times.
//!
//! --prompts DIR reads ATHENA.md, ATHENA.PLAN.md and ATHENA.REFLECT.md from DIR instead of context/agents/ (try a
//! variant without touching the live prompts). Transcripts (plan, every tool call and result, reviews, answer) go
//! to agents/memory/evals/<time>-<tag>/, and one summary line per eval to agents/memory/evals/history.jsonl.

use std::{path::{Path, PathBuf}, time::Instant};

use chrono::{Local, TimeZone};
use regex::RegexBuilder;
use serde_json::{Value, json};

use crate::{Athena, agent, fixture::Fixture, mailbox::Mailbox, mcp::Mcp};

struct Opts {
    cases: PathBuf,
    prompts: PathBuf,
    tag: String,
    only: Vec<String>,
    repeat: usize,
}

fn opts(home: &Path, args: &[String]) -> Result<Opts, String> {
    let mut o = Opts {
        cases: home.join("agents/athena/eval/cases.json"),
        prompts: home.join("context/agents"),
        tag: "run".into(),
        only: Vec::new(),
        repeat: 1,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--cases" => o.cases = PathBuf::from(val()?),
            "--prompts" => o.prompts = PathBuf::from(val()?),
            "--tag" => o.tag = val()?,
            "--only" => o.only = val()?.split(',').map(String::from).collect(),
            "--repeat" => o.repeat = val()?.parse().map_err(|_| "--repeat takes a number".to_string())?,
            other => return Err(format!("unknown option {other}")),
        }
    }
    Ok(o)
}

/// Plain ASCII spacing and dashes: models write "December\u{202f}20" or "61\u{2011}48213", which a regex with a
/// space or hyphen wouldn't match.
fn normal(s: &str) -> String {
    s.chars().map(|c| match c {
        '\u{00a0}' | '\u{2007}' | '\u{2009}' | '\u{200a}' | '\u{202f}' | '\u{205f}' => ' ',
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2212}' => '-',
        '\u{2018}' | '\u{2019}' => '\'',
        c => c,
    }).collect()
}

fn score(answer: &str, must: &[String], must_not: &[String]) -> (f64, Vec<String>, Vec<String>) {
    let answer = normal(answer);
    let answer = answer.as_str();
    let hit = |p: &str| RegexBuilder::new(p).case_insensitive(true).build().map(|r| r.is_match(answer)).unwrap_or(false);
    let missed: Vec<String> = must.iter().filter(|p| !hit(p)).cloned().collect();
    let leaked: Vec<String> = must_not.iter().filter(|p| hit(p)).cloned().collect();
    let s = if must.is_empty() { 1.0 } else { (must.len() - missed.len()) as f64 / must.len() as f64 };
    (if leaked.is_empty() { s } else { 0.0 }, missed, leaked)
}

pub async fn main(athena: &Athena, args: &[String]) -> Result<(), String> {
    let o = opts(&athena.home, args)?;
    let suite: Value = serde_json::from_str(&std::fs::read_to_string(&o.cases).map_err(|e| format!("{}: {e}", o.cases.display()))?)
        .map_err(|e| format!("{}: {e}", o.cases.display()))?;
    let fixture = match suite["fixture"].as_str() {
        Some(f) => Some(Fixture::load(&o.cases.parent().unwrap_or(Path::new(".")).join(f))?),
        None if suite["live"] == true => None,
        None => return Err("the cases file needs \"fixture\" or \"live\": true".into()),
    };
    let cases: Vec<Value> = suite["cases"].as_array().cloned().unwrap_or_default().into_iter()
        .filter(|c| o.only.is_empty() || o.only.iter().any(|id| c["id"].as_str() == Some(id)))
        .collect();
    let stamp = Local::now().format("%Y%m%d-%H%M%S").to_string();
    let out = athena.home.join(format!("agents/memory/evals/{stamp}-{}", o.tag));
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let sel = board::llm::load(&athena.home);
    println!("eval {} | model {} | prompts {} | {} cases x {} | {}", o.tag, sel.model, o.prompts.display(),
             cases.len(), o.repeat, if fixture.is_some() { "sandbox" } else { "LIVE mailboxes" });

    let mut rows = Vec::new();
    for c in &cases {
        let id = c["id"].as_str().unwrap_or("?");
        let request = c["request"].as_str().unwrap_or("");
        let strs = |k: &str| -> Vec<String> { c[k].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(String::from)).collect() };
        let (must, must_not) = (strs("must"), strs("must_not"));
        for n in 1..=o.repeat {
            let (mut mb, now) = match &fixture {
                Some(f) => (Mailbox::Fixture(f.clone()), Local.from_utc_datetime(&f.now().naive_utc())),
                None => (Mailbox::Live(Mcp::connect(&athena.http, &athena.mcp_url, athena.mcp_key.clone()).await?), Local::now()),
            };
            let key = if sel.provider == "openrouter" { Some(athena.openrouter_key().await?) } else { None };
            let ctx = agent::Ctx { http: &athena.http, sel: board::llm::load(&athena.home), key, prompts: &o.prompts, now, db: None };
            let started = Instant::now();
            let result = agent::run(&ctx, &mut mb, request).await;
            mb.close().await;
            let secs = started.elapsed().as_secs_f64();
            let row = match result {
                Ok(r) => {
                    let (s, missed, leaked) = score(&r.text, &must, &must_not);
                    let pass = missed.is_empty() && leaked.is_empty();
                    println!("{} {id:<16} #{n} score {:.2}  {:>3.0}s  {:>2} model calls  {:>2} tools{}{}", if pass { "PASS" } else { "FAIL" },
                             s, secs, r.model_calls, r.tool_calls,
                             if missed.is_empty() { String::new() } else { format!("  missed {missed:?}") },
                             if leaked.is_empty() { String::new() } else { format!("  LEAKED {leaked:?}") });
                    let row = json!({"id": id, "n": n, "pass": pass, "score": s, "secs": secs, "model_calls": r.model_calls,
                                     "tool_calls": r.tool_calls, "missed": missed, "leaked": leaked, "answer": r.text});
                    let mut full = row.clone();
                    full["trace"] = json!(r.trace);
                    let _ = std::fs::write(out.join(format!("{id}-{n}.json")), serde_json::to_string_pretty(&full).unwrap());
                    row
                }
                Err(e) => {
                    println!("ERR  {id:<16} #{n} {e}");
                    json!({"id": id, "n": n, "pass": false, "score": 0.0, "secs": secs, "error": e})
                }
            };
            rows.push(row);
        }
    }
    let total = rows.len().max(1) as f64;
    let passed = rows.iter().filter(|r| r["pass"] == true).count();
    let avg = |k: &str| rows.iter().filter_map(|r| r[k].as_f64()).sum::<f64>() / total;
    let summary = json!({"tag": o.tag, "at": stamp, "model": sel.model, "prompts": o.prompts, "cases": o.cases,
                         "runs": rows.len(), "passed": passed, "pass_rate": passed as f64 / total, "avg_score": avg("score"),
                         "avg_secs": avg("secs"), "avg_model_calls": avg("model_calls"), "avg_tool_calls": avg("tool_calls")});
    println!("== {} | passed {passed}/{} | avg score {:.2} | avg {:.0}s, {:.1} model calls, {:.1} tools | {}", o.tag, rows.len(),
             avg("score"), avg("secs"), avg("model_calls"), avg("tool_calls"), out.display());
    let _ = std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&json!({"summary": summary, "runs": rows})).unwrap());
    let hist = athena.home.join("agents/memory/evals/history.jsonl");
    let line = summary.to_string() + "\n";
    let _ = std::fs::OpenOptions::new().create(true).append(true).open(&hist)
        .and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::score;

    #[test]
    fn scoring() {
        let m = vec!["amtrak".to_string(), "7:05".to_string()];
        assert_eq!(score("Amtrak at 7:05 AM", &m, &[]).0, 1.0);
        assert_eq!(score("Amtrak only", &m, &[]).0, 0.5);
        assert_eq!(score("Amtrak 7:05, code 5521", &m, &["5521".to_string()]).0, 0.0);
        assert_eq!(score("Amtrak 7:05\u{202f}AM", &["7:05 am".to_string()], &[]).0, 1.0);
    }
}
