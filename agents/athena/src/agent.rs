//! Athena's harness for a request from Max: Plan, Act, Reflect.
//!
//! Her tools are offered on every model call, in every phase, so she always plans and reviews knowing exactly what
//! she can do; a tool call made while planning or reviewing simply runs (it counts as acting).
//!
//!   1. Plan: she works out what a complete answer must contain, sketches a few different ways to find
//!      the evidence (a small tree of thoughts) and picks an order. The plan stays in her context.
//!   2. Act (ReAct): she reasons, calls read-only tools, reads the results, and adjusts. Each tool result carries a
//!      budget line (tool calls used and left), so she can pace herself.
//!   3. Reflect: when she drafts an answer, a review pass checks it against the plan and the evidence and
//!      answers FINAL (with the answer, corrected) or CONTINUE (with what is missing and how to get it). CONTINUE
//!      goes back to Act, at most MAX_REFLECT times.
//!   Out of budget: one last tool-less call answers with what she has.
//!
//! Guardrails (nudges added to a tool result at the moment she drifts, which a small model follows better than a
//! rule in a long prompt): three searches in a row without reading anything, and searching for a date as text.
//!
//! Her prompts are files, re-read every request, so they can be tuned without a rebuild: ATHENA.md (system),
//! ATHENA.PLAN.md and ATHENA.REFLECT.md in the prompts folder (context/agents/, or another for an evaluation).
//! Every run returns a trace of all phases, for evaluations and debugging.

use std::path::Path;

use board::llm;
use chrono::{DateTime, Local};
use serde_json::{Value, json};

use crate::{mailbox::Mailbox, parse};

const MAX_MODEL_CALLS: usize = 26;
const MAX_TOOL_CALLS: usize = 20;
const MAX_REFLECT: usize = 2;
const READ_CHARS: u32 = 6000;

const SYSTEM: &str = "You are Athena, the email sub-agent of Max, the user's assistant. Work out Max's request with \
your read-only email tools and answer him in plain sentences. {today}";
const PLAN: &str = "Before using any tool, plan. What exactly is being asked, and what must a complete answer contain? \
Where would the evidence be (which emails, senders, words, which account)? Sketch two or three different ways to find \
it and the order you'll try them. Reply with the plan only.";
const REFLECT: &str = "Review your draft against your plan and what the tools showed. Is every part of the request \
covered? Is every fact from an email you read? Anything contradictory or unverified? Start your reply with one line, \
FINAL or CONTINUE. FINAL: then the answer for Max, corrected. CONTINUE: then what is missing and the next searches to \
run.";

pub struct Ctx<'a> {
    pub http: &'a reqwest::Client,
    pub sel: llm::Sel,
    pub key: Option<String>,
    pub prompts: &'a Path,
    pub now: DateTime<Local>,
    pub db: Option<&'a Path>, // agents.db: record context use on the board (None in evaluations)
}

pub struct Outcome {
    pub text: String,
    pub view: Option<Value>,
    pub trace: Vec<Value>,
    pub model_calls: usize,
    pub tool_calls: usize,
}

struct Run<'a, 'b> {
    ctx: &'a Ctx<'b>,
    model_calls: usize,
    trace: Vec<Value>,
    searches_unread: usize, // searches since she last read an email
}

impl Run<'_, '_> {
    async fn chat(&mut self, messages: &[Value], tools: &Value) -> Result<llm::Reply, String> {
        self.model_calls += 1;
        let r = llm::chat(self.ctx.http, &self.ctx.sel, self.ctx.key.as_deref(), messages, tools).await?;
        if let Some(db) = self.ctx.db {
            board::usage(db, "athena", r.tokens, self.ctx.sel.num_ctx, &self.ctx.sel.model);
        }
        Ok(r)
    }
}

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

fn assistant(text: &str) -> Value {
    json!({"role": "assistant", "content": text})
}

pub async fn run(ctx: &Ctx<'_>, mb: &mut Mailbox, request: &str) -> Result<Outcome, String> {
    let vars = [("today", board::today_at(ctx.now)), ("now", ctx.now.format("%A %Y-%m-%d %H:%M").to_string())];
    let system = board::prompt_in(ctx.prompts, "athena", SYSTEM, &vars);
    let plan_prompt = board::prompt_in(ctx.prompts, "athena.plan", PLAN, &vars);
    let reflect_prompt = board::prompt_in(ctx.prompts, "athena.reflect", REFLECT, &vars);
    let specs = tools();
    let mut r = Run { ctx, model_calls: 0, trace: vec![json!({"phase": "request", "text": request})], searches_unread: 0 };
    let mut messages = vec![json!({"role": "system", "content": system}), user(request)];

    let (mut tool_calls, mut reflections, mut view) = (0, 0, None);

    // 1. Plan (tools offered: if she starts acting right away, those calls run)
    messages.push(user(&plan_prompt));
    let reply = r.chat(&messages, &specs).await?;
    let plan = reply.content.trim().to_string();
    r.trace.push(json!({"phase": "plan", "text": plan}));
    if reply.calls.is_empty() {
        messages.push(assistant(&plan));
    } else {
        messages.push(reply.message.clone());
        run_calls(&mut r, mb, &reply.calls, &mut messages, &mut tool_calls, &mut view).await;
    }
    messages.push(user("Now carry out your plan with your tools, adjusting it as you learn. When you have what you \
                        need, write your answer for Max."));

    // 2. Act, 3. Reflect
    let answer = loop {
        if r.model_calls >= MAX_MODEL_CALLS || tool_calls >= MAX_TOOL_CALLS {
            messages.push(user("You are out of steps. Answer Max now with what you found, and say what you couldn't \
                                finish."));
            let a = r.chat(&messages, &json!([])).await?.content.trim().to_string();
            r.trace.push(json!({"phase": "forced", "text": a}));
            break a;
        }
        let reply = r.chat(&messages, &specs).await?;
        if reply.calls.is_empty() {
            let draft = reply.content.trim().to_string();
            r.trace.push(json!({"phase": "draft", "text": draft}));
            if reflections >= MAX_REFLECT {
                break draft;
            }
            reflections += 1;
            messages.push(assistant(&draft));
            messages.push(user(&reflect_prompt));
            let reply = r.chat(&messages, &specs).await?;
            let review = reply.content.trim().to_string();
            if !reply.calls.is_empty() {
                // the review went straight to more searching: that's a CONTINUE, already under way
                r.trace.push(json!({"phase": "reflect", "verdict": "CONTINUE", "text": review}));
                messages.push(reply.message.clone());
                run_calls(&mut r, mb, &reply.calls, &mut messages, &mut tool_calls, &mut view).await;
                continue;
            }
            let (verdict, rest) = verdict(&review);
            r.trace.push(json!({"phase": "reflect", "verdict": verdict, "text": review}));
            messages.push(assistant(&review));
            if verdict == "CONTINUE" && tool_calls < MAX_TOOL_CALLS {
                messages.push(user("Go on: do what your review says, then write your answer for Max."));
                continue;
            }
            break if rest.trim().is_empty() { draft } else { rest.trim().to_string() };
        }
        messages.push(reply.message.clone());
        run_calls(&mut r, mb, &reply.calls, &mut messages, &mut tool_calls, &mut view).await;
    };
    let answer = strip_verdict(&answer);
    let text = if answer.is_empty() { "(no answer)".to_string() } else { answer };
    r.trace.push(json!({"phase": "answer", "text": text}));
    Ok(Outcome { text, view, trace: r.trace, model_calls: r.model_calls, tool_calls })
}

/// Run a round of tool calls; each result goes back with a budget line.
async fn run_calls(r: &mut Run<'_, '_>, mb: &mut Mailbox, calls: &[llm::Call], messages: &mut Vec<Value>,
                   tool_calls: &mut usize, view: &mut Option<Value>) {
    for call in calls {
        *tool_calls += 1;
        let out = tool(mb, &call.name, &call.args, view).await;
        let left = MAX_TOOL_CALLS.saturating_sub(*tool_calls);
        let mut budget = format!("\n[tool calls: {tool_calls} used, {left} left]");
        match call.name.as_str() {
            "read_email" => r.searches_unread = 0,
            "search_emails" | "list_emails" => {
                r.searches_unread += 1;
                if call.args["query"].as_str().is_some_and(has_date_text) {
                    budget.push_str("\n[note: dates are written many ways inside emails (Sat, Nov 14 / 11/14 / November \
                                     14th), so searching for one as text rarely works. Search by kind, sender or place, \
                                     and read the emails to find their dates.]");
                }
                if r.searches_unread >= 3 {
                    budget.push_str("\n[note: that's several searches in a row without reading anything. Before you \
                                     search again, open the results above that could be what you need with read_email.]");
                }
            }
            _ => {}
        }
        r.trace.push(json!({"phase": "tool", "name": call.name, "args": call.args,
                            "result": out.chars().take(3000).collect::<String>()}));
        messages.push(llm::tool_message(&r.ctx.sel, call, &(out + &budget)));
    }
}

/// Does a search look for a date written as text (2026-11-14, 11/14, Nov 14, November 14th, Saturday, ...)?
fn has_date_text(q: &str) -> bool {
    let q = q.to_lowercase();
    let without_ops: String = q.split_whitespace()
        .filter(|w| !["after:", "before:", "newer_than:", "older_than:"].iter().any(|op| w.trim_start_matches(['-', '(']).starts_with(op)))
        .collect::<Vec<_>>().join(" ");
    let months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
    let words: Vec<&str> = without_ops.split(|c: char| !c.is_alphanumeric() && c != '/' && c != '-').filter(|w| !w.is_empty()).collect();
    words.windows(2).any(|w| months.iter().any(|m| w[0].starts_with(m)) && w[1].chars().next().is_some_and(|c| c.is_ascii_digit()))
        || words.iter().any(|w| {
            let parts: Vec<&str> = w.split(['/', '-']).collect();
            parts.len() >= 2 && parts.iter().all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
        })
}

/// The review's verdict (its first line: FINAL or CONTINUE, maybe with the answer after it on that line) and the
/// text after it. No verdict at all: FINAL with no text, so the draft stands (the review was commentary).
fn verdict(review: &str) -> (&'static str, String) {
    let t = review.trim_start();
    let (first, rest) = t.split_once('\n').unwrap_or((t, ""));
    let up = first.to_uppercase();
    for v in ["CONTINUE", "FINAL"] {
        if let Some(i) = up.find(v) {
            let head = first.get(i + v.len()..).unwrap_or("").trim_start_matches([':', '-', ' ', '*']);
            return (v, format!("{head}\n{rest}").trim().to_string());
        }
    }
    ("FINAL", String::new())
}

/// Drop a leading FINAL/CONTINUE line the model sometimes repeats into a draft.
fn strip_verdict(text: &str) -> String {
    let mut t = text.trim();
    loop {
        let up = t.to_uppercase();
        let Some(w) = ["FINAL", "CONTINUE"].iter().find(|w| up.starts_with(**w)) else { break };
        t = t[w.len()..].trim_start_matches([':', '-', ' ', '*', '\n', '\r']);
    }
    t.to_string()
}

/// Common HTML entities left in email text, so the model reads plain words.
fn clean(s: &str) -> String {
    s.replace("&#39;", "'").replace("&quot;", "\"").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        .replace("&nbsp;", " ")
}

/// Run one tool against the mailbox. Lists and searches also become the view (the last one wins).
async fn tool(mb: &mut Mailbox, name: &str, a: &Value, view: &mut Option<Value>) -> String {
    let account = match a["account"].as_str() {
        Some("outlook") => "outlook",
        Some("gmail") => "gmail",
        _ => return "error: account must be \"outlook\" or \"gmail\"".into(),
    };
    let count = a["count"].as_u64().unwrap_or(10).clamp(1, 25);
    let (tool, args, label) = match name {
        "list_emails" => {
            let mut args = json!({"folder": a["folder"].as_str().unwrap_or("inbox"), "limit": count,
                                  "unread": a["unread_only"].as_bool().unwrap_or(false)});
            for k in ["from", "since"] {
                if let Some(v) = a[k].as_str().filter(|v| !v.is_empty()) {
                    args[k] = json!(v);
                }
            }
            (format!("{account}_list"), args, None)
        }
        "search_emails" => {
            let Some(q) = a["query"].as_str().filter(|q| !q.trim().is_empty()) else {
                return "error: search_emails needs a query".into();
            };
            (format!("{account}_search"), json!({"query": q, "limit": count}), Some(q.to_string()))
        }
        "read_email" => {
            let Some(id) = a["id"].as_str() else { return "error: read_email needs an id (from a list)".into() };
            return match mb.call(&format!("{account}_read"), json!({"id": id, "max_chars": READ_CHARS})).await {
                Ok(t) => clean(&t),
                Err(e) => format!("error: {e}"),
            };
        }
        _ => return format!("error: unknown tool {name:?}"),
    };
    match mb.call(&tool, args).await {
        Ok(text) => {
            let mails = parse(account, &text);
            *view = Some(json!({"kind": "email.list", "by": "athena", "account": account, "query": label,
                "unread_only": a["unread_only"].as_bool().unwrap_or(false), "count": count,
                "at": chrono::Local::now().format("%Y-%m-%d %H:%M").to_string(),
                "emails": mails.iter().map(|m| json!({"from": clean(&m.from), "subject": clean(&m.subject),
                    "when": m.when, "preview": clean(&m.preview), "unread": m.unread, "attachment": m.attachment}))
                    .collect::<Vec<_>>()}));
            clean(&text)
        }
        Err(e) => format!("error: {e}"),
    }
}

pub fn tools() -> Value {
    let f = |name: &str, desc: &str, props: Value, req: &[&str]| json!({"type": "function", "function": {
        "name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": req}}});
    let account = json!({"type": "string", "enum": ["outlook", "gmail"]});
    json!([
        f("list_emails", "The newest emails in a folder (default the inbox), newest first.",
          json!({"account": account, "count": {"type": "integer", "description": "1 to 25, default 10"},
                 "unread_only": {"type": "boolean"}, "from": {"type": "string", "description": "sender address"},
                 "since": {"type": "string", "description": "sent on or after, YYYY-MM-DD"},
                 "folder": {"type": "string", "description": "default inbox"}}), &["account"]),
        f("search_emails", "Search one mailbox. Gmail takes Gmail search syntax (words, \"phrases\", OR, ( ), -word, \
           from:, subject:, newer_than:30d, after:YYYY/MM/DD, category:updates, -category:promotions). Outlook takes \
           plain words (all must match), from:name and subject:word. Date filters are about when an email was SENT, \
           not the date of a trip or event inside it.",
          json!({"account": account, "query": {"type": "string"},
                 "count": {"type": "integer", "description": "1 to 25, default 10"}}), &["account", "query"]),
        f("read_email", "Read one email in full (reading doesn't mark it read). The id comes from a list or search.",
          json!({"account": account, "id": {"type": "string"}}), &["account", "id"]),
    ])
}

#[cfg(test)]
mod tests {
    use super::verdict;

    #[test]
    fn verdicts() {
        assert_eq!(verdict("CONTINUE\nsearch for the return trip").0, "CONTINUE");
        let (v, a) = verdict("FINAL\nYou have a bus on Saturday.");
        assert_eq!((v, a.trim()), ("FINAL", "You have a bus on Saturday."));
        let (v, a) = verdict("FINAL: You have a bus.");
        assert_eq!((v, a.trim()), ("FINAL", "You have a bus."));
        assert_eq!(verdict("Looks complete. You have a bus."), ("FINAL", String::new()));
        assert_eq!(verdict("**FINAL**\nYou have a bus.").1, "You have a bus.");
        assert_eq!(super::strip_verdict("FINAL  \nFINAL\nYou have a bus."), "You have a bus.");
        use super::has_date_text;
        assert!(has_date_text("Nov 14 2026 travel confirmation"));
        assert!(has_date_text("trip 2026-11-14"));
        assert!(has_date_text("\"November 14th\" booking"));
        assert!(!has_date_text("(confirmation OR ticket) newer_than:60d after:2026/11/01 -category:promotions"));
        assert!(!has_date_text("from:amtrak eticket"));
    }
}
