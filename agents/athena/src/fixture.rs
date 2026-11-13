//! A sandbox mailbox for evaluations: a fixed set of emails (a JSON file) behind the same six tools email-mcp offers
//! (gmail_/outlook_ list, search, read), answering in the same text format, at a frozen "now". Nothing leaves the
//! machine and every run sees the same mail, so prompt and harness changes can be compared.
//!
//! Gmail search supports what the model is told it can use: words and "phrases" (matched in sender, recipient,
//! subject and body), OR, ( ), -negation, from: to: subject: (also with a (group)), after:/before: (YYYY/MM/DD),
//! newer_than:/older_than: (Nd, Nm, Ny), category:, label:/in:, is:unread/read, has:attachment. An operator Gmail
//! doesn't have (like since:) is taken as a plain word and, as in Gmail, matches nothing. Outlook search takes plain
//! words (all must match) plus from: and subject:, as the Graph search does.

use chrono::{DateTime, Duration, FixedOffset, NaiveDate, NaiveDateTime};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize, Clone)]
pub struct Email {
    pub id: String,
    pub account: String, // gmail | outlook
    pub from: String,
    #[serde(default)]
    pub to: String,
    pub date: String, // "YYYY-MM-DD HH:MM", UTC, as email-mcp shows it
    pub subject: String,
    #[serde(default)]
    pub labels: Vec<String>, // INBOX, CATEGORY_UPDATES, CATEGORY_PROMOTIONS, ...
    #[serde(default)]
    pub unread: bool,
    #[serde(default)]
    pub attachment: bool,
    pub body: String,
}

#[derive(Deserialize, Clone)]
pub struct Fixture {
    pub now: String, // RFC 3339, the frozen "now" of this mailbox
    pub emails: Vec<Email>,
}

impl Fixture {
    pub fn load(path: &std::path::Path) -> Result<Fixture, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn now(&self) -> DateTime<FixedOffset> {
        DateTime::parse_from_rfc3339(&self.now).expect("fixture now is RFC 3339")
    }

    fn sent(e: &Email) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(&e.date, "%Y-%m-%d %H:%M").unwrap_or_default()
    }

    /// The same contract as email-mcp's tools.
    pub fn call(&self, tool: &str, a: &Value) -> Result<String, String> {
        let (account, op) = tool.split_once('_').ok_or_else(|| format!("unknown tool {tool}"))?;
        if account != "gmail" && account != "outlook" {
            return Err(format!("unknown tool {tool}"));
        }
        let limit = a["limit"].as_u64().unwrap_or(20).clamp(1, 50) as usize;
        let mine = self.emails.iter().filter(|e| e.account == account);
        match op {
            "list" => {
                let folder = a["folder"].as_str().unwrap_or("inbox").to_uppercase();
                let unread = a["unread"].as_bool().unwrap_or(false);
                let from = a["from"].as_str().map(str::to_lowercase);
                let since = a["since"].as_str().and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
                let hits: Vec<&Email> = mine
                    .filter(|e| e.labels.iter().any(|l| l.to_uppercase() == folder) || (folder == "INBOX" && account == "outlook"))
                    .filter(|e| !unread || e.unread)
                    .filter(|e| from.as_ref().is_none_or(|f| e.from.to_lowercase().contains(f)))
                    .filter(|e| since.is_none_or(|d| Self::sent(e).date() >= d))
                    .collect();
                Ok(show(hits, limit))
            }
            "search" => {
                let q = a["query"].as_str().unwrap_or("");
                let now = self.now().naive_utc();
                let hits: Vec<&Email> = if account == "gmail" {
                    let node = parse_gmail(q);
                    mine.filter(|e| matches(&node, e, now)).collect()
                } else {
                    mine.filter(|e| outlook_match(q, e)).collect()
                };
                Ok(show(hits, limit))
            }
            "read" => {
                let id = a["id"].as_str().unwrap_or("");
                let e = self.emails.iter().find(|e| e.id == id && e.account == account)
                    .ok_or_else(|| format!("no message with id {id}"))?;
                let max = a["max_chars"].as_u64().unwrap_or(20000) as usize;
                let body: String = e.body.chars().take(max).collect();
                Ok(format!("From: {}\nTo: {}\nDate: {} UTC\nSubject: {}\nLabels: {}\n{}\n\n{body}",
                           e.from, if e.to.is_empty() { "me" } else { &e.to }, e.date, e.subject,
                           if e.unread { format!("UNREAD, {}", e.labels.join(", ")) } else { e.labels.join(", ") },
                           if e.attachment { "Attachments: attachment.pdf" } else { "" }))
            }
            _ => Err(format!("unknown tool {tool}")),
        }
    }
}

/// A result list in email-mcp's format, newest first.
fn show(mut hits: Vec<&Email>, limit: usize) -> String {
    if hits.is_empty() {
        return "No messages.".into();
    }
    hits.sort_by(|a, b| b.date.cmp(&a.date));
    hits.iter().take(limit).enumerate().map(|(i, e)| {
        let preview: String = e.body.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(100).collect();
        format!("{}. {}{}{} | {} | {}\n   {preview}\n   id={}", i + 1, if e.unread { "UNREAD " } else { "" },
                if e.attachment { "[attachment] " } else { "" }, e.date, e.from, e.subject, e.id)
    }).collect::<Vec<_>>().join("\n")
}

fn outlook_match(q: &str, e: &Email) -> bool {
    let all = format!("{} {} {}", e.from, e.subject, e.body).to_lowercase();
    q.split_whitespace().all(|w| {
        let w = w.to_lowercase();
        match w.split_once(':') {
            Some(("from", v)) => e.from.to_lowercase().contains(v),
            Some(("subject", v)) => e.subject.to_lowercase().contains(v),
            _ => all.contains(w.trim_matches('"')),
        }
    })
}

// ---------------------------------------------------------------- Gmail query language (the subset above)

#[derive(Debug)]
enum Node {
    And(Vec<Node>),
    Or(Vec<Node>),
    Not(Box<Node>),
    Term { field: Option<String>, value: String },
}

#[derive(Debug, PartialEq)]
enum Tok {
    LParen,
    RParen,
    Neg,
    Or,
    Field(String), // "from:" directly before "("
    Word(String),  // a word, "a phrase", or field:value
}

fn tokens(q: &str) -> Vec<Tok> {
    let c: Vec<char> = q.chars().collect();
    let (mut i, mut out) = (0, Vec::new());
    while i < c.len() {
        match c[i] {
            ch if ch.is_whitespace() => i += 1,
            '(' => { out.push(Tok::LParen); i += 1; }
            ')' => { out.push(Tok::RParen); i += 1; }
            '-' if i + 1 < c.len() && !c[i + 1].is_whitespace() && (i == 0 || c[i - 1].is_whitespace() || c[i - 1] == '(') => {
                out.push(Tok::Neg);
                i += 1;
            }
            '"' => {
                let end = c[i + 1..].iter().position(|&x| x == '"').map(|p| i + 1 + p).unwrap_or(c.len());
                out.push(Tok::Word(c[i + 1..end].iter().collect()));
                i = end + 1;
            }
            _ => {
                let start = i;
                while i < c.len() && !c[i].is_whitespace() && c[i] != '(' && c[i] != ')' {
                    if c[i] == '"' {
                        // field:"a phrase"
                        i += 1;
                        while i < c.len() && c[i] != '"' { i += 1; }
                    }
                    i += 1;
                }
                let w: String = c[start..i.min(c.len())].iter().collect();
                if w == "OR" || w == "|" {
                    out.push(Tok::Or);
                } else if w == "AND" {
                } else if w.ends_with(':') && i < c.len() && c[i] == '(' {
                    out.push(Tok::Field(w.trim_end_matches(':').to_lowercase()));
                } else {
                    out.push(Tok::Word(w));
                }
            }
        }
    }
    out
}

fn parse_gmail(q: &str) -> Node {
    let toks = tokens(q);
    let mut pos = 0;
    parse_or(&toks, &mut pos, None)
}

fn parse_or(t: &[Tok], p: &mut usize, field: Option<&str>) -> Node {
    let mut alts = vec![parse_and(t, p, field)];
    while *p < t.len() && t[*p] == Tok::Or {
        *p += 1;
        alts.push(parse_and(t, p, field));
    }
    if alts.len() == 1 { alts.pop().unwrap() } else { Node::Or(alts) }
}

fn parse_and(t: &[Tok], p: &mut usize, field: Option<&str>) -> Node {
    let mut all = Vec::new();
    while *p < t.len() && t[*p] != Tok::Or && t[*p] != Tok::RParen {
        all.push(parse_unary(t, p, field));
    }
    if all.len() == 1 { all.pop().unwrap() } else { Node::And(all) }
}

fn parse_unary(t: &[Tok], p: &mut usize, field: Option<&str>) -> Node {
    let tok = &t[*p];
    *p += 1;
    match tok {
        Tok::Neg if *p < t.len() => Node::Not(Box::new(parse_unary(t, p, field))),
        Tok::LParen => {
            let n = parse_or(t, p, field);
            if *p < t.len() && t[*p] == Tok::RParen { *p += 1; }
            n
        }
        Tok::Field(f) => {
            if *p < t.len() && t[*p] == Tok::LParen { *p += 1; }
            let n = parse_or(t, p, Some(f));
            if *p < t.len() && t[*p] == Tok::RParen { *p += 1; }
            n
        }
        Tok::Word(w) => match w.split_once(':') {
            Some((f, v)) if !f.is_empty() && !v.is_empty() && f.chars().all(|c| c.is_ascii_alphabetic() || c == '_') => {
                Node::Term { field: Some(f.to_lowercase()), value: v.trim_matches('"').to_string() }
            }
            _ => Node::Term { field: field.map(String::from), value: w.clone() },
        },
        _ => Node::And(vec![]), // a stray ) or -: matches everything, like an empty query
    }
}

fn matches(n: &Node, e: &Email, now: NaiveDateTime) -> bool {
    match n {
        Node::And(v) => v.iter().all(|x| matches(x, e, now)),
        Node::Or(v) => v.iter().any(|x| matches(x, e, now)),
        Node::Not(x) => !matches(x, e, now),
        Node::Term { field, value } => term(field.as_deref(), value, e, now),
    }
}

fn has_label(e: &Email, l: &str) -> bool {
    e.labels.iter().any(|x| x.eq_ignore_ascii_case(l))
}

fn term(field: Option<&str>, value: &str, e: &Email, now: NaiveDateTime) -> bool {
    let v = value.to_lowercase();
    let sent = Fixture::sent(e);
    let day = |s: &str| NaiveDate::parse_from_str(&s.replace('-', "/"), "%Y/%m/%d").ok();
    let span = |s: &str| -> Option<Duration> {
        let (n, unit) = s.split_at(s.len().checked_sub(1)?);
        let n: i64 = n.parse().ok()?;
        Some(match unit { "d" => Duration::days(n), "m" => Duration::days(30 * n), "y" => Duration::days(365 * n), _ => return None })
    };
    match field {
        None => format!("{} {} {} {}", e.from, e.to, e.subject, e.body).to_lowercase().contains(&v),
        Some("from") => e.from.to_lowercase().contains(&v),
        Some("to") => e.to.to_lowercase().contains(&v),
        Some("subject") => e.subject.to_lowercase().contains(&v),
        Some("after") => day(&v).is_some_and(|d| sent.date() >= d),
        Some("before") => day(&v).is_some_and(|d| sent.date() < d),
        Some("newer_than") => span(&v).is_some_and(|d| sent >= now - d),
        Some("older_than") => span(&v).is_some_and(|d| sent < now - d),
        Some("category") => match v.as_str() {
            "primary" => !e.labels.iter().any(|l| l.starts_with("CATEGORY_")) || has_label(e, "CATEGORY_PERSONAL"),
            c => has_label(e, &format!("CATEGORY_{}", c.to_uppercase())),
        },
        Some("label") | Some("in") => v == "anywhere" || has_label(e, &v.to_uppercase()),
        Some("is") => match v.as_str() { "unread" => e.unread, "read" => !e.unread, _ => false },
        Some("has") => v == "attachment" && e.attachment,
        // not a Gmail operator: Gmail would look for the literal text, which no email has
        Some(f) => format!("{} {}", e.subject, e.body).to_lowercase().contains(&format!("{f}:{v}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mail(id: &str, from: &str, subject: &str, labels: &[&str], date: &str, body: &str) -> Email {
        Email { id: id.into(), account: "gmail".into(), from: from.into(), to: String::new(), date: date.into(),
                subject: subject.into(), labels: labels.iter().map(|s| s.to_string()).collect(), unread: false,
                attachment: false, body: body.into() }
    }

    #[test]
    fn gmail_queries() {
        let fx = Fixture { now: "2026-11-13T09:30:00-05:00".into(), emails: vec![
            mail("a", "Amtrak <tickets@amtrak.com>", "Your eTicket", &["INBOX", "CATEGORY_UPDATES"], "2026-11-02 18:12", "Boston 7:05 AM"),
            mail("b", "Amtrak <deals@amtrak.com>", "Fares from $25", &["INBOX", "CATEGORY_PROMOTIONS"], "2026-11-10 12:00", "Boston deals"),
            mail("c", "Priya <p@x.example>", "Moving our 1:1?", &["INBOX"], "2026-08-01 10:00", "Tuesday"),
        ]};
        let ids = |q: &str| -> Vec<String> {
            let n = parse_gmail(q);
            let now = fx.now().naive_utc();
            fx.emails.iter().filter(|e| matches(&n, e, now)).map(|e| e.id.clone()).collect()
        };
        assert_eq!(ids("boston -category:promotions"), vec!["a"]);
        assert_eq!(ids("from:(amtrak OR priya) newer_than:30d"), vec!["a", "b"]);
        assert_eq!(ids("(eticket OR \"1:1\") after:2026/07/01"), vec!["a", "c"]);
        assert!(ids("boston since:2026-11-01").is_empty()); // not a Gmail operator
        assert_eq!(ids("subject:\"our 1:1\""), vec!["c"]);
        assert_eq!(ids("category:primary"), vec!["c"]);
    }
}
