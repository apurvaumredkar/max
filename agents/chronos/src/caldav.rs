//! Apple (iCloud) Calendar over CalDAV, with nothing but reqwest and string handling.
//!
//! Sign-in is the Apple ID plus an app-specific password (appleid.apple.com -> Sign-In and Security -> App-Specific
//! Passwords), secrets ICLOUD_USERNAME and ICLOUD_APP_PASSWORD from the secrets Worker. Discovery follows the
//! standard steps: current-user-principal -> calendar-home-set -> the calendars in it that hold events (VEVENT;
//! Reminders lists are skipped). Events come from a calendar-query REPORT with `expand`, so the server turns
//! repeating events into their instances, in UTC.
//!
//! Writing: a new event is PUT as its own .ics; an update re-reads the event's .ics, replaces the changed
//! properties and PUTs it back with If-Match; a delete is a DELETE. Repeating events are only read, not edited.

use chrono::{DateTime, Duration, Local, NaiveDate, NaiveDateTime, TimeZone, Utc};
use reqwest::Method;

const ROOT: &str = "https://caldav.icloud.com/";

pub struct Cal {
    http: reqwest::Client,
    user: String,
    pass: String,
    calendars: Vec<Calendar>,
}

#[derive(Clone)]
pub struct Calendar {
    pub url: String,
    pub name: String,
}

#[derive(Clone, Copy, PartialEq)]
pub enum When {
    At(DateTime<Local>),
    Day(NaiveDate), // all-day
}

impl When {
    /// "YYYY-MM-DD HH:MM" (a time) or "YYYY-MM-DD" (all day), local.
    pub fn parse(s: &str) -> Option<When> {
        let s = s.trim();
        if s.len() <= 10 {
            return NaiveDate::parse_from_str(s, "%Y-%m-%d").ok().map(When::Day);
        }
        crate::jobs::parse_local(s).map(When::At)
    }

    pub fn show(&self) -> String {
        match self {
            When::At(t) => t.format("%a %b %-d, %-I:%M %p").to_string(),
            When::Day(d) => d.format("%a %b %-d (all day)").to_string(),
        }
    }

    /// Sort key / comparison point: the moment it starts (all-day: local midnight).
    pub fn instant(&self) -> DateTime<Local> {
        match self {
            When::At(t) => *t,
            When::Day(d) => Local.from_local_datetime(&d.and_hms_opt(0, 0, 0).unwrap()).earliest().unwrap_or_else(Local::now),
        }
    }

    fn ics(&self, name: &str) -> String {
        match self {
            When::At(t) => format!("{name}:{}", t.with_timezone(&Utc).format("%Y%m%dT%H%M%SZ")),
            When::Day(d) => format!("{name};VALUE=DATE:{}", d.format("%Y%m%d")),
        }
    }
}

#[derive(Clone)]
pub struct Event {
    pub uid: String,
    pub href: String, // the .ics resource (absolute URL)
    pub etag: String,
    pub title: String,
    pub start: When,
    pub end: Option<When>,
    pub location: String,
    pub calendar: String,
    pub recurring: bool,
}

impl Cal {
    pub fn new(http: reqwest::Client, user: String, pass: String) -> Cal {
        Cal { http, user, pass, calendars: Vec::new() }
    }

    pub fn calendars(&self) -> &[Calendar] {
        &self.calendars
    }

    async fn dav(&self, method: &str, url: &str, depth: &str, body: String) -> Result<(String, String), String> {
        let r = self.http.request(Method::from_bytes(method.as_bytes()).unwrap(), url)
            .basic_auth(&self.user, Some(&self.pass))
            .header("Depth", depth)
            .header("Content-Type", "application/xml; charset=utf-8")
            .body(body)
            .send().await.map_err(|e| format!("caldav {method}: {e}"))?;
        let (status, final_url) = (r.status(), r.url().to_string());
        let text = r.text().await.map_err(|e| format!("caldav {method}: {e}"))?;
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err("iCloud refused the sign-in (check ICLOUD_USERNAME / ICLOUD_APP_PASSWORD)".into());
        }
        if !status.is_success() {
            return Err(format!("caldav {method} {url}: HTTP {status}"));
        }
        Ok((text, final_url))
    }

    /// Find the event calendars (once; again after an error).
    pub async fn discover(&mut self) -> Result<(), String> {
        if !self.calendars.is_empty() {
            return Ok(());
        }
        let (xml, base) = self.dav("PROPFIND", ROOT, "0",
            r#"<d:propfind xmlns:d="DAV:"><d:prop><d:current-user-principal/></d:prop></d:propfind>"#.into()).await?;
        let principal = elements(&xml, "current-user-principal").first().and_then(|p| elements(p, "href").first().copied())
            .map(|h| resolve(&base, &unescape(h))).ok_or("caldav: no current-user-principal")?;
        let (xml, base) = self.dav("PROPFIND", &principal, "0",
            r#"<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav"><d:prop><c:calendar-home-set/></d:prop></d:propfind>"#.into()).await?;
        let home = elements(&xml, "calendar-home-set").first().and_then(|p| elements(p, "href").first().copied())
            .map(|h| resolve(&base, &unescape(h))).ok_or("caldav: no calendar-home-set")?;
        let (xml, base) = self.dav("PROPFIND", &home, "1",
            r#"<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav"><d:prop><d:resourcetype/><d:displayname/><c:supported-calendar-component-set/></d:prop></d:propfind>"#.into()).await?;
        for r in elements(&xml, "response") {
            let rt = elements(r, "resourcetype").first().copied().unwrap_or("");
            let comps = elements(r, "supported-calendar-component-set").first().copied().unwrap_or("VEVENT");
            if elements(rt, "calendar").is_empty() || !comps.contains("VEVENT") {
                continue;
            }
            let Some(href) = elements(r, "href").first().map(|h| resolve(&base, &unescape(h))) else { continue };
            let name = elements(r, "displayname").first().map(|n| unescape(n)).filter(|n| !n.is_empty())
                .unwrap_or_else(|| "Calendar".into());
            self.calendars.push(Calendar { url: if href.ends_with('/') { href } else { href + "/" }, name });
        }
        if self.calendars.is_empty() {
            return Err("caldav: no event calendars found".into());
        }
        Ok(())
    }

    /// Every event (repeating ones as instances) starting between `from` and `to`, all calendars, by start.
    pub async fn events(&mut self, from: DateTime<Local>, to: DateTime<Local>) -> Result<Vec<Event>, String> {
        self.discover().await?;
        let fmt = |t: DateTime<Local>| t.with_timezone(&Utc).format("%Y%m%dT%H%M%SZ").to_string();
        let (a, b) = (fmt(from), fmt(to));
        let body = format!(r#"<c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav"><d:prop><d:getetag/><c:calendar-data><c:expand start="{a}" end="{b}"/></c:calendar-data></d:prop><c:filter><c:comp-filter name="VCALENDAR"><c:comp-filter name="VEVENT"><c:time-range start="{a}" end="{b}"/></c:comp-filter></c:comp-filter></c:filter></c:calendar-query>"#);
        let mut out = Vec::new();
        for cal in self.calendars.clone() {
            let (xml, base) = match self.dav("REPORT", &cal.url, "1", body.clone()).await {
                Ok(x) => x,
                Err(e) => {
                    self.calendars.clear(); // rediscover next time (iCloud moves things around now and then)
                    return Err(e);
                }
            };
            for r in elements(&xml, "response") {
                let Some(href) = elements(r, "href").first().map(|h| resolve(&base, &unescape(h))) else { continue };
                let etag = elements(r, "getetag").first().map(|e| unescape(e)).unwrap_or_default();
                let Some(data) = elements(r, "calendar-data").first().map(|d| unescape(d)) else { continue };
                for ev in vevents(&data) {
                    let Some(start) = ev.when("DTSTART") else { continue };
                    out.push(Event {
                        uid: ev.get("UID").unwrap_or_default(),
                        href: href.clone(),
                        etag: etag.clone(),
                        title: ev.get("SUMMARY").unwrap_or_else(|| "(no title)".into()),
                        start,
                        end: ev.when("DTEND"),
                        location: ev.get("LOCATION").unwrap_or_default(),
                        calendar: cal.name.clone(),
                        recurring: ev.has("RECURRENCE-ID") || ev.has("RRULE"),
                    });
                }
            }
        }
        out.sort_by_key(|e| e.start.instant());
        Ok(out)
    }

    fn calendar(&self, name: Option<&str>) -> Result<Calendar, String> {
        let want = name.map(str::to_lowercase).or_else(|| std::env::var("CHRONOS_CALENDAR").ok().map(|n| n.to_lowercase()));
        match want {
            Some(w) => self.calendars.iter().find(|c| c.name.to_lowercase() == w).cloned().ok_or_else(|| {
                format!("no calendar named {w:?}; there are: {}",
                        self.calendars.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", "))
            }),
            None => self.calendars.first().cloned().ok_or_else(|| "no calendars".into()),
        }
    }

    /// Add an event; returns its uid and the calendar it went in.
    pub async fn create(&mut self, calendar: Option<&str>, title: &str, start: When, end: Option<When>, location: &str)
        -> Result<(String, String), String> {
        self.discover().await?;
        let cal = self.calendar(calendar)?;
        let end = end.unwrap_or(match start {
            When::At(t) => When::At(t + Duration::hours(1)),
            When::Day(d) => When::Day(d + Duration::days(1)),
        });
        let uid = format!("{}-{}@chronos", Local::now().timestamp_millis(), std::process::id());
        let mut lines = vec!["BEGIN:VCALENDAR".to_string(), "VERSION:2.0".into(), "PRODID:-//max//chronos//EN".into(),
                             "BEGIN:VEVENT".into(), format!("UID:{uid}"), stamp(), start.ics("DTSTART"), end.ics("DTEND"),
                             format!("SUMMARY:{}", esc(title))];
        if !location.is_empty() {
            lines.push(format!("LOCATION:{}", esc(location)));
        }
        lines.extend(["END:VEVENT".to_string(), "END:VCALENDAR".into()]);
        let url = format!("{}{uid}.ics", cal.url);
        let r = self.http.put(&url).basic_auth(&self.user, Some(&self.pass))
            .header("Content-Type", "text/calendar; charset=utf-8").header("If-None-Match", "*")
            .body(lines.join("\r\n") + "\r\n").send().await.map_err(|e| format!("caldav PUT: {e}"))?;
        if !r.status().is_success() {
            return Err(format!("caldav PUT: HTTP {}", r.status()));
        }
        Ok((uid, cal.name))
    }

    /// Change an event's title, times or location (None: keep). Moving the start keeps the length unless an end is
    /// given.
    pub async fn update(&mut self, ev: &Event, title: Option<&str>, start: Option<When>, end: Option<When>,
                        location: Option<&str>) -> Result<(), String> {
        let r = self.http.get(&ev.href).basic_auth(&self.user, Some(&self.pass)).send().await
            .map_err(|e| format!("caldav GET: {e}"))?;
        if !r.status().is_success() {
            return Err(format!("caldav GET: HTTP {}", r.status()));
        }
        let etag = r.headers().get("etag").and_then(|v| v.to_str().ok()).map(String::from);
        let ics = r.text().await.map_err(|e| format!("caldav GET: {e}"))?;
        if ics.contains("\nRRULE") {
            return Err("that is a repeating event; change it in the Calendar app".into());
        }
        let lines = unfold(&ics);
        let old = vevents(&ics).into_iter().next().ok_or("caldav: the event has no VEVENT")?;
        let (old_start, old_end) = (old.when("DTSTART"), old.when("DTEND"));
        let end = end.or_else(|| match (start, old_start, old_end) {
            (Some(When::At(n)), Some(When::At(s)), Some(When::At(e))) => Some(When::At(n + (e - s))),
            (Some(When::Day(n)), Some(When::Day(s)), Some(When::Day(e))) => Some(When::Day(n + (e - s))),
            (Some(When::At(n)), _, _) => Some(When::At(n + Duration::hours(1))),
            (Some(When::Day(n)), _, _) => Some(When::Day(n + Duration::days(1))),
            _ => None,
        });
        let mut set: Vec<(&str, String)> = vec![("DTSTAMP", stamp())];
        if let Some(t) = title { set.push(("SUMMARY", format!("SUMMARY:{}", esc(t)))) }
        if let Some(s) = start { set.push(("DTSTART", s.ics("DTSTART"))) }
        if let Some(e) = end { set.push(("DTEND", e.ics("DTEND"))) }
        if let Some(l) = location { set.push(("LOCATION", format!("LOCATION:{}", esc(l)))) }
        // rewrite the (first) VEVENT: drop the lines being replaced, add the new ones before its END
        let (mut out, mut inside, mut done) = (Vec::new(), false, false);
        for l in lines {
            let name = l.split([':', ';']).next().unwrap_or("").to_ascii_uppercase();
            if l == "BEGIN:VEVENT" && !done {
                inside = true;
            } else if l == "END:VEVENT" && inside {
                out.extend(set.iter().map(|(_, v)| v.clone()));
                (inside, done) = (false, true);
            } else if inside && set.iter().any(|(n, _)| *n == name) {
                continue;
            }
            out.push(l);
        }
        let mut req = self.http.put(&ev.href).basic_auth(&self.user, Some(&self.pass))
            .header("Content-Type", "text/calendar; charset=utf-8");
        if let Some(t) = etag {
            req = req.header("If-Match", t);
        }
        let r = req.body(out.join("\r\n") + "\r\n").send().await.map_err(|e| format!("caldav PUT: {e}"))?;
        if !r.status().is_success() {
            return Err(format!("caldav PUT: HTTP {}", r.status()));
        }
        Ok(())
    }

    pub async fn delete(&mut self, ev: &Event) -> Result<(), String> {
        if ev.recurring {
            return Err("that is a repeating event; delete it in the Calendar app".into());
        }
        let r = self.http.delete(&ev.href).basic_auth(&self.user, Some(&self.pass)).send().await
            .map_err(|e| format!("caldav DELETE: {e}"))?;
        if !r.status().is_success() && r.status() != reqwest::StatusCode::NOT_FOUND {
            return Err(format!("caldav DELETE: HTTP {}", r.status()));
        }
        Ok(())
    }
}

fn stamp() -> String {
    format!("DTSTAMP:{}", Utc::now().format("%Y%m%dT%H%M%SZ"))
}

/// An href against the URL it came from: absolute as is, a path on the same origin.
fn resolve(base: &str, href: &str) -> String {
    if href.starts_with("http://") || href.starts_with("https://") {
        return href.to_string();
    }
    let origin_end = base.find("://").map(|i| i + 3).and_then(|i| base[i..].find('/').map(|j| i + j)).unwrap_or(base.len());
    format!("{}{}", &base[..origin_end], href)
}

/// The contents of every element with local name `name` (any namespace prefix), in document order; a self-closing
/// element gives "". Good enough for CalDAV multistatus replies (no same-name nesting).
fn elements<'a>(xml: &'a str, name: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(lt) = xml[i..].find('<').map(|p| p + i) {
        let rest = &xml[lt + 1..];
        let tag_end = rest.find(|c: char| c.is_whitespace() || c == '>' || c == '/').unwrap_or(rest.len());
        let qname = &rest[..tag_end];
        let local = qname.rsplit(':').next().unwrap_or(qname);
        let Some(gt) = rest.find('>') else { break };
        i = lt + 1 + gt + 1;
        if qname.is_empty() || qname.starts_with(['/', '?', '!']) || local != name {
            continue;
        }
        if rest[..gt].ends_with('/') {
            out.push("");
            continue;
        }
        let close = format!("</{qname}>");
        if let Some(c) = xml[i..].find(&close) {
            out.push(&xml[i..i + c]);
            i += c + close.len();
        }
    }
    out
}

fn unescape(s: &str) -> String {
    let s = s.trim();
    if let Some(inner) = s.strip_prefix("<![CDATA[").and_then(|t| t.strip_suffix("]]>")) {
        return inner.to_string();
    }
    s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&apos;", "'")
        .replace("&#13;", "\r").replace("&#xD;", "\r").replace("&#xd;", "\r").replace("&amp;", "&")
}

/// iCalendar text escaping (RFC 5545 3.3.11).
fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace(';', "\\;").replace(',', "\\,").replace('\n', "\\n")
}

fn unesc(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('n' | 'N') => out.push('\n'),
            Some(o) => out.push(o),
            None => {}
        }
    }
    out
}

/// Lines with RFC 5545 folding undone.
fn unfold(ics: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for l in ics.split('\n').map(|l| l.trim_end_matches('\r')) {
        match (l.strip_prefix([' ', '\t']), out.last_mut()) {
            (Some(cont), Some(prev)) => prev.push_str(cont),
            _ if l.is_empty() => {}
            _ => out.push(l.to_string()),
        }
    }
    out
}

/// One VEVENT's properties: (NAME, params, value).
struct VEvent(Vec<(String, String, String)>);

impl VEvent {
    fn prop(&self, name: &str) -> Option<&(String, String, String)> {
        self.0.iter().find(|(n, _, _)| n == name)
    }
    fn has(&self, name: &str) -> bool {
        self.prop(name).is_some()
    }
    fn get(&self, name: &str) -> Option<String> {
        self.prop(name).map(|(_, _, v)| unesc(v))
    }
    fn when(&self, name: &str) -> Option<When> {
        let (_, params, v) = self.prop(name)?;
        let v = v.trim();
        if params.to_ascii_uppercase().contains("VALUE=DATE") && !params.to_ascii_uppercase().contains("VALUE=DATE-TIME")
            || v.len() == 8 {
            return NaiveDate::parse_from_str(v, "%Y%m%d").ok().map(When::Day);
        }
        let naive = NaiveDateTime::parse_from_str(v.trim_end_matches('Z'), "%Y%m%dT%H%M%S").ok()?;
        if v.ends_with('Z') {
            Some(When::At(Utc.from_utc_datetime(&naive).with_timezone(&Local)))
        } else {
            // floating or TZID time (expand normally turns these into UTC): taken as local time
            Local.from_local_datetime(&naive).earliest().map(When::At)
        }
    }
}

fn vevents(ics: &str) -> Vec<VEvent> {
    let (mut out, mut cur): (Vec<VEvent>, Option<Vec<(String, String, String)>>) = (Vec::new(), None);
    let mut depth = 0; // nested components inside a VEVENT (VALARM) are skipped
    for l in unfold(ics) {
        match l.as_str() {
            "BEGIN:VEVENT" => cur = Some(Vec::new()),
            "END:VEVENT" => out.extend(cur.take().map(VEvent)),
            _ if l.starts_with("BEGIN:") && cur.is_some() => depth += 1,
            _ if l.starts_with("END:") && cur.is_some() && depth > 0 => depth -= 1,
            _ => {
                if let (Some(c), 0) = (cur.as_mut(), depth) {
                    if let Some((head, value)) = l.split_once(':') {
                        let (name, params) = head.split_once(';').unwrap_or((head, ""));
                        c.push((name.to_ascii_uppercase(), params.to_string(), value.to_string()));
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_multistatus_and_ics() {
        let xml = r#"<?xml version="1.0"?><multistatus xmlns="DAV:"><response><href>/1/calendars/home/a.ics</href>
            <propstat><prop><getetag>"e1"</getetag><C:calendar-data xmlns:C="urn:ietf:params:xml:ns:caldav">BEGIN:VCALENDAR
BEGIN:VEVENT
UID:abc
SUMMARY:Dentist\, Dr. Lee
DTSTART:20261117T200000Z
DTEND:20261117T210000Z
BEGIN:VALARM
SUMMARY:alarm
END:VALARM
END:VEVENT
END:VCALENDAR
</C:calendar-data></prop></propstat></response></multistatus>"#;
        let rs = elements(xml, "response");
        assert_eq!(rs.len(), 1);
        assert_eq!(elements(rs[0], "href")[0], "/1/calendars/home/a.ics");
        let data = unescape(elements(rs[0], "calendar-data")[0]);
        let ev = &vevents(&data)[0];
        assert_eq!(ev.get("SUMMARY").unwrap(), "Dentist, Dr. Lee");
        assert!(matches!(ev.when("DTSTART"), Some(When::At(_))));
        assert_eq!(resolve("https://p1-caldav.icloud.com/1/principal/", "/1/calendars/"), "https://p1-caldav.icloud.com/1/calendars/");
        assert!(elements(r#"<d:resourcetype><d:collection/><cal:calendar/></d:resourcetype>"#, "calendar").len() == 1);
    }
}
