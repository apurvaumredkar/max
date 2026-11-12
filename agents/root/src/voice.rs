//! Voice mode: a WebSocket session (GET /voice) where the browser streams its microphone in and gets Max's voice out.
//!
//! Browser -> agent: binary frames of 16 kHz mono s16le PCM (the mic), and JSON text {"type":"interrupt"} to cut the
//! current turn short (stops generation and speech; the partial reply is kept). Closing the socket ends the session.
//! Agent -> browser: JSON text events, plus binary frames of s16le PCM for Max's speech:
//!   {"type":"ready"}                                  STT connected, start sending audio
//!   {"type":"heard","text"}                           an utterance the STT finished (becomes the user turn)
//!   {"type":"thinking_delta","text"} / {"type":"reply_delta","text"}   the turn as the model writes it
//!   {"type":"speaking","rate":24000} then <PCM...>    Max's voice, sentence by sentence as the reply streams
//!   {"type":"reply","text","thinking","interrupted"}  the finished (or cut) reply, as stored in agents.db
//!   {"type":"done"}                                   turn over: all audio sent
//!   {"type":"error","error"}
//!
//! Mic audio is relayed to canary-stt-server's /ws (Silero VAD ends an utterance after ~0.7 s of silence). Each
//! utterance runs a streamed chat turn; complete sentences go to tts-server's /stream (Phonon, voice VOICE) while the
//! model is still writing. Phonon generates one request at a time (a second one gets 429), so speech is synthesized
//! in order, one chunk at a time; each chunk takes every sentence finished by then, so the first sentence starts
//! fast and later ones come in longer, smoother pieces.

use std::{sync::{Arc, Mutex}, time::{Duration, Instant}};

use agentlog::{error, info, warn};
use axum::{
    extract::{State, ws::{Message, WebSocket, WebSocketUpgrade}},
    response::Response,
};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio_tungstenite::tungstenite::Message as SttMsg;
use tokio_util::sync::CancellationToken;

use crate::{Agent, Delta};

pub const VOICE: &str = "Russell"; // Phonon catalog voice, American male

type Out = UnboundedSender<Message>;

fn event(out: &Out, v: Value) {
    let _ = out.send(Message::Text(v.to_string().into()));
}

pub async fn upgrade(ws: WebSocketUpgrade, State(agent): State<Arc<Agent>>) -> Response {
    ws.on_upgrade(move |socket| session(agent, socket))
}

async fn session(agent: Arc<Agent>, socket: WebSocket) {
    let (mut to_browser, mut from_browser) = socket.split();
    let (out, mut out_rx) = unbounded_channel::<Message>();
    let writer = tokio::spawn(async move {
        while let Some(m) = out_rx.recv().await {
            if to_browser.send(m).await.is_err() {
                break;
            }
        }
    });

    let stt = match tokio_tungstenite::connect_async(agent.stt_url.as_str()).await {
        Ok((stt, _)) => stt,
        Err(e) => {
            error!(error = %e, url = %agent.stt_url, "voice: STT connect failed");
            event(&out, json!({"type": "error", "error": format!("speech-to-text unavailable: {e}")}));
            drop(out);
            let _ = writer.await;
            return;
        }
    };
    info!(voice = VOICE, "voice session started");
    let (mut to_stt, mut from_stt) = stt.split();
    event(&out, json!({"type": "ready"}));

    // the running turn's cancel token; an interrupt from the browser cancels it
    let current = Arc::new(Mutex::new(CancellationToken::new()));

    // browser -> STT (mic) and interrupts
    let mut mic = {
        let current = current.clone();
        tokio::spawn(async move {
            while let Some(Ok(m)) = from_browser.next().await {
                match m {
                    Message::Binary(pcm) => {
                        if to_stt.send(SttMsg::Binary(pcm)).await.is_err() {
                            break;
                        }
                    }
                    Message::Text(t) => {
                        let v: Value = serde_json::from_str(t.as_str()).unwrap_or_default();
                        if v["type"] == "interrupt" {
                            info!("voice: interrupted");
                            current.lock().unwrap().cancel();
                        }
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
            let _ = to_stt.close().await;
        })
    };

    // utterances -> streamed turns -> speech, one at a time
    let turns = {
        let (agent, out, current) = (agent.clone(), out.clone(), current.clone());
        async move {
            while let Some(Ok(m)) = from_stt.next().await {
                let SttMsg::Text(t) = m else { continue };
                let text = serde_json::from_str::<Value>(t.as_str()).ok()
                    .and_then(|v| v["text"].as_str().map(|s| s.trim().to_string()))
                    .unwrap_or_default();
                if text.is_empty() {
                    continue;
                }
                let cancel = CancellationToken::new();
                *current.lock().unwrap() = cancel.clone();
                info!(chars = text.chars().count(), "voice: heard");
                event(&out, json!({"type": "heard", "text": text}));
                turn(&agent, &text, &out, &cancel).await;
                event(&out, json!({"type": "done"}));
            }
        }
    };

    tokio::select! {
        _ = &mut mic => {}                  // browser left (or STT stopped accepting audio)
        _ = turns => { mic.abort(); }       // STT closed the stream
    }
    current.lock().unwrap().cancel();
    drop(out);
    let _ = writer.await;
    info!("voice session ended");
}

/// One spoken turn: stream the reply, speaking each finished sentence while the rest is still being written.
async fn turn(agent: &Agent, text: &str, out: &Out, cancel: &CancellationToken) {
    let (sentences, queue) = unbounded_channel::<String>();
    let model = async {
        let mut split = Splitter::default();
        let result = agent.chat_stream(text, cancel, |d| match d {
            Delta::Thinking(t) => event(out, json!({"type": "thinking_delta", "text": t})),
            Delta::Content(c) => {
                event(out, json!({"type": "reply_delta", "text": c}));
                for s in split.push(c) {
                    let _ = sentences.send(s);
                }
            }
        }).await;
        if !cancel.is_cancelled() {
            if let Some(rest) = split.finish() {
                let _ = sentences.send(rest);
            }
        }
        drop(sentences); // lets the speaker finish once the queue is empty
        result
    };
    let (result, ()) = tokio::join!(model, speaker(agent, queue, out, cancel));
    match result {
        Ok((reply, thinking, interrupted)) => {
            event(out, json!({"type": "reply", "text": reply, "thinking": thinking, "interrupted": interrupted}))
        }
        Err(e) => event(out, json!({"type": "error", "error": e})),
    }
}

/// Speak queued sentences in order, one TTS request at a time, batching whatever has queued up meanwhile.
async fn speaker(agent: &Agent, mut queue: UnboundedReceiver<String>, out: &Out, cancel: &CancellationToken) {
    let mut announced = false;
    loop {
        let mut chunk = tokio::select! {
            s = queue.recv() => match s { Some(s) => s, None => break },
            _ = cancel.cancelled() => break,
        };
        while let Ok(more) = queue.try_recv() {
            chunk.push(' ');
            chunk.push_str(&more);
        }
        speak(agent, &chunk, out, cancel, &mut announced).await;
        if cancel.is_cancelled() {
            break;
        }
    }
}

/// Stream one chunk of text through the TTS as `VOICE`, forwarding PCM to the browser as it arrives.
async fn speak(agent: &Agent, text: &str, out: &Out, cancel: &CancellationToken, announced: &mut bool) {
    let started = Instant::now();
    let mut tries = 0;
    let resp = loop {
        let req = agent.http.post(&agent.tts_url).json(&json!({"text": text, "voice": VOICE})).send();
        let resp = tokio::select! { r = req => r, _ = cancel.cancelled() => return };
        match resp {
            // the engine is busy with another request (one generation at a time): wait briefly and retry
            Ok(r) if r.status().as_u16() == 429 && tries < 25 => {
                tries += 1;
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            Ok(r) => break r,
            Err(e) => {
                error!(error = %e, "voice: TTS request failed");
                event(out, json!({"type": "error", "error": format!("text-to-speech unavailable: {e}")}));
                return;
            }
        }
    };
    if !resp.status().is_success() {
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        if status.as_u16() == 400 {
            warn!(detail = %detail, "voice: nothing speakable in chunk"); // e.g. emoji only
        } else {
            error!(%status, detail = %detail, "voice: TTS error");
            event(out, json!({"type": "error", "error": format!("text-to-speech {status}: {detail}")}));
        }
        return;
    }
    let rate: u32 = resp.headers().get("x-sample-rate")
        .and_then(|v| v.to_str().ok()?.parse().ok())
        .unwrap_or(24000);
    if !*announced {
        event(out, json!({"type": "speaking", "rate": rate}));
        *announced = true;
    }

    let (mut stream, mut carry, mut bytes, mut first_ms) = (resp.bytes_stream(), None::<u8>, 0usize, None);
    loop {
        let chunk = tokio::select! { c = stream.next() => c, _ = cancel.cancelled() => break };
        let Some(Ok(chunk)) = chunk else { break };
        // keep frames on whole 16-bit samples
        let mut buf: Vec<u8> = carry.take().into_iter().chain(chunk.iter().copied()).collect();
        if buf.len() % 2 == 1 {
            carry = buf.pop();
        }
        if buf.is_empty() {
            continue;
        }
        first_ms.get_or_insert(started.elapsed().as_millis() as u64);
        bytes += buf.len();
        if out.send(Message::Binary(buf.into())).is_err() {
            return; // browser gone
        }
    }
    info!(chars = text.chars().count(), first_audio_ms = first_ms.unwrap_or(0), retries = tries,
          ms = started.elapsed().as_millis() as u64, audio_ms = (bytes as u64 / 2) * 1000 / rate as u64,
          cut = cancel.is_cancelled(), "voice: spoke");
}

/// Cuts streamed reply text into sentences: after . ! ? (or …) followed by whitespace, and at newlines. Never inside
/// an open ``` code fence, so a fence reaches the TTS whole (and gets dropped there instead of read out).
#[derive(Default)]
struct Splitter {
    buf: String,
}

impl Splitter {
    fn push(&mut self, piece: &str) -> Vec<String> {
        self.buf.push_str(piece);
        let mut out = Vec::new();
        while let Some(end) = self.boundary() {
            let s = self.buf[..end].trim().to_string();
            self.buf.drain(..end);
            if !s.is_empty() {
                out.push(s);
            }
        }
        out
    }

    /// Byte index just past the first usable sentence end, if the text after it has started.
    fn boundary(&self) -> Option<usize> {
        let mut chars = self.buf.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            let end = i + c.len_utf8();
            let ends = c == '\n'
                || (matches!(c, '.' | '!' | '?' | '…') && chars.peek().is_some_and(|&(_, n)| n.is_whitespace()));
            if ends && self.buf[..end].matches("```").count() % 2 == 0 {
                return Some(end);
            }
        }
        None
    }

    fn finish(self) -> Option<String> {
        Some(self.buf.trim().to_string()).filter(|s| !s.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::Splitter;

    fn split(pieces: &[&str]) -> Vec<String> {
        let mut s = Splitter::default();
        let mut out: Vec<String> = pieces.iter().flat_map(|p| s.push(p)).collect();
        out.extend(s.finish());
        out
    }

    #[test]
    fn sentences_across_tokens() {
        assert_eq!(split(&["Hey", " there! How", "'s it", " going? Fine", "."]),
                   ["Hey there!", "How's it going?", "Fine."]);
    }

    #[test]
    fn waits_for_text_after_the_period() {
        let mut s = Splitter::default();
        assert!(s.push("It costs 3.").is_empty()); // could be "3.5"
        assert_eq!(s.push("5 dollars. Ok"), ["It costs 3.5 dollars."]);
    }

    #[test]
    fn newlines_and_fences() {
        assert_eq!(split(&["List:\n- one\n```\nx = 1. y = 2.\n```\nDone."]),
                   ["List:", "- one", "```\nx = 1. y = 2.\n```", "Done."]);
    }
}
