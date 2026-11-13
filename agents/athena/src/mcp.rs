//! A minimal MCP client over Streamable HTTP: initialize, then `tools/call`. Responses come back as JSON or as an
//! SSE stream of JSON-RPC messages; both are handled. One session per check (the server expires idle ones).

use reqwest::{Client, Response};
use serde_json::{Value, json};

pub struct Mcp {
    http: Client,
    url: String,
    key: Option<String>, // X-Api-Key, when the server sets MCP_AUTH_TOKEN
    session: Option<String>,
    version: String,
    next_id: u64,
}

impl Mcp {
    pub async fn connect(http: &Client, url: &str, key: Option<String>) -> Result<Mcp, String> {
        let mut m = Mcp {
            http: http.clone(),
            url: url.to_string(),
            key,
            session: None,
            version: "2025-03-26".into(),
            next_id: 0,
        };
        let init = m
            .rpc("initialize", json!({"protocolVersion": m.version, "capabilities": {},
                                      "clientInfo": {"name": "athena", "version": env!("CARGO_PKG_VERSION")}}))
            .await?;
        if let Some(v) = init["protocolVersion"].as_str() {
            m.version = v.to_string();
        }
        m.post(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).await?;
        Ok(m)
    }

    /// Call a tool; its text content on success, the text as Err when the tool reports an error.
    pub async fn call(&mut self, tool: &str, args: Value) -> Result<String, String> {
        let r = self.rpc("tools/call", json!({"name": tool, "arguments": args})).await?;
        let text = r["content"]
            .as_array()
            .map(|c| c.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("\n"))
            .unwrap_or_default();
        if r["isError"].as_bool() == Some(true) { Err(format!("{tool}: {text}")) } else { Ok(text) }
    }

    /// End the session (best effort).
    pub async fn close(self) {
        if let Some(s) = &self.session {
            let _ = self.http.delete(&self.url).header("mcp-session-id", s).send().await;
        }
    }

    async fn post(&self, body: &Value) -> Result<Response, String> {
        let mut req = self
            .http
            .post(&self.url)
            .header("accept", "application/json, text/event-stream")
            .header("mcp-protocol-version", &self.version)
            .json(body);
        if let Some(s) = &self.session {
            req = req.header("mcp-session-id", s);
        }
        if let Some(k) = &self.key {
            req = req.header("x-api-key", k);
        }
        let resp = req.send().await.map_err(|e| format!("email-mcp: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            return Err(format!("email-mcp {status}: {}", resp.text().await.unwrap_or_default()));
        }
        Ok(resp)
    }

    async fn rpc(&mut self, method: &str, params: Value) -> Result<Value, String> {
        self.next_id += 1;
        let id = self.next_id;
        let resp = self.post(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})).await?;
        if self.session.is_none() {
            self.session = resp.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()).map(String::from);
        }
        let sse = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|t| t.starts_with("text/event-stream"));
        let body = resp.text().await.map_err(|e| format!("email-mcp: {e}"))?;
        let messages: Vec<Value> = if sse {
            body.lines().filter_map(|l| l.strip_prefix("data:")).filter_map(|d| serde_json::from_str(d.trim()).ok()).collect()
        } else {
            serde_json::from_str(&body).map(|v| vec![v]).map_err(|e| format!("email-mcp: bad JSON: {e}"))?
        };
        let msg = messages
            .into_iter()
            .find(|m| m["id"].as_u64() == Some(id))
            .ok_or_else(|| format!("email-mcp: no response to {method}"))?;
        if let Some(e) = msg.get("error") {
            return Err(format!("email-mcp {method}: {}", e["message"].as_str().unwrap_or("error")));
        }
        Ok(msg["result"].clone())
    }
}
