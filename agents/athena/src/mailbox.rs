//! Where Athena's tools read mail from: the live email-mcp server, or a sandbox fixture (evaluations). The agent
//! can't tell them apart: both take email-mcp's tool names and arguments and answer in its text format.

use serde_json::Value;

use crate::{fixture::Fixture, mcp::Mcp};

pub enum Mailbox {
    Live(Mcp),
    Fixture(Fixture),
}

impl Mailbox {
    pub async fn call(&mut self, tool: &str, args: Value) -> Result<String, String> {
        match self {
            Mailbox::Live(m) => m.call(tool, args).await,
            Mailbox::Fixture(f) => f.call(tool, &args),
        }
    }

    pub async fn close(self) {
        if let Mailbox::Live(m) = self {
            m.close().await;
        }
    }
}
