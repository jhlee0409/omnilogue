//! Headless session-format audit: `--audit-providers [--sessions N]`.
//!
//! Assistants change how they store sessions without notice, and the parsers
//! then drop content silently (a tool's output moved to another field, a new
//! record type, a missing title). This loads the latest sessions of every
//! assistant found on this machine, through the same loaders the app uses,
//! and prints counts that show such drift. Nothing from a conversation is
//! printed: no text, paths, project names or ids; only provider ids,
//! message and block type names, tool names and counts. The output is meant
//! to be pasted into a bug report.

use std::collections::{BTreeMap, HashSet};

use serde::Serialize;
use serde_json::Value;

use crate::cli_args::extract_flag_value;
use crate::commands::multi_provider::{
    detect_providers, load_provider_messages, load_provider_sessions, scan_all_projects,
};
use crate::models::ClaudeMessage;

/// Sessions read per assistant when `--sessions` is not given.
const DEFAULT_SESSIONS: usize = 10;

/// Content block types the transcript renders with a dedicated view; any
/// other type is reported as unknown.
const KNOWN_BLOCK_TYPES: &[&str] = &[
    "text",
    "thinking",
    "redacted_thinking",
    "tool_use",
    "tool_result",
    "image",
    "document",
    "search_result",
    "server_tool_use",
    "web_search_tool_result",
    "web_fetch_tool_result",
    "code_execution_tool_result",
    "bash_code_execution_tool_result",
    "text_editor_code_execution_tool_result",
    "tool_search_tool_result",
    "mcp_tool_use",
    "mcp_tool_result",
    "container_upload",
    "opencode_step",
    "advisor_tool_result",
];

const USAGE: &str = "Usage: --audit-providers [--sessions N] [--output <file>]\n\
Prints counts (no conversation text, paths or names) for the latest N sessions of each assistant found on this computer, or writes them to <file>.";

/// What one assistant's sampled sessions look like.
#[derive(Debug, Default, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ProviderAudit {
    pub projects: usize,
    pub sessions_read: usize,
    pub messages: usize,
    /// Message `type` → count.
    pub message_types: BTreeMap<String, usize>,
    /// Content block `type` → count (`<string>` for plain text content).
    pub block_types: BTreeMap<String, usize>,
    /// Block types with no dedicated view.
    pub unknown_block_types: BTreeMap<String, usize>,
    /// Tool name → calls.
    pub tool_names: BTreeMap<String, usize>,
    pub tool_calls: usize,
    /// Tool calls whose result is not in the same session.
    pub tool_calls_without_result: usize,
    /// Messages with no content at all.
    pub empty_messages: usize,
    /// Messages whose recorded role is neither user nor assistant but that
    /// are shown as user or assistant messages (e.g. instructions).
    pub role_mismatches: usize,
    pub sessions_without_title: usize,
    /// Sessions listed with messages that loaded none.
    pub sessions_loaded_empty: usize,
    pub load_errors: usize,
}

impl ProviderAudit {
    /// Count one loaded session.
    pub fn add_session(
        &mut self,
        listed_messages: usize,
        title: Option<&str>,
        messages: &[ClaudeMessage],
    ) {
        self.sessions_read += 1;
        if title.is_none_or(|title| title.trim().is_empty()) {
            self.sessions_without_title += 1;
        }
        if messages.is_empty() && listed_messages > 0 {
            self.sessions_loaded_empty += 1;
        }
        let mut calls = HashSet::new();
        let mut results = HashSet::new();
        for message in messages {
            self.messages += 1;
            *self
                .message_types
                .entry(message.message_type.clone())
                .or_default() += 1;
            if matches!(message.message_type.as_str(), "user" | "assistant")
                && message
                    .role
                    .as_deref()
                    .is_some_and(|role| !matches!(role, "user" | "assistant"))
            {
                self.role_mismatches += 1;
            }
            match &message.content {
                None | Some(Value::Null) => self.empty_messages += 1,
                Some(Value::String(text)) if text.is_empty() => self.empty_messages += 1,
                Some(Value::Array(blocks)) if blocks.is_empty() => self.empty_messages += 1,
                Some(Value::String(_)) => {
                    *self.block_types.entry("<string>".into()).or_default() += 1;
                }
                Some(Value::Array(blocks)) => {
                    for block in blocks {
                        self.add_block(block, &mut calls, &mut results);
                    }
                }
                Some(_) => *self.block_types.entry("<other>".into()).or_default() += 1,
            }
        }
        self.tool_calls_without_result += calls.difference(&results).count();
    }

    fn add_block(
        &mut self,
        block: &Value,
        calls: &mut HashSet<String>,
        results: &mut HashSet<String>,
    ) {
        let kind = block
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("<untyped>");
        *self.block_types.entry(kind.to_string()).or_default() += 1;
        if !KNOWN_BLOCK_TYPES.contains(&kind) {
            *self
                .unknown_block_types
                .entry(kind.to_string())
                .or_default() += 1;
        }
        match kind {
            "tool_use" | "server_tool_use" | "mcp_tool_use" => {
                self.tool_calls += 1;
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("<unnamed>");
                *self.tool_names.entry(name.to_string()).or_default() += 1;
                match block.get("id").and_then(Value::as_str) {
                    Some(id) => {
                        calls.insert(id.to_string());
                    }
                    // Nothing can pair with a call that has no id.
                    None => self.tool_calls_without_result += 1,
                }
            }
            _ => {
                if let Some(id) = block.get("tool_use_id").and_then(Value::as_str) {
                    results.insert(id.to_string());
                }
            }
        }
    }
}

/// Entry point for `--audit-providers`; returns the process exit code.
pub fn run_audit(args: &[String]) -> i32 {
    let sessions = match extract_flag_value(args, "--sessions").map(|value| value.parse::<usize>())
    {
        None => DEFAULT_SESSIONS,
        Some(Ok(count)) if count > 0 => count,
        Some(_) => {
            eprintln!("{USAGE}");
            return 2;
        }
    };
    let report = tauri::async_runtime::block_on(audit(sessions));
    let json = match serde_json::to_string_pretty(&report) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Could not print the report: {error}");
            return 1;
        }
    };
    // The Windows desktop build has no console to print to: `--output`.
    let output_given = args
        .iter()
        .any(|a| a == "--output" || a.starts_with("--output="));
    let Some(path) = extract_flag_value(args, "--output").filter(|path| !path.is_empty()) else {
        if output_given {
            eprintln!("{USAGE}");
            return 2;
        }
        println!("{json}");
        return 0;
    };
    if let Err(error) = std::fs::write(&path, format!("{json}\n")) {
        eprintln!("Could not write {path}: {error}");
        return 1;
    }
    eprintln!("Report written to {path}");
    0
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AuditReport {
    app_version: &'static str,
    sessions_per_provider: usize,
    providers: BTreeMap<String, ProviderAudit>,
}

async fn audit(sessions_per_provider: usize) -> AuditReport {
    let available: Vec<String> = detect_providers()
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|provider| provider.is_available)
        .map(|provider| provider.id)
        .collect();
    let projects = scan_all_projects(None, Some(available.clone()), None, None, None)
        .await
        .unwrap_or_default();

    let mut providers: BTreeMap<String, ProviderAudit> = BTreeMap::new();
    for id in &available {
        let mut audit = ProviderAudit::default();
        let mut mine: Vec<_> = projects
            .iter()
            .map(|listed| &listed.project)
            .filter(|project| project.provider.as_deref().unwrap_or("claude") == id)
            .collect();
        audit.projects = mine.len();
        mine.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));

        // Newest projects first; a project's sessions are no newer than the
        // project itself, so stop once the N newest sessions found so far are
        // all newer than the next project.
        let mut sessions = Vec::new();
        let mut remaining = mine.into_iter().peekable();
        while let Some(project) = remaining.next() {
            match load_provider_sessions(id.clone(), project.path.clone(), None).await {
                Ok(listed) => sessions.extend(listed.into_iter().map(|listed| listed.session)),
                Err(_) => audit.load_errors += 1,
            }
            sessions.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
            let enough = sessions.get(sessions_per_provider.saturating_sub(1));
            if let (Some(nth), Some(next)) = (enough, remaining.peek()) {
                if next.last_modified <= nth.last_modified {
                    break;
                }
            }
        }
        for session in sessions.into_iter().take(sessions_per_provider) {
            match load_provider_messages(id.clone(), session.file_path.clone()).await {
                Ok(messages) => {
                    audit.add_session(session.message_count, session.summary.as_deref(), &messages);
                }
                Err(_) => audit.load_errors += 1,
            }
        }
        providers.insert(id.clone(), audit);
    }

    AuditReport {
        app_version: env!("CARGO_PKG_VERSION"),
        sessions_per_provider,
        providers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn message(message_type: &str, role: Option<&str>, content: Value) -> ClaudeMessage {
        let mut message = crate::utils::build_provider_message(
            "test",
            "m".into(),
            "s",
            "2026-11-25T00:00:00Z".into(),
            message_type,
            role,
            Some(content),
            None,
        );
        if message.content == Some(Value::Null) {
            message.content = None;
        }
        message
    }

    /// The counts that reveal format drift: a tool call without its result,
    /// an unknown block type, instructions shown as a reply, a missing title.
    #[test]
    fn counts_the_signs_of_format_drift() {
        let mut audit = ProviderAudit::default();
        let messages = [
            message("user", Some("user"), json!("fix it")),
            message(
                "assistant",
                Some("developer"),
                json!([{ "type": "text", "text": "instructions" }]),
            ),
            message(
                "assistant",
                Some("assistant"),
                json!([
                    { "type": "tool_use", "id": "a", "name": "Read", "input": {} },
                    { "type": "tool_use", "id": "b", "name": "Bash", "input": {} },
                    { "type": "tool_use", "name": "Grep", "input": {} },
                    { "type": "new_block" }
                ]),
            ),
            message(
                "user",
                Some("user"),
                json!([{ "type": "tool_result", "tool_use_id": "a", "content": "ok" }]),
            ),
            message("assistant", Some("assistant"), Value::Null),
        ];
        audit.add_session(5, None, &messages);
        audit.add_session(3, Some("Title"), &[]);

        assert_eq!(audit.sessions_read, 2);
        assert_eq!(audit.sessions_without_title, 1);
        assert_eq!(audit.sessions_loaded_empty, 1);
        assert_eq!(audit.messages, 5);
        assert_eq!(audit.tool_calls, 3);
        assert_eq!(audit.tool_calls_without_result, 2);
        assert_eq!(audit.unknown_block_types.get("new_block"), Some(&1));
        assert_eq!(audit.role_mismatches, 1);
        assert_eq!(audit.empty_messages, 1);
        assert_eq!(audit.tool_names.get("Bash"), Some(&1));
    }

    /// The report carries no conversation text.
    #[test]
    fn report_holds_no_conversation_text() {
        let mut audit = ProviderAudit::default();
        audit.add_session(
            1,
            Some("secret title"),
            &[message("user", Some("user"), json!("secret prompt"))],
        );
        let printed = serde_json::to_string(&audit).unwrap();
        assert!(!printed.contains("secret"));
    }
}
