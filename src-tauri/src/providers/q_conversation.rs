//! Shared `ConversationState` → message conversion for the Amazon Q CLI lineage.
//!
//! Amazon Q CLI (`amazon-q/data.sqlite3`, table `conversations`) and its
//! rebrand Kiro CLI (`kiro-cli/data.sqlite3`, table `conversations_v2`) store
//! the **same** serialized `ConversationState` in their `value` column — only
//! the surrounding table/row layout differs. This module owns the value→message
//! conversion both providers share; it's parameterized by `provider` so each
//! tags its own id.
//!
//! `value` JSON: `{ "history": [ { "user": {...}, "assistant": {...} }, ... ] }`
//! with externally-tagged enums:
//! - `user.content`: `Prompt{prompt}` | `ToolUseResults{tool_use_results}` |
//!   `CancelledToolUses{prompt?, tool_use_results}`; each result is
//!   `{tool_use_id, content: [{Text} | {Json}], status: Success | Error}`
//! - `assistant`: `Response{message_id?, content}` |
//!   `ToolUse{message_id?, content, tool_uses[]{id,name,args}}`
//!
//! Source: `aws/amazon-q-developer-cli` `crates/chat-cli/src/cli/chat/`
//! (`message.rs`, `conversation.rs`, `tools/`) at `15cc8f3c`. While the CLI is
//! in tangent mode, `history` is the main conversation followed by the
//! tangent's turns (`tangent_state.main_history` is that prefix), so the
//! saved row still holds the whole main conversation.

use crate::models::ClaudeMessage;
use crate::utils::build_provider_message;
use serde_json::{json, Value};

/// Parse a `ConversationState` `value` JSON string into messages, in order.
pub(crate) fn parse_history(
    provider: &str,
    value_json: &str,
    session_id: &str,
) -> Vec<ClaudeMessage> {
    let Ok(json) = serde_json::from_str::<Value>(value_json) else {
        return Vec::new();
    };
    let Some(history) = json.get("history").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut messages = Vec::new();
    for (i, entry) in history.iter().enumerate() {
        if let Some(user) = entry.get("user") {
            if let Some(msg) = convert_user_message(provider, user, session_id, i) {
                messages.push(msg);
            }
        }
        if let Some(assistant) = entry.get("assistant") {
            if let Some(msg) = convert_assistant_message(provider, assistant, session_id, i) {
                messages.push(msg);
            }
        }
    }
    messages
}

/// The (first, last) user-turn ISO timestamps in a history, if any. Amazon Q's
/// v1 `conversations` table has no `created_at`/`updated_at` columns, so session
/// times are derived from these. `last` falls back to `first` for single-turn
/// conversations.
pub(crate) fn history_time_bounds(value_json: &str) -> (Option<String>, Option<String>) {
    let Ok(json) = serde_json::from_str::<Value>(value_json) else {
        return (None, None);
    };
    let Some(history) = json.get("history").and_then(Value::as_array) else {
        return (None, None);
    };
    let mut times = history
        .iter()
        .filter_map(|e| {
            e.get("user")
                .and_then(|u| u.get("timestamp"))
                .and_then(Value::as_str)
        })
        .filter(|s| !s.is_empty());
    let first = times.next().map(str::to_string);
    // next_back() (not last()) avoids re-walking the whole iterator; gives the
    // last remaining timestamp, falling back to `first` for single-turn chats.
    let last = times
        .next_back()
        .map(str::to_string)
        .or_else(|| first.clone());
    (first, last)
}

/// Build a short session summary from the first user prompt in the history.
pub(crate) fn first_prompt_summary(value_json: &str, max_chars: usize) -> Option<String> {
    let json = serde_json::from_str::<Value>(value_json).ok()?;
    let history = json.get("history").and_then(Value::as_array)?;
    history.iter().find_map(|e| {
        e.get("user")?
            .get("content")?
            .get("Prompt")?
            .get("prompt")?
            .as_str()
            .map(|s| s.chars().take(max_chars).collect::<String>())
    })
}

/// Number of renderable messages a history would produce (user + assistant).
pub(crate) fn message_count(value_json: &str) -> usize {
    parse_history("", value_json, "count").len()
}

/// `(has_tool_use, has_errors)` of parsed messages: any tool call, and any
/// tool result recorded with `status: "Error"`.
pub(crate) fn session_flags(messages: &[ClaudeMessage]) -> (bool, bool) {
    let blocks = || {
        messages
            .iter()
            .filter_map(|m| m.content.as_ref()?.as_array())
            .flatten()
    };
    let has_tool_use = blocks().any(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"));
    let has_errors = blocks().any(|b| b.get("is_error").and_then(Value::as_bool) == Some(true));
    (has_tool_use, has_errors)
}

pub(crate) fn convert_user_message(
    provider: &str,
    user: &Value,
    session_id: &str,
    idx: usize,
) -> Option<ClaudeMessage> {
    let timestamp = user
        .get("timestamp")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let content_obj = user.get("content")?;
    let mut blocks: Vec<Value> = Vec::new();

    if let Some(prompt) = content_obj.get("Prompt") {
        if let Some(text) = prompt.get("prompt").and_then(Value::as_str) {
            if !text.is_empty() {
                blocks.push(json!({"type": "text", "text": text}));
            }
        }
    } else if let Some(tool_results) = content_obj.get("ToolUseResults") {
        push_tool_results(&mut blocks, tool_results);
    } else if let Some(cancelled) = content_obj.get("CancelledToolUses") {
        // A cancelled turn may carry a prompt plus the partial tool results.
        if let Some(text) = cancelled.get("prompt").and_then(Value::as_str) {
            if !text.is_empty() {
                blocks.push(json!({"type": "text", "text": text}));
            }
        }
        push_tool_results(&mut blocks, cancelled);
    }

    if blocks.is_empty() {
        return None;
    }

    Some(build_provider_message(
        provider,
        format!("{session_id}-user-{idx}"),
        session_id,
        timestamp,
        "user",
        Some("user"),
        Some(Value::Array(blocks)),
        None,
    ))
}

/// Append `tool_result` blocks from an object holding `tool_use_results[]`.
///
/// A result's `content` is a list of `{"Text": string}` / `{"Json": value}`
/// blocks (`ToolUseResultBlock`); all of them are shown, a `Json` one as its
/// JSON text, as the CLI itself sends them on. `status: "Error"` (a failed or
/// cancelled call) marks the result as an error.
fn push_tool_results(blocks: &mut Vec<Value>, holder: &Value) {
    let Some(results) = holder.get("tool_use_results").and_then(Value::as_array) else {
        return;
    };
    for tr in results {
        let tool_use_id = tr.get("tool_use_id").and_then(Value::as_str).unwrap_or("");
        let text = tr
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|block| match (block.get("Text"), block.get("Json")) {
                (Some(Value::String(text)), _) => Some(text.clone()),
                (_, Some(json)) => Some(json.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let mut block = json!({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": text
        });
        if tr.get("status").and_then(Value::as_str) == Some("Error") {
            block["is_error"] = Value::Bool(true);
        }
        blocks.push(block);
    }
}

pub(crate) fn convert_assistant_message(
    provider: &str,
    assistant: &Value,
    session_id: &str,
    idx: usize,
) -> Option<ClaudeMessage> {
    let mut blocks: Vec<Value> = Vec::new();

    if let Some(response) = assistant.get("Response") {
        let text = response
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !text.is_empty() {
            blocks.push(json!({"type": "text", "text": text}));
        }
    } else if let Some(tool_use) = assistant.get("ToolUse") {
        let text = tool_use
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !text.is_empty() {
            blocks.push(json!({"type": "text", "text": text}));
        }
        if let Some(tools) = tool_use.get("tool_uses").and_then(Value::as_array) {
            for tool in tools {
                let id = tool.get("id").and_then(Value::as_str).unwrap_or("");
                let name = tool
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let args = tool
                    .get("args")
                    .cloned()
                    .unwrap_or(Value::Object(serde_json::Map::default()));
                let (name, input) = canonical_tool(name, args);
                blocks.push(json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": input
                }));
            }
        }
    }

    if blocks.is_empty() {
        return None;
    }

    let msg_id = assistant
        .get("Response")
        .or_else(|| assistant.get("ToolUse"))
        .and_then(|v| v.get("message_id"))
        .and_then(Value::as_str)
        .unwrap_or("");

    Some(build_provider_message(
        provider,
        if msg_id.is_empty() {
            format!("{session_id}-asst-{idx}")
        } else {
            msg_id.to_string()
        },
        session_id,
        String::new(),
        "assistant",
        Some("assistant"),
        Some(Value::Array(blocks)),
        None,
    ))
}

/// A tool call as the viewer's card for the same primitive reads it: the
/// canonical name, with the input in that card's shape. A call keeps its own
/// name and arguments when no card takes the same input.
///
/// Native tools (`tools/mod.rs` `NATIVE_TOOLS`): `execute_bash`
/// (`execute_cmd` on Windows) `{command, summary?}`; `fs_read`
/// `{operations: [{mode, path, …}], summary?}` (older builds: one operation's
/// fields at the top level); `fs_write` `{command, path, …}`.
fn canonical_tool(name: &str, args: Value) -> (String, Value) {
    let path = |op: &Value| op.get("path").and_then(Value::as_str).map(String::from);
    match name {
        "execute_bash" | "execute_cmd" => ("Bash".into(), args),
        // One line-range read of one file is the Read card.
        "fs_read" => {
            let op = match args.get("operations").and_then(Value::as_array) {
                Some(ops) if ops.len() == 1 => &ops[0],
                Some(_) => return (name.into(), args),
                None => &args,
            };
            match (op.get("mode").and_then(Value::as_str), path(op)) {
                (Some("Line"), Some(_)) => ("Read".into(), with_read_range(op.clone())),
                _ => (name.into(), args),
            }
        }
        "fs_write" => {
            let Some(file) = path(&args) else {
                return (name.into(), args);
            };
            let text = |key: &str| args.get(key).and_then(Value::as_str);
            match args.get("command").and_then(Value::as_str) {
                // The CLI takes `new_str` when the model leaves out `file_text`.
                Some("create") => match text("file_text").or_else(|| text("new_str")) {
                    Some(content) => ("Write".into(), json!({"path": file, "content": content})),
                    None => (name.into(), args),
                },
                Some("str_replace") => match (text("old_str"), text("new_str")) {
                    (Some(old), Some(new)) => (
                        "Edit".into(),
                        json!({"path": file, "old_string": old, "new_string": new}),
                    ),
                    _ => (name.into(), args),
                },
                _ => (name.into(), args),
            }
        }
        "read_file" | "file_read" => ("Read".into(), args),
        "write_file" | "file_write" | "create_file" => ("Write".into(), args),
        "list_directory" => ("Glob".into(), args),
        "search_files" | "grep" => ("Grep".into(), args),
        "web_search" => ("WebSearch".into(), args),
        "web_fetch" => ("WebFetch".into(), args),
        _ => (name.into(), args),
    }
}

/// A `Line` read's 1-based `start_line`/`end_line` as the Read card's
/// `offset`/`limit`, when both are positive (`end_line: -1` means the end of
/// the file, which has no length to show).
fn with_read_range(mut op: Value) -> Value {
    let line = |key: &str| op.get(key).and_then(Value::as_i64).filter(|n| *n >= 1);
    let (start, end) = (line("start_line"), line("end_line"));
    if let Some(map) = op.as_object_mut() {
        if let Some(start) = start {
            map.insert("offset".into(), start.into());
            if let Some(end) = end.filter(|end| *end >= start) {
                map.insert("limit".into(), (end - start + 1).into());
            }
        }
    }
    op
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_user_prompt() {
        let user = json!({
            "content": {"Prompt": {"prompt": "hello world"}},
            "timestamp": "2025-10-08T10:50:49.220865-07:00"
        });
        let msg = convert_user_message("amazonq", &user, "sess-1", 0).unwrap();
        assert_eq!(msg.message_type, "user");
        assert_eq!(msg.provider.as_deref(), Some("amazonq"));
        assert_eq!(msg.uuid, "sess-1-user-0");
    }

    #[test]
    fn convert_assistant_response() {
        let asst = json!({"Response": {"message_id": "abc", "content": "Hello!"}});
        let msg = convert_assistant_message("kiro", &asst, "sess-1", 0).unwrap();
        assert_eq!(msg.message_type, "assistant");
        assert_eq!(msg.provider.as_deref(), Some("kiro"));
        assert_eq!(msg.uuid, "abc"); // message_id wins over the synthetic id
        let arr = msg.content.unwrap();
        assert_eq!(arr[0]["text"], "Hello!");
    }

    #[test]
    fn convert_assistant_tool_use_maps_name() {
        let asst = json!({
            "ToolUse": {
                "message_id": "xyz",
                "content": "Let me run that",
                "tool_uses": [{"id": "t1", "name": "execute_bash", "args": {"command": "ls"}}]
            }
        });
        let msg = convert_assistant_message("amazonq", &asst, "sess-1", 1).unwrap();
        let arr = msg.content.unwrap().as_array().unwrap().clone();
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[1]["type"], "tool_use");
        assert_eq!(arr[1]["name"], "Bash");
    }

    #[test]
    fn convert_user_tool_results() {
        let user = json!({
            "content": {"ToolUseResults": {"tool_use_results": [
                {"tool_use_id": "t1", "content": [{"Text": "output here"}]}
            ]}},
            "timestamp": "2025-10-08T10:51:00-07:00"
        });
        let msg = convert_user_message("amazonq", &user, "sess-1", 1).unwrap();
        let arr = msg.content.unwrap().as_array().unwrap().clone();
        assert_eq!(arr[0]["type"], "tool_result");
        assert_eq!(arr[0]["tool_use_id"], "t1");
        assert_eq!(arr[0]["content"], "output here");
    }

    #[test]
    fn convert_cancelled_tool_uses_prompt_and_results() {
        let user = json!({
            "content": {"CancelledToolUses": {
                "prompt": "stop",
                "tool_use_results": [{"tool_use_id": "t9", "content": [{"Text": "partial"}]}]
            }},
            "timestamp": ""
        });
        let msg = convert_user_message("amazonq", &user, "s", 0).unwrap();
        let arr = msg.content.unwrap().as_array().unwrap().clone();
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[0]["text"], "stop");
        assert_eq!(arr[1]["type"], "tool_result");
        assert_eq!(arr[1]["tool_use_id"], "t9");
    }

    #[test]
    fn parse_history_walks_user_and_assistant() {
        let value = json!({
            "history": [
                {
                    "user": {"content": {"Prompt": {"prompt": "hi"}}, "timestamp": "2025-10-08T10:00:00Z"},
                    "assistant": {"Response": {"message_id": "m1", "content": "hello"}}
                }
            ]
        })
        .to_string();
        let msgs = parse_history("amazonq", &value, "sess-1");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role.as_deref(), Some("user"));
        assert_eq!(msgs[1].role.as_deref(), Some("assistant"));
        assert_eq!(message_count(&value), 2);
        let (first, last) = history_time_bounds(&value);
        assert_eq!(first.as_deref(), Some("2025-10-08T10:00:00Z"));
        assert_eq!(last.as_deref(), Some("2025-10-08T10:00:00Z")); // single turn -> last = first
        assert_eq!(first_prompt_summary(&value, 80).as_deref(), Some("hi"));
    }

    /// A `history` entry pair as `serde_json::to_string(&ConversationState)`
    /// writes them (`message.rs`, `tools/fs_read.rs`, `tools/fs_write.rs`).
    fn tool_history() -> String {
        json!({"history": [
            {
                "user": {"content": {"Prompt": {"prompt": "fix it"}}, "timestamp": "2025-10-08T10:00:00Z"},
                "assistant": {"ToolUse": {"message_id": "m1", "content": "", "tool_uses": [
                    {"id": "r1", "name": "fs_read", "args": {"operations": [{"mode": "Line", "path": "/repo/a.rs", "start_line": 1, "end_line": 20}]}},
                    {"id": "r2", "name": "fs_read", "args": {"operations": [{"mode": "Directory", "path": "/repo"}]}},
                    {"id": "w1", "name": "fs_write", "args": {"command": "create", "path": "/repo/b.rs", "file_text": "fn main() {}"}},
                    {"id": "e1", "name": "fs_write", "args": {"command": "str_replace", "path": "/repo/a.rs", "old_str": "a", "new_str": "b"}},
                    {"id": "x1", "name": "use_aws", "args": {"service_name": "s3", "operation_name": "list-buckets"}}
                ]}}
            },
            {
                "user": {"content": {"ToolUseResults": {"tool_use_results": [
                    {"tool_use_id": "x1", "content": [{"Json": {"exit_status": "0", "stdout": "bucket-a"}}, {"Text": "done"}], "status": "Success"},
                    {"tool_use_id": "w1", "content": [{"Text": "permission denied"}], "status": "Error"}
                ]}}, "timestamp": "2025-10-08T10:01:00Z"}
            }
        ]})
        .to_string()
    }

    #[test]
    fn native_tools_reach_their_cards_and_keep_their_own_shape_otherwise() {
        let msgs = parse_history("amazonq", &tool_history(), "s");
        let calls = msgs[1].content.as_ref().unwrap().as_array().unwrap();
        let call = |id: &str| calls.iter().find(|c| c["id"] == id).unwrap().clone();

        let read = call("r1");
        assert_eq!(read["name"], "Read");
        assert_eq!(read["input"]["path"], "/repo/a.rs");
        // Lines 1–20 as the Read card's range.
        assert_eq!(read["input"]["offset"], 1);
        assert_eq!(read["input"]["limit"], 20);
        // A directory listing is not a file read.
        assert_eq!(call("r2")["name"], "fs_read");
        assert_eq!(
            call("w1")["input"],
            json!({"path": "/repo/b.rs", "content": "fn main() {}"})
        );
        assert_eq!(call("w1")["name"], "Write");
        assert_eq!(
            call("e1")["input"],
            json!({"path": "/repo/a.rs", "old_string": "a", "new_string": "b"})
        );
        assert_eq!(call("e1")["name"], "Edit");
        assert_eq!(call("x1")["name"], "use_aws");
    }

    #[test]
    fn tool_results_keep_every_block_and_their_status() {
        let msgs = parse_history("amazonq", &tool_history(), "s");
        let results = msgs[2].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(
            results[0]["content"],
            "{\"exit_status\":\"0\",\"stdout\":\"bucket-a\"}\ndone"
        );
        assert!(results[0].get("is_error").is_none());
        assert_eq!(results[1]["content"], "permission denied");
        assert_eq!(results[1]["is_error"], true);
        assert_eq!(session_flags(&msgs), (true, true));
    }
}
