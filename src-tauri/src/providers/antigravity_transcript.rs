//! Step lines of an Antigravity transcript (`brain/<id>/.system_generated/
//! logs/transcript_full.jsonl`), shared by the desktop app and the CLI, which
//! write the same records.
//!
//! Every line has `step_index`, `source`, `type`, `status` and `created_at`
//! (Heeler's live census of `agy` 1.5.3/1.5.9, `docs/research/
//! antigravity-cli-transcript-format.md`, "Records and turns"):
//!
//! - `USER_INPUT`: `content` is `<USER_REQUEST>…</USER_REQUEST>` followed by
//!   `<ADDITIONAL_METADATA>` (and `<USER_SETTINGS_CHANGE>` on the first
//!   message); the prompt is the text inside `<USER_REQUEST>`.
//! - `PLANNER_RESPONSE`: `thinking`, `content` and `tool_calls: [{name,
//!   args}]` in any mix. `args` is a JSON value in the full transcript, a
//!   JSON-encoded string in the compact one. Calls carry no id.
//! - `GENERIC` from `MODEL`: a tool result. A planner response with K calls
//!   is followed by K results, the kth for the kth call. `content` opens with
//!   `Created At:` / `Completed At:` lines. `status` is `DONE`, `RUNNING` (sent
//!   to the background; its end arrives later as a system message) or `ERROR`
//!   (with `error`). Older builds wrote a result under another type, such as
//!   `SEARCH_WEB`.
//! - `SYSTEM_MESSAGE`, `CHECKPOINT`, `ERROR_MESSAGE` and `CONVERSATION_HISTORY`
//!   are not turns of the conversation.

use crate::models::ClaudeMessage;
use crate::utils::build_provider_message;
use serde_json::{json, Value};
use std::collections::VecDeque;

/// Reads a transcript's step lines in order, pairing each tool result with
/// the call it answers.
pub(crate) struct StepReader<'a> {
    provider: &'a str,
    session_id: &'a str,
    /// Ids of the latest planner response's calls not yet answered.
    pending: VecDeque<String>,
}

impl<'a> StepReader<'a> {
    pub(crate) fn new(provider: &'a str, session_id: &'a str) -> Self {
        Self {
            provider,
            session_id,
            pending: VecDeque::new(),
        }
    }

    /// The message one step record shows, if any. `step_index` falls back to
    /// `line_idx`.
    pub(crate) fn step(
        &mut self,
        rec: &Value,
        line_idx: u64,
        timestamp: &str,
    ) -> Option<ClaudeMessage> {
        let source = rec.get("source").and_then(Value::as_str).unwrap_or("");
        let kind = rec.get("type").and_then(Value::as_str).unwrap_or("");
        let step_index = rec
            .get("step_index")
            .and_then(Value::as_u64)
            .unwrap_or(line_idx);
        let uuid = format!("{}-step-{step_index}", self.session_id);
        let text = |key: &str| {
            rec.get(key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
        };

        let (role, blocks) = if kind == "USER_INPUT" || source == "USER_EXPLICIT" {
            self.pending.clear();
            let prompt = user_request(text("content")?);
            if prompt.is_empty() {
                return None;
            }
            ("user", vec![json!({"type": "text", "text": prompt})])
        } else if kind == "PLANNER_RESPONSE" || (source == "MODEL" && kind.is_empty()) {
            self.pending.clear();
            let mut blocks = Vec::new();
            if let Some(thinking) = text("thinking") {
                blocks.push(json!({"type": "thinking", "thinking": thinking}));
            }
            if let Some(content) = text("content") {
                blocks.push(json!({"type": "text", "text": content}));
            }
            let calls = rec.get("tool_calls").and_then(Value::as_array);
            for (k, call) in calls.into_iter().flatten().enumerate() {
                let id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .map_or_else(|| format!("{uuid}-tool-{k}"), ToOwned::to_owned);
                blocks.push(json!({
                    "type": "tool_use",
                    "id": id,
                    "name": call.get("name").and_then(Value::as_str).unwrap_or("tool"),
                    "input": call_args(call.get("args")),
                }));
                self.pending.push_back(id);
            }
            if blocks.is_empty() {
                return None;
            }
            ("assistant", blocks)
        } else if source == "MODEL" {
            // A tool result: the oldest unanswered call's.
            let id = self.pending.pop_front().unwrap_or_else(|| uuid.clone());
            let status = rec.get("status").and_then(Value::as_str).unwrap_or("");
            let body = text("content").map(result_body).unwrap_or_default();
            let (content, is_error) = match status {
                "ERROR" => (text("error").map(str::to_string).unwrap_or(body), true),
                // The rest of a background start is instructions to the
                // model; the first line names the task.
                "RUNNING" => (body.lines().next().unwrap_or_default().to_string(), false),
                // A command's own failure: `Command exited with code N.`
                _ => {
                    let failed = body
                        .strip_prefix("Command exited with code ")
                        .and_then(|rest| rest.split('.').next())
                        .is_some_and(|code| code.trim() != "0");
                    (body, failed)
                }
            };
            let mut result = json!({
                "type": "tool_result",
                "tool_use_id": id,
                "content": content,
            });
            if is_error {
                result["is_error"] = Value::Bool(true);
            }
            ("user", vec![result])
        } else {
            return None;
        };

        Some(build_provider_message(
            self.provider,
            uuid,
            self.session_id,
            timestamp.to_string(),
            role,
            Some(role),
            Some(Value::Array(blocks)),
            None,
        ))
    }
}

/// The prompt inside `<USER_REQUEST>…</USER_REQUEST>`, without the metadata
/// blocks after it; the whole text when there is no such block.
pub(crate) fn user_request(content: &str) -> &str {
    content
        .split_once("<USER_REQUEST>")
        .and_then(|(_, rest)| rest.split_once("</USER_REQUEST>"))
        .map_or(content, |(inner, _)| inner)
        .trim()
}

/// A result's body without its `Created At:` / `Completed At:` lines.
fn result_body(content: &str) -> String {
    content
        .lines()
        .skip_while(|line| line.starts_with("Created At:") || line.starts_with("Completed At:"))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// A call's `args`: the value itself, or the object a JSON-encoded string
/// holds (the compact transcript); other text is kept as `{input}`.
fn call_args(args: Option<&Value>) -> Value {
    match args {
        Some(Value::String(text)) => match serde_json::from_str(text) {
            Ok(Value::Object(map)) => Value::Object(map),
            _ => json!({ "input": text }),
        },
        Some(value) if !value.is_null() => value.clone(),
        _ => json!({}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lines in the shapes of Heeler's `agy` census.
    fn turn() -> Vec<Value> {
        [
            r#"{"step_index":0,"source":"USER_EXPLICIT","type":"USER_INPUT","status":"DONE","created_at":"2026-11-07T15:53:30Z","content":"<USER_REQUEST>\nlist and read\n</USER_REQUEST>\n<ADDITIONAL_METADATA>\nThe current local time is: 2026-11-07T16:53:30+01:00.\n</ADDITIONAL_METADATA>\n<USER_SETTINGS_CHANGE>\nmodel: gemini\n</USER_SETTINGS_CHANGE>"}"#,
            r#"{"step_index":1,"source":"MODEL","type":"PLANNER_RESPONSE","status":"DONE","created_at":"2026-11-07T15:53:32Z","thinking":"Two calls.","tool_calls":[{"name":"run_command","args":{"CommandLine":"ls nope"}},{"name":"view_file","args":"{\"AbsolutePath\":\"/w/a.rs\"}"}]}"#,
            r#"{"step_index":2,"source":"MODEL","type":"GENERIC","status":"DONE","created_at":"2026-11-07T15:53:33Z","content":"Created At: 2026-11-07T16:53:33+01:00\nCompleted At: 2026-11-07T16:53:33+01:00\nCommand exited with code 1.\nCommand output:\nls: nope: No such file"}"#,
            r#"{"step_index":3,"source":"MODEL","type":"GENERIC","status":"ERROR","created_at":"2026-11-07T15:53:34Z","content":"Created At: 2026-11-07T16:53:34+01:00\nEncountered error in step execution: permission check failed","error":"permission check failed for view_file: user denied permission"}"#,
            r#"{"step_index":4,"source":"SYSTEM","type":"SYSTEM_MESSAGE","status":"DONE","created_at":"2026-11-07T15:53:35Z","content":"The following is a <SYSTEM_MESSAGE> not actually sent by the user."}"#,
            r#"{"step_index":5,"source":"MODEL","type":"PLANNER_RESPONSE","status":"DONE","created_at":"2026-11-07T15:53:36Z","content":"Done."}"#,
        ]
        .iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
    }

    #[test]
    fn results_pair_with_their_calls_by_position() {
        let mut reader = StepReader::new("antigravity", "s");
        let messages: Vec<ClaudeMessage> = turn()
            .iter()
            .enumerate()
            .filter_map(|(i, rec)| reader.step(rec, i as u64, ""))
            .collect();
        let content = |i: usize| messages[i].content.clone().unwrap();

        assert_eq!(messages.len(), 5, "the system message is not a turn");
        assert_eq!(content(0)[0]["text"], "list and read");

        let calls = content(1);
        assert_eq!(calls[0]["thinking"], "Two calls.");
        assert_eq!(calls[1]["input"], json!({"CommandLine": "ls nope"}));
        assert_eq!(calls[2]["input"], json!({"AbsolutePath": "/w/a.rs"}));

        let first = content(2);
        assert_eq!(first[0]["tool_use_id"], calls[1]["id"]);
        assert_eq!(
            first[0]["content"],
            "Command exited with code 1.\nCommand output:\nls: nope: No such file"
        );
        assert_eq!(first[0]["is_error"], true, "the command exited 1");
        let second = content(3);
        assert_eq!(second[0]["tool_use_id"], calls[2]["id"]);
        assert_eq!(second[0]["is_error"], true);
        assert_eq!(
            second[0]["content"],
            "permission check failed for view_file: user denied permission"
        );
        assert_eq!(content(4)[0]["text"], "Done.");
    }

    #[test]
    fn a_background_start_shows_its_task_line() {
        let mut reader = StepReader::new("antigravity", "s");
        let planner: Value = serde_json::from_str(
            r#"{"step_index":1,"source":"MODEL","type":"PLANNER_RESPONSE","tool_calls":[{"name":"run_command","args":{}}]}"#,
        )
        .unwrap();
        let running: Value = serde_json::from_str(
            r#"{"step_index":2,"source":"MODEL","type":"GENERIC","status":"RUNNING","content":"Created At: x\nTool is running as a background task with task id: c/task-2\nYOU MUST TAKE ONE OF THE FOLLOWING TWO ACTIONS"}"#,
        )
        .unwrap();
        reader.step(&planner, 1, "");
        let result = reader.step(&running, 2, "").unwrap();
        assert_eq!(
            result.content.unwrap()[0]["content"],
            "Tool is running as a background task with task id: c/task-2"
        );
    }
}
