//! Continue-family providers (Continue.dev and its fork `PearAI`).
//!
//! Both read chat sessions written by the Continue VS Code / `JetBrains`
//! extension (and CLI) under `<global-dir>/sessions/`. Each session is a single
//! JSON object in `<sessionId>.json`; `sessions.json` is a lightweight index
//! that we intentionally ignore (its on-disk shape is less stable than the
//! per-session files).
//!
//! Continue defaults to `~/.continue` (overridable via `CONTINUE_GLOBAL_DIR`);
//! `PearAI` rebrands the directory to `~/.pearai` (see [`super::pearai`]). The
//! shared logic is parameterized by `Family`.
//!
//! Session file shape (Continue `Session`):
//! ```json
//! {
//!   "sessionId": "uuid",
//!   "title": "Fix the login bug",
//!   "workspaceDirectory": "/Users/jack/client/my-project",
//!   "history": [
//!     { "message": { "role": "user", "content": "..." }, "contextItems": [] },
//!     { "message": { "role": "assistant", "content": [{ "type": "text", "text": "..." }] },
//!       "toolCallStates": [] }
//!   ]
//! }
//! ```
//! `content` is either a plain string or an array of message parts
//! (`{type:"text",text}` / `{type:"imageUrl",...}`); we pass it through to the
//! existing content renderers unchanged. Continue records no per-message
//! timestamp, so message times fall back to the session file's mtime.
//!
//! Projects are derived by grouping sessions on `workspaceDirectory` (the
//! virtual project path is `<scheme><workspaceDirectory>`), mirroring how the
//! Codex provider groups rollouts by `cwd`.

use super::ProviderInfo;
use crate::models::{ClaudeMessage, ClaudeProject, ClaudeSession};
use crate::utils::{build_provider_message, search_json_value_case_insensitive};
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Index file that sits beside the per-session JSON files; never a session.
const INDEX_FILE: &str = "sessions.json";

/// Max characters of the first user prompt used as a session title fallback.
const SUMMARY_MAX_CHARS: usize = 80;

/// `cwd` bucket for sessions with no `workspaceDirectory`.
const UNKNOWN_WORKSPACE: &str = "unknown";

/// Titles Continue gives a session before it is named: the extension's
/// `NEW_SESSION_TITLE` (`core/util/constants.ts`) and the CLI's
/// `DEFAULT_SESSION_TITLE` (`extensions/cli/src/constants/session.ts`).
const PLACEHOLDER_TITLES: [&str; 2] = ["New Session", "Untitled Session"];

/// Configuration distinguishing the members of the Continue family.
pub(crate) struct Family {
    /// Provider id stamped on projects/sessions/messages (e.g. `"continue"`).
    pub provider_id: &'static str,
    /// Human-facing name (e.g. `"Continue"`).
    pub display_name: &'static str,
    /// Home-relative global dir (e.g. `".continue"` / `".pearai"`).
    pub home_subdir: &'static str,
    /// Env var that overrides the global dir, if the tool honors one.
    pub global_dir_env: Option<&'static str>,
    /// Virtual project-path prefix (e.g. `"continue://"`).
    pub scheme: &'static str,
}

/// Continue.dev. Honors `CONTINUE_GLOBAL_DIR`.
pub(crate) const CONTINUE: Family = Family {
    provider_id: "continue",
    display_name: "Continue",
    home_subdir: ".continue",
    global_dir_env: Some("CONTINUE_GLOBAL_DIR"),
    scheme: "continue://",
};

// ============================================================================
// Continue public API (Continue-specific wrappers around the family core)
// ============================================================================

/// Detect a Continue installation.
pub fn detect() -> Option<ProviderInfo> {
    detect_for(&CONTINUE)
}

/// Base path for Continue sessions: `<global-dir>/sessions`.
pub fn get_base_path() -> Option<String> {
    base_path_for(&CONTINUE)
}

/// Scan Continue projects under the default sessions root.
pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    scan_projects_for(&CONTINUE)
}

/// [`scan_projects`] parameterized by the sessions root (for tests).
pub fn scan_projects_in(base: &Path) -> Result<Vec<ClaudeProject>, String> {
    scan_in_for(&CONTINUE, base)
}

/// Load the sessions belonging to one Continue project.
pub fn load_sessions(
    project_path: &str,
    exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    load_sessions_for(&CONTINUE, project_path, exclude_sidechain)
}

/// Load all messages from a single Continue session file.
pub fn load_messages(session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    load_messages_for(&CONTINUE, session_path)
}

/// Search across all Continue sessions.
pub fn search(query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    search_for(&CONTINUE, query, limit)
}

// ============================================================================
// Family core (shared by Continue and PearAI)
// ============================================================================

pub(crate) fn detect_for(f: &Family) -> Option<ProviderInfo> {
    let base = base_path_for(f)?;
    Some(ProviderInfo {
        id: f.provider_id.to_string(),
        display_name: f.display_name.to_string(),
        is_available: has_any_session(Path::new(&base)),
        base_path: base,
    })
}

/// The family's global directory (env override or `~/<home_subdir>`).
fn global_dir_for(f: &Family) -> Option<PathBuf> {
    if let Some(env) = f.global_dir_env {
        if let Ok(dir) = std::env::var(env) {
            let dir = dir.trim();
            if !dir.is_empty() {
                return Some(PathBuf::from(dir));
            }
        }
    }
    Some(crate::utils::home_dir()?.join(f.home_subdir))
}

/// Base path (`<global-dir>/sessions`); `None` unless the directory exists.
pub(crate) fn base_path_for(f: &Family) -> Option<String> {
    let sessions = global_dir_for(f)?.join("sessions");
    if sessions.is_dir() {
        Some(sessions.to_string_lossy().to_string())
    } else {
        None
    }
}

pub(crate) fn scan_projects_for(f: &Family) -> Result<Vec<ClaudeProject>, String> {
    let base =
        base_path_for(f).ok_or_else(|| format!("{} sessions path not found", f.display_name))?;
    scan_in_for(f, Path::new(&base))
}

// Returns Result for parity with the other providers' scan API (and the public
// Continue/PearAI wrappers); the body currently cannot fail.
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn scan_in_for(f: &Family, base: &Path) -> Result<Vec<ClaudeProject>, String> {
    struct Agg {
        session_count: usize,
        message_count: usize,
        last_modified: String,
    }
    let mut by_workspace: HashMap<String, Agg> = HashMap::new();

    for file in session_files(base) {
        let Ok(data) = fs::read_to_string(&file) else {
            continue;
        };
        let mtime = file_mtime_rfc3339(&file);
        let Some(doc) = parse_session_doc(&data, f.provider_id, &file_stem(&file), &mtime) else {
            continue;
        };
        if doc.messages.is_empty() {
            continue;
        }
        let workspace = workspace_key(&doc);

        let entry = by_workspace.entry(workspace).or_insert_with(|| Agg {
            session_count: 0,
            message_count: 0,
            last_modified: String::new(),
        });
        entry.session_count += 1;
        entry.message_count += doc.messages.len();
        if mtime > entry.last_modified {
            entry.last_modified = mtime;
        }
    }

    let mut projects: Vec<ClaudeProject> = by_workspace
        .into_iter()
        .map(|(workspace, agg)| ClaudeProject {
            name: project_name_for(&workspace),
            path: project_path_for(f, &workspace),
            actual_path: workspace,
            session_count: agg.session_count,
            message_count: agg.message_count,
            last_modified: agg.last_modified,
            git_info: None,
            provider: Some(f.provider_id.to_string()),
            storage_type: Some("json".to_string()),
            custom_directory_label: None,
        })
        .collect();

    projects.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(projects)
}

pub(crate) fn load_sessions_for(
    f: &Family,
    project_path: &str,
    _exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    let base =
        base_path_for(f).ok_or_else(|| format!("{} sessions path not found", f.display_name))?;
    let target = project_path.strip_prefix(f.scheme).unwrap_or(project_path);

    let mut sessions = Vec::new();
    for file in session_files(Path::new(&base)) {
        let Ok(data) = fs::read_to_string(&file) else {
            continue;
        };
        let mtime = file_mtime_rfc3339(&file);
        let Some(doc) = parse_session_doc(&data, f.provider_id, &file_stem(&file), &mtime) else {
            continue;
        };
        if doc.messages.is_empty() {
            continue;
        }
        let workspace = workspace_key(&doc);
        if workspace != target {
            continue;
        }

        let project_name = Path::new(&workspace)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let summary = doc
            .title
            .clone()
            .filter(|t| !t.trim().is_empty())
            .or_else(|| doc.first_user_text.clone().map(|t| summarize(&t)))
            .or_else(|| Some(doc.session_id.clone()));

        sessions.push(ClaudeSession {
            session_id: file.to_string_lossy().to_string(),
            actual_session_id: doc.session_id,
            file_path: file.to_string_lossy().to_string(),
            project_name,
            message_count: doc.messages.len(),
            first_message_time: mtime.clone(),
            last_message_time: mtime.clone(),
            last_modified: mtime,
            has_tool_use: doc.has_tool_use,
            has_errors: doc.has_errors,
            summary,
            is_renamed: doc.title.is_some(),
            provider: Some(f.provider_id.to_string()),
            storage_type: Some("json".to_string()),
            entrypoint: None,
        });
    }

    sessions.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(sessions)
}

pub(crate) fn load_messages_for(
    f: &Family,
    session_path: &str,
) -> Result<Vec<ClaudeMessage>, String> {
    let path = Path::new(session_path);
    if !path.exists() {
        return Err(format!("Session file not found: {session_path}"));
    }
    validate_under_base(f, path)?;
    if is_symlink(path) {
        return Err("Session file must not be a symlink".to_string());
    }

    let data = fs::read_to_string(path).map_err(|e| format!("Failed to read session file: {e}"))?;
    let timestamp = file_mtime_rfc3339(path);
    Ok(
        parse_session_doc(&data, f.provider_id, &file_stem(path), &timestamp)
            .map(|doc| doc.messages)
            .unwrap_or_default(),
    )
}

// Returns Result for parity with the other providers' search API; the body
// currently cannot fail (a missing base path yields an empty result set).
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn search_for(
    f: &Family,
    query: &str,
    limit: usize,
) -> Result<Vec<ClaudeMessage>, String> {
    let Some(base) = base_path_for(f) else {
        return Ok(vec![]);
    };
    Ok(search_in_for(f, Path::new(&base), query, limit))
}

/// [`search_for`] parameterized by the sessions root (for tests).
fn search_in_for(f: &Family, base: &Path, query: &str, limit: usize) -> Vec<ClaudeMessage> {
    let query_lower = query.to_lowercase();
    let mut results = Vec::new();

    for file in session_files(base) {
        let Ok(data) = fs::read_to_string(&file) else {
            continue;
        };
        let mtime = file_mtime_rfc3339(&file);
        let Some(doc) = parse_session_doc(&data, f.provider_id, &file_stem(&file), &mtime) else {
            continue;
        };
        // Same workspace bucket, name and path as `scan_in_for`.
        let workspace = workspace_key(&doc);
        let project_name = project_name_for(&workspace);
        let project_path = project_path_for(f, &workspace);

        // Reuse the exact messages (and their deterministic UUIDs) produced by
        // load_messages so global-search navigation resolves back correctly.
        for mut msg in doc.messages {
            let matched = msg
                .content
                .as_ref()
                .map(|c| search_json_value_case_insensitive(c, &query_lower))
                .unwrap_or(false);
            if !matched {
                continue;
            }
            msg.project_name = Some(project_name.clone());
            msg.project_path = Some(project_path.clone());
            results.push(msg);
            if results.len() >= limit {
                return results;
            }
        }
    }

    results
}

// ============================================================================
// Shared helpers
// ============================================================================

/// The workspace a session is grouped under (`unknown` when it records none).
fn workspace_key(doc: &SessionDoc) -> String {
    doc.workspace_directory
        .clone()
        .filter(|w| !w.is_empty())
        .unwrap_or_else(|| UNKNOWN_WORKSPACE.to_string())
}

/// Display name of a workspace project, shared by scan and search.
fn project_name_for(workspace: &str) -> String {
    Path::new(workspace)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| workspace.to_string())
}

/// Opaque project identity of a workspace, shared by scan and search.
fn project_path_for(f: &Family, workspace: &str) -> String {
    format!("{}{workspace}", f.scheme)
}

/// True if at least one `<sessionId>.json` (excluding the index) exists.
fn has_any_session(base: &Path) -> bool {
    !session_files(base).is_empty()
}

/// Immediate `*.json` session files under `base`, excluding `sessions.json`.
fn session_files(base: &Path) -> Vec<PathBuf> {
    WalkDir::new(base)
        .min_depth(1)
        .max_depth(1)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.path().to_path_buf())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .filter(|p| p.file_name().and_then(|n| n.to_str()) != Some(INDEX_FILE))
        .filter(|p| !is_symlink(p))
        .collect()
}

/// A parsed Continue session: metadata + normalized messages.
struct SessionDoc {
    session_id: String,
    title: Option<String>,
    workspace_directory: Option<String>,
    first_user_text: Option<String>,
    has_tool_use: bool,
    /// A tool call ended `errored` or `canceled`.
    has_errors: bool,
    messages: Vec<ClaudeMessage>,
}

/// Parse one session JSON document into normalized messages + metadata.
/// `fallback_session_id` (the file stem) is used when the JSON omits
/// `sessionId`; `timestamp` (the file mtime) is stamped on every message since
/// Continue records no per-message time. `provider_id` tags the messages.
fn parse_session_doc(
    data: &str,
    provider_id: &str,
    fallback_session_id: &str,
    timestamp: &str,
) -> Option<SessionDoc> {
    let root: Value = serde_json::from_str(data).ok()?;
    let obj = root.as_object()?;

    let session_id = obj
        .get("sessionId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or(fallback_session_id)
        .to_string();
    // Placeholders, not names: the extension's `NEW_SESSION_TITLE` and the
    // CLI's `DEFAULT_SESSION_TITLE`.
    let title = obj
        .get("title")
        .and_then(Value::as_str)
        .map(str::to_string)
        .filter(|t| !t.trim().is_empty() && !PLACEHOLDER_TITLES.contains(&t.trim()));
    let workspace_directory = obj
        .get("workspaceDirectory")
        .and_then(Value::as_str)
        .map(str::to_string);

    let history = obj.get("history").and_then(Value::as_array);

    let mut messages = Vec::new();
    let mut first_user_text: Option<String> = None;
    let mut has_tool_use = false;
    let mut has_errors = false;
    // Results already shown from a call's `toolCallStates`, so a separate
    // `role: "tool"` item for the same call is not shown twice.
    let mut shown_results: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut idx = 0u64;

    if let Some(history) = history {
        for item in history {
            let Some(message) = item.get("message") else {
                continue;
            };
            let Some(role) = message.get("role").and_then(Value::as_str) else {
                continue;
            };

            let (role, content) = match role {
                "user" | "system" => {
                    let content = message.get("content").cloned();
                    if first_user_text.is_none() && role == "user" {
                        first_user_text = content.as_ref().and_then(extract_text);
                    }
                    (role, content)
                }
                "assistant" => {
                    let blocks = assistant_blocks(item, message, &mut shown_results);
                    has_tool_use |= blocks.iter().any(|b| b["type"] == "tool_use");
                    has_errors |= blocks.iter().any(|b| b["is_error"] == true);
                    if blocks.is_empty() {
                        continue;
                    }
                    (role, Some(Value::Array(blocks)))
                }
                "thinking" => {
                    let Some(text) = message.get("content").and_then(extract_text) else {
                        continue;
                    };
                    (
                        "assistant",
                        Some(serde_json::json!([{"type": "thinking", "thinking": text}])),
                    )
                }
                "tool" => {
                    let id = message
                        .get("toolCallId")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if shown_results.contains(id) {
                        continue;
                    }
                    let text = message
                        .get("content")
                        .and_then(extract_text)
                        .unwrap_or_default();
                    (
                        "user",
                        Some(serde_json::json!([
                            {"type": "tool_result", "tool_use_id": id, "content": text}
                        ])),
                    )
                }
                _ => continue,
            };

            messages.push(build_provider_message(
                provider_id,
                format!("{session_id}-{idx}"),
                &session_id,
                timestamp.to_string(),
                role,
                Some(role),
                content,
                None,
            ));
            idx += 1;
        }
    }

    Some(SessionDoc {
        session_id,
        title,
        workspace_directory,
        first_user_text,
        has_tool_use,
        has_errors,
        messages,
    })
}

/// An assistant history item as content blocks: its reasoning, its text,
/// then each tool call with its result.
///
/// A call and its result come from the item's `toolCallStates`
/// (`ToolCallState {toolCallId, toolCall: {function: {name, arguments}},
/// status, parsedArgs, output?: ContextItem[]}`); a call the message lists in
/// `toolCalls` without a state is shown without a result. A call whose
/// status is `errored` or `canceled` is a failure.
fn assistant_blocks(
    item: &Value,
    message: &Value,
    shown_results: &mut std::collections::HashSet<String>,
) -> Vec<Value> {
    let mut blocks = Vec::new();
    if let Some(text) = item
        .get("reasoning")
        .and_then(|r| r.get("text"))
        .and_then(Value::as_str)
        .filter(|t| !t.trim().is_empty())
    {
        blocks.push(serde_json::json!({"type": "thinking", "thinking": text}));
    }
    match message.get("content") {
        Some(Value::String(text)) if !text.trim().is_empty() => {
            blocks.push(serde_json::json!({"type": "text", "text": text}));
        }
        Some(Value::Array(parts)) => blocks.extend(parts.iter().cloned()),
        _ => {}
    }

    let states = item
        .get("toolCallStates")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let listed = message
        .get("toolCalls")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let state_ids: std::collections::HashSet<&str> = states
        .iter()
        .filter_map(|s| s.get("toolCallId").and_then(Value::as_str))
        .collect();

    for state in states {
        let call = state.get("toolCall").unwrap_or(&Value::Null);
        let id = state
            .get("toolCallId")
            .or_else(|| call.get("id"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let input = state
            .get("parsedArgs")
            .filter(|args| args.is_object())
            .cloned()
            .unwrap_or_else(|| call_arguments(call));
        blocks.push(serde_json::json!({
            "type": "tool_use",
            "id": id,
            "name": call_name(call),
            "input": input,
        }));
        if let Some(output) = state.get("output").and_then(Value::as_array) {
            let text = output
                .iter()
                .filter_map(|ctx| ctx.get("content").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n");
            let mut result = serde_json::json!({
                "type": "tool_result",
                "tool_use_id": id,
                "content": text,
            });
            if matches!(
                state.get("status").and_then(Value::as_str),
                Some("errored" | "canceled")
            ) {
                result["is_error"] = Value::Bool(true);
            }
            blocks.push(result);
            shown_results.insert(id.to_string());
        }
    }
    for call in listed {
        let id = call.get("id").and_then(Value::as_str).unwrap_or_default();
        if state_ids.contains(id) {
            continue;
        }
        blocks.push(serde_json::json!({
            "type": "tool_use",
            "id": id,
            "name": call_name(call),
            "input": call_arguments(call),
        }));
    }
    blocks
}

fn call_name(call: &Value) -> &str {
    call.get("function")
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("tool")
}

/// `function.arguments`, a JSON string, as an object (`{}` when it does not
/// parse to one).
fn call_arguments(call: &Value) -> Value {
    call.get("function")
        .and_then(|f| f.get("arguments"))
        .and_then(Value::as_str)
        .and_then(|args| serde_json::from_str::<Value>(args).ok())
        .filter(Value::is_object)
        .unwrap_or_else(|| serde_json::json!({}))
}

/// Concatenate the text of a `content` value (plain string, or an array of
/// `{type:"text",text}` parts).
fn extract_text(content: &Value) -> Option<String> {
    match content {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if !out.is_empty() {
                        out.push(' ');
                    }
                    out.push_str(text);
                }
            }
            if out.is_empty() {
                None
            } else {
                Some(out)
            }
        }
        _ => None,
    }
}

/// Build a short session title from the first user prompt.
fn summarize(text: &str) -> String {
    let cleaned = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.chars().count() > SUMMARY_MAX_CHARS {
        let truncated: String = cleaned.chars().take(SUMMARY_MAX_CHARS).collect();
        format!("{truncated}…")
    } else {
        cleaned
    }
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn is_symlink(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
}

/// Confine `path` to the family's sessions root (defense-in-depth against
/// traversal / symlink escapes). Canonicalizes both sides.
fn validate_under_base(f: &Family, path: &Path) -> Result<(), String> {
    let base =
        base_path_for(f).ok_or_else(|| format!("{} sessions path not found", f.display_name))?;
    let canon_base = Path::new(&base)
        .canonicalize()
        .map_err(|e| format!("Failed to resolve {} base: {e}", f.display_name))?;
    let canon_path = path
        .canonicalize()
        .map_err(|e| format!("Failed to resolve session path: {e}"))?;
    if canon_path.starts_with(&canon_base) {
        Ok(())
    } else {
        Err(format!(
            "Path is outside the {} sessions root: {}",
            f.display_name,
            path.display()
        ))
    }
}

fn file_mtime_rfc3339(path: &Path) -> String {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::SystemTime::UNIX_EPOCH).ok())
        .map(|d| ts_to_rfc3339(d.as_secs()))
        .unwrap_or_default()
}

#[allow(clippy::cast_possible_wrap)]
fn ts_to_rfc3339(secs: u64) -> String {
    if secs == 0 {
        return Utc::now().to_rfc3339();
    }
    DateTime::from_timestamp(secs as i64, 0)
        .unwrap_or_else(Utc::now)
        .to_rfc3339()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const SESSION_A: &str = r#"{
        "sessionId": "sess-a",
        "title": "Fix the login bug",
        "workspaceDirectory": "/Users/jack/client/my-project",
        "history": [
            { "message": { "role": "user", "content": "why does LOGIN fail?" }, "contextItems": [] },
            { "message": { "role": "assistant", "content": [{ "type": "text", "text": "Checking ZmagicToken" }] }, "toolCallStates": [{ "toolCallId": "t1" }] }
        ]
    }"#;

    // No title -> summary derived from first user prompt. Same workspace as A.
    const SESSION_B: &str = r#"{
        "sessionId": "sess-b",
        "workspaceDirectory": "/Users/jack/client/my-project",
        "history": [
            { "message": { "role": "system", "content": "system prompt" } },
            { "message": { "role": "user", "content": "second session question" } }
        ]
    }"#;

    // Different workspace.
    const SESSION_C: &str = r#"{
        "sessionId": "sess-c",
        "workspaceDirectory": "/Users/jack/client/other",
        "history": [
            { "message": { "role": "user", "content": "other project" } }
        ]
    }"#;

    fn write_session(base: &Path, name: &str, body: &str) {
        fs::write(base.join(name), body).unwrap();
    }

    #[test]
    fn scan_groups_sessions_by_workspace() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        write_session(base, "sess-a.json", SESSION_A);
        write_session(base, "sess-b.json", SESSION_B);
        write_session(base, "sess-c.json", SESSION_C);
        // The index file must never be treated as a session.
        write_session(base, "sessions.json", r#"[{"sessionId":"sess-a"}]"#);

        let projects = scan_projects_in(base).unwrap();
        assert_eq!(projects.len(), 2, "two distinct workspaces");

        let my = projects
            .iter()
            .find(|p| p.name == "my-project")
            .expect("my-project grouped");
        assert_eq!(my.session_count, 2);
        assert_eq!(my.provider.as_deref(), Some("continue"));
        assert_eq!(my.storage_type.as_deref(), Some("json"));
        assert_eq!(my.path, "continue:///Users/jack/client/my-project");
        assert_eq!(my.actual_path, "/Users/jack/client/my-project");
    }

    #[test]
    fn pearai_family_tags_provider_and_scheme() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        write_session(base, "sess-a.json", SESSION_A);

        let projects = scan_in_for(&super::super::pearai::PEARAI, base).unwrap();
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].provider.as_deref(), Some("pearai"));
        assert_eq!(projects[0].path, "pearai:///Users/jack/client/my-project");
    }

    #[test]
    fn parse_workspace_filtering_and_titles() {
        let mut docs: Vec<SessionDoc> = [SESSION_A, SESSION_B, SESSION_C]
            .iter()
            .filter_map(|d| parse_session_doc(d, "continue", "fallback", "2026-06-21T00:00:00Z"))
            .collect();
        docs.retain(|d| d.workspace_directory.as_deref() == Some("/Users/jack/client/my-project"));
        assert_eq!(docs.len(), 2);

        let a = docs.iter().find(|d| d.session_id == "sess-a").unwrap();
        assert!(a.has_tool_use, "toolCallStates marks tool use");
        assert_eq!(a.title.as_deref(), Some("Fix the login bug"));

        let b = docs.iter().find(|d| d.session_id == "sess-b").unwrap();
        assert!(b.title.is_none());
        assert_eq!(
            b.first_user_text.as_deref(),
            Some("second session question")
        );
    }

    #[test]
    fn parse_maps_roles_with_deterministic_uuids_and_passthrough_content() {
        let doc =
            parse_session_doc(SESSION_A, "continue", "fallback", "2026-06-21T00:00:00Z").unwrap();
        assert_eq!(doc.session_id, "sess-a");
        assert_eq!(doc.messages.len(), 2);

        assert_eq!(doc.messages[0].role.as_deref(), Some("user"));
        assert_eq!(doc.messages[0].message_type, "user");
        assert_eq!(doc.messages[0].uuid, "sess-a-0");
        assert_eq!(doc.messages[0].provider.as_deref(), Some("continue"));

        assert_eq!(doc.messages[1].role.as_deref(), Some("assistant"));
        assert_eq!(doc.messages[1].uuid, "sess-a-1");
        let content = doc.messages[1].content.as_ref().unwrap();
        assert!(content.to_string().contains("ZmagicToken"));
    }

    #[test]
    fn parse_honors_given_provider_id() {
        let doc = parse_session_doc(SESSION_A, "pearai", "fallback", "").unwrap();
        assert_eq!(doc.messages[0].provider.as_deref(), Some("pearai"));
    }

    #[test]
    fn parse_includes_system_and_uses_fallback_session_id() {
        let doc = parse_session_doc(SESSION_B, "continue", "file-stem-id", "").unwrap();
        assert_eq!(doc.session_id, "sess-b");
        assert_eq!(doc.messages.len(), 2);
        assert_eq!(doc.messages[0].role.as_deref(), Some("system"));

        let no_id = r#"{ "history": [ { "message": { "role": "user", "content": "hi" } } ] }"#;
        let doc = parse_session_doc(no_id, "continue", "file-stem-id", "").unwrap();
        assert_eq!(doc.session_id, "file-stem-id");
        assert_eq!(doc.messages[0].uuid, "file-stem-id-0");
    }

    #[test]
    fn parse_skips_invalid_and_non_chat_roles() {
        let data = r#"{
            "sessionId": "x",
            "history": [
                { "message": { "role": "user", "content": "ok" } },
                { "message": { "role": "function", "content": "ignored" } },
                { "contextItems": [] },
                { "message": { "content": "no role" } }
            ]
        }"#;
        let doc = parse_session_doc(data, "continue", "x", "").unwrap();
        assert_eq!(doc.messages.len(), 1);
        assert_eq!(doc.messages[0].role.as_deref(), Some("user"));
    }

    #[test]
    fn parse_rejects_non_object_json() {
        assert!(parse_session_doc("[]", "continue", "x", "").is_none());
        assert!(parse_session_doc("not json", "continue", "x", "").is_none());
    }

    #[test]
    fn session_files_excludes_index_and_non_json() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        write_session(base, "sess-a.json", SESSION_A);
        write_session(base, "sessions.json", "[]");
        write_session(base, "notes.txt", "ignore me");

        let files = session_files(base);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].file_name().unwrap().to_str(), Some("sess-a.json"));
    }

    #[test]
    fn extract_text_handles_string_and_parts() {
        assert_eq!(
            extract_text(&Value::String("hello".into())).as_deref(),
            Some("hello")
        );
        let parts = serde_json::json!([
            { "type": "text", "text": "a" },
            { "type": "imageUrl", "imageUrl": { "url": "x" } },
            { "type": "text", "text": "b" }
        ]);
        assert_eq!(extract_text(&parts).as_deref(), Some("a b"));
        assert!(extract_text(&Value::Null).is_none());
    }

    #[test]
    fn summarize_truncates_long_prompts() {
        assert_eq!(summarize("  fix   the bug  "), "fix the bug");
        let long = "x".repeat(200);
        let s = summarize(&long);
        assert!(s.chars().count() <= SUMMARY_MAX_CHARS + 1);
        assert!(s.ends_with('…'));
    }

    #[test]
    fn search_hits_carry_scanned_project_path_for_same_named_projects() {
        for family in [&CONTINUE, &super::super::pearai::PEARAI] {
            let tmp = TempDir::new().unwrap();
            let base = tmp.path();
            for (id, workspace) in [
                ("sess-a", "/work/client-a/app"),
                ("sess-b", "/work/client-b/app"),
            ] {
                let body = serde_json::json!({
                    "sessionId": id,
                    "workspaceDirectory": workspace,
                    "history": [{ "message": { "role": "user", "content": "shared needle" } }]
                });
                write_session(base, &format!("{id}.json"), &body.to_string());
            }

            let projects = scan_in_for(family, base).unwrap();
            assert_eq!(projects.len(), 2);
            assert!(projects.iter().all(|p| p.name == "app"));
            assert_ne!(projects[0].path, projects[1].path);

            let hits = search_in_for(family, base, "needle", 10);
            assert_eq!(hits.len(), 2);
            for hit in &hits {
                let workspace = if hit.session_id == "sess-a" {
                    "/work/client-a/app"
                } else {
                    "/work/client-b/app"
                };
                let project = projects
                    .iter()
                    .find(|p| p.actual_path == workspace)
                    .expect("hit session belongs to a scanned project");
                assert_eq!(hit.project_path.as_deref(), Some(project.path.as_str()));
                assert_eq!(hit.project_name.as_deref(), Some("app"));
            }
        }
    }

    /// A session the Continue CLI 1.5.47 wrote (`cn -p … --auto` against a
    /// scripted model): Bash `seq 1 5`, Bash exiting 3, Read of a missing
    /// file, then List and Search in one turn. Only the workspace path is
    /// rewritten.
    const CLI_SESSION: &str = include_str!("../../tests/fixtures/continue-cli-1.5.47-tools.json");

    #[test]
    fn recorded_tool_calls_show_with_their_results_and_failures() {
        let doc = parse_session_doc(CLI_SESSION, "continue", "x", "").unwrap();
        // The prompt and five replies; none is empty.
        assert_eq!(doc.messages.len(), 6);
        assert!(doc.has_tool_use);
        assert!(doc.has_errors);

        let blocks = |i: usize| doc.messages[i].content.clone().unwrap();
        let seq = blocks(1);
        assert_eq!(seq[0]["type"], "tool_use");
        assert_eq!(seq[0]["name"], "Bash");
        assert_eq!(seq[0]["input"], serde_json::json!({"command": "seq 1 5"}));
        assert_eq!(seq[1]["type"], "tool_result");
        assert_eq!(seq[1]["tool_use_id"], seq[0]["id"]);
        assert_eq!(seq[1]["content"], "1\n2\n3\n4\n5\n");
        assert!(seq[1].get("is_error").is_none());

        let failed = blocks(2);
        assert_eq!(failed[1]["is_error"], true);
        assert!(failed[1]["content"]
            .as_str()
            .unwrap()
            .contains("exit code 3"));

        let two = blocks(4);
        let kinds: Vec<&str> = two
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["type"].as_str().unwrap())
            .collect();
        assert_eq!(
            kinds,
            ["tool_use", "tool_result", "tool_use", "tool_result"]
        );
        assert_eq!(blocks(5)[0]["text"], "All steps done.");

        // "Untitled Session" is the CLI's placeholder: the prompt titles it.
        assert_eq!(doc.title, None);
        assert_eq!(
            doc.first_user_text.as_deref(),
            Some("RUN-T please run the steps")
        );
    }

    #[test]
    fn reasoning_thinking_items_and_tool_items_are_shown_once() {
        let data = serde_json::json!({
            "sessionId": "s",
            "title": "New Session",
            "history": [
                {"message": {"role": "user", "content": "go"}},
                {"message": {"role": "thinking", "content": "Plan first."}},
                {"message": {"role": "assistant", "content": "",
                    "toolCalls": [{"id": "c1", "type": "function",
                        "function": {"name": "Read", "arguments": "{\"filepath\":\"a\"}"}}]},
                 "reasoning": {"active": false, "text": "Need the file."}},
                {"message": {"role": "tool", "toolCallId": "c1", "content": "file a"}},
                {"message": {"role": "assistant", "content": "ok"},
                 "toolCallStates": [{"toolCallId": "c2",
                    "toolCall": {"id": "c2", "function": {"name": "Bash", "arguments": "{}"}},
                    "status": "canceled", "parsedArgs": {"command": "rm -rf /"},
                    "output": [{"content": "Canceled by the user"}]}]},
                {"message": {"role": "tool", "toolCallId": "c2", "content": "Canceled by the user"}}
            ]
        })
        .to_string();
        let doc = parse_session_doc(&data, "continue", "s", "").unwrap();
        let content: Vec<Value> = doc
            .messages
            .iter()
            .map(|m| m.content.clone().unwrap())
            .collect();

        assert_eq!(content[1][0]["thinking"], "Plan first.");
        assert_eq!(content[2][0]["thinking"], "Need the file.");
        // A call listed on the message, with no state: no result there...
        assert_eq!(content[2][1]["input"], serde_json::json!({"filepath": "a"}));
        // ...so the tool item carries it.
        assert_eq!(content[3][0]["type"], "tool_result");
        assert_eq!(content[3][0]["tool_use_id"], "c1");
        // A canceled call is a failure, and its tool item is not repeated.
        assert_eq!(content[4][2]["is_error"], true);
        assert_eq!(doc.messages.len(), 5);
        assert_eq!(doc.title, None, "\"New Session\" is a placeholder");
    }
}
