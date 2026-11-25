//! Goose provider (Block's open-source on-machine agent).
//!
//! Reads the single `SQLite` store Goose has used since ~v1.10.0:
//! `<data-dir>/sessions/sessions.db`, with a `sessions` table (one row per
//! session, grouped here by `working_dir`) and a `messages` table (one row per
//! message; `content_json` holds a JSON array of Goose `MessageContent` items).
//!
//! Path resolution mirrors Goose's `etcetera` strategy with a tolerant fallback
//! list (the macOS location is reported inconsistently upstream, so we probe
//! both the XDG and Apple locations). `GOOSE_PATH_ROOT` overrides the root.
//!
//! `content_json` items are internally tagged (`{"type":"text",...}` /
//! `{"type":"toolRequest",...}` / `{"type":"toolResponse",...}` /
//! `{"type":"thinking",...}`); we map them to the viewer's Claude-style content
//! blocks so the existing renderers light up.

use crate::models::{ClaudeMessage, ClaudeProject, ClaudeSession};
use crate::providers::ProviderInfo;
use crate::utils::{build_provider_message, ms_to_iso, search_json_value_case_insensitive};
use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Value};
use std::path::PathBuf;

const PROVIDER: &str = "goose";
const SCHEME: &str = "goose://";

/// Candidate `sessions.db` locations, most-specific first.
fn candidate_db_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // GOOSE_PATH_ROOT override → <root>/data/sessions/sessions.db
    if let Ok(root) = std::env::var("GOOSE_PATH_ROOT") {
        let root = root.trim();
        if !root.is_empty() {
            paths.push(PathBuf::from(root).join("data/sessions/sessions.db"));
        }
    }
    // $XDG_DATA_HOME/goose/sessions/sessions.db
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        let xdg = xdg.trim();
        if !xdg.is_empty() {
            paths.push(PathBuf::from(xdg).join("goose/sessions/sessions.db"));
        }
    }
    if let Some(home) = crate::utils::home_dir() {
        // XDG default (Linux, and macOS under Goose's etcetera strategy).
        paths.push(home.join(".local/share/goose/sessions/sessions.db"));
        // macOS Apple-strategy fallback.
        #[cfg(target_os = "macos")]
        paths.push(home.join("Library/Application Support/Block/goose/sessions/sessions.db"));
    }
    // Windows: %APPDATA%\Block\goose\data\sessions\sessions.db
    #[cfg(target_os = "windows")]
    if let Some(data) = dirs::data_dir() {
        paths.push(data.join("Block/goose/data/sessions/sessions.db"));
    }

    paths
}

fn get_db_path() -> Option<PathBuf> {
    candidate_db_paths().into_iter().find(|p| p.is_file())
}

/// Detect a Goose installation (only when the sessions DB exists).
pub fn detect() -> Option<ProviderInfo> {
    let db = get_db_path()?;
    Some(ProviderInfo {
        id: PROVIDER.to_string(),
        display_name: "Goose".to_string(),
        base_path: db
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default(),
        is_available: true,
    })
}

/// Base path (the `sessions` dir holding `sessions.db`), for the file watcher.
pub fn get_base_path() -> Option<String> {
    Some(get_db_path()?.parent()?.to_string_lossy().to_string())
}

fn open_db() -> Result<Connection, String> {
    let path = get_db_path().ok_or("Goose sessions DB not found")?;
    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("Failed to open Goose DB: {e}"))?;
    conn.busy_timeout(std::time::Duration::from_secs(5))
        .map_err(|e| format!("Failed to set busy timeout: {e}"))?;
    Ok(conn)
}

/// Scan Goose projects (grouped by `working_dir`).
pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    scan_in_conn(&open_db()?)
}

fn scan_in_conn(conn: &Connection) -> Result<Vec<ClaudeProject>, String> {
    let schema = Schema::probe(conn);
    // Counts cover only what the viewer lists: sub-agent/hidden sessions and
    // agent-only messages are excluded (see `Schema`).
    let mut stmt = conn
        .prepare(&format!(
            "SELECT COALESCE(NULLIF(s.working_dir, ''), 'unknown') AS working_dir, \
                    COUNT(DISTINCT s.id) AS sess_cnt, \
                    COUNT(m.id) AS msg_cnt, MAX(s.updated_at) AS last_upd \
             FROM sessions s LEFT JOIN messages m ON m.session_id = s.id AND {} \
             WHERE {} \
             GROUP BY COALESCE(NULLIF(s.working_dir, ''), 'unknown') ORDER BY last_upd DESC",
            schema.visible_message_sql("m"),
            schema.listed_session_sql("s"),
        ))
        .map_err(|e| e.to_string())?;

    let projects = stmt
        .query_map([], |row| {
            let working_dir: Option<String> = row.get(0)?;
            let sess_cnt: usize = crate::providers::sql_unsigned(row, 1)?;
            let msg_cnt: usize = crate::providers::sql_unsigned(row, 2)?;
            let last_upd: Option<String> = row.get(3)?;
            Ok((working_dir, sess_cnt, msg_cnt, last_upd))
        })
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|(working_dir, sess_cnt, msg_cnt, last_upd)| {
            let cwd = working_dir_key(working_dir);
            ClaudeProject {
                name: project_name_for(&cwd),
                path: project_path_for(&cwd),
                actual_path: cwd,
                session_count: sess_cnt,
                message_count: msg_cnt,
                last_modified: normalize_ts(last_upd.as_deref().unwrap_or_default()),
                git_info: None,
                provider: Some(PROVIDER.to_string()),
                storage_type: Some("sqlite".to_string()),
                custom_directory_label: None,
            }
        })
        .collect();

    Ok(projects)
}

/// Load the sessions for one Goose project (`working_dir`).
pub fn load_sessions(
    project_path: &str,
    _exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    let working_dir = project_path.strip_prefix(SCHEME).unwrap_or(project_path);
    load_sessions_conn(&open_db()?, working_dir)
}

fn load_sessions_conn(conn: &Connection, working_dir: &str) -> Result<Vec<ClaudeSession>, String> {
    let schema = Schema::probe(conn);
    let mut stmt = conn
        .prepare(&format!(
            "SELECT s.id, s.name, s.description, s.created_at, s.updated_at, \
                    (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id AND {}) AS msg_cnt \
             FROM sessions s \
             WHERE COALESCE(NULLIF(s.working_dir, ''), 'unknown') = ?1 AND {} \
             ORDER BY s.updated_at DESC",
            schema.visible_message_sql("m"),
            schema.listed_session_sql("s"),
        ))
        .map_err(|e| e.to_string())?;

    let project_name = PathBuf::from(working_dir)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    let sessions = stmt
        .query_map([working_dir], |row| {
            let id: String = row.get(0)?;
            let name: Option<String> = row.get(1)?;
            let description: Option<String> = row.get(2)?;
            let created: Option<String> = row.get(3)?;
            let updated: Option<String> = row.get(4)?;
            let msg_cnt: usize = crate::providers::sql_unsigned(row, 5)?;
            Ok((id, name, description, created, updated, msg_cnt))
        })
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|(id, name, description, created, updated, msg_cnt)| {
            let summary = description
                .filter(|d| !d.trim().is_empty())
                .or(name.filter(|n| !n.trim().is_empty()))
                .or_else(|| Some(id.clone()));
            let created_iso = normalize_ts(created.as_deref().unwrap_or_default());
            let updated_iso = normalize_ts(updated.as_deref().unwrap_or_default());
            ClaudeSession {
                session_id: format!("{SCHEME}{id}"),
                actual_session_id: id.clone(),
                file_path: format!("{SCHEME}{id}"),
                project_name: project_name.clone(),
                message_count: msg_cnt,
                first_message_time: created_iso.clone(),
                last_message_time: updated_iso.clone(),
                last_modified: updated_iso,
                has_tool_use: false,
                has_errors: false,
                summary,
                is_renamed: false,
                provider: Some(PROVIDER.to_string()),
                storage_type: Some("sqlite".to_string()),
                entrypoint: None,
            }
        })
        .collect();

    Ok(sessions)
}

/// Load messages for one Goose session.
pub fn load_messages(session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    let session_id = session_path.strip_prefix(SCHEME).unwrap_or(session_path);
    load_messages_conn(&open_db()?, session_id)
}

fn load_messages_conn(conn: &Connection, session_id: &str) -> Result<Vec<ClaudeMessage>, String> {
    let schema = Schema::probe(conn);
    let mut stmt = conn
        .prepare(&format!(
            "SELECT id, role, content_json, created_timestamp, message_id, {} \
             FROM messages WHERE session_id = ?1 ORDER BY id",
            schema.metadata_column("messages"),
        ))
        .map_err(|e| e.to_string())?;

    let rows = stmt
        .query_map([session_id], |row| {
            let row_id: i64 = row.get(0)?;
            let role: String = row.get(1)?;
            let content_json: String = row.get(2)?;
            let created: i64 = row.get(3)?;
            let message_id: Option<String> = row.get(4)?;
            let metadata_json: Option<String> = row.get(5)?;
            Ok((
                row_id,
                role,
                content_json,
                created,
                message_id,
                metadata_json,
            ))
        })
        .map_err(|e| e.to_string())?;

    let mut messages = Vec::new();
    for row in rows.flatten() {
        let (row_id, role, content_json, created, message_id, metadata_json) = row;
        if let Some(msg) = build_message(
            session_id,
            row_id,
            &role,
            &content_json,
            created,
            message_id,
            metadata_json.as_deref(),
        ) {
            messages.push(msg);
        }
    }
    Ok(messages)
}

/// Search across all Goose messages.
pub fn search(query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    if query.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    search_conn(&open_db()?, query, limit)
}

fn search_conn(conn: &Connection, query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    let pattern = format!("%{query}%");
    let query_lower = query.to_lowercase();
    let schema = Schema::probe(conn);

    let mut stmt = conn
        .prepare(&format!(
            "SELECT s.working_dir, m.id, m.session_id, m.role, m.content_json, \
                    m.created_timestamp, m.message_id, {} \
             FROM messages m JOIN sessions s ON s.id = m.session_id \
             WHERE m.content_json LIKE ?1 AND {}",
            schema.metadata_column("m"),
            schema.listed_session_sql("s"),
        ))
        .map_err(|e| e.to_string())?;

    let rows = stmt
        .query_map([&pattern], |row| {
            let working_dir: Option<String> = row.get(0)?;
            let row_id: i64 = row.get(1)?;
            let session_id: String = row.get(2)?;
            let role: String = row.get(3)?;
            let content_json: String = row.get(4)?;
            let created: i64 = row.get(5)?;
            let message_id: Option<String> = row.get(6)?;
            let metadata_json: Option<String> = row.get(7)?;
            Ok((
                working_dir,
                row_id,
                session_id,
                role,
                content_json,
                created,
                message_id,
                metadata_json,
            ))
        })
        .map_err(|e| e.to_string())?;

    let mut results = Vec::new();
    for row in rows.flatten() {
        let (
            working_dir,
            row_id,
            session_id,
            role,
            content_json,
            created,
            message_id,
            metadata_json,
        ) = row;
        let Some(mut msg) = build_message(
            &session_id,
            row_id,
            &role,
            &content_json,
            created,
            message_id,
            metadata_json.as_deref(),
        ) else {
            continue;
        };
        let matched = msg
            .content
            .as_ref()
            .map(|c| search_json_value_case_insensitive(c, &query_lower))
            .unwrap_or(false);
        if !matched {
            continue;
        }
        // Same `unknown` fallback, name and path as `scan_in_conn`.
        let cwd = working_dir_key(working_dir);
        msg.project_name = Some(project_name_for(&cwd));
        msg.project_path = Some(project_path_for(&cwd));
        results.push(msg);
        if results.len() >= limit {
            break;
        }
    }
    Ok(results)
}

/// The project key a session's `working_dir` is grouped under (`unknown` when
/// empty), matching the SQL `COALESCE(NULLIF(working_dir, ''), 'unknown')`.
fn working_dir_key(working_dir: Option<String>) -> String {
    working_dir
        .filter(|w| !w.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Display name of a project key, shared by scan and search.
fn project_name_for(cwd: &str) -> String {
    PathBuf::from(cwd)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| cwd.to_string())
}

/// Opaque project identity of a project key, shared by scan and search.
fn project_path_for(cwd: &str) -> String {
    format!("{SCHEME}{cwd}")
}

// ============================================================================
// Schema tolerance and visibility
// ============================================================================

/// Optional columns of the Goose store. Goose adds them by migration
/// (`messages.metadata_json` in schema 3, `sessions.session_type` in 5), so a
/// store that Goose has not opened since upgrading may lack them; queries then
/// degrade to "everything visible / every session a user session".
struct Schema {
    message_metadata: bool,
    session_type: bool,
}

impl Schema {
    fn probe(conn: &Connection) -> Self {
        Self {
            message_metadata: has_column(conn, "messages", "metadata_json"),
            session_type: has_column(conn, "sessions", "session_type"),
        }
    }

    /// The `metadata_json` select expression for the `messages` table/alias.
    fn metadata_column(&self, alias: &str) -> String {
        if self.message_metadata {
            format!("{alias}.metadata_json")
        } else {
            "NULL".to_string()
        }
    }

    /// SQL condition: the message is shown to the user. Mirrors Goose's own
    /// `user_visible_message_sql` (`COALESCE(userVisible, 1) != 0`): rows
    /// Goose marks `userVisible:false` (per-turn `<turn-context>` events,
    /// sub-agent delivery notices, `final_output` nudges, compaction
    /// summaries) are agent-only context, not conversation.
    fn visible_message_sql(&self, alias: &str) -> String {
        if self.message_metadata {
            format!(
                "(CASE WHEN json_valid({alias}.metadata_json) \
                 THEN COALESCE(json_extract({alias}.metadata_json, '$.userVisible'), 1) \
                 ELSE 1 END) != 0"
            )
        } else {
            "1".to_string()
        }
    }

    /// SQL condition: the session is a top-level conversation. Delegated
    /// `sub_agent` runs (also folded into the parent as a delivery message
    /// and a tool result) and `hidden` sessions (`--no-session` runs) are not
    /// listed, as the other providers skip their sub-agent sessions.
    fn listed_session_sql(&self, alias: &str) -> String {
        if self.session_type {
            format!("COALESCE({alias}.session_type, 'user') NOT IN ('sub_agent', 'hidden')")
        } else {
            "1".to_string()
        }
    }
}

fn has_column(conn: &Connection, table: &str, column: &str) -> bool {
    conn.prepare("SELECT 1 FROM pragma_table_info(?1) WHERE name = ?2")
        .and_then(|mut stmt| stmt.exists([table, column]))
        .unwrap_or(false)
}

/// Whether a row's `metadata_json` marks it `userVisible:false`.
fn is_agent_only(metadata_json: Option<&str>) -> bool {
    metadata_json
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .and_then(|meta| meta.get("userVisible").and_then(Value::as_bool))
        == Some(false)
}

// ============================================================================
// Pure conversion (unit-testable)
// ============================================================================

/// Build one `ClaudeMessage` from a `messages` row. `None` when the row carries
/// no renderable content.
fn build_message(
    session_id: &str,
    row_id: i64,
    role: &str,
    content_json: &str,
    created: i64,
    message_id: Option<String>,
    metadata_json: Option<&str>,
) -> Option<ClaudeMessage> {
    if !matches!(role, "user" | "assistant" | "system") || is_agent_only(metadata_json) {
        return None;
    }
    let blocks = map_content(content_json);
    if blocks.is_empty() {
        return None;
    }
    // Fall back to the stable DB row id (not an enumeration index) so a message's
    // UUID is identical whether produced by load_messages or search.
    let uuid = message_id
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| format!("{session_id}-{row_id}"));
    Some(build_provider_message(
        PROVIDER,
        uuid,
        session_id,
        epoch_to_iso(created),
        role,
        Some(role),
        Some(Value::Array(blocks)),
        None,
    ))
}

/// Map Goose `content_json` (a JSON array of internally-tagged `MessageContent`
/// items) into the viewer's Claude-style content blocks.
fn map_content(content_json: &str) -> Vec<Value> {
    let Ok(Value::Array(items)) = serde_json::from_str::<Value>(content_json) else {
        return Vec::new();
    };
    items.iter().filter_map(map_content_item).collect()
}

fn map_content_item(item: &Value) -> Option<Value> {
    match item.get("type").and_then(Value::as_str)? {
        "text" => {
            let text = item.get("text").and_then(Value::as_str).unwrap_or("");
            if text.is_empty() {
                None
            } else {
                Some(json!({ "type": "text", "text": text }))
            }
        }
        "thinking" => {
            let thinking = item.get("thinking").and_then(Value::as_str).unwrap_or("");
            Some(json!({
                "type": "thinking",
                "thinking": thinking,
                "signature": item.get("signature").and_then(Value::as_str).unwrap_or("")
            }))
        }
        "redactedThinking" => Some(json!({
            "type": "redacted_thinking",
            "data": item.get("data").and_then(Value::as_str).unwrap_or("")
        })),
        "toolRequest" => {
            let id = item.get("id").and_then(Value::as_str).unwrap_or("");
            let tool_call = item.get("toolCall");
            let value = tool_call.and_then(|tc| tc.get("value"));
            let name = value
                .and_then(|v| v.get("name"))
                .and_then(Value::as_str)
                // A call Goose could not decode is recorded as
                // `{"status":"error","error":..}` without a name.
                .or_else(|| {
                    tool_call
                        .and_then(|tc| tc.get("error"))
                        .and_then(Value::as_str)
                        .and_then(|error| errored_call_name(error, id))
                })
                .unwrap_or("unknown");
            let input = value
                .and_then(|v| v.get("arguments"))
                .cloned()
                .unwrap_or_else(|| json!({}));
            Some(json!({ "type": "tool_use", "id": id, "name": name, "input": input }))
        }
        "toolResponse" => {
            let id = item.get("id").and_then(Value::as_str).unwrap_or("");
            let result = item.get("toolResult");
            // `status:"error"` is a tool that returned no result; a tool that
            // ran and failed is `status:"success"` with `value.isError:true`
            // (a shell command's non-zero exit, an unavailable tool, a call
            // that could not be parsed).
            let failed_call =
                result.and_then(|r| r.get("status")).and_then(Value::as_str) == Some("error");
            let flag = result
                .and_then(|r| r.get("value"))
                .and_then(|v| v.get("isError"))
                .and_then(Value::as_bool);
            let is_error = failed_call || flag == Some(true);
            let content = if failed_call {
                json!(result
                    .and_then(|r| r.get("error"))
                    .and_then(Value::as_str)
                    .unwrap_or(""))
            } else {
                tool_result_content(
                    result
                        .and_then(|r| r.get("value"))
                        .and_then(|v| v.get("content"))
                        .and_then(Value::as_array),
                )
            };
            Some(json!({
                "type": "tool_result",
                "tool_use_id": id,
                "content": content,
                "is_error": is_error,
                // Goose writes `isError` (rmcp `CallToolResult`) and the
                // `status` tag itself; only a bare success is derived.
                "is_error_source": if failed_call || flag.is_some() { "producer" } else { "derived" }
            }))
        }
        // An assistant message that failed (provider error, exhausted turn
        // budget) holds only this block.
        "error" => {
            let message = item.get("message").and_then(Value::as_str).unwrap_or("");
            if message.is_empty() {
                None
            } else {
                Some(json!({ "type": "text", "text": format!("[Error] {message}") }))
            }
        }
        _ => None,
    }
}

/// The tool name inside a decode-failure message. Goose's OpenAI/Databricks
/// formats word it `Tool arguments for <name> (id <id>) ...`; the error is
/// stored as `<code>: <message>` (the code may repeat). Only a message that
/// *starts* with that wording names the tool: truncation errors end in the
/// model's raw arguments (`cut off at: <arguments>`), which may hold any text.
fn errored_call_name<'a>(error: &'a str, id: &str) -> Option<&'a str> {
    const PREFIX: &str = "Tool arguments for ";
    let mut message = error;
    while let Some((code, tail)) = message.split_once(": ") {
        let digits = code.strip_prefix('-').unwrap_or(code);
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            break;
        }
        message = tail;
    }
    let rest = message.strip_prefix(PREFIX)?;
    let (name, tail) = rest.split_once(" (id ")?;
    let tail = tail.strip_prefix(id)?;
    (tail.starts_with(')') && !name.is_empty() && !name.contains(char::is_whitespace))
        .then_some(name)
}

/// A tool result's `content` (MCP content items). Text only stays a joined
/// string; with an image it becomes Claude-style `text`/`image` blocks so the
/// picture is shown.
fn tool_result_content(items: Option<&Vec<Value>>) -> Value {
    let items = items.map(Vec::as_slice).unwrap_or_default();
    let has_image = items
        .iter()
        .any(|c| c.get("type").and_then(Value::as_str) == Some("image"));
    if !has_image {
        return json!(items
            .iter()
            .filter_map(|c| c.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"));
    }
    Value::Array(
        items
            .iter()
            .filter_map(|c| match c.get("type").and_then(Value::as_str) {
                Some("image") => {
                    let data = c.get("data").and_then(Value::as_str)?;
                    Some(json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": c
                                .get("mimeType")
                                .and_then(Value::as_str)
                                .unwrap_or("image/png"),
                            "data": data,
                        }
                    }))
                }
                _ => c
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|t| !t.is_empty())
                    .map(|t| json!({ "type": "text", "text": t })),
            })
            .collect(),
    )
}

/// Convert an epoch timestamp (seconds or ms) to RFC3339.
fn epoch_to_iso(n: i64) -> String {
    if n <= 0 {
        return String::new();
    }
    let ms = if n > 1_000_000_000_000 { n } else { n * 1000 };
    ms_to_iso(ms as u64)
}

/// Normalize a `SQLite` `TIMESTAMP` text ("YYYY-MM-DD HH:MM:SS", UTC) to ISO
/// 8601. Values that already look ISO-ish (contain `T`) pass through.
fn normalize_ts(ts: &str) -> String {
    let ts = ts.trim();
    if ts.is_empty() || ts.contains('T') {
        return ts.to_string();
    }
    match ts.split_once(' ') {
        Some((date, time)) => format!("{date}T{time}Z"),
        None => ts.to_string(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const CWD: &str = "/home/user/demo";
    const VISIBLE: &str = r#"{"userVisible":true,"agentVisible":true}"#;
    const AGENT_ONLY: &str = r#"{"userVisible":false,"agentVisible":true}"#;

    /// The columns of Goose's schema-18 `sessions`/`messages` tables that the
    /// parser reads (types and defaults as `CREATE TABLE` in goose
    /// `session_manager.rs`).
    const SCHEMA: &str =
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '', \
            description TEXT NOT NULL DEFAULT '', user_set_name INTEGER DEFAULT 0, \
            session_type TEXT NOT NULL DEFAULT 'user', working_dir TEXT NOT NULL, \
            created_at TEXT, updated_at TEXT, parent_session_id TEXT);
         CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT, \
            session_id TEXT NOT NULL, role TEXT NOT NULL, content_json TEXT NOT NULL, \
            created_timestamp INTEGER NOT NULL, metadata_json TEXT);";

    // Real records from a Goose 1.60.2 store (paths neutralised).
    const TURN_CONTEXT: &str = r#"[{"type":"text","text":"<turn-context>\n<current-time>2026-11-25 12:32:00 +09:00</current-time>\n<working-directory>/home/user/demo</working-directory>\n</turn-context>"}]"#;
    const DELIVERY: &str = r#"[{"type":"text","text":"Subagent 20261125_13 completed: {\"summary\":\"src contains math.js\"}"}]"#;
    const NUDGE: &str = r#"[{"type":"text","text":"You MUST call the `final_output` tool NOW with the final output for the user."}]"#;
    const TOOL_CALL: &str = r#"[{"type":"toolRequest","id":"6bc1adc4","toolCall":{"status":"success","value":{"name":"shell","arguments":{"command":"npm test"}}},"_meta":{"goose_extension":"developer","goose.executable":true}}]"#;
    const FAILED_SHELL: &str = r#"[{"type":"toolResponse","id":"6bc1adc4","toolResult":{"status":"success","value":{"resultType":"complete","content":[{"type":"text","text":"\n> demo@1.0.0 test\n> node test.js\n\n4 !== 5\n"}],"structuredContent":{"stdout":"4 !== 5","stderr":"","exit_code":1},"isError":true}}}]"#;
    const IMAGE_RESULT: &str = r#"{"type":"toolResponse","id":"call_3388a612","toolResult":{"status":"success","value":{"resultType":"complete","content":[{"type":"text","text":"Loaded image from /home/user/demo/logo.png (70 bytes, image/png, 1x1).","annotations":{"priority":0.0}},{"type":"image","data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==","mimeType":"image/png"}],"isError":false}}}"#;
    const BROKEN_CALL: &str = r#"{"type":"toolRequest","id":"call_83536c45","toolCall":{"status":"error","error":"-32603: -32602: The model's response was truncated — it hit the output token limit while generating this tool call. Try increasing max_tokens for this provider or breaking the task into smaller steps.\nReceived 15 characters; cut off at: {not valid json"},"_meta":{"goose.toolCall.providerIndex":0}}"#;
    const EXHAUSTED: &str = r#"[{"type":"error","kind":"resourceExhausted","message":"I've reached the maximum number of actions I can do without user input. Would you like me to continue?"}]"#;
    const PROVIDER_FAILURE: &str = r#"[{"type":"error","kind":"other","message":"Ran into this error: Server error: Server error (500 Internal Server Error) at http://127.0.0.1:11878/v1/chat/completions: scripted provider failure.\n\nPlease retry if you think this is a transient or recoverable error."}]"#;

    fn empty_db(schema: &str) -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(schema).unwrap();
        conn
    }

    fn add_session(conn: &Connection, id: &str, name: &str, session_type: &str, cwd: &str) {
        conn.execute(
            "INSERT INTO sessions (id, name, session_type, working_dir, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, '2026-06-20 09:00:00', '2026-06-21 10:00:00')",
            [id, name, session_type, cwd],
        )
        .unwrap();
    }

    fn add_message(
        conn: &Connection,
        id: &str,
        session: &str,
        role: &str,
        content: &str,
        meta: &str,
    ) {
        conn.execute(
            "INSERT INTO messages (message_id, session_id, role, content_json, created_timestamp, metadata_json) \
             VALUES (?1, ?2, ?3, ?4, 1750500000, ?5)",
            [id, session, role, content, meta],
        )
        .unwrap();
    }

    /// A user session with a prompt, its stored turn-context, a shell call and
    /// a failing result, plus the other session kinds Goose writes.
    fn fixture_db() -> Connection {
        let conn = empty_db(SCHEMA);
        add_session(&conn, "s1", "title-a", "user", CWD);
        add_session(&conn, "s2", "title-b", "user", CWD);
        add_session(&conn, "s3", "other", "user", "/home/user/other");
        add_session(&conn, "sub", "Delegated task", "sub_agent", CWD);
        add_session(&conn, "hid", "Demo session", "hidden", CWD);
        let prompt = r#"[{"type":"text","text":"Run npm test and tell me why it LOGIN fails."}]"#;
        add_message(&conn, "m1", "s1", "user", prompt, VISIBLE);
        add_message(
            &conn,
            "m1t",
            "s1",
            "user",
            TURN_CONTEXT,
            r#"{"userVisible":false,"agentVisible":true,"turnContext":true}"#,
        );
        add_message(&conn, "m2", "s1", "assistant", TOOL_CALL, VISIBLE);
        add_message(&conn, "m3", "s1", "user", FAILED_SHELL, VISIBLE);
        add_message(
            &conn,
            "m4",
            "s1",
            "user",
            DELIVERY,
            r#"{"userVisible":false,"agentVisible":true,"operations":{"foreground_subagent":{"delivered":"sub"}}}"#,
        );
        add_message(&conn, "m5", "s2", "user", prompt, VISIBLE);
        add_message(
            &conn,
            "sub1",
            "sub",
            "user",
            r#"[{"type":"text","text":"Subagent ID: sub\n\nSubtask: list files LOGIN"}]"#,
            VISIBLE,
        );
        add_message(&conn, "hid1", "hid", "user", prompt, VISIBLE);
        conn
    }

    #[test]
    fn scan_groups_by_working_dir_without_sub_agent_hidden_or_agent_only() {
        let conn = fixture_db();
        let projects = scan_in_conn(&conn).unwrap();
        assert_eq!(projects.len(), 2);
        let proj = projects.iter().find(|p| p.name == "demo").unwrap();
        assert_eq!(
            proj.session_count, 2,
            "sub_agent and hidden sessions are not listed"
        );
        // s1: prompt, call, result (turn-context + delivery excluded); s2: prompt.
        assert_eq!(proj.message_count, 4);
        assert_eq!(proj.provider.as_deref(), Some("goose"));
        assert_eq!(proj.storage_type.as_deref(), Some("sqlite"));
        assert_eq!(proj.path, "goose:///home/user/demo");
        assert_eq!(proj.last_modified, "2026-06-21T10:00:00Z");
    }

    #[test]
    fn project_with_only_sub_agent_sessions_is_not_listed() {
        let conn = empty_db(SCHEMA);
        add_session(&conn, "sub", "Delegated task", "sub_agent", CWD);
        add_session(&conn, "hid", "Demo session", "hidden", CWD);
        assert!(scan_in_conn(&conn).unwrap().is_empty());
    }

    #[test]
    fn load_sessions_lists_top_level_sessions_with_visible_counts() {
        let conn = fixture_db();
        let sessions = load_sessions_conn(&conn, CWD).unwrap();
        let mut ids: Vec<_> = sessions
            .iter()
            .map(|s| s.actual_session_id.as_str())
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, ["s1", "s2"]);
        let s1 = sessions
            .iter()
            .find(|s| s.actual_session_id == "s1")
            .unwrap();
        assert_eq!(s1.message_count, 3);
        assert_eq!(s1.summary.as_deref(), Some("title-a"));
        assert_eq!(s1.file_path, "goose://s1");
    }

    /// Older Goose stores kept the session title in `description`; it still
    /// wins when set, and an unnamed session falls back to its id.
    #[test]
    fn legacy_description_is_preferred_and_empty_names_fall_back_to_id() {
        let conn = empty_db(SCHEMA);
        add_session(&conn, "s1", "title-a", "user", CWD);
        add_session(&conn, "s2", "", "user", CWD);
        conn.execute(
            "UPDATE sessions SET description = 'Fix login' WHERE id = 's1'",
            [],
        )
        .unwrap();
        let sessions = load_sessions_conn(&conn, CWD).unwrap();
        let summary = |id: &str| {
            sessions
                .iter()
                .find(|s| s.actual_session_id == id)
                .unwrap()
                .summary
                .clone()
        };
        assert_eq!(summary("s1").as_deref(), Some("Fix login"));
        assert_eq!(summary("s2").as_deref(), Some("s2"));
    }

    #[test]
    fn load_messages_drops_agent_only_rows() {
        let conn = fixture_db();
        add_message(&conn, "m6", "s1", "user", NUDGE, AGENT_ONLY);
        let msgs = load_messages_conn(&conn, "s1").unwrap();
        let uuids: Vec<_> = msgs.iter().map(|m| m.uuid.as_str()).collect();
        assert_eq!(
            uuids,
            ["m1", "m2", "m3"],
            "turn-context, delivery and nudge are agent-only"
        );
        assert_eq!(msgs[0].role.as_deref(), Some("user"));
    }

    #[test]
    fn load_messages_maps_text_and_tools() {
        let conn = fixture_db();
        let msgs = load_messages_conn(&conn, "s1").unwrap();

        let asst = msgs[1].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(asst[0]["type"], "tool_use");
        assert_eq!(asst[0]["name"], "shell");
        assert_eq!(asst[0]["id"], "6bc1adc4");
        assert_eq!(asst[0]["input"]["command"], "npm test");

        // `status:"success"` + `isError:true` is a failed call.
        let tr = msgs[2].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(tr[0]["type"], "tool_result");
        assert_eq!(tr[0]["tool_use_id"], "6bc1adc4");
        assert!(tr[0]["content"].as_str().unwrap().contains("4 !== 5"));
        assert_eq!(tr[0]["is_error"], true);
        assert_eq!(tr[0]["is_error_source"], "producer");
    }

    #[test]
    fn tool_result_flags_follow_the_producer() {
        let ok = r#"{"type":"toolResponse","id":"c","toolResult":{"status":"success","value":{"resultType":"complete","content":[{"type":"text","text":"x"}],"isError":false}}}"#;
        let block = map_content_item(&serde_json::from_str(ok).unwrap()).unwrap();
        assert_eq!(block["is_error"], false);
        assert_eq!(block["is_error_source"], "producer");
        // A tool that returned no result: `status:"error"`.
        let item = json!({
            "type": "toolResponse",
            "id": "c1",
            "toolResult": {"status": "error", "error": "boom"}
        });
        let block = map_content_item(&item).unwrap();
        assert_eq!(block["is_error"], true);
        assert_eq!(block["content"], "boom");
        assert_eq!(block["is_error_source"], "producer");
    }

    #[test]
    fn tool_result_image_is_kept() {
        let item: Value = serde_json::from_str(IMAGE_RESULT).unwrap();
        let block = map_content_item(&item).unwrap();
        assert_eq!(block["is_error"], false);
        let parts = block["content"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert!(parts[0]["text"]
            .as_str()
            .unwrap()
            .starts_with("Loaded image from"));
        assert_eq!(parts[1]["type"], "image");
        assert_eq!(parts[1]["source"]["type"], "base64");
        assert_eq!(parts[1]["source"]["media_type"], "image/png");
        assert!(parts[1]["source"]["data"]
            .as_str()
            .unwrap()
            .starts_with("iVBORw0KGgo"));
    }

    #[test]
    fn error_blocks_are_shown() {
        for (content, needle) in [
            (EXHAUSTED, "maximum number of actions"),
            (PROVIDER_FAILURE, "scripted provider failure"),
        ] {
            let msg = build_message(
                "s",
                1,
                "assistant",
                content,
                1750500000,
                None,
                Some(r#"{"userVisible":true,"agentVisible":false}"#),
            )
            .expect("an error-only assistant message is not dropped");
            let blocks = msg.content.unwrap();
            let text = blocks[0]["text"].as_str().unwrap();
            assert!(
                text.starts_with("[Error] ") && text.contains(needle),
                "{text}"
            );
        }
    }

    #[test]
    fn undecodable_call_keeps_what_the_record_names() {
        // The real record carries no name: the tool stays `unknown`.
        let item: Value = serde_json::from_str(BROKEN_CALL).unwrap();
        let block = map_content_item(&item).unwrap();
        assert_eq!(block["type"], "tool_use");
        assert_eq!(block["name"], "unknown");
        assert_eq!(block["id"], "call_83536c45");
        // goose-provider-types `formats/openai.rs` words other decode failures
        // `Tool arguments for <name> (id <id>) ...`, stored as `<code>: <message>`.
        let named = json!({
            "type": "toolRequest",
            "id": "call_1",
            "toolCall": {"status": "error", "error": "-32602: Tool arguments for shell (id call_1) must be a JSON object, got an array. Raw arguments: '[1]'"}
        });
        assert_eq!(map_content_item(&named).unwrap()["name"], "shell");
        // Another call's id never names this one.
        let other = json!({
            "type": "toolRequest",
            "id": "call_2",
            "toolCall": {"status": "error", "error": "Tool arguments for shell (id call_1) must be a JSON object"}
        });
        assert_eq!(map_content_item(&other).unwrap()["name"], "unknown");
        // Truncated arguments quoting the wording do not name the tool
        // (`cut off at: <arguments>`, goose-provider-types `json.rs:211`).
        let quoted = json!({
            "type": "toolRequest",
            "id": "call_1",
            "toolCall": {"status": "error", "error": "-32603: -32602: The model's response was truncated. Try again.\nReceived 40 characters; cut off at: {\"content\":\"Tool arguments for shell (id call_1)"}
        });
        assert_eq!(map_content_item(&quoted).unwrap()["name"], "unknown");
    }

    #[test]
    fn search_skips_agent_only_messages_and_sub_agent_sessions() {
        let conn = fixture_db();
        let results = search_conn(&conn, "LOGIN", 10).unwrap();
        let mut uuids: Vec<_> = results.iter().map(|m| m.uuid.as_str()).collect();
        uuids.sort_unstable();
        assert_eq!(
            uuids,
            ["m1", "m5"],
            "sub_agent and hidden sessions are not searched"
        );
        assert_eq!(results[0].project_name.as_deref(), Some("demo"));
        assert!(search_conn(&conn, "turn-context", 10).unwrap().is_empty());
        assert!(search_conn(&conn, "nonexistent-xyz", 10)
            .unwrap()
            .is_empty());
    }

    /// Stores Goose has not migrated lack `metadata_json` and `session_type`.
    #[test]
    fn legacy_schema_without_optional_columns_still_loads() {
        let conn = empty_db(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, name TEXT, description TEXT, \
                 working_dir TEXT NOT NULL, created_at TEXT, updated_at TEXT);
             CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT, \
                 session_id TEXT NOT NULL, role TEXT NOT NULL, content_json TEXT NOT NULL, \
                 created_timestamp INTEGER NOT NULL);",
        );
        conn.execute(
            "INSERT INTO sessions VALUES ('s1','t','','/home/user/demo','2026-06-20 09:00:00','2026-06-20 09:30:00')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (message_id,session_id,role,content_json,created_timestamp) VALUES ('m1','s1','user',?1,1750500000)",
            [r#"[{"type":"text","text":"hi"}]"#],
        )
        .unwrap();
        assert_eq!(scan_in_conn(&conn).unwrap()[0].message_count, 1);
        assert_eq!(load_sessions_conn(&conn, CWD).unwrap().len(), 1);
        assert_eq!(load_messages_conn(&conn, "s1").unwrap().len(), 1);
        assert_eq!(search_conn(&conn, "hi", 10).unwrap().len(), 1);
    }

    #[test]
    fn map_thinking_and_skips_unknown() {
        let think = json!({"type":"thinking","thinking":"hmm","signature":"sig"});
        let b = map_content_item(&think).unwrap();
        assert_eq!(b["type"], "thinking");
        assert_eq!(b["thinking"], "hmm");
        assert!(map_content_item(&json!({"type":"image","image":{}})).is_none());
        assert!(map_content_item(&json!({"type":"text","text":""})).is_none());
        assert!(map_content_item(&json!({"type":"error","kind":"other","message":""})).is_none());
    }

    #[test]
    fn normalize_ts_and_epoch() {
        assert_eq!(normalize_ts("2026-06-21 10:00:00"), "2026-06-21T10:00:00Z");
        assert_eq!(normalize_ts("2026-06-21T10:00:00Z"), "2026-06-21T10:00:00Z");
        assert_eq!(normalize_ts(""), "");
        // epoch seconds -> iso (non-empty)
        assert_ne!(epoch_to_iso(1750500000), "");
        assert_eq!(epoch_to_iso(0), "");
    }

    /// An empty `working_dir` must round-trip: it's grouped under "unknown" in
    /// scan and resolvable by `load_sessions("unknown")`.
    #[test]
    fn empty_working_dir_rounds_trips_as_unknown() {
        let conn = empty_db(SCHEMA);
        add_session(&conn, "s1", "", "user", "");
        add_message(
            &conn,
            "m1",
            "s1",
            "user",
            r#"[{"type":"text","text":"hi"}]"#,
            VISIBLE,
        );

        let projects = scan_in_conn(&conn).unwrap();
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].actual_path, "unknown");
        assert_eq!(projects[0].path, "goose://unknown");

        // The key the project advertises must resolve back to its sessions.
        let sessions = load_sessions_conn(&conn, "unknown").unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].actual_session_id, "s1");

        let hits = search_conn(&conn, "hi", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].project_path.as_deref(), Some("goose://unknown"));
    }

    #[test]
    fn search_hits_carry_scanned_project_path_for_same_named_projects() {
        let conn = empty_db(SCHEMA);
        for (id, cwd) in [("sa", "/work/client-a/app"), ("sb", "/work/client-b/app")] {
            add_session(&conn, id, "", "user", cwd);
            add_message(
                &conn,
                id,
                id,
                "user",
                r#"[{"type":"text","text":"shared needle"}]"#,
                VISIBLE,
            );
        }

        let projects = scan_in_conn(&conn).unwrap();
        assert_eq!(projects.len(), 2);
        assert!(projects.iter().all(|p| p.name == "app"));
        assert_ne!(projects[0].path, projects[1].path);

        let hits = search_conn(&conn, "needle", 10).unwrap();
        assert_eq!(hits.len(), 2);
        for hit in &hits {
            let project = projects
                .iter()
                .find(|p| {
                    load_sessions_conn(&conn, &p.actual_path)
                        .unwrap()
                        .iter()
                        .any(|s| s.actual_session_id == hit.session_id)
                })
                .expect("hit session belongs to a scanned project");
            assert_eq!(hit.project_path.as_deref(), Some(project.path.as_str()));
        }
    }

    /// A NULL `message_id` must get the same UUID from load and search (stable
    /// DB row id), not divergent enumeration indices.
    #[test]
    fn null_message_id_uuid_is_stable_across_load_and_search() {
        let conn = empty_db(SCHEMA);
        add_session(&conn, "s1", "", "user", "/p");
        // A decoy message first so the search-filtered index differs from the
        // session index, then the NULL-message_id message we assert on.
        add_message(
            &conn,
            "keep",
            "s1",
            "user",
            r#"[{"type":"text","text":"decoy"}]"#,
            VISIBLE,
        );
        conn.execute(
            "INSERT INTO messages (message_id,session_id,role,content_json,created_timestamp,metadata_json) VALUES (NULL,'s1','assistant',?1,1750500001,?2)",
            [r#"[{"type":"text","text":"NEEDLE here"}]"#, VISIBLE],
        )
        .unwrap();

        let loaded = load_messages_conn(&conn, "s1").unwrap();
        let loaded_uuid = &loaded[1].uuid; // the NULL-message_id row
        assert!(loaded_uuid.starts_with("s1-"));

        let found = search_conn(&conn, "NEEDLE", 10).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(&found[0].uuid, loaded_uuid, "load and search must agree");
    }
}

#[cfg(test)]
mod outcome_preservation_tests {
    use super::*;
    #[test]
    fn producer_outcome_evidence_is_not_fabricated_or_dropped() {
        // No `isError` and no `status:"error"`: nothing recorded, so derived.
        let block=map_content_item(&serde_json::json!({"type":"toolResponse","id":"c","toolResult":{"value":{"content":[]}}})).unwrap();
        assert_eq!(block["is_error_source"], "derived");
    }
}
