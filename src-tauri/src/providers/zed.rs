//! Zed provider (Agent Panel thread history).
//!
//! Zed stores agent threads in a single `SQLite` DB under its data dir, which
//! mirrors Zed's own `paths::data_dir()`: macOS
//! `~/Library/Application Support/Zed`, Linux/FreeBSD `$XDG_DATA_HOME/zed`
//! (lowercase!), Windows `%LOCALAPPDATA%\Zed` — then `/threads/threads.db`.
//! The `threads` table holds metadata plus a `data` BLOB that is the serialized
//! `DbThread` JSON — stored either plain (`data_type = "json"`) or
//! Zstd-compressed (`data_type = "zstd"`). The optional columns `folder_paths`
//! and `created_at` are absent on older Zed schemas, so the SELECTs adapt via
//! `PRAGMA table_info`.
//!
//! `DbThread.messages` is an array of externally-tagged `Message` values:
//! `{"User":{content:[{Text}|{Mention}|{Image}]}}` /
//! `{"Agent":{content:[{Text}|{Thinking}|{RedactedThinking}|{ToolUse}],
//!  tool_results:{<id>:{...}}}}` / `"Resume"` / `{"Compaction":...}`. A legacy
//! `SerializedThread` shape (internally-tagged segments) is handled as a
//! fallback. Tool results live in a sibling `tool_results` map keyed by
//! tool-use id and are surfaced as a following user turn (Claude convention).
//!
//! Projects group by the thread's display-first workspace folder
//! (`folder_paths`, see `workspace_of`). Sub-agent threads (`parent_id` set)
//! are not listed as sessions; the `SubAgent` panel lists them under the
//! thread that spawned them ([`load_subagents`]). Schema is
//! undocumented/unstable across Zed releases (see the `zed-chat-export`
//! reference tool); tool results are treated as opaque JSON.

use crate::models::{ClaudeMessage, ClaudeProject, ClaudeSession};
use crate::providers::opencode::ChildSession;
use crate::providers::ProviderInfo;
use crate::utils::{build_provider_message, search_json_value_case_insensitive};
use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

const PROVIDER: &str = "zed";
const SCHEME: &str = "zed://";
const UNKNOWN_WORKSPACE: &str = "unknown";
/// Longest session title taken from a thread's first prompt.
const SUMMARY_MAX_CHARS: usize = 100;

fn get_db_path() -> Option<PathBuf> {
    // Mirror Zed's own `paths::data_dir()`. macOS uses ~/Library/Application
    // Support (== dirs::data_dir); Linux/FreeBSD and Windows use the *local*
    // data dir (XDG_DATA_HOME / %LOCALAPPDATA%, == dirs::data_local_dir), NOT
    // the roaming dir. The app folder is lowercase "zed" only on Linux/FreeBSD.
    let base = if cfg!(target_os = "macos") {
        dirs::data_dir()?
    } else {
        dirs::data_local_dir()?
    };
    let app_name = if cfg!(any(target_os = "linux", target_os = "freebsd")) {
        "zed"
    } else {
        "Zed"
    };
    Some(base.join(app_name).join("threads").join("threads.db"))
}

/// Detect a Zed installation.
pub fn detect() -> Option<ProviderInfo> {
    let db = get_db_path()?;
    Some(ProviderInfo {
        id: PROVIDER.to_string(),
        display_name: "Zed".to_string(),
        base_path: db
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default(),
        is_available: db.is_file(),
    })
}

/// Base path (the `threads` dir holding `threads.db`), for the file watcher.
pub fn get_base_path() -> Option<String> {
    get_db_path()?
        .parent()
        .map(|p| p.to_string_lossy().to_string())
}

fn open_db() -> Result<Connection, String> {
    let path = get_db_path().ok_or("Zed threads DB not found")?;
    if !path.is_file() {
        return Err("Zed threads database not found".to_string());
    }
    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("Failed to open Zed DB: {e}"))?;
    conn.busy_timeout(std::time::Duration::from_secs(5))
        .map_err(|e| format!("Failed to set busy timeout: {e}"))?;
    Ok(conn)
}

/// Display-first workspace folder of a thread, used as the project grouping
/// key. Zed writes `folder_paths` the way `PathList::serialize` does
/// (`crates/util/src/path_list.rs`): the paths sorted lexicographically and
/// joined with `\n`, with `folder_paths_order` giving each path's display
/// position as comma-separated indices. The path at position 0 is the one Zed
/// shows first. Like `PathList::deserialize` (which compares `PathBuf`s), an
/// order that does not fit the paths falls back to the first path in sorted
/// order.
fn workspace_of(folder_paths: Option<&str>, folder_paths_order: Option<&str>) -> String {
    let Some(serialized) = folder_paths.filter(|s| !s.is_empty()) else {
        return UNKNOWN_WORKSPACE.to_string();
    };
    let mut paths: Vec<&Path> = serialized.split('\n').map(Path::new).collect();
    let order: Vec<usize> = folder_paths_order
        .unwrap_or("")
        .split(',')
        .filter_map(|ix| ix.trim().parse().ok())
        .collect();
    let first = if paths.is_sorted() && order.len() == paths.len() {
        order
            .iter()
            .zip(&paths)
            .min_by_key(|(ix, _)| **ix)
            .map(|(_, path)| *path)
    } else {
        paths.sort_unstable();
        paths.first().copied()
    };
    first
        .and_then(Path::to_str)
        .filter(|path| !path.is_empty())
        .map_or_else(|| UNKNOWN_WORKSPACE.to_string(), str::to_string)
}

/// Column names present in `table` (via `PRAGMA table_info`). Lets SELECTs adapt
/// to Zed's schema drift — older `threads` tables lack `folder_paths`/`created_at`.
fn table_columns(conn: &Connection, table: &str) -> HashSet<String> {
    let mut cols = HashSet::new();
    if let Ok(mut stmt) = conn.prepare(&format!("PRAGMA table_info({table})")) {
        if let Ok(rows) = stmt.query_map([], |row| row.get::<_, String>(1)) {
            cols.extend(rows.flatten());
        }
    }
    cols
}

/// `name` when the column exists, else `NULL AS name` so the SELECT keeps the
/// expected column positions on older schemas. `name` is always a hardcoded
/// literal, so the interpolated SQL is safe.
fn optional_col(cols: &HashSet<String>, name: &str) -> String {
    if cols.contains(name) {
        name.to_string()
    } else {
        format!("NULL AS {name}")
    }
}

/// `WHERE` clause keeping only threads listed as sessions. Zed sets
/// `parent_id` on a sub-agent thread (`DbThread.subagent_context
/// .parent_thread_id`); [`load_subagents`] lists those under their parent.
/// The parent must exist, so a run whose parent was deleted is still listed
/// rather than lost. Older schemas have no sub-agents and no column. Search
/// uses it too: a hit in a thread that is not listed could not be opened.
fn top_level_filter(cols: &HashSet<String>) -> &'static str {
    if cols.contains("parent_id") {
        "WHERE NOT (COALESCE(threads.parent_id, '') <> '' \
         AND EXISTS (SELECT 1 FROM threads p WHERE p.id = threads.parent_id))"
    } else {
        ""
    }
}

/// Project identity for a workspace key from [`workspace_of`] — shared by
/// scan and search so search hits group under the scanned project (#672).
fn project_path_for(ws: &str) -> String {
    format!("{SCHEME}{ws}")
}

/// Display name for a workspace key (its folder leaf, else the key).
fn project_name_for(ws: &str) -> String {
    PathBuf::from(ws)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| ws.to_string())
}

/// Scan Zed projects (threads grouped by first workspace folder).
pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    scan_projects_conn(&open_db()?)
}

fn scan_projects_conn(conn: &Connection) -> Result<Vec<ClaudeProject>, String> {
    let cols = table_columns(conn, "threads");
    let sql = format!(
        "SELECT {}, {}, updated_at FROM threads {}",
        optional_col(&cols, "folder_paths"),
        optional_col(&cols, "folder_paths_order"),
        top_level_filter(&cols),
    );
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;

    struct Agg {
        session_count: usize,
        last_modified: String,
    }
    let mut by_ws: HashMap<String, Agg> = HashMap::new();

    let rows = stmt
        .query_map([], |row| {
            let folder_paths: Option<String> = row.get(0)?;
            let folder_paths_order: Option<String> = row.get(1)?;
            let updated_at: Option<String> = row.get(2)?;
            Ok((folder_paths, folder_paths_order, updated_at))
        })
        .map_err(|e| e.to_string())?;
    for (folder_paths, folder_paths_order, updated_at) in rows.flatten() {
        let ws = workspace_of(folder_paths.as_deref(), folder_paths_order.as_deref());
        let entry = by_ws.entry(ws).or_insert_with(|| Agg {
            session_count: 0,
            last_modified: String::new(),
        });
        entry.session_count += 1;
        let last = updated_at.unwrap_or_default();
        if last > entry.last_modified {
            entry.last_modified = last;
        }
    }

    let mut projects: Vec<ClaudeProject> = by_ws
        .into_iter()
        .map(|(ws, agg)| {
            let name = project_name_for(&ws);
            ClaudeProject {
                name,
                path: project_path_for(&ws),
                actual_path: ws,
                session_count: agg.session_count,
                message_count: 0,
                last_modified: agg.last_modified,
                git_info: None,
                provider: Some(PROVIDER.to_string()),
                storage_type: Some("sqlite".to_string()),
                custom_directory_label: None,
            }
        })
        .collect();
    projects.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(projects)
}

/// Load the threads (sessions) for one Zed project (workspace folder).
pub fn load_sessions(
    project_path: &str,
    _exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    let target_ws = project_path.strip_prefix(SCHEME).unwrap_or(project_path);
    load_sessions_conn(&open_db()?, target_ws)
}

fn load_sessions_conn(conn: &Connection, target_ws: &str) -> Result<Vec<ClaudeSession>, String> {
    let cols = table_columns(conn, "threads");
    // `data` only for threads without a title, which fall back to their
    // first prompt.
    let sql = format!(
        "SELECT id, summary, {}, {}, {}, updated_at, data_type, \
                CASE WHEN summary IS NULL OR trim(summary) = '' THEN data END \
         FROM threads {} ORDER BY updated_at DESC",
        optional_col(&cols, "folder_paths"),
        optional_col(&cols, "folder_paths_order"),
        optional_col(&cols, "created_at"),
        top_level_filter(&cols),
    );
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;

    let project_name = PathBuf::from(target_ws)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    struct Row {
        id: String,
        summary: Option<String>,
        folder_paths: Option<String>,
        folder_paths_order: Option<String>,
        created_at: Option<String>,
        updated_at: Option<String>,
        data_type: Option<String>,
        untitled_data: Option<Vec<u8>>,
    }

    let sessions = stmt
        .query_map([], |row| {
            Ok(Row {
                id: row.get(0)?,
                summary: row.get(1)?,
                folder_paths: row.get(2)?,
                folder_paths_order: row.get(3)?,
                created_at: row.get(4)?,
                updated_at: row.get(5)?,
                data_type: row.get(6)?,
                untitled_data: row.get(7)?,
            })
        })
        .map_err(|e| e.to_string())?
        .flatten()
        .filter(|row| {
            workspace_of(
                row.folder_paths.as_deref(),
                row.folder_paths_order.as_deref(),
            ) == target_ws
        })
        .map(|row| {
            let created = row.created_at.unwrap_or_default();
            let updated = row.updated_at.unwrap_or_default();
            let summary = row
                .summary
                .filter(|s| !s.trim().is_empty())
                .or_else(|| {
                    let data = row.untitled_data.as_deref()?;
                    let thread = decode_thread_data(row.data_type.as_deref()?, data).ok()?;
                    first_prompt_title(&thread)
                })
                .unwrap_or_else(|| row.id.clone());
            ClaudeSession {
                session_id: format!("{SCHEME}{}", row.id),
                file_path: format!("{SCHEME}{}", row.id),
                actual_session_id: row.id,
                project_name: project_name.clone(),
                message_count: 0,
                first_message_time: created,
                last_message_time: updated.clone(),
                last_modified: updated,
                has_tool_use: false,
                has_errors: false,
                summary: Some(summary),
                is_renamed: false,
                provider: Some(PROVIDER.to_string()),
                storage_type: Some("sqlite".to_string()),
                entrypoint: None,
            }
        })
        .collect();

    Ok(sessions)
}

/// Title for a thread Zed has not titled yet (`DbThread.title` is empty until
/// title generation succeeds): its first typed prompt, cut to
/// [`SUMMARY_MAX_CHARS`].
fn first_prompt_title(thread: &Value) -> Option<String> {
    let text = thread
        .get("messages")?
        .as_array()?
        .iter()
        .filter_map(|msg| msg.get("User")?.get("content")?.as_array())
        .flatten()
        .filter_map(|item| item.get("Text")?.as_str())
        .map(str::trim)
        .find(|text| !text.is_empty())?;
    Some(match text.char_indices().nth(SUMMARY_MAX_CHARS) {
        Some((idx, _)) => format!("{}...", &text[..idx]),
        None => text.to_string(),
    })
}

/// Load messages for one Zed thread (`zed://<thread_id>`).
pub fn load_messages(session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    let id = session_path.strip_prefix(SCHEME).unwrap_or(session_path);
    let conn = open_db()?;
    let (data_type, data, updated_at): (String, Vec<u8>, Option<String>) = conn
        .query_row(
            "SELECT data_type, data, updated_at FROM threads WHERE id = ?1",
            [id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .map_err(|e| format!("Thread not found: {e}"))?;

    let json = decode_thread_data(&data_type, &data)?;
    Ok(parse_thread(&json, id, updated_at.as_deref().unwrap_or("")))
}

/// A sub-agent thread, for the `SubAgent` panel of the thread that spawned it.
pub struct ZedSubagent {
    pub session: ChildSession,
    /// The parent's `spawn_agent` call that started the thread, when its
    /// result names the thread.
    pub tool_use_id: Option<String>,
}

/// The sub-agent threads `session_path` (`zed://<thread_id>`) spawned:
/// threads whose `parent_id` is the thread, oldest first. A deeper run is
/// listed by its own parent. Empty on schemas without `parent_id`.
pub fn load_subagents(session_path: &str) -> Result<Vec<ZedSubagent>, String> {
    let id = session_path.strip_prefix(SCHEME).unwrap_or(session_path);
    load_subagents_conn(&open_db()?, id)
}

fn load_subagents_conn(conn: &Connection, parent_id: &str) -> Result<Vec<ZedSubagent>, String> {
    let cols = table_columns(conn, "threads");
    if !cols.contains("parent_id") {
        return Ok(Vec::new());
    }
    let spawned_by = conn
        .query_row(
            "SELECT data_type, data FROM threads WHERE id = ?1",
            [parent_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )
        .ok()
        .and_then(|(data_type, data)| decode_thread_data(&data_type, &data).ok())
        .map(|thread| spawning_calls(&thread))
        .unwrap_or_default();

    let sql = format!(
        "SELECT id, summary, {}, updated_at, data_type, data FROM threads \
         WHERE parent_id = ?1 AND id != ?1 ORDER BY 3, updated_at",
        optional_col(&cols, "created_at"),
    );
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([parent_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, Vec<u8>>(5)?,
            ))
        })
        .map_err(|e| e.to_string())?;

    let mut subagents = Vec::new();
    for (id, summary, created_at, updated_at, data_type, data) in rows.flatten() {
        let thread = decode_thread_data(&data_type, &data).ok();
        let updated_at = updated_at.unwrap_or_default();
        let title = summary
            .filter(|s| !s.trim().is_empty())
            .or_else(|| thread.as_ref().and_then(first_prompt_title))
            .unwrap_or_default();
        let message_count = thread
            .as_ref()
            .map_or(0, |thread| parse_thread(thread, &id, &updated_at).len());
        subagents.push(ZedSubagent {
            tool_use_id: spawned_by.get(&id).cloned(),
            session: ChildSession {
                created_at: created_at.unwrap_or_else(|| updated_at.clone()),
                updated_at,
                id,
                title,
                message_count,
            },
        });
    }
    Ok(subagents)
}

/// Sub-agent thread id -> the `spawn_agent` call that started it. The call's
/// result keeps the tool's `SpawnAgentToolOutput` in `output`, whose
/// `session_id` is the sub-agent thread's id
/// (`crates/agent/src/tools/spawn_agent_tool.rs`). A follow-up call to the
/// same agent names it again; the first call is kept.
fn spawning_calls(thread: &Value) -> HashMap<String, String> {
    let mut calls = HashMap::new();
    let messages = thread.get("messages").and_then(Value::as_array);
    for msg in messages.into_iter().flatten() {
        let Some(results) = msg
            .get("Agent")
            .and_then(|agent| agent.get("tool_results"))
            .and_then(Value::as_object)
        else {
            continue;
        };
        let calls_in_message = msg
            .get("Agent")
            .and_then(|agent| agent.get("content"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|item| item.get("ToolUse")?.get("id")?.as_str());
        for call_id in calls_in_message {
            let Some(child) = results
                .get(call_id)
                .and_then(|result| result.get("output"))
                .and_then(|output| output.get("session_id"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            calls
                .entry(child.to_string())
                .or_insert_with(|| call_id.to_string());
        }
    }
    calls
}

/// Search across all Zed threads.
pub fn search(query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    if query.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    search_conn(&open_db()?, query, limit)
}

fn search_conn(conn: &Connection, query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    let query_lower = query.to_lowercase();
    let cols = table_columns(conn, "threads");
    let sql = format!(
        "SELECT id, data_type, data, updated_at, {}, {} FROM threads {}",
        optional_col(&cols, "folder_paths"),
        optional_col(&cols, "folder_paths_order"),
        top_level_filter(&cols),
    );
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        })
        .map_err(|e| e.to_string())?;

    let mut results = Vec::new();
    for (id, data_type, data, updated_at, folder_paths, folder_paths_order) in rows.flatten() {
        let Ok(json) = decode_thread_data(&data_type, &data) else {
            continue;
        };
        let ws = workspace_of(folder_paths.as_deref(), folder_paths_order.as_deref());
        let project_name = project_name_for(&ws);
        let project_path = project_path_for(&ws);
        for mut msg in parse_thread(&json, &id, updated_at.as_deref().unwrap_or("")) {
            if results.len() >= limit {
                return Ok(results);
            }
            let matched = msg
                .content
                .as_ref()
                .map(|c| search_json_value_case_insensitive(c, &query_lower))
                .unwrap_or(false);
            if matched {
                msg.project_name = Some(project_name.clone());
                msg.project_path = Some(project_path.clone());
                results.push(msg);
            }
        }
    }
    Ok(results)
}

// ============================================================================
// Decoding + parsing (pure where possible)
// ============================================================================

/// Decode a `threads.data` BLOB to its `DbThread` JSON string.
fn decode_thread_data(data_type: &str, data: &[u8]) -> Result<Value, String> {
    let bytes = if data_type == "zstd" {
        zstd::decode_all(data).map_err(|e| format!("Failed to zstd-decode Zed thread: {e}"))?
    } else {
        data.to_vec()
    };
    serde_json::from_slice(&bytes).map_err(|e| format!("Failed to parse Zed thread JSON: {e}"))
}

/// Parse a decoded `DbThread` (or legacy `SerializedThread`) JSON into messages.
fn parse_thread(thread: &Value, thread_id: &str, ts: &str) -> Vec<ClaudeMessage> {
    let Some(messages) = thread.get("messages").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (idx, msg) in messages.iter().enumerate() {
        convert_message(msg, thread_id, idx, ts, &mut out);
    }
    out
}

/// Convert one `Message` (current `DbThread` shape, or legacy `SerializedMessage`)
/// and push the resulting message(s).
fn convert_message(
    msg: &Value,
    thread_id: &str,
    idx: usize,
    ts: &str,
    out: &mut Vec<ClaudeMessage>,
) {
    // Current externally-tagged shape: {"User":{...}} / {"Agent":{...}}.
    if let Some(user) = msg.get("User") {
        let blocks = user_blocks(user.get("content"));
        if !blocks.is_empty() {
            out.push(make_msg(thread_id, idx, "user", ts, blocks, None));
        }
        return;
    }
    if let Some(agent) = msg.get("Agent") {
        let blocks = agent_blocks(agent.get("content"));
        if !blocks.is_empty() {
            out.push(make_msg(thread_id, idx, "assistant", ts, blocks, None));
        }
        // Tool results live in a sibling map keyed by tool-use id -> surface as a
        // following user turn (Claude convention).
        if let Some(results) = agent.get("tool_results").and_then(Value::as_object) {
            let tr_blocks: Vec<Value> = results
                .iter()
                .map(|(tool_use_id, res)| {
                    let content = res
                        .get("content")
                        .map(tool_result_content)
                        .unwrap_or_default();
                    let mut block = json!({ "type": "tool_result", "tool_use_id": tool_use_id, "content": content });
                    super::result_evidence::copy_error_flag(&mut block, res, "is_error");
                    block
                })
                .collect();
            if !tr_blocks.is_empty() {
                out.push(make_msg(thread_id, idx, "user", ts, tr_blocks, Some("-tr")));
            }
        }
        return;
    }
    // Legacy SerializedMessage: {"role":"user"|"assistant"|"system","segments":[{"type":"text",...}],...}
    if let Some(role) = msg.get("role").and_then(Value::as_str) {
        let blocks = legacy_segments(msg.get("segments"));
        if !blocks.is_empty() {
            let mapped = if role == "assistant" {
                "assistant"
            } else {
                "user"
            };
            out.push(make_msg(thread_id, idx, mapped, ts, blocks, None));
        }
    }
    // `{"Compaction":{"Summary":"…"}}`: the summary that replaced the earlier
    // turns (`CompactionInfo::Summary`), shown as a compaction marker. A
    // `ProviderNative` compaction keeps only provider items, so the marker has
    // no summary. `"Resume"` carries nothing to show.
    if let Some(compaction) = msg.get("Compaction") {
        let summary = compaction
            .get("Summary")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let text = summary.map_or_else(
            || "Context compacted".to_string(),
            |summary| format!("Context compacted\n\n{summary}"),
        );
        let mut message = build_provider_message(
            PROVIDER,
            format!("{thread_id}-{idx}"),
            thread_id,
            ts.to_string(),
            "system",
            None,
            Some(Value::String(text)),
            None,
        );
        message.subtype = Some("compact_boundary".to_string());
        message.level = Some("info".to_string());
        out.push(message);
    }
}

fn user_blocks(content: Option<&Value>) -> Vec<Value> {
    let Some(items) = content.and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut blocks = Vec::new();
    for item in items {
        if let Some(text) = item.get("Text").and_then(Value::as_str) {
            if !text.is_empty() {
                blocks.push(json!({ "type": "text", "text": text }));
            }
        } else if let Some(mention) = item.get("Mention") {
            if let Some(text) = mention.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    blocks.push(json!({ "type": "text", "text": text }));
                }
            }
        } else if let Some(source) = item
            .get("Image")
            .and_then(|image| image.get("source"))
            .and_then(Value::as_str)
            .filter(|source| !source.is_empty())
        {
            // `LanguageModelImage.source`: base64 PNG bytes without a
            // `data:` prefix (`crates/language_model/src/request.rs`).
            blocks.push(json!({
                "type": "image",
                "source": { "type": "base64", "media_type": "image/png", "data": source }
            }));
        }
    }
    blocks
}

fn agent_blocks(content: Option<&Value>) -> Vec<Value> {
    let Some(items) = content.and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut blocks = Vec::new();
    for item in items {
        if let Some(text) = item.get("Text").and_then(Value::as_str) {
            if !text.is_empty() {
                blocks.push(json!({ "type": "text", "text": text }));
            }
        } else if let Some(thinking) = item.get("Thinking") {
            let text = thinking.get("text").and_then(Value::as_str).unwrap_or("");
            blocks.push(json!({
                "type": "thinking",
                "thinking": text,
                "signature": thinking.get("signature").and_then(Value::as_str).unwrap_or("")
            }));
        } else if let Some(rt) = item.get("RedactedThinking").and_then(Value::as_str) {
            blocks.push(json!({ "type": "redacted_thinking", "data": rt }));
        } else if let Some(tool) = item.get("ToolUse") {
            let id = tool.get("id").and_then(Value::as_str).unwrap_or("");
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let input = tool_input(tool);
            blocks.push(json!({ "type": "tool_use", "id": id, "name": name, "input": input }));
        }
    }
    blocks
}

/// Arguments of a `ToolUse`. Zed serializes `LanguageModelToolUse.input` as
/// `{"type":"json","value":<args>}` or `{"type":"text","value":"…"}`
/// (`crates/language_model_core/src/language_model_core.rs`), and reads any
/// other value as the arguments themselves; `raw_input` holds the arguments
/// as the model streamed them. Text input that parses as JSON is shown as
/// that JSON.
fn tool_input(tool: &Value) -> Value {
    let parse = |text: &str| serde_json::from_str::<Value>(text).ok();
    if let Some(input) = tool.get("input") {
        let wrapped = input
            .as_object()
            .filter(|object| object.len() == 2)
            .and_then(|object| Some((object.get("type")?.as_str()?, object.get("value")?)));
        return match wrapped {
            Some(("json", value)) => value.clone(),
            Some(("text", Value::String(text))) => {
                parse(text).unwrap_or_else(|| Value::String(text.clone()))
            }
            _ => input.clone(),
        };
    }
    match tool.get("raw_input") {
        Some(Value::String(raw)) => parse(raw).unwrap_or_else(|| Value::String(raw.clone())),
        Some(other) => other.clone(),
        None => json!({}),
    }
}

fn legacy_segments(segments: Option<&Value>) -> Vec<Value> {
    let Some(items) = segments.and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut blocks = Vec::new();
    for seg in items {
        match seg.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(t) = seg.get("text").and_then(Value::as_str) {
                    if !t.is_empty() {
                        blocks.push(json!({ "type": "text", "text": t }));
                    }
                }
            }
            Some("thinking") => {
                let t = seg.get("text").and_then(Value::as_str).unwrap_or("");
                blocks.push(json!({
                    "type": "thinking",
                    "thinking": t,
                    "signature": seg.get("signature").and_then(Value::as_str).unwrap_or("")
                }));
            }
            _ => {}
        }
    }
    blocks
}

fn make_msg(
    thread_id: &str,
    idx: usize,
    role: &str,
    ts: &str,
    blocks: Vec<Value>,
    suffix: Option<&str>,
) -> ClaudeMessage {
    let uuid = format!("{thread_id}-{idx}{}", suffix.unwrap_or(""));
    build_provider_message(
        PROVIDER,
        uuid,
        thread_id,
        ts.to_string(),
        role,
        Some(role),
        Some(Value::Array(blocks)),
        None,
    )
}

/// Extract human-readable text from a Zed tool-result `content`. Zed serializes
/// this as `Vec<LanguageModelToolResultContent>` — a JSON array of externally
/// tagged items `{"Text": "..."}` / `{"Image": {...}}`. The wire format also
/// tolerates a bare string or a single item, so handle all three rather than
/// JSON-encoding the value verbatim (which would leak the enum tags into the
/// rendered result, e.g. `[{"Text":"auth.rs:42"}]`).
fn tool_result_content(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .map(tool_result_content_part)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(_) => tool_result_content_part(v),
        other => other.to_string(),
    }
}

/// One `LanguageModelToolResultContent` item -> text. `{"Text": "..."}` yields the
/// text (Zed matches the key case-insensitively); `{"Image": ...}` becomes a
/// placeholder; a bare string passes through; anything else falls back to JSON.
fn tool_result_content_part(item: &Value) -> String {
    if let Value::String(s) = item {
        return s.clone();
    }
    if let Some(text) = item
        .get("Text")
        .or_else(|| item.get("text"))
        .and_then(Value::as_str)
    {
        return text.to_string();
    }
    if item.get("Image").is_some() || item.get("image").is_some() {
        return "[image]".to_string();
    }
    item.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db_thread() -> Value {
        json!({
            "title": "Fix LOGIN",
            "messages": [
                { "User": { "id": "u1", "content": [{ "Text": "why does LOGIN fail?" }, { "Mention": { "uri": "x", "content": "see auth.rs" } }] } },
                { "Agent": {
                    "content": [
                        { "Text": "Let me look" },
                        { "Thinking": { "text": "reasoning", "signature": "sig" } },
                        { "ToolUse": { "id": "t1", "name": "grep", "input": { "q": "login" } } }
                    ],
                    "tool_results": { "t1": { "content": [{ "Text": "auth.rs:42" }], "is_error": false } }
                } },
                "Resume"
            ]
        })
    }

    #[test]
    fn parse_thread_user_agent_and_tool_results() {
        let msgs = parse_thread(&db_thread(), "thread-1", "2026-06-20T10:00:00Z");
        // user, assistant, tool-result(user) ; "Resume" ignored
        assert_eq!(msgs.len(), 3);

        assert_eq!(msgs[0].role.as_deref(), Some("user"));
        assert_eq!(msgs[0].provider.as_deref(), Some("zed"));
        let u = msgs[0].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(u[0]["text"], "why does LOGIN fail?");
        assert_eq!(u[1]["text"], "see auth.rs"); // Mention -> text

        let a = msgs[1].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(a[0]["type"], "text");
        assert_eq!(a[1]["type"], "thinking");
        assert_eq!(a[2]["type"], "tool_use");
        assert_eq!(a[2]["id"], "t1");

        // tool_results -> following user turn with tool_result block joined by id
        assert_eq!(msgs[2].role.as_deref(), Some("user"));
        let tr = msgs[2].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(tr[0]["type"], "tool_result");
        assert_eq!(tr[0]["tool_use_id"], "t1");
        assert_eq!(tr[0]["content"], "auth.rs:42");
    }

    #[test]
    fn tool_result_content_extracts_text_from_all_shapes() {
        // Legacy bare string passes through.
        assert_eq!(tool_result_content(&json!("auth.rs:42")), "auth.rs:42");
        // Modern Vec<LanguageModelToolResultContent>: array of externally-tagged items.
        assert_eq!(
            tool_result_content(&json!([{ "Text": "auth.rs:42" }])),
            "auth.rs:42"
        );
        // Multiple parts incl. an image placeholder, joined by newline.
        assert_eq!(
            tool_result_content(&json!([
                { "Text": "line 1" },
                { "Image": { "source": "data:...", "size": { "width": 1, "height": 1 } } },
                { "Text": "line 2" }
            ])),
            "line 1\n[image]\nline 2"
        );
        // A single item that wasn't wrapped in an array.
        assert_eq!(tool_result_content(&json!({ "Text": "single" })), "single");
        // The old behaviour would have leaked the enum tag; ensure it does not.
        assert!(!tool_result_content(&json!([{ "Text": "x" }])).contains("Text"));
    }

    #[test]
    fn parse_legacy_serialized_thread() {
        let legacy = json!({
            "version": "0.1.0",
            "summary": "old",
            "messages": [
                { "id": 1, "role": "user", "segments": [{ "type": "text", "text": "hi" }] },
                { "id": 2, "role": "assistant", "segments": [
                    { "type": "thinking", "text": "hmm", "signature": "s" },
                    { "type": "text", "text": "hello" }
                ] }
            ]
        });
        let msgs = parse_thread(&legacy, "t", "");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role.as_deref(), Some("user"));
        assert_eq!(msgs[1].role.as_deref(), Some("assistant"));
        let a = msgs[1].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(a[0]["type"], "thinking");
        assert_eq!(a[1]["type"], "text");
    }

    // `PathList::serialize` (crates/util/src/path_list.rs): sorted paths
    // joined with `\n`, display positions in `folder_paths_order`.
    #[test]
    fn workspace_is_the_display_first_folder() {
        assert_eq!(
            workspace_of(Some("/Users/jack/proj"), Some("0")),
            "/Users/jack/proj"
        );
        // Zed shows `/zz/api` first: position 0 belongs to the second path.
        assert_eq!(
            workspace_of(Some("/aa/web\n/zz/api"), Some("1,0")),
            "/zz/api"
        );
        assert_eq!(
            workspace_of(Some("/aa/web\n/zz/api"), Some("0,1")),
            "/aa/web"
        );
        // An order that does not fit falls back to the first sorted path.
        assert_eq!(workspace_of(Some("/aa/web\n/zz/api"), None), "/aa/web");
        assert_eq!(
            workspace_of(Some("/zz/api\n/aa/web"), Some("0,1")),
            "/aa/web"
        );
        assert_eq!(workspace_of(Some(""), None), "unknown");
        assert_eq!(workspace_of(None, None), "unknown");
    }

    #[test]
    fn db_path_uses_platform_app_name() {
        let Some(p) = get_db_path() else { return };
        let s = p.to_string_lossy().replace('\\', "/");
        assert!(s.ends_with("threads/threads.db"), "got {s}");
        if cfg!(any(target_os = "linux", target_os = "freebsd")) {
            assert!(
                s.contains("/zed/threads"),
                "linux must use lowercase zed: {s}"
            );
        } else {
            assert!(s.contains("/Zed/threads"), "got {s}");
        }
    }

    #[test]
    fn table_columns_and_optional_col() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute("CREATE TABLE threads (id TEXT, summary TEXT)", [])
            .unwrap();
        let cols = table_columns(&conn, "threads");
        assert!(cols.contains("id") && cols.contains("summary"));
        assert!(!cols.contains("folder_paths"));
        assert_eq!(optional_col(&cols, "id"), "id");
        assert_eq!(optional_col(&cols, "folder_paths"), "NULL AS folder_paths");
    }

    fn insert_thread(conn: &Connection, cols: &str, values: &[&dyn rusqlite::ToSql]) {
        let placeholders = (1..=values.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(",");
        conn.execute(
            &format!("INSERT INTO threads ({cols}) VALUES ({placeholders})"),
            values,
        )
        .unwrap();
    }

    #[test]
    fn scan_and_load_tolerate_old_5col_schema() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE threads (id TEXT PRIMARY KEY, summary TEXT NOT NULL, \
             updated_at TEXT NOT NULL, data_type TEXT NOT NULL, data BLOB NOT NULL)",
            [],
        )
        .unwrap();
        let data = serde_json::to_vec(&json!({ "messages": [] })).unwrap();
        insert_thread(
            &conn,
            "id, summary, updated_at, data_type, data",
            &[
                &"t1",
                &"Old thread",
                &"2026-06-20T10:00:00Z",
                &"json",
                &data,
            ],
        );

        // No folder_paths / created_at columns -> must not error.
        let projects = scan_projects_conn(&conn).unwrap();
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].actual_path, "unknown"); // no folder_paths -> grouped as unknown
        assert_eq!(projects[0].session_count, 1);

        let sessions = load_sessions_conn(&conn, "unknown").unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].actual_session_id, "t1");
        assert_eq!(sessions[0].summary.as_deref(), Some("Old thread"));
    }

    #[test]
    fn scan_and_load_use_new_schema_columns() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE threads (id TEXT PRIMARY KEY, summary TEXT NOT NULL, \
             updated_at TEXT NOT NULL, data_type TEXT NOT NULL, data BLOB NOT NULL, \
             folder_paths TEXT, created_at TEXT)",
            [],
        )
        .unwrap();
        let data = serde_json::to_vec(&json!({ "messages": [] })).unwrap();
        insert_thread(
            &conn,
            "id, summary, updated_at, data_type, data, folder_paths, created_at",
            &[
                &"t1",
                &"New",
                &"2026-06-20T10:00:00Z",
                &"json",
                &data,
                &"/Users/jack/proj",
                &"2026-06-20T09:00:00Z",
            ],
        );

        let projects = scan_projects_conn(&conn).unwrap();
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].actual_path, "/Users/jack/proj");

        let sessions = load_sessions_conn(&conn, "/Users/jack/proj").unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].first_message_time, "2026-06-20T09:00:00Z"); // created_at
    }

    #[test]
    fn search_hits_carry_scanned_project_path_for_same_named_folders() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE threads (id TEXT PRIMARY KEY, summary TEXT NOT NULL, \
             updated_at TEXT NOT NULL, data_type TEXT NOT NULL, data BLOB NOT NULL, \
             folder_paths TEXT)",
            [],
        )
        .unwrap();
        for (id, folders) in [("ta", "/work/client-a/app"), ("tb", "/work/client-b/app")] {
            let data = serde_json::to_vec(
                &json!({ "messages": [{ "User": { "content": [{ "Text": "needle" }] } }] }),
            )
            .unwrap();
            insert_thread(
                &conn,
                "id, summary, updated_at, data_type, data, folder_paths",
                &[&id, &"s", &"2026-06-20T10:00:00Z", &"json", &data, &folders],
            );
        }

        let projects = scan_projects_conn(&conn).unwrap();
        let path_of = |ws: &str| {
            projects
                .iter()
                .find(|p| p.actual_path == ws)
                .map(|p| p.path.clone())
                .unwrap()
        };
        let a = path_of("/work/client-a/app");
        let b = path_of("/work/client-b/app");
        assert_ne!(a, b);

        let hits = search_conn(&conn, "needle", 10).unwrap();
        assert_eq!(hits.len(), 2);
        for hit in &hits {
            let expected = if hit.session_id == "ta" { &a } else { &b };
            assert_eq!(hit.project_path.as_ref(), Some(expected));
            assert_eq!(hit.project_name.as_deref(), Some("app"));
        }
    }

    #[test]
    fn decode_plain_and_zstd() {
        let value = json!({ "messages": [] });
        let json_bytes = serde_json::to_vec(&value).unwrap();
        // plain
        let got = decode_thread_data("json", &json_bytes).unwrap();
        assert!(got.get("messages").is_some());
        // zstd
        let compressed = zstd::encode_all(&json_bytes[..], 3).unwrap();
        let got = decode_thread_data("zstd", &compressed).unwrap();
        assert!(got.get("messages").is_some());
    }

    /// A `threads` table with the columns `save_thread_sync` writes
    /// (crates/agent/src/db.rs).
    fn current_schema() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE threads (id TEXT PRIMARY KEY, parent_id TEXT, folder_paths TEXT, \
             folder_paths_order TEXT, summary TEXT NOT NULL, updated_at TEXT NOT NULL, \
             data_type TEXT NOT NULL, data BLOB NOT NULL, created_at TEXT)",
            [],
        )
        .unwrap();
        conn
    }

    fn insert_current(
        conn: &Connection,
        id: &str,
        parent_id: Option<&str>,
        summary: &str,
        created_at: &str,
        thread: &Value,
    ) {
        let data = zstd::encode_all(&serde_json::to_vec(thread).unwrap()[..], 3).unwrap();
        insert_thread(
            conn,
            "id, parent_id, folder_paths, folder_paths_order, summary, updated_at, data_type, \
             data, created_at",
            &[
                &id,
                &parent_id,
                &"/aa/web\n/zz/api",
                &"1,0",
                &summary,
                &created_at,
                &"zstd",
                &data,
                &created_at,
            ],
        );
    }

    #[test]
    fn newline_folder_paths_group_threads_into_their_project() {
        let conn = current_schema();
        let thread = json!({ "messages": [{ "User": { "content": [{ "Text": "needle" }] } }] });
        insert_current(&conn, "t1", None, "Fix it", "2026-11-26T10:00:00Z", &thread);

        let projects = scan_projects_conn(&conn).unwrap();
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].actual_path, "/zz/api");
        assert_eq!(projects[0].name, "api");
        assert_eq!(load_sessions_conn(&conn, "/zz/api").unwrap().len(), 1);
        let hits = search_conn(&conn, "needle", 10).unwrap();
        assert_eq!(hits[0].project_path.as_deref(), Some("zed:///zz/api"));
    }

    #[test]
    fn untitled_thread_is_named_by_its_first_prompt() {
        let conn = current_schema();
        let long = "x".repeat(150);
        let thread = json!({ "title": "", "messages": [
            { "User": { "content": [{ "Mention": { "uri": "file:///a", "content": "ctx" } }, { "Text": "  " }] } },
            { "User": { "content": [{ "Text": long }] } }
        ] });
        insert_current(&conn, "t1", None, "", "2026-11-26T10:00:00Z", &thread);
        insert_current(&conn, "t2", None, "Named", "2026-11-26T09:00:00Z", &thread);

        let sessions = load_sessions_conn(&conn, "/zz/api").unwrap();
        let title = |id: &str| {
            sessions
                .iter()
                .find(|s| s.actual_session_id == id)
                .and_then(|s| s.summary.clone())
                .unwrap()
        };
        assert_eq!(title("t1"), format!("{}...", "x".repeat(100)));
        assert_eq!(title("t2"), "Named");
    }

    #[test]
    fn sub_agent_threads_are_listed_under_the_thread_that_spawned_them() {
        let conn = current_schema();
        let parent = json!({ "messages": [
            { "User": { "content": [{ "Text": "split the work" }] } },
            { "Agent": {
                "content": [
                    { "ToolUse": { "id": "call-1", "name": "spawn_agent", "raw_input": "{}", "input": { "type": "json", "value": {} } } }
                ],
                "tool_results": { "call-1": {
                    "tool_use_id": "call-1", "tool_name": "spawn_agent", "is_error": false,
                    "content": [{ "Text": "{\"session_id\":\"child\",\"output\":\"done\"}" }],
                    "output": { "session_id": "child", "output": "done", "session_info": {} }
                } }
            } }
        ] });
        let child = json!({ "messages": [
            { "User": { "content": [{ "Text": "list the files" }] } },
            { "Agent": { "content": [{ "Text": "a.rs" }], "tool_results": {} } }
        ] });
        insert_current(
            &conn,
            "parent",
            None,
            "Parent",
            "2026-11-26T10:00:00Z",
            &parent,
        );
        insert_current(
            &conn,
            "child",
            Some("parent"),
            "",
            "2026-11-26T10:01:00Z",
            &child,
        );
        insert_current(
            &conn,
            "orphan",
            Some("gone"),
            "Orphan",
            "2026-11-26T10:02:00Z",
            &child,
        );

        let listed: Vec<String> = load_sessions_conn(&conn, "/zz/api")
            .unwrap()
            .into_iter()
            .map(|s| s.actual_session_id)
            .collect();
        assert_eq!(listed, ["orphan", "parent"]);
        assert_eq!(scan_projects_conn(&conn).unwrap()[0].session_count, 2);
        let hits = search_conn(&conn, "list the files", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].session_id, "orphan");

        let subagents = load_subagents_conn(&conn, "parent").unwrap();
        assert_eq!(subagents.len(), 1);
        assert_eq!(subagents[0].session.id, "child");
        assert_eq!(subagents[0].session.title, "list the files");
        assert_eq!(subagents[0].session.message_count, 2);
        assert_eq!(subagents[0].session.created_at, "2026-11-26T10:01:00Z");
        assert_eq!(subagents[0].tool_use_id.as_deref(), Some("call-1"));
        assert!(load_subagents_conn(&conn, "child").unwrap().is_empty());
    }

    // `LanguageModelToolUseInput` serialization
    // (crates/language_model_core/src/language_model_core.rs).
    #[test]
    fn tool_input_is_unwrapped_from_its_type_tag() {
        let input = |tool: Value| tool_input(&tool);
        assert_eq!(
            input(json!({ "input": { "type": "json", "value": { "path": "a.rs" } } })),
            json!({ "path": "a.rs" })
        );
        assert_eq!(
            input(json!({ "input": { "type": "text", "value": "{\"path\":\"a.rs\"}" } })),
            json!({ "path": "a.rs" })
        );
        assert_eq!(
            input(json!({ "input": { "type": "text", "value": "not json" } })),
            json!("not json")
        );
        // Older threads stored the arguments directly.
        assert_eq!(
            input(json!({ "input": { "path": "a.rs" } })),
            json!({ "path": "a.rs" })
        );
        assert_eq!(
            input(json!({ "raw_input": "{\"path\":\"a.rs\"}" })),
            json!({ "path": "a.rs" })
        );
    }

    #[test]
    fn compaction_summary_and_prompt_images_are_shown() {
        let thread = json!({ "messages": [
            { "Compaction": { "Summary": "Earlier: fixed login" } },
            { "Compaction": { "ProviderNative": { "provider": "anthropic", "items": [] } } },
            { "User": { "content": [{ "Text": "and this?" }, { "Image": { "source": "iVBORw0KGgo=" } }] } },
            "Resume"
        ] });
        let msgs = parse_thread(&thread, "t", "2026-11-26T10:00:00Z");
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].subtype.as_deref(), Some("compact_boundary"));
        assert_eq!(
            msgs[0].content,
            Some(json!("Context compacted\n\nEarlier: fixed login"))
        );
        assert_eq!(msgs[1].content, Some(json!("Context compacted")));
        let blocks = msgs[2].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(
            blocks[1],
            json!({ "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo=" } })
        );
    }
}

#[cfg(test)]
mod outcome_preservation_tests {
    use super::*;
    #[test]
    fn producer_outcome_evidence_is_not_fabricated_or_dropped() {
        let mut out = Vec::new();
        convert_message(
            &serde_json::json!({"Agent":{"tool_results":{"c":{"content":[]}}}}),
            "s",
            0,
            "",
            &mut out,
        );
        assert_ne!(
            out.last().unwrap().content.as_ref().unwrap()[0]["is_error"],
            serde_json::json!(false)
        );
    }
}
