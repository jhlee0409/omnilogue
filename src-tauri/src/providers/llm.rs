//! `llm` provider (Simon Willison's `llm` CLI, `github.com/simonw/llm`).
//!
//! Reads the `SQLite` log at `<app-dir>/logs.db` where `<app-dir>` =
//! `click.get_app_dir("io.datasette.llm")` (which matches `dirs::config_dir()`
//! on every OS) — overridable via `LLM_USER_PATH`. Logging is on by default.
//!
//! `llm` has no project/`cwd` concept, so everything is surfaced under one
//! synthetic project. The log holds two generations of tables:
//!
//! - **llm >= 0.32** writes a content-addressed store only: `threads`,
//!   `turns`, `messages` and `parts`. Each `threads` row is a session. A
//!   message has no thread, timestamp or usage of its own (identical content
//!   is one shared row), so a thread's transcript is the chain of `messages`
//!   from `threads.tip_message_hash` up through `parent_hash`; the parts of
//!   a message (`text`, `reasoning`, `tool_call`, `tool_result`,
//!   `attachment`) come from `parts` in `position` order, and time, model and
//!   token usage from the `turns` row whose tip is the message.
//! - **Older versions** wrote `conversations` and `responses`. Each
//!   `conversations` row is a session; responses with no conversation
//!   (`conversation_id IS NULL`, the common one-shot case) are bucketed into
//!   a single "Ungrouped" session. Each `responses` row becomes a user turn
//!   (the prompt) followed by an assistant turn (the response, carrying token
//!   usage + model). New llm still creates these tables, empty, and never
//!   writes them again.
//!
//! A thread is named after the conversation it continues, so a database that
//! holds both generations lists such a session once, from the thread.

use crate::models::{ClaudeMessage, ClaudeProject, ClaudeSession, TokenUsage};
use crate::providers::scan_report::ScanReport;
use crate::providers::ProviderInfo;
use crate::utils::{build_provider_message, search_json_value_case_insensitive};
use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const PROVIDER: &str = "llm";
const SCHEME: &str = "llm://";
/// Single synthetic project bucket.
const PROJECT_KEY: &str = "__all__";
/// Sentinel session id for legacy responses with no conversation.
const NO_CONVERSATION: &str = "__none__";

fn get_db_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LLM_USER_PATH") {
        let p = p.trim();
        if !p.is_empty() {
            return Some(PathBuf::from(p).join("logs.db"));
        }
    }
    Some(dirs::config_dir()?.join("io.datasette.llm").join("logs.db"))
}

/// Detect an `llm` installation (only when the logs DB exists).
pub fn detect() -> Option<ProviderInfo> {
    let db = get_db_path()?;
    if !db.is_file() {
        return None;
    }
    Some(ProviderInfo {
        id: PROVIDER.to_string(),
        display_name: "llm".to_string(),
        base_path: db
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default(),
        is_available: true,
    })
}

/// Base path (the dir holding `logs.db`), for the file watcher.
pub fn get_base_path() -> Option<String> {
    Some(get_db_path()?.parent()?.to_string_lossy().to_string())
}

fn open_db() -> Result<Connection, String> {
    let path = get_db_path().ok_or("llm logs DB not found")?;
    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("Failed to open llm DB: {e}"))?;
    conn.busy_timeout(std::time::Duration::from_secs(5))
        .map_err(|e| format!("Failed to set busy timeout: {e}"))?;
    Ok(conn)
}

/// The logs database for a scan that records what it cannot read: `None`
/// when there is none, or when it is there but cannot be opened (recorded
/// in `report`, as is a directory that may not be looked into).
fn open_db_reporting(report: &ScanReport) -> Option<(PathBuf, Connection)> {
    let path = get_db_path()?;
    if !report.is_file(&path) {
        return None;
    }
    let conn = report.db(
        &path,
        Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY),
    )?;
    report.db(&path, conn.busy_timeout(std::time::Duration::from_secs(5)))?;
    Some((path, conn))
}

/// `datetime_utc` is a UTC value stored without a timezone designator (e.g.
/// `2026-06-20T10:00:00`). Mark it explicitly UTC so downstream date parsing
/// doesn't treat it as local time. Values that already carry a tz, or have no
/// time component, are returned unchanged.
///
/// llm >= 0.32 writes `turns.datetime_utc` as `2026-11-25T03:33:10.474738+00:00`
/// and `threads.datetime_utc` as `2026-11-25 03:33:59.752893+00:00` (a space
/// separator); both carry an offset, and both come out in the `T` form.
fn normalize_utc_ts(ts: &str) -> String {
    let t = ts.trim().replace(' ', "T");
    if t.is_empty() || t.ends_with('Z') {
        return t;
    }
    match t.split_once('T') {
        // A time component with an explicit offset (+hh:mm / -hh:mm) is left as-is.
        Some((_, time)) if time.contains('+') || time.contains('-') => t,
        Some((_, _)) => format!("{t}Z"),
        None => t, // date-only / unparseable: don't fabricate a time zone
    }
}

/// Tables of the content-addressed store written by llm >= 0.32.
const STORE_TABLES: [&str; 4] = ["threads", "turns", "messages", "parts"];

/// Which generations of tables a logs database has. A fresh llm >= 0.32
/// database still creates the empty legacy tables, but a database from a
/// future version, or a trimmed copy, need have neither generation.
struct Layout {
    /// `threads`, `turns`, `messages` and `parts` (llm >= 0.32).
    store: bool,
    /// `responses` (older versions; the orphan bucket lives here).
    legacy: bool,
    /// `conversations`, which names the legacy sessions.
    conversations: bool,
}

fn has_table(conn: &Connection, name: &str) -> rusqlite::Result<bool> {
    conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
        [name],
        |r| r.get::<_, i64>(0),
    )
    .map(|n| n > 0)
}

/// The generations of tables the database `db` has. A database that cannot
/// be read is recorded in `report` and is an error.
fn detect_layout(conn: &Connection, db: &Path, report: &ScanReport) -> Result<Layout, String> {
    let read = || -> rusqlite::Result<Layout> {
        let mut store = true;
        for table in STORE_TABLES {
            store &= has_table(conn, table)?;
        }
        Ok(Layout {
            store,
            legacy: has_table(conn, "responses")?,
            conversations: has_table(conn, "conversations")?,
        })
    };
    read().map_err(|e| {
        let message = e.to_string();
        report.db::<()>(db, Err(e));
        message
    })
}

/// Restricts a `responses r` query to rows no thread owns. llm >= 0.32
/// names a thread after the conversation it continues, and a conversation
/// continued after the upgrade (or logged by an alpha that wrote both
/// generations) has its whole history in the thread, so the thread is the
/// session and the legacy rows must not list it a second time.
fn legacy_scope(layout: &Layout) -> &'static str {
    if layout.store {
        " AND (r.conversation_id IS NULL OR r.conversation_id NOT IN (SELECT id FROM threads))"
    } else {
        ""
    }
}

/// Milliseconds since the epoch of a normalized timestamp, for ordering
/// values that do not share a format (`Z` against `+00:00`).
fn ts_millis(ts: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(ts).map_or(i64::MIN, |d| d.timestamp_millis())
}

/// The later of `current` and `candidate`.
fn later(current: Option<String>, candidate: String) -> Option<String> {
    match current {
        _ if candidate.is_empty() => current,
        Some(c) if ts_millis(&c) >= ts_millis(&candidate) => Some(c),
        _ => Some(candidate),
    }
}

/// One synthetic project containing every `llm` conversation.
pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    scan_in_conn(&open_db()?)
}

/// [`scan_projects`], recording in `report` a logs database that is there
/// but cannot be read. Unlike [`scan_projects`], no database (or one that
/// cannot be opened) is no projects, not an error: for a caller that counts
/// the report only.
pub(crate) fn scan_projects_reporting(report: &ScanReport) -> Result<Vec<ClaudeProject>, String> {
    match open_db_reporting(report) {
        Some((path, conn)) => scan_in_conn_reporting(&conn, &path, report),
        None => Ok(Vec::new()),
    }
}

/// Identity of the single synthetic project — shared by scan and search so
/// search hits group under the scanned project (#672).
fn project_path() -> String {
    format!("{SCHEME}{PROJECT_KEY}")
}

fn scan_in_conn(conn: &Connection) -> Result<Vec<ClaudeProject>, String> {
    scan_in_conn_reporting(conn, Path::new(""), &ScanReport::default())
}

fn scan_in_conn_reporting(
    conn: &Connection,
    db: &Path,
    report: &ScanReport,
) -> Result<Vec<ClaudeProject>, String> {
    let (session_count, message_count, last) = project_stats(conn, db, report)?;
    if message_count == 0 {
        return Ok(Vec::new());
    }
    Ok(vec![ClaudeProject {
        name: "llm".to_string(),
        path: project_path(),
        actual_path: "llm".to_string(),
        session_count,
        message_count,
        last_modified: last,
        git_info: None,
        provider: Some(PROVIDER.to_string()),
        storage_type: Some("sqlite".to_string()),
        custom_directory_label: None,
    }])
}

/// Session-count / message-count / latest-time for the synthetic project. A
/// count the database `db` cannot be read for is recorded in `report`.
fn project_stats(
    conn: &Connection,
    db: &Path,
    report: &ScanReport,
) -> Result<(usize, usize, String), String> {
    let layout = detect_layout(conn, db, report)?;
    let mut sessions = 0;
    let mut messages = 0;
    let mut last: Option<String> = None;
    if layout.store {
        for session in store_sessions(conn, db, report)? {
            sessions += 1;
            messages += session.message_count;
            last = later(last, session.last_message_time);
        }
    }
    if layout.legacy {
        let (legacy_sessions, legacy_messages, legacy_last) =
            legacy_stats(conn, db, report, &layout)?;
        sessions += legacy_sessions;
        messages += legacy_messages;
        last = later(last, legacy_last);
    }
    Ok((sessions, messages, last.unwrap_or_default()))
}

/// Session-count / message-count / latest-time of the legacy `responses`.
fn legacy_stats(
    conn: &Connection,
    db: &Path,
    report: &ScanReport,
    layout: &Layout,
) -> Result<(usize, usize, String), String> {
    let scope = legacy_scope(layout);
    // Sessions = distinct conversations + 1 for the orphan bucket (if any exist).
    let convs: usize = report
        .db(
            db,
            conn.query_row(
                &format!("SELECT COUNT(DISTINCT r.conversation_id) FROM responses r WHERE r.conversation_id IS NOT NULL{scope}"),
                [],
                |r| crate::providers::sql_unsigned(r, 0),
            ),
        )
        .unwrap_or(0);
    let orphans: usize = report
        .db(
            db,
            conn.query_row(
                "SELECT COUNT(*) FROM responses WHERE conversation_id IS NULL",
                [],
                |r| crate::providers::sql_unsigned(r, 0),
            ),
        )
        .unwrap_or(0);
    // Each response yields a user + assistant message.
    let responses: usize = conn
        .query_row(
            &format!("SELECT COUNT(*) FROM responses r WHERE 1 = 1{scope}"),
            [],
            |r| crate::providers::sql_unsigned(r, 0),
        )
        .map_err(|e| e.to_string())?;
    let last: Option<String> = report
        .db(
            db,
            conn.query_row(
                &format!("SELECT MAX(r.datetime_utc) FROM responses r WHERE 1 = 1{scope}"),
                [],
                |r| r.get(0),
            ),
        )
        .unwrap_or(None);
    Ok((
        convs + usize::from(orphans > 0),
        responses * 2,
        normalize_utc_ts(&last.unwrap_or_default()),
    ))
}

/// Load the sessions (threads and conversations) for the synthetic `llm`
/// project.
pub fn load_sessions(
    _project_path: &str,
    _exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    load_sessions_conn(&open_db()?)
}

/// [`load_sessions`], recording in `report` a logs database, or a row of
/// it, that cannot be read. Unlike [`load_sessions`], no database (or one
/// that cannot be opened) is no sessions, not an error: for a caller that
/// counts the report only.
pub(crate) fn load_sessions_reporting(
    _project_path: &str,
    report: &ScanReport,
) -> Result<Vec<ClaudeSession>, String> {
    match open_db_reporting(report) {
        Some((path, conn)) => load_sessions_conn_reporting(&conn, &path, report),
        None => Ok(Vec::new()),
    }
}

fn load_sessions_conn(conn: &Connection) -> Result<Vec<ClaudeSession>, String> {
    load_sessions_conn_reporting(conn, Path::new(""), &ScanReport::default())
}

fn load_sessions_conn_reporting(
    conn: &Connection,
    db: &Path,
    report: &ScanReport,
) -> Result<Vec<ClaudeSession>, String> {
    let layout = detect_layout(conn, db, report)?;
    let mut sessions = Vec::new();
    if layout.store {
        sessions.extend(store_sessions(conn, db, report)?);
    }
    if layout.legacy {
        sessions.extend(legacy_sessions(conn, db, report, &layout)?);
    }
    sessions.sort_by_key(|s| std::cmp::Reverse(ts_millis(&s.last_message_time)));
    Ok(sessions)
}

/// Every thread of the content-addressed store as a session. A thread's
/// messages are the chain from `threads.tip_message_hash` to the root:
/// messages are content-addressed and shared across threads, so the chain is
/// the only way to tell which messages a thread holds (`llm/logs.py`
/// `LogStore.load_chain`). Threads without a message are not sessions.
fn store_sessions(
    conn: &Connection,
    db: &Path,
    report: &ScanReport,
) -> Result<Vec<ClaudeSession>, String> {
    let mut stmt = conn
        .prepare(
            "WITH RECURSIVE chain(tid, hash, parent) AS ( \
                 SELECT t.id, m.hash, m.parent_hash \
                 FROM threads t JOIN messages m ON m.hash = t.tip_message_hash \
                 UNION ALL \
                 SELECT c.tid, m.hash, m.parent_hash \
                 FROM chain c JOIN messages m ON m.hash = c.parent) \
             SELECT t.id, t.name, t.datetime_utc, tt.first, tt.last, \
                    COALESCE(SUM(EXISTS(SELECT 1 FROM parts p WHERE p.message_hash = c.hash)), 0), \
                    COALESCE(MAX(EXISTS(SELECT 1 FROM parts p \
                        WHERE p.message_hash = c.hash AND p.type = 'tool_call')), 0), \
                    COALESCE(MAX(EXISTS(SELECT 1 FROM parts p \
                        WHERE p.message_hash = c.hash AND p.type = 'tool_result' \
                          AND CASE WHEN json_valid(p.payload) \
                               THEN json_extract(p.payload, '$.exception') IS NOT NULL \
                               ELSE 0 END)), 0) \
             FROM threads t \
             LEFT JOIN chain c ON c.tid = t.id \
             LEFT JOIN (SELECT thread_id, MIN(datetime_utc) AS first, MAX(datetime_utc) AS last \
                        FROM turns GROUP BY thread_id) tt ON tt.thread_id = t.id \
             GROUP BY t.id",
        )
        .map_err(|e| e.to_string())?;

    let sessions = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Option<String>>(4)?,
                crate::providers::sql_unsigned::<usize>(row, 5)?,
                row.get::<_, i64>(6)? != 0,
                row.get::<_, i64>(7)? != 0,
            ))
        })
        .map_err(|e| e.to_string())?
        .filter_map(|row| report.db(db, row))
        .filter(|(.., count, _, _)| *count > 0)
        .map(
            |(id, name, thread_time, first, last, count, has_tool_use, has_errors)| {
                // `turns.datetime_utc` is when a response started;
                // `threads.datetime_utc` is when its first turn finished, and
                // is only the answer for a thread that has no turn.
                let first =
                    normalize_utc_ts(&first.or_else(|| thread_time.clone()).unwrap_or_default());
                let last = normalize_utc_ts(&last.or(thread_time).unwrap_or_default());
                let summary = name
                    .filter(|n| !n.trim().is_empty())
                    .or_else(|| Some(id.clone()));
                ClaudeSession {
                    session_id: format!("{SCHEME}{id}"),
                    actual_session_id: id.clone(),
                    file_path: format!("{SCHEME}{id}"),
                    project_name: "llm".to_string(),
                    message_count: count,
                    first_message_time: first,
                    last_message_time: last.clone(),
                    last_modified: last,
                    has_tool_use,
                    has_errors,
                    summary,
                    is_renamed: false,
                    provider: Some(PROVIDER.to_string()),
                    storage_type: Some("sqlite".to_string()),
                    entrypoint: None,
                }
            },
        )
        .collect();
    Ok(sessions)
}

/// The sessions of the legacy `responses` / `conversations` tables, minus
/// the conversations a thread owns.
fn legacy_sessions(
    conn: &Connection,
    db: &Path,
    report: &ScanReport,
    layout: &Layout,
) -> Result<Vec<ClaudeSession>, String> {
    let (name, join) = if layout.conversations {
        (
            "c.name",
            "LEFT JOIN conversations c ON c.id = r.conversation_id",
        )
    } else {
        ("NULL", "")
    };
    let scope = legacy_scope(layout);
    let mut stmt = conn
        .prepare(&format!(
            "SELECT r.conversation_id, {name}, MIN(r.datetime_utc) AS first, \
                    MAX(r.datetime_utc) AS last, COUNT(*) AS cnt \
             FROM responses r {join} WHERE 1 = 1{scope} \
             GROUP BY r.conversation_id ORDER BY last DESC",
        ))
        .map_err(|e| e.to_string())?;

    let sessions = stmt
        .query_map([], |row| {
            let conv_id: Option<String> = row.get(0)?;
            let name: Option<String> = row.get(1)?;
            let first: Option<String> = row.get(2)?;
            let last: Option<String> = row.get(3)?;
            let cnt: usize = crate::providers::sql_unsigned(row, 4)?;
            Ok((conv_id, name, first, last, cnt))
        })
        .map_err(|e| e.to_string())?
        .filter_map(|row| report.db(db, row))
        .map(|(conv_id, name, first, last, cnt)| {
            let id = conv_id
                .clone()
                .unwrap_or_else(|| NO_CONVERSATION.to_string());
            let summary = if conv_id.is_none() {
                Some("Ungrouped prompts".to_string())
            } else {
                name.filter(|n| !n.trim().is_empty())
                    .or_else(|| Some(id.clone()))
            };
            let first = normalize_utc_ts(&first.unwrap_or_default());
            let last = normalize_utc_ts(&last.unwrap_or_default());
            ClaudeSession {
                session_id: format!("{SCHEME}{id}"),
                actual_session_id: id.clone(),
                file_path: format!("{SCHEME}{id}"),
                project_name: "llm".to_string(),
                message_count: cnt * 2,
                first_message_time: first,
                last_message_time: last.clone(),
                last_modified: last,
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

/// Load messages for one `llm` session (`llm://<thread or conversation id>`).
pub fn load_messages(session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    let id = session_path.strip_prefix(SCHEME).unwrap_or(session_path);
    load_messages_conn(&open_db()?, id)
}

fn load_messages_conn(conn: &Connection, id: &str) -> Result<Vec<ClaudeMessage>, String> {
    let layout = detect_layout(conn, Path::new(""), &ScanReport::default())?;
    // A thread wins over a conversation of the same id: it holds the whole
    // history (see `legacy_scope`).
    if layout.store && id != NO_CONVERSATION {
        let is_thread = conn
            .query_row("SELECT COUNT(*) FROM threads WHERE id = ?1", [id], |r| {
                r.get::<_, i64>(0)
            })
            .map_err(|e| e.to_string())?
            > 0;
        if is_thread {
            return load_thread_messages(conn, id);
        }
    }
    if layout.legacy {
        return load_legacy_messages(conn, id);
    }
    Ok(Vec::new())
}

/// The chain of messages ending at `?1`, as `(hash, parent, role, depth)`.
const CHAIN_CTE: &str = "WITH RECURSIVE chain(hash, parent, role, depth) AS ( \
        SELECT hash, parent_hash, role, 0 FROM messages WHERE hash = ?1 \
        UNION ALL \
        SELECT m.hash, m.parent_hash, m.role, c.depth + 1 \
        FROM chain c JOIN messages m ON m.hash = c.parent) ";

/// What a turn records about the response that ended at its tip message.
struct TurnInfo {
    thread_id: Option<String>,
    model: Option<String>,
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    duration_ms: Option<i64>,
    datetime: Option<String>,
}

/// One row of `parts`.
struct PartRow {
    position: i64,
    kind: String,
    tool_name: Option<String>,
    text: Option<String>,
    payload: Option<String>,
}

/// The messages of one thread, oldest first. Messages carry no timestamp,
/// model or token count of their own (rows are shared between threads), so
/// those come from the turn whose span of the chain the message falls in: a
/// turn answers the messages after the previous turn's tip, up to its own
/// tip.
fn load_thread_messages(conn: &Connection, thread_id: &str) -> Result<Vec<ClaudeMessage>, String> {
    let (tip, thread_time): (Option<String>, Option<String>) = conn
        .query_row(
            "SELECT tip_message_hash, datetime_utc FROM threads WHERE id = ?1",
            [thread_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(|e| e.to_string())?;
    let Some(tip) = tip else {
        return Ok(Vec::new());
    };

    let chain: Vec<(String, Option<String>)> = conn
        .prepare(&format!(
            "{CHAIN_CTE}SELECT hash, role FROM chain ORDER BY depth DESC"
        ))
        .map_err(|e| e.to_string())?
        .query_map([&tip], |r| Ok((r.get(0)?, r.get(1)?)))
        .map_err(|e| e.to_string())?
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;

    // A forked thread reaches into its parent's turns, so a tip can belong
    // to a turn of another thread (or of several threads that answered
    // identically): prefer this thread's turn, then the earliest.
    let mut turns: HashMap<String, TurnInfo> = HashMap::new();
    let mut stmt = conn
        .prepare(&format!(
            "{CHAIN_CTE}SELECT t.tip_message_hash, t.thread_id, t.model, t.input_tokens, \
                    t.output_tokens, t.duration_ms, t.datetime_utc \
             FROM turns t WHERE t.tip_message_hash IN (SELECT hash FROM chain) \
             ORDER BY t.datetime_utc"
        ))
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([&tip], |r| {
            Ok((
                r.get::<_, String>(0)?,
                TurnInfo {
                    thread_id: r.get(1)?,
                    model: r.get(2)?,
                    input_tokens: r.get(3)?,
                    output_tokens: r.get(4)?,
                    duration_ms: r.get(5)?,
                    datetime: r.get(6)?,
                },
            ))
        })
        .map_err(|e| e.to_string())?;
    for row in rows {
        let (hash, turn) = row.map_err(|e| e.to_string())?;
        let own = turn.thread_id.as_deref() == Some(thread_id);
        match turns.get(&hash) {
            Some(known) if known.thread_id.as_deref() == Some(thread_id) || !own => {}
            _ => {
                turns.insert(hash, turn);
            }
        }
    }

    let mut owner: Vec<Option<&TurnInfo>> = vec![None; chain.len()];
    let mut pending = Vec::new();
    let mut last_turn = None;
    for (i, (hash, _)) in chain.iter().enumerate() {
        pending.push(i);
        if let Some(turn) = turns.get(hash) {
            for j in pending.drain(..) {
                owner[j] = Some(turn);
            }
            last_turn = Some(turn);
        }
    }
    for j in pending {
        owner[j] = last_turn;
    }

    let mut parts_stmt = conn
        .prepare(
            "SELECT position, type, tool_name, text, payload FROM parts \
             WHERE message_hash = ?1 ORDER BY position",
        )
        .map_err(|e| e.to_string())?;
    let mut messages = Vec::new();
    for (i, (hash, role)) in chain.iter().enumerate() {
        // A tool result is a `tool` message: surface it in the user lane
        // (Claude convention) so the tool_result renderer picks it up.
        let (message_type, role) = match role.as_deref() {
            Some("assistant") => ("assistant", "assistant"),
            Some("system") => ("system", "system"),
            Some("user" | "tool") => ("user", "user"),
            _ => continue,
        };
        let parts: Vec<PartRow> = parts_stmt
            .query_map([hash], |r| {
                Ok(PartRow {
                    position: r.get(0)?,
                    kind: r.get(1)?,
                    tool_name: r.get(2)?,
                    text: r.get(3)?,
                    payload: r.get(4)?,
                })
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        let blocks: Vec<Value> = parts
            .iter()
            .filter_map(|part| part_block(conn, hash, part))
            .collect();
        if blocks.is_empty() {
            continue;
        }
        let turn = owner[i];
        let ts = normalize_utc_ts(
            turn.and_then(|t| t.datetime.as_deref())
                .or(thread_time.as_deref())
                .unwrap_or_default(),
        );
        let model = turn
            .filter(|_| message_type == "assistant")
            .and_then(|t| t.model.clone());
        let mut msg = build_provider_message(
            PROVIDER,
            hash.clone(),
            thread_id,
            ts,
            message_type,
            Some(role),
            Some(Value::Array(blocks)),
            model,
        );
        // Usage and duration belong to the response that ended at the tip.
        if let Some(turn) = turns.get(hash).filter(|_| message_type == "assistant") {
            if turn.input_tokens.is_some() || turn.output_tokens.is_some() {
                msg.usage = Some(TokenUsage {
                    input_tokens: turn.input_tokens.map(|t| t.max(0) as u32),
                    output_tokens: turn.output_tokens.map(|t| t.max(0) as u32),
                    ..Default::default()
                });
            }
            msg.duration_ms = turn
                .duration_ms
                .and_then(|d| u64::try_from(d).ok())
                .filter(|d| *d > 0);
        }
        messages.push(msg);
    }
    Ok(messages)
}

/// The content block for one part (`llm/parts.py` `Part.from_dict`), or
/// `None` for a part with nothing to show. The `type`, literal `text` and
/// `tool_name` are columns; everything else is the JSON `payload`.
fn part_block(conn: &Connection, hash: &str, part: &PartRow) -> Option<Value> {
    let payload: Value = part
        .payload
        .as_deref()
        .and_then(|p| serde_json::from_str(p).ok())
        .unwrap_or(Value::Null);
    let name = || {
        payload
            .get("name")
            .and_then(Value::as_str)
            .or(part.tool_name.as_deref())
            .unwrap_or("unknown")
            .to_string()
    };
    // Calls and results are paired by the producer's `tool_call_id`.
    let call_id = || {
        payload
            .get("tool_call_id")
            .and_then(Value::as_str)
            .map_or_else(|| format!("{hash}:{}", part.position), String::from)
    };
    match part.kind.as_str() {
        "text" => {
            let text = part_text(conn, part, &payload);
            (!text.is_empty()).then(|| json!({ "type": "text", "text": text }))
        }
        "reasoning" => {
            // An opaque (redacted) reasoning part has no text to show.
            let text = part_text(conn, part, &payload);
            (!text.is_empty())
                .then(|| json!({ "type": "thinking", "thinking": text, "signature": "" }))
        }
        "tool_call" => Some(json!({
            "type": "tool_use",
            "id": call_id(),
            "name": name(),
            "input": payload.get("arguments").cloned().unwrap_or_else(|| json!({})),
        })),
        "tool_result" => {
            let mut content = match payload.get("output") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Null) | None => String::new(),
                Some(other) => other.to_string(),
            };
            for id in payload
                .get("attachments")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|a| a.get("id").and_then(Value::as_str))
            {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(&attachment_label(conn, id));
            }
            // `ToolResultPart.to_dict` writes `exception` only for a tool
            // call that raised (or was cancelled), so its presence is the
            // producer's own failure marker.
            let is_error = payload.get("exception").is_some_and(|e| !e.is_null());
            Some(json!({
                "type": "tool_result",
                "tool_use_id": call_id(),
                "content": content,
                "is_error": is_error,
                "is_error_source": "producer",
            }))
        }
        "attachment" => {
            let label = payload
                .get("attachment")
                .and_then(|a| a.get("id"))
                .and_then(Value::as_str)
                .map_or_else(
                    || "[Attachment]".to_string(),
                    |id| attachment_label(conn, id),
                );
            Some(json!({ "type": "text", "text": label }))
        }
        _ => None,
    }
}

/// The text of a `text` / `reasoning` part: the `text` column, or — when
/// the text borrows prompt fragments (`llm -f`) — the `text_ref` pieces of
/// the payload with each fragment's content looked up (`llm/logs.py`
/// `_decode_text_refs`).
fn part_text(conn: &Connection, part: &PartRow, payload: &Value) -> String {
    if let Some(text) = &part.text {
        return text.clone();
    }
    let mut text = String::new();
    for piece in payload
        .get("text_ref")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(literal) = piece.get("literal").and_then(Value::as_str) {
            text.push_str(literal);
        } else if let Some(id) = piece.get("fragment").and_then(Value::as_i64) {
            if let Ok(content) =
                conn.query_row("SELECT content FROM fragments WHERE id = ?1", [id], |r| {
                    r.get::<_, String>(0)
                })
            {
                text.push_str(&content);
            }
        }
    }
    text
}

/// A one-line note for an attachment row (the bytes are not shown).
fn attachment_label(conn: &Connection, id: &str) -> String {
    conn.query_row(
        "SELECT type, path, url FROM attachments WHERE id = ?1",
        [id],
        |r| {
            Ok((
                r.get::<_, Option<String>>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<String>>(2)?,
            ))
        },
    )
    .map_or_else(
        |_| "[Attachment]".to_string(),
        |(kind, path, url)| {
            let detail = [kind, path, url]
                .into_iter()
                .flatten()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            format!("[Attachment: {detail}]")
        },
    )
}

/// Messages of a legacy conversation (or, for [`NO_CONVERSATION`], the
/// responses with none): a user message per prompt and an assistant message
/// per response.
fn load_legacy_messages(conn: &Connection, conv_id: &str) -> Result<Vec<ClaudeMessage>, String> {
    // The orphan bucket selects rows with a NULL conversation_id; named
    // conversations select by id.
    let sql = if conv_id == NO_CONVERSATION {
        "SELECT id, prompt, response, model, datetime_utc, input_tokens, output_tokens \
         FROM responses WHERE conversation_id IS NULL ORDER BY datetime_utc, id"
    } else {
        "SELECT id, prompt, response, model, datetime_utc, input_tokens, output_tokens \
         FROM responses WHERE conversation_id = ?1 ORDER BY datetime_utc, id"
    };
    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;

    let map_row = |row: &rusqlite::Row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, Option<String>>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<i64>>(5)?,
            row.get::<_, Option<i64>>(6)?,
        ))
    };
    let rows = if conv_id == NO_CONVERSATION {
        stmt.query_map([], map_row)
    } else {
        stmt.query_map([conv_id], map_row)
    }
    .map_err(|e| e.to_string())?;

    let mut messages = Vec::new();
    for row in rows.flatten() {
        let (id, prompt, response, model, dt, input_tokens, output_tokens) = row;
        let ts = normalize_utc_ts(&dt.unwrap_or_default());
        if let Some(p) = prompt.filter(|p| !p.is_empty()) {
            messages.push(build_provider_message(
                PROVIDER,
                format!("{id}-user"),
                conv_id,
                ts.clone(),
                "user",
                Some("user"),
                Some(json!([{ "type": "text", "text": p }])),
                None,
            ));
        }
        if let Some(r) = response.filter(|r| !r.is_empty()) {
            let mut msg = build_provider_message(
                PROVIDER,
                format!("{id}-asst"),
                conv_id,
                ts,
                "assistant",
                Some("assistant"),
                Some(json!([{ "type": "text", "text": r }])),
                model,
            );
            if input_tokens.is_some() || output_tokens.is_some() {
                msg.usage = Some(TokenUsage {
                    input_tokens: input_tokens.map(|t| t.max(0) as u32),
                    output_tokens: output_tokens.map(|t| t.max(0) as u32),
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                    reasoning_tokens: None,
                    service_tier: None,
                    ..Default::default()
                });
            }
            messages.push(msg);
        }
    }
    Ok(messages)
}

/// Search across all `llm` prompts and responses.
pub fn search(query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    if query.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    search_conn(&open_db()?, query, limit)
}

fn search_conn(conn: &Connection, query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    let layout = detect_layout(conn, Path::new(""), &ScanReport::default())?;
    let mut results = Vec::new();
    // `turn_search` is the producer's own search text (`prompt` and
    // `response` of each turn); it is created together with the store.
    if layout.store && has_table(conn, "turn_search").map_err(|e| e.to_string())? {
        results.extend(search_store(conn, query, limit)?);
    }
    if layout.legacy {
        results.extend(search_legacy(conn, &layout, query, limit)?);
    }
    results.sort_by_key(|m| std::cmp::Reverse(ts_millis(&m.timestamp)));
    results.truncate(limit);
    Ok(results)
}

/// One search hit: whichever side of the exchange matched `query_lower`.
fn push_hits(
    results: &mut Vec<ClaudeMessage>,
    limit: usize,
    query_lower: &str,
    session_id: &str,
    ts: &str,
    sides: [(String, &str, Option<String>, Option<String>); 2],
) {
    let project_path = project_path();
    for (uuid, role, text, model) in sides {
        if results.len() >= limit {
            break;
        }
        let Some(text) = text.filter(|t| !t.is_empty()) else {
            continue;
        };
        let content = json!([{ "type": "text", "text": text }]);
        if !search_json_value_case_insensitive(&content, query_lower) {
            continue;
        }
        let mut msg = build_provider_message(
            PROVIDER,
            uuid,
            session_id,
            ts.to_string(),
            role,
            Some(role),
            Some(content),
            model,
        );
        msg.project_name = Some("llm".to_string());
        msg.project_path = Some(project_path.clone());
        results.push(msg);
    }
}

/// Hits in the turns of the content-addressed store. The user side is
/// anchored at the message the turn answers and the assistant side at its
/// tip, so a hit lands on a message of the loaded thread.
fn search_store(
    conn: &Connection,
    query: &str,
    limit: usize,
) -> Result<Vec<ClaudeMessage>, String> {
    let pattern = format!("%{query}%");
    let query_lower = query.to_lowercase();
    let mut stmt = conn
        .prepare(
            "SELECT s.turn_id, t.thread_id, t.parent_message_hash, t.tip_message_hash, \
                    s.prompt, s.response, t.model, t.datetime_utc \
             FROM turn_search s JOIN turns t ON t.id = s.turn_id \
             WHERE s.prompt LIKE ?1 OR s.response LIKE ?1 \
             ORDER BY t.datetime_utc DESC",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([&pattern], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, Option<String>>(6)?,
                row.get::<_, Option<String>>(7)?,
            ))
        })
        .map_err(|e| e.to_string())?;

    let mut results = Vec::new();
    for row in rows.flatten() {
        if results.len() >= limit {
            break;
        }
        let (turn_id, thread_id, parent, tip, prompt, response, model, dt) = row;
        let Some(thread_id) = thread_id else {
            continue;
        };
        let ts = normalize_utc_ts(&dt.unwrap_or_default());
        push_hits(
            &mut results,
            limit,
            &query_lower,
            &thread_id,
            &ts,
            [
                (
                    parent.unwrap_or_else(|| format!("{turn_id}-user")),
                    "user",
                    prompt,
                    None,
                ),
                (
                    tip.unwrap_or_else(|| format!("{turn_id}-asst")),
                    "assistant",
                    response,
                    model,
                ),
            ],
        );
    }
    Ok(results)
}

/// Hits in the legacy `responses`, minus the conversations a thread owns.
fn search_legacy(
    conn: &Connection,
    layout: &Layout,
    query: &str,
    limit: usize,
) -> Result<Vec<ClaudeMessage>, String> {
    let pattern = format!("%{query}%");
    let query_lower = query.to_lowercase();
    let scope = legacy_scope(layout);

    let mut stmt = conn
        .prepare(&format!(
            "SELECT r.id, r.conversation_id, r.prompt, r.response, r.model, r.datetime_utc \
             FROM responses r WHERE (r.prompt LIKE ?1 OR r.response LIKE ?1){scope} \
             ORDER BY r.datetime_utc DESC",
        ))
        .map_err(|e| e.to_string())?;

    let rows = stmt
        .query_map([&pattern], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        })
        .map_err(|e| e.to_string())?;

    let mut results = Vec::new();
    for row in rows.flatten() {
        if results.len() >= limit {
            break;
        }
        let (id, conv_id, prompt, response, model, dt) = row;
        let conv = conv_id.unwrap_or_else(|| NO_CONVERSATION.to_string());
        let ts = normalize_utc_ts(&dt.unwrap_or_default());
        push_hits(
            &mut results,
            limit,
            &query_lower,
            &conv,
            &ts,
            [
                (format!("{id}-user"), "user", prompt, None),
                (format!("{id}-asst"), "assistant", response, model),
            ],
        );
    }
    Ok(results)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, name TEXT, model TEXT);
             CREATE TABLE responses (id TEXT PRIMARY KEY, model TEXT, prompt TEXT, response TEXT, \
                 conversation_id TEXT, datetime_utc TEXT, input_tokens INTEGER, output_tokens INTEGER);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO conversations VALUES ('c1','Debug LOGIN','gpt-4')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO responses VALUES ('r1','gpt-4','why does LOGIN fail?','Checking auth','c1','2026-06-20T10:00:00',12,34)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO responses VALUES ('r2','gpt-4','more detail?','Here it is','c1','2026-06-20T10:01:00',5,8)",
            [],
        )
        .unwrap();
        // Orphan (no conversation).
        conn.execute(
            "INSERT INTO responses VALUES ('r3','gpt-4','one-shot question','one-shot answer',NULL,'2026-06-19T09:00:00',NULL,NULL)",
            [],
        )
        .unwrap();
        conn
    }

    #[test]
    fn scan_single_project_counts_messages() {
        let conn = fixture_db();
        let projects = scan_in_conn(&conn).unwrap();
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].name, "llm");
        assert_eq!(projects[0].path, "llm://__all__");
        // 1 named conversation + 1 orphan bucket.
        assert_eq!(projects[0].session_count, 2);
        // 3 responses * 2 messages each.
        assert_eq!(projects[0].message_count, 6);
        assert_eq!(projects[0].provider.as_deref(), Some("llm"));
    }

    #[test]
    fn load_sessions_groups_conversations_and_orphans() {
        let conn = fixture_db();
        let sessions = load_sessions_conn(&conn).unwrap();
        assert_eq!(sessions.len(), 2);
        let c1 = sessions
            .iter()
            .find(|s| s.actual_session_id == "c1")
            .unwrap();
        assert_eq!(c1.summary.as_deref(), Some("Debug LOGIN"));
        assert_eq!(c1.message_count, 4); // 2 responses * 2
        let orphan = sessions
            .iter()
            .find(|s| s.actual_session_id == NO_CONVERSATION)
            .unwrap();
        assert_eq!(orphan.summary.as_deref(), Some("Ungrouped prompts"));
        assert_eq!(orphan.message_count, 2);
    }

    #[test]
    fn load_messages_makes_user_assistant_pairs_with_usage() {
        let conn = fixture_db();
        let msgs = load_messages_conn(&conn, "c1").unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role.as_deref(), Some("user"));
        assert_eq!(msgs[0].uuid, "r1-user");
        assert_eq!(msgs[1].role.as_deref(), Some("assistant"));
        assert_eq!(msgs[1].uuid, "r1-asst");
        assert_eq!(msgs[1].model.as_deref(), Some("gpt-4"));
        let usage = msgs[1].usage.as_ref().unwrap();
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.output_tokens, Some(34));
    }

    #[test]
    fn load_messages_orphan_bucket_selects_null_conversation() {
        let conn = fixture_db();
        let msgs = load_messages_conn(&conn, NO_CONVERSATION).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].uuid, "r3-user");
        // Orphan response had NULL tokens -> no usage attached.
        assert!(msgs[1].usage.is_none());
    }

    #[test]
    fn search_matches_prompt_or_response_and_tags_project() {
        let conn = fixture_db();
        let results = search_conn(&conn, "LOGIN", 10).unwrap();
        assert_eq!(results.len(), 1); // only the prompt of r1 matches
        assert_eq!(results[0].project_name.as_deref(), Some("llm"));
        assert_eq!(results[0].role.as_deref(), Some("user"));

        let resp = search_conn(&conn, "one-shot answer", 10).unwrap();
        assert_eq!(resp.len(), 1);
        assert_eq!(resp[0].role.as_deref(), Some("assistant"));
    }

    #[test]
    fn search_hits_carry_scanned_project_path() {
        let conn = fixture_db();
        let projects = scan_in_conn(&conn).unwrap();
        let results = search_conn(&conn, "one-shot", 10).unwrap();
        assert_eq!(results.len(), 2);
        for hit in &results {
            assert_eq!(hit.project_path.as_ref(), Some(&projects[0].path));
        }
    }

    #[test]
    fn timestamps_are_normalized_to_utc() {
        assert_eq!(
            normalize_utc_ts("2026-06-20T10:00:00"),
            "2026-06-20T10:00:00Z"
        );
        assert_eq!(
            normalize_utc_ts("2026-06-20 10:00:00"),
            "2026-06-20T10:00:00Z"
        );
        // Already-zoned values pass through unchanged.
        assert_eq!(
            normalize_utc_ts("2026-06-20T10:00:00Z"),
            "2026-06-20T10:00:00Z"
        );
        assert_eq!(
            normalize_utc_ts("2026-06-20T10:00:00+09:00"),
            "2026-06-20T10:00:00+09:00"
        );
        assert_eq!(normalize_utc_ts(""), "");
        // llm >= 0.32 writes both shapes into one database, both with an
        // explicit offset: `threads.datetime_utc` has a space separator,
        // `turns.datetime_utc` a `T`.
        assert_eq!(
            normalize_utc_ts("2026-11-25 03:33:59.752893+00:00"),
            "2026-11-25T03:33:59.752893+00:00"
        );
        assert_eq!(
            normalize_utc_ts("2026-11-25T03:33:10.474738+00:00"),
            "2026-11-25T03:33:10.474738+00:00"
        );

        // The fixture's bare datetimes surface with a Z on real messages.
        let conn = fixture_db();
        let msgs = load_messages_conn(&conn, "c1").unwrap();
        assert!(msgs[0].timestamp.ends_with('Z'));
    }

    // ------------------------------------------------------------------
    // llm >= 0.32: threads / turns / messages / parts. The rows below are
    // records of a real llm 0.38 `logs.db` (`llm-ollama` 0.20, `llm-echo`
    // 0.5, written headless in a sandbox); message hashes, ids, tool-call
    // ids, timestamps and token counts are as written by llm. Producer:
    // `simonw/llm@46f0461d9b1ea9e898cac8a1d7333ab9bc6c8f7f`
    // `llm/logs.py` (`LogStore.log`, `_write_part`, `load_chain`).
    // ------------------------------------------------------------------

    /// The 0.38 DDL of the tables read here (foreign keys dropped).
    const STORE_DDL: &str = r"
        CREATE TABLE threads (id TEXT PRIMARY KEY, name TEXT, tip_message_hash TEXT,
            forked_from TEXT, datetime_utc TEXT);
        CREATE TABLE messages (hash TEXT PRIMARY KEY, parent_hash TEXT, role TEXT,
            provider_metadata TEXT);
        CREATE INDEX idx_messages_parent_hash ON messages (parent_hash);
        CREATE TABLE parts (id INTEGER PRIMARY KEY, message_hash TEXT, position INTEGER,
            type TEXT, tool_name TEXT, text TEXT, payload TEXT);
        CREATE UNIQUE INDEX idx_parts_message_hash_position ON parts (message_hash, position);
        CREATE TABLE turns (id TEXT PRIMARY KEY, thread_id TEXT, parent_message_hash TEXT,
            tip_message_hash TEXT, model TEXT, resolved_model TEXT, options_json TEXT,
            schema_id TEXT, input_tokens INTEGER, output_tokens INTEGER, token_details TEXT,
            duration_ms INTEGER, datetime_utc TEXT, response_json TEXT);
        CREATE INDEX idx_turns_thread_id ON turns (thread_id);
        CREATE TABLE turn_search (id INTEGER PRIMARY KEY, turn_id TEXT, prompt TEXT,
            response TEXT);
        CREATE TABLE fragments (id INTEGER PRIMARY KEY, hash TEXT, content TEXT,
            datetime_utc TEXT, source TEXT);
        CREATE TABLE attachments (id TEXT PRIMARY KEY, type TEXT, path TEXT, url TEXT,
            content BLOB);
    ";

    /// The legacy tables a fresh 0.38 database still creates, empty.
    const LEGACY_DDL: &str = "CREATE TABLE conversations (id TEXT PRIMARY KEY, name TEXT, model TEXT);
         CREATE TABLE responses (id TEXT PRIMARY KEY, model TEXT, prompt TEXT, response TEXT, \
             conversation_id TEXT, datetime_utc TEXT, input_tokens INTEGER, output_tokens INTEGER);";

    type PartSpec<'a> = (&'a str, Option<&'a str>, Option<&'a str>, Option<&'a str>);

    fn insert_message(
        conn: &Connection,
        hash: &str,
        parent: Option<&str>,
        role: &str,
        parts: &[PartSpec],
    ) {
        conn.execute(
            "INSERT INTO messages (hash, parent_hash, role) VALUES (?1, ?2, ?3)",
            rusqlite::params![hash, parent, role],
        )
        .unwrap();
        for (position, (kind, tool_name, text, payload)) in parts.iter().enumerate() {
            conn.execute(
                "INSERT INTO parts (message_hash, position, type, tool_name, text, payload) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    hash,
                    i64::try_from(position).unwrap(),
                    kind,
                    tool_name,
                    text,
                    payload
                ],
            )
            .unwrap();
        }
    }

    struct TurnSpec<'a> {
        id: &'a str,
        thread: &'a str,
        parent: &'a str,
        tip: &'a str,
        model: &'a str,
        tokens: (i64, i64),
        duration_ms: i64,
        datetime: &'a str,
    }

    fn insert_turn(conn: &Connection, t: &TurnSpec) {
        conn.execute(
            "INSERT INTO turns (id, thread_id, parent_message_hash, tip_message_hash, model, \
                 input_tokens, output_tokens, duration_ms, datetime_utc) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                t.id,
                t.thread,
                t.parent,
                t.tip,
                t.model,
                t.tokens.0,
                t.tokens.1,
                t.duration_ms,
                t.datetime
            ],
        )
        .unwrap();
    }

    fn insert_thread(conn: &Connection, id: &str, name: &str, tip: &str, datetime: &str) {
        conn.execute(
            "INSERT INTO threads (id, name, tip_message_hash, datetime_utc) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![id, name, tip, datetime],
        )
        .unwrap();
    }

    fn insert_search(conn: &Connection, turn_id: &str, prompt: &str, response: &str) {
        conn.execute(
            "INSERT INTO turn_search (turn_id, prompt, response) VALUES (?1, ?2, ?3)",
            rusqlite::params![turn_id, prompt, response],
        )
        .unwrap();
    }

    // Thread 1 of the sample: `llm -m echo -T llm_time '{"tool_calls":…}' --td`
    // then `llm -m echo -c --functions 'def boom…' '{"tool_calls":[boom]}' --td`.
    const ECHO_THREAD: &str = "01m88cgkykx9xne6hx6xgn43hk";
    const U_TIME: &str = "b2:721f23bbc9e38c48d469740cb41e611a";
    const A_TIME: &str = "b2:dd927276f1b483eaac0b5f8c75949bf8";
    const T_TIME: &str = "b2:5b3d482c3675e15db12261dc48e367d4";
    const A_TIME2: &str = "b2:62db736a1c003a70bb48d99f4f3e47fd";
    const U_BOOM: &str = "b2:dfbaf52972f8949a2f7c65c26fc422a1";
    const A_BOOM: &str = "b2:2a57e3ffffec6563dc1c64f001e2baf5";
    const T_BOOM: &str = "b2:33afe4936cfa2c5b568b724e930f3f02";
    const A_BOOM2: &str = "b2:2800cdb4f5aabbec933bbd941578abba";
    const TIME_CALL: &str = "tc_01m88cgkymh0eg56jesvy39233";
    const BOOM_CALL: &str = "tc_01m88cgqq19px5m0cnmq1a383j";

    const ECHO_FIRST: &str = "{\n  \"prompt\": \"\",\n  \"system\": \"\",\n  \"attachments\": [],\n  \"stream\": true,\n  \"previous\": []\n}";
    const ECHO_AFTER_TIME: &str = "{\n  \"prompt\": \"\",\n  \"system\": \"\",\n  \"attachments\": [],\n  \"stream\": true,\n  \"previous\": [\n    {\n      \"prompt\": \"{\\\"tool_calls\\\": [{\\\"name\\\": \\\"llm_time\\\", \\\"arguments\\\": {}}]}\"\n    }\n  ],\n  \"tool_results\": [\n    {\n      \"name\": \"llm_time\",\n      \"output\": \"{\\\"utc_time\\\": \\\"2026-11-25 03:38:12 UTC\\\"}\",\n      \"tool_call_id\": \"tc_01m88cgkymh0eg56jesvy39233\"\n    }\n  ]\n}";
    const ECHO_AFTER_BOOM: &str = "{\n  \"prompt\": \"\",\n  \"system\": \"\",\n  \"attachments\": [],\n  \"stream\": true,\n  \"previous\": [\n    {\n      \"prompt\": \"{\\\"tool_calls\\\": [{\\\"name\\\": \\\"boom\\\", \\\"arguments\\\": {\\\"x\\\": 1}}]}\"\n    }\n  ],\n  \"tool_results\": [\n    {\n      \"name\": \"boom\",\n      \"output\": \"Error: boom failed\",\n      \"tool_call_id\": \"tc_01m88cgqq19px5m0cnmq1a383j\"\n    }\n  ]\n}";
    const TIME_PROMPT: &str = r#"{"tool_calls": [{"name": "llm_time", "arguments": {}}]}"#;
    const BOOM_PROMPT: &str = r#"{"tool_calls": [{"name": "boom", "arguments": {"x": 1}}]}"#;

    fn insert_echo_thread(conn: &Connection) {
        let time_call =
            format!(r#"{{"name": "llm_time", "arguments": {{}}, "tool_call_id": "{TIME_CALL}"}}"#);
        let time_result = format!(
            r#"{{"name": "llm_time", "output": "{{\"utc_time\": \"2026-11-25 03:38:12 UTC\"}}", "tool_call_id": "{TIME_CALL}"}}"#
        );
        let boom_call = format!(
            r#"{{"name": "boom", "arguments": {{"x": 1}}, "tool_call_id": "{BOOM_CALL}"}}"#
        );
        // The producer writes `exception` only for a call that raised.
        let boom_result = format!(
            r#"{{"name": "boom", "output": "Error: boom failed", "tool_call_id": "{BOOM_CALL}", "exception": "ValueError: boom failed"}}"#
        );
        insert_message(
            conn,
            U_TIME,
            None,
            "user",
            &[("text", None, Some(TIME_PROMPT), None)],
        );
        insert_message(
            conn,
            A_TIME,
            Some(U_TIME),
            "assistant",
            &[
                ("text", None, Some(ECHO_FIRST), None),
                ("tool_call", Some("llm_time"), None, Some(&time_call)),
            ],
        );
        insert_message(
            conn,
            T_TIME,
            Some(A_TIME),
            "tool",
            &[("tool_result", Some("llm_time"), None, Some(&time_result))],
        );
        insert_message(
            conn,
            A_TIME2,
            Some(T_TIME),
            "assistant",
            &[("text", None, Some(ECHO_AFTER_TIME), None)],
        );
        insert_message(
            conn,
            U_BOOM,
            Some(A_TIME2),
            "user",
            &[("text", None, Some(BOOM_PROMPT), None)],
        );
        insert_message(
            conn,
            A_BOOM,
            Some(U_BOOM),
            "assistant",
            &[
                ("text", None, Some(ECHO_FIRST), None),
                ("tool_call", Some("boom"), None, Some(&boom_call)),
            ],
        );
        insert_message(
            conn,
            T_BOOM,
            Some(A_BOOM),
            "tool",
            &[("tool_result", Some("boom"), None, Some(&boom_result))],
        );
        insert_message(
            conn,
            A_BOOM2,
            Some(T_BOOM),
            "assistant",
            &[("text", None, Some(ECHO_AFTER_BOOM), None)],
        );
        insert_thread(
            conn,
            ECHO_THREAD,
            r#"{"tool_calls": [{"name": "llm_t…"#,
            A_BOOM2,
            "2026-11-25 03:38:12.820509+00:00",
        );
        for (id, parent, tip, tokens, datetime) in [
            (
                "01m88cgkymh0eg56jesvy39232",
                U_TIME,
                A_TIME,
                (5, 12),
                "2026-11-25T03:38:12.820036+00:00",
            ),
            (
                "01m88cgkypmcb5sepbcvgp7z2a",
                T_TIME,
                A_TIME2,
                (15, 46),
                "2026-11-25T03:38:12.822436+00:00",
            ),
            (
                "01m88cgqq19px5m0cnmq1a383h",
                U_BOOM,
                A_BOOM,
                (6, 12),
                "2026-11-25T03:38:16.673574+00:00",
            ),
            (
                "01m88cgqq4qcyn8m6pfrfbvmxk",
                T_BOOM,
                A_BOOM2,
                (3, 35),
                "2026-11-25T03:38:16.676067+00:00",
            ),
        ] {
            insert_turn(
                conn,
                &TurnSpec {
                    id,
                    thread: ECHO_THREAD,
                    parent,
                    tip,
                    model: "echo",
                    tokens,
                    duration_ms: 0,
                    datetime,
                },
            );
        }
        insert_search(conn, "01m88cgkymh0eg56jesvy39232", TIME_PROMPT, ECHO_FIRST);
        insert_search(conn, "01m88cgkypmcb5sepbcvgp7z2a", "", ECHO_AFTER_TIME);
        insert_search(conn, "01m88cgqq19px5m0cnmq1a383h", BOOM_PROMPT, ECHO_FIRST);
        insert_search(conn, "01m88cgqq4qcyn8m6pfrfbvmxk", "", ECHO_AFTER_BOOM);
    }

    // `llm -m ollama/qwen2.5-coder:7b -s "You are terse. …" "What does module.exports do in Node?"`
    const SYSTEM_THREAD: &str = "01m88cgwjjkfa6rrf7jd6vr5m0";

    fn insert_system_thread(conn: &Connection) {
        insert_message(
            conn,
            "b2:71ab06a59bb60708e295865ccb53a36b",
            None,
            "system",
            &[(
                "text",
                None,
                Some("You are terse. Answer in one sentence."),
                None,
            )],
        );
        insert_message(
            conn,
            "b2:405da203466b99d4034c35f3ddcd3532",
            Some("b2:71ab06a59bb60708e295865ccb53a36b"),
            "user",
            &[(
                "text",
                None,
                Some("What does module.exports do in Node?"),
                None,
            )],
        );
        insert_message(
            conn,
            "b2:a12cfaefad603b07eed3e7c0fc703e07",
            Some("b2:405da203466b99d4034c35f3ddcd3532"),
            "assistant",
            &[(
                "text",
                None,
                Some(
                    "It makes functions, objects, or variables available for use in other modules.",
                ),
                None,
            )],
        );
        insert_thread(
            conn,
            SYSTEM_THREAD,
            "What does module.exports do in …",
            "b2:a12cfaefad603b07eed3e7c0fc703e07",
            "2026-11-25 03:38:22.987377+00:00",
        );
        insert_turn(
            conn,
            &TurnSpec {
                id: "01m88cgwjkv06pvd9nhw0rv7ws",
                thread: SYSTEM_THREAD,
                parent: "b2:405da203466b99d4034c35f3ddcd3532",
                tip: "b2:a12cfaefad603b07eed3e7c0fc703e07",
                model: "ollama/qwen2.5-coder:7b",
                tokens: (30, 16),
                duration_ms: 1335,
                datetime: "2026-11-25T03:38:21.651198+00:00",
            },
        );
        insert_search(
            conn,
            "01m88cgwjkv06pvd9nhw0rv7ws",
            "What does module.exports do in Node?",
            "It makes functions, objects, or variables available for use in other modules.",
        );
    }

    // The same prompt run twice is one shared user message under two threads.
    const SHARED_USER: &str = "b2:5516d2caf22009fd17839997909bd74a";

    fn insert_shared_threads(conn: &Connection) {
        insert_message(
            conn,
            SHARED_USER,
            None,
            "user",
            &[(
                "text",
                None,
                Some("What time is it right now and which llm version is installed?"),
                None,
            )],
        );
        for (thread, id, tip, text, datetime) in [
            (
                "01m88cg7eyex7em4x981p6wxhh",
                "01m88cg7ez14s2zjgwb1a5zgyd",
                "b2:fbde77b7d035daafd13f2cbfd9a1ae72",
                "{\"name\": \"llm_time\", \"arguments\": {}}\n{\"name\": \"llm_version\", \"arguments\": {}}",
                "2026-11-25T03:38:00.031881+00:00",
            ),
            (
                "01m88cg8yyw15z1aqn1rx6pbaw",
                "01m88cg8yzh3h8s9p8kbk9j7ag",
                "b2:b7da624838eeaab8faba81f5d3b8859c",
                "[{\"name\": \"llm_time\", \"arguments\": {}}, {\"name\": \"llm_version\", \"arguments\": {}}]",
                "2026-11-25T03:38:01.567147+00:00",
            ),
        ] {
            insert_message(conn, tip, Some(SHARED_USER), "assistant", &[("text", None, Some(text), None)]);
            insert_thread(
                conn,
                thread,
                "What time is it right now and w…",
                tip,
                "2026-11-25 03:38:01.322216+00:00",
            );
            insert_turn(
                conn,
                &TurnSpec {
                    id,
                    thread,
                    parent: SHARED_USER,
                    tip,
                    model: "ollama/qwen2.5-coder:7b",
                    tokens: (197, 27),
                    duration_ms: 1289,
                    datetime,
                },
            );
        }
    }

    /// A store-only database: no legacy table at all.
    fn store_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(STORE_DDL).unwrap();
        insert_echo_thread(&conn);
        insert_system_thread(&conn);
        insert_shared_threads(&conn);
        conn
    }

    fn blocks(msg: &ClaudeMessage) -> &Vec<Value> {
        msg.content.as_ref().and_then(Value::as_array).unwrap()
    }

    #[test]
    fn store_threads_are_sessions_and_projects_count_them() {
        let conn = store_db();
        let projects = scan_in_conn(&conn).unwrap();
        assert_eq!(projects.len(), 1);
        // echo thread, system thread and the two threads sharing a user message.
        assert_eq!(projects[0].session_count, 4);
        // 8 + 3 + 2 + 2 messages: a shared message counts once per thread.
        assert_eq!(projects[0].message_count, 15);
        // The system thread was logged last.
        assert_eq!(
            projects[0].last_modified,
            "2026-11-25T03:38:21.651198+00:00"
        );

        let sessions = load_sessions_conn(&conn).unwrap();
        assert_eq!(sessions.len(), 4);
        // Newest first.
        assert_eq!(sessions[0].actual_session_id, SYSTEM_THREAD);
        assert_eq!(sessions[1].actual_session_id, ECHO_THREAD);
        let echo = &sessions[1];
        assert_eq!(echo.session_id, format!("llm://{ECHO_THREAD}"));
        assert_eq!(echo.message_count, 8);
        assert!(echo.has_tool_use);
        assert!(echo.has_errors);
        // Time spans the turns (start of the first, start of the last), not
        // the space-separated `threads.datetime_utc`.
        assert_eq!(echo.first_message_time, "2026-11-25T03:38:12.820036+00:00");
        assert_eq!(echo.last_message_time, "2026-11-25T03:38:16.676067+00:00");
        assert_eq!(
            echo.summary.as_deref(),
            Some(r#"{"tool_calls": [{"name": "llm_t…"#)
        );
        let system = sessions
            .iter()
            .find(|s| s.actual_session_id == SYSTEM_THREAD)
            .unwrap();
        assert_eq!(system.message_count, 3);
        assert!(!system.has_tool_use);
        assert!(!system.has_errors);
        assert_eq!(
            system.summary.as_deref(),
            Some("What does module.exports do in …")
        );
    }

    #[test]
    fn thread_transcript_pairs_tool_calls_with_results_and_flags_exceptions() {
        let conn = store_db();
        let msgs = load_messages_conn(&conn, ECHO_THREAD).unwrap();
        assert_eq!(msgs.len(), 8);
        let order: Vec<&str> = msgs.iter().map(|m| m.uuid.as_str()).collect();
        assert_eq!(
            order,
            [U_TIME, A_TIME, T_TIME, A_TIME2, U_BOOM, A_BOOM, T_BOOM, A_BOOM2]
        );
        assert!(msgs.iter().all(|m| m.session_id == ECHO_THREAD));

        // The assistant message holds its text part and the tool call.
        let call_msg = &msgs[1];
        assert_eq!(call_msg.role.as_deref(), Some("assistant"));
        assert_eq!(blocks(call_msg).len(), 2);
        assert_eq!(blocks(call_msg)[0]["type"], "text");
        assert_eq!(blocks(call_msg)[1]["type"], "tool_use");
        assert_eq!(blocks(call_msg)[1]["id"], TIME_CALL);
        assert_eq!(blocks(call_msg)[1]["name"], "llm_time");
        assert_eq!(blocks(call_msg)[1]["input"], json!({}));

        // A `tool` message is a result in the user lane, paired by id; no
        // `exception` key means the call succeeded.
        let ok = &msgs[2];
        assert_eq!(ok.role.as_deref(), Some("user"));
        assert_eq!(blocks(ok)[0]["type"], "tool_result");
        assert_eq!(blocks(ok)[0]["tool_use_id"], TIME_CALL);
        assert_eq!(blocks(ok)[0]["is_error"], false);
        assert_eq!(blocks(ok)[0]["is_error_source"], "producer");

        // The `exception` key of the payload is the producer's failure marker.
        assert_eq!(blocks(&msgs[5])[1]["input"], json!({"x": 1}));
        let failed = &blocks(&msgs[6])[0];
        assert_eq!(failed["tool_use_id"], BOOM_CALL);
        assert_eq!(failed["content"], "Error: boom failed");
        assert_eq!(failed["is_error"], true);
        assert_eq!(failed["is_error_source"], "producer");
    }

    #[test]
    fn thread_messages_take_time_model_and_usage_from_their_turn() {
        let conn = store_db();
        let msgs = load_messages_conn(&conn, ECHO_THREAD).unwrap();
        // The prompt belongs to the turn that answers it; the tool message
        // to the turn it starts.
        assert_eq!(msgs[0].timestamp, "2026-11-25T03:38:12.820036+00:00");
        assert_eq!(msgs[1].timestamp, "2026-11-25T03:38:12.820036+00:00");
        assert_eq!(msgs[2].timestamp, "2026-11-25T03:38:12.822436+00:00");
        assert_eq!(msgs[3].timestamp, "2026-11-25T03:38:12.822436+00:00");
        assert_eq!(msgs[7].timestamp, "2026-11-25T03:38:16.676067+00:00");

        assert_eq!(msgs[1].model.as_deref(), Some("echo"));
        let usage = msgs[1].usage.as_ref().unwrap();
        assert_eq!(usage.input_tokens, Some(5));
        assert_eq!(usage.output_tokens, Some(12));
        assert!(msgs[0].model.is_none() && msgs[0].usage.is_none());
        // `duration_ms` 0 is not a duration worth showing.
        assert!(msgs[1].duration_ms.is_none());

        let system = load_messages_conn(&conn, SYSTEM_THREAD).unwrap();
        assert_eq!(system.len(), 3);
        assert_eq!(system[0].message_type, "system");
        assert_eq!(system[0].role.as_deref(), Some("system"));
        assert_eq!(system[1].role.as_deref(), Some("user"));
        assert_eq!(system[2].duration_ms, Some(1335));
        assert_eq!(system[2].model.as_deref(), Some("ollama/qwen2.5-coder:7b"));
    }

    #[test]
    fn shared_messages_belong_to_each_thread_that_reaches_them() {
        let conn = store_db();
        let a = load_messages_conn(&conn, "01m88cg7eyex7em4x981p6wxhh").unwrap();
        let b = load_messages_conn(&conn, "01m88cg8yyw15z1aqn1rx6pbaw").unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
        assert_eq!(a[0].uuid, SHARED_USER);
        assert_eq!(b[0].uuid, SHARED_USER);
        assert_eq!(a[1].uuid, "b2:fbde77b7d035daafd13f2cbfd9a1ae72");
        assert_eq!(b[1].uuid, "b2:b7da624838eeaab8faba81f5d3b8859c");
        assert_ne!(a[1].timestamp, b[1].timestamp);
    }

    #[test]
    fn thread_without_turns_falls_back_to_the_thread_time() {
        let conn = store_db();
        conn.execute("DELETE FROM turns WHERE thread_id = ?1", [SYSTEM_THREAD])
            .unwrap();
        let sessions = load_sessions_conn(&conn).unwrap();
        let system = sessions
            .iter()
            .find(|s| s.actual_session_id == SYSTEM_THREAD)
            .unwrap();
        // `threads.datetime_utc` uses a space separator.
        assert_eq!(
            system.first_message_time,
            "2026-11-25T03:38:22.987377+00:00"
        );
        let msgs = load_messages_conn(&conn, SYSTEM_THREAD).unwrap();
        assert_eq!(msgs[0].timestamp, "2026-11-25T03:38:22.987377+00:00");
        assert!(msgs[2].usage.is_none());
    }

    #[test]
    fn store_search_anchors_hits_on_thread_messages() {
        let conn = store_db();
        let hits = search_conn(&conn, "boom", 10).unwrap();
        // The prompt of the boom turn and the response that echoes the failure.
        assert_eq!(hits.len(), 2);
        let user = hits
            .iter()
            .find(|h| h.role.as_deref() == Some("user"))
            .unwrap();
        assert_eq!(user.uuid, U_BOOM);
        assert_eq!(user.session_id, ECHO_THREAD);
        assert_eq!(user.project_name.as_deref(), Some("llm"));
        let projects = scan_in_conn(&conn).unwrap();
        assert_eq!(user.project_path.as_ref(), Some(&projects[0].path));
        let asst = hits
            .iter()
            .find(|h| h.role.as_deref() == Some("assistant"))
            .unwrap();
        assert_eq!(asst.uuid, A_BOOM2);
        assert_eq!(asst.model.as_deref(), Some("echo"));
        // The hit's uuid is a message of the loaded thread.
        let loaded = load_messages_conn(&conn, ECHO_THREAD).unwrap();
        assert!(loaded.iter().any(|m| m.uuid == user.uuid));
        assert!(loaded.iter().any(|m| m.uuid == asst.uuid));

        assert_eq!(search_conn(&conn, "BOOM", 1).unwrap().len(), 1);
    }

    #[test]
    fn database_with_both_generations_lists_a_conversation_once() {
        let conn = store_db();
        conn.execute_batch(LEGACY_DDL).unwrap();
        // The system thread continues a legacy conversation of the same id
        // (the thread holds its whole history); `c-old` is only legacy.
        conn.execute_batch(&format!(
            "INSERT INTO conversations VALUES ('{SYSTEM_THREAD}', 'Old name', 'gpt-4');
             INSERT INTO responses VALUES ('r1', 'gpt-4', 'legacy prompt', 'legacy answer',
                 '{SYSTEM_THREAD}', '2026-06-20T10:00:00', 1, 2);
             INSERT INTO conversations VALUES ('c-old', 'Old chat', 'gpt-4');
             INSERT INTO responses VALUES ('r2', 'gpt-4', 'old prompt', 'old answer',
                 'c-old', '2026-06-21T10:00:00', 1, 2);
             INSERT INTO responses VALUES ('r3', 'gpt-4', 'one-shot', 'one-shot answer',
                 NULL, '2026-06-22T10:00:00', 1, 2);"
        ))
        .unwrap();

        let sessions = load_sessions_conn(&conn).unwrap();
        let ids: Vec<&str> = sessions
            .iter()
            .map(|s| s.actual_session_id.as_str())
            .collect();
        // 4 threads + `c-old` + the orphan bucket; the shared id once.
        assert_eq!(sessions.len(), 6);
        assert_eq!(ids.iter().filter(|id| **id == SYSTEM_THREAD).count(), 1);
        assert!(ids.contains(&"c-old") && ids.contains(&NO_CONVERSATION));
        let system = sessions
            .iter()
            .find(|s| s.actual_session_id == SYSTEM_THREAD)
            .unwrap();
        assert_eq!(
            system.summary.as_deref(),
            Some("What does module.exports do in …")
        );
        assert_eq!(system.message_count, 3);

        let projects = scan_in_conn(&conn).unwrap();
        assert_eq!(projects[0].session_count, 6);
        // 15 thread messages + 2 responses * 2 (`c-old`, orphan).
        assert_eq!(projects[0].message_count, 19);
        // Sorted newest first across both generations: the legacy sessions
        // (June) come after every thread (November).
        let last_two: Vec<&str> = ids[ids.len() - 2..].to_vec();
        assert_eq!(last_two, [NO_CONVERSATION, "c-old"]);

        // The shared id loads the thread, not the legacy rows.
        let msgs = load_messages_conn(&conn, SYSTEM_THREAD).unwrap();
        assert_eq!(msgs.len(), 3);
        assert!(msgs.iter().all(|m| !m.uuid.starts_with("r1")));
        // Legacy-only sessions still load.
        assert_eq!(load_messages_conn(&conn, "c-old").unwrap().len(), 2);
        assert_eq!(load_messages_conn(&conn, NO_CONVERSATION).unwrap().len(), 2);

        // Search lists the shared conversation's text once, from the thread.
        let hits = search_conn(&conn, "legacy", 10).unwrap();
        assert!(hits.is_empty());
        let hits = search_conn(&conn, "answer", 10).unwrap();
        let sessions_hit: Vec<&str> = hits.iter().map(|h| h.session_id.as_str()).collect();
        assert!(sessions_hit.contains(&"c-old") && sessions_hit.contains(&NO_CONVERSATION));
    }

    #[test]
    fn fresh_database_with_empty_legacy_tables_shows_its_threads() {
        // llm 0.38 creates `conversations` / `responses` empty next to the store.
        let conn = store_db();
        conn.execute_batch(LEGACY_DDL).unwrap();
        let projects = scan_in_conn(&conn).unwrap();
        assert_eq!(projects[0].session_count, 4);
        assert_eq!(projects[0].message_count, 15);
        assert_eq!(load_sessions_conn(&conn).unwrap().len(), 4);
    }

    #[test]
    fn database_without_any_known_table_is_empty_not_an_error() {
        let conn = Connection::open_in_memory().unwrap();
        assert!(scan_in_conn(&conn).unwrap().is_empty());
        assert!(load_sessions_conn(&conn).unwrap().is_empty());
        assert!(load_messages_conn(&conn, "x").unwrap().is_empty());
        assert!(search_conn(&conn, "x", 5).unwrap().is_empty());
    }

    // The shapes below are not in the captured sample (it used no `-f`, no
    // attachment and no reasoning model); they follow the producer's
    // `Part.to_dict` / `_encode_text_refs` / `_write_part` in
    // `simonw/llm@46f0461d9b1ea9e898cac8a1d7333ab9bc6c8f7f`.
    #[test]
    fn fragment_reasoning_and_attachment_parts_are_resolved() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(STORE_DDL).unwrap();
        conn.execute_batch(
            "INSERT INTO fragments (id, content) VALUES (7, 'fn main() {}');
             INSERT INTO attachments (id, type, path) VALUES ('att1', 'image/png', '/home/user/demo/shot.png');",
        )
        .unwrap();
        insert_message(
            &conn,
            "b2:u",
            None,
            "user",
            &[
                (
                    "text",
                    None,
                    None,
                    Some(r#"{"text_ref": [{"literal": "Explain: "}, {"fragment": 7}]}"#),
                ),
                (
                    "attachment",
                    None,
                    None,
                    Some(r#"{"attachment": {"id": "att1"}}"#),
                ),
            ],
        );
        insert_message(
            &conn,
            "b2:a",
            Some("b2:u"),
            "assistant",
            &[
                ("reasoning", None, Some("thinking it over"), None),
                ("reasoning", None, Some(""), Some(r#"{"redacted": true}"#)),
                ("text", None, Some("Done."), None),
            ],
        );
        insert_thread(
            &conn,
            "t1",
            "Explain: fn main() {}",
            "b2:a",
            "2026-11-25 03:00:00+00:00",
        );
        insert_turn(
            &conn,
            &TurnSpec {
                id: "turn1",
                thread: "t1",
                parent: "b2:u",
                tip: "b2:a",
                model: "m",
                tokens: (1, 1),
                duration_ms: 10,
                datetime: "2026-11-25T03:00:00+00:00",
            },
        );
        let msgs = load_messages_conn(&conn, "t1").unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(blocks(&msgs[0])[0]["text"], "Explain: fn main() {}");
        assert_eq!(
            blocks(&msgs[0])[1]["text"],
            "[Attachment: image/png /home/user/demo/shot.png]"
        );
        // The redacted (empty) reasoning part is not a block.
        assert_eq!(blocks(&msgs[1]).len(), 2);
        assert_eq!(blocks(&msgs[1])[0]["type"], "thinking");
        assert_eq!(blocks(&msgs[1])[0]["thinking"], "thinking it over");
        assert_eq!(blocks(&msgs[1])[1]["text"], "Done.");
    }
}
