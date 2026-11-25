//! Crush provider (Charmbracelet's agentic coding TUI).
//!
//! Crush stores history **per project** in `<data_dir>/crush.db` (`SQLite`),
//! where `data_dir` is `<project>/.crush` unless the project was started with
//! `--data-dir` / `options.data_directory` or inherits an enclosing project's
//! `.crush`. There is no global history root, but Crush keeps a global
//! registry of every project it has opened in `<global data dir>/projects.json`
//! (`[{path, data_dir, last_accessed}]`). Discovery reads that registry first
//! (so projects outside the scanned roots, or with a custom data dir, are
//! found) and then adds a depth-limited scan of common code roots
//! (`~/{client,projects,code,src,dev,work,repos}` + `$HOME`) for
//! `.crush/crush.db`, mirroring the Aider provider.
//!
//! Tables: `sessions(id,parent_session_id,title,...,created_at,updated_at
//! INTEGER)` and `messages(id,session_id,role,parts TEXT,created_at,...)`.
//! `parts` is a JSON array of `{"type":..,"data":..}` items
//! (`text`/`tool_call`/`tool_result`/`shell_command`/`reasoning`/…); we map
//! them to the viewer's Claude-style content blocks.
//!
//! Sub-agent runs (`agent_dispatch`, `agentic_fetch`) are separate `sessions`
//! rows whose `parent_session_id` names the spawning session and whose id is
//! `<assistant message id>$$<tool call id>`. They are not listed as sessions
//! of their own; the `SubAgent` panel lists them through
//! [`load_child_sessions`], like Z Code's `subagent_child` rows.
//!
//! NOTE: because DBs are scattered, Crush is not tracked by the file watcher or
//! the `WebUI` session-path allowlist (the desktop IPC path works; the `WebUI`
//! headless server cannot read arbitrary per-project DBs).

use crate::commands::session::SearchFilters;
use crate::models::{ClaudeMessage, ClaudeProject, ClaudeSession};
use crate::providers::opencode::ChildSession;
use crate::providers::ProviderInfo;
use crate::utils::{build_provider_message, ms_to_iso, search_json_value_case_insensitive};
use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const PROVIDER: &str = "crush";
const SCHEME: &str = "crush://";
/// Separator between the project dir and session id in a session path.
const SESSION_SEP: char = '#';
/// Max `.crush/crush.db` files to discover (guards the recursive scan).
const MAX_DBS: usize = 200;
const MAX_DEPTH: usize = 4;

/// Detect a Crush installation (a registered project, else a shallow scan for
/// any `.crush/crush.db`).
pub fn detect() -> Option<ProviderInfo> {
    let dbs = discover_projects(1);
    let db = dbs.first()?;
    let base = db
        .db_path
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    Some(ProviderInfo {
        id: PROVIDER.to_string(),
        display_name: "Crush".to_string(),
        base_path: base,
        is_available: true,
    })
}

/// Crush has no single root (per-project DBs), so the watcher / `WebUI`
/// allowlist don't track it.
pub fn get_base_path() -> Option<String> {
    None
}

/// Scan Crush projects — one per discovered Crush database.
pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    // Up to MAX_DBS per-project DBs, each open parking up to 5s on a locked
    // DB — scanned on a bounded pool instead of stacking those waits
    // sequentially. A broken DB just yields None.
    let mut projects: Vec<ClaudeProject> =
        crate::utils::par_map_bounded(discover_projects(MAX_DBS), |db| scan_db(&db))
            .into_iter()
            .flatten()
            .collect();
    projects.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(projects)
}

/// One Crush database → one project, or `None` when the DB is unreadable or
/// empty.
fn scan_db(db: &CrushDb) -> Option<ClaudeProject> {
    let project_dir = db.project_dir.clone();
    let conn = open_db(&db.db_path).ok()?;
    let (session_count, message_count, last_modified) = project_stats(&conn).ok()?;
    if session_count == 0 {
        return None;
    }
    Some(ClaudeProject {
        name: project_name_for(&project_dir),
        path: project_path_for(&project_dir),
        actual_path: project_dir,
        session_count,
        message_count,
        last_modified,
        git_info: None,
        provider: Some(PROVIDER.to_string()),
        storage_type: Some("sqlite".to_string()),
        custom_directory_label: None,
    })
}

/// Load the sessions for one Crush project (`crush://<project_dir>`).
pub fn load_sessions(
    project_path: &str,
    _exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    let project_dir = project_path.strip_prefix(SCHEME).unwrap_or(project_path);
    let db_path = db_path_for(project_dir)?;
    let conn = open_db(&db_path)?;
    load_sessions_conn(&conn, project_dir)
}

fn load_sessions_conn(conn: &Connection, project_dir: &str) -> Result<Vec<ClaudeSession>, String> {
    let mut stmt = conn
        .prepare(&format!(
            "{} ORDER BY updated_at DESC, id",
            session_rows_sql(conn)?
        ))
        .map_err(|e| e.to_string())?;

    let project_name = project_name_for(project_dir);

    let sessions = stmt
        .query_map([], |row| {
            let id: String = row.get(0)?;
            let title: Option<String> = row.get(1)?;
            let created: Option<i64> = row.get(2)?;
            let updated: Option<i64> = row.get(3)?;
            let msg_cnt: usize = crate::providers::sql_unsigned(row, 4)?;
            Ok((id, title, created, updated, msg_cnt))
        })
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|(id, title, created, updated, msg_cnt)| {
            let created_iso = epoch_to_iso(created.unwrap_or(0));
            let updated_iso = epoch_to_iso(updated.unwrap_or(0));
            let session_path = session_path_for(project_dir, &id);
            ClaudeSession {
                session_id: session_path.clone(),
                actual_session_id: id.clone(),
                file_path: session_path,
                project_name: project_name.clone(),
                message_count: msg_cnt,
                first_message_time: created_iso.clone(),
                last_message_time: updated_iso.clone(),
                last_modified: updated_iso,
                has_tool_use: false,
                has_errors: false,
                summary: title.filter(|t| !t.trim().is_empty()).or(Some(id)),
                is_renamed: false,
                provider: Some(PROVIDER.to_string()),
                storage_type: Some("sqlite".to_string()),
                entrypoint: None,
            }
        })
        .collect();

    Ok(sessions)
}

/// Load messages for one Crush session (`crush://<project_dir>#<session_id>`).
pub fn load_messages(session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    let (project_dir, session_id) = parse_session_path(session_path)?;
    let db_path = db_path_for(&project_dir)?;
    let conn = open_db(&db_path)?;
    load_messages_conn(&conn, &session_id)
}

fn load_messages_conn(conn: &Connection, session_id: &str) -> Result<Vec<ClaudeMessage>, String> {
    // Insertion order is the conversation order. `created_at` has one-second
    // resolution and `id` is a random UUID, so ordering by them can put a
    // reply before its own prompt (and scrambles tool turns inside one
    // second); `rowid` is assigned as Crush appends each message.
    let mut stmt = conn
        .prepare(
            "SELECT id, role, parts, created_at FROM messages \
             WHERE session_id = ?1 ORDER BY rowid",
        )
        .map_err(|e| e.to_string())?;

    let rows = stmt
        .query_map([session_id], |row| {
            let id: String = row.get(0)?;
            let role: String = row.get(1)?;
            let parts: String = row.get(2)?;
            let created: Option<i64> = row.get(3)?;
            Ok((id, role, parts, created))
        })
        .map_err(|e| e.to_string())?;

    let mut messages = Vec::new();
    for row in rows.flatten() {
        let (id, role, parts, created) = row;
        if let Some(msg) = build_message(session_id, &id, &role, &parts, created.unwrap_or(0)) {
            messages.push(msg);
        }
    }
    Ok(messages)
}

/// The sub-agent runs `session_path` (`crush://<project_dir>#<session_id>`)
/// spawned: `sessions` rows whose `parent_session_id` is the session, oldest
/// first. Crush keeps them in its database rather than as sidechain files, so
/// the `SubAgent` panel reads them from there. A deeper run is listed by its
/// own parent. Empty when the store predates `parent_session_id`.
pub fn load_child_sessions(session_path: &str) -> Result<Vec<ChildSession>, String> {
    let (project_dir, session_id) = parse_session_path(session_path)?;
    let conn = open_db(&db_path_for(&project_dir)?)?;
    load_child_sessions_conn(&conn, &session_id)
}

fn load_child_sessions_conn(
    conn: &Connection,
    session_id: &str,
) -> Result<Vec<ChildSession>, String> {
    if !has_parent_session_id(conn)? {
        return Ok(Vec::new());
    }
    let mut stmt = conn
        .prepare(
            "SELECT s.id, s.title, s.created_at, s.updated_at, \
                    (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) \
             FROM sessions s \
             WHERE s.parent_session_id = ?1 AND s.id != ?1 \
             ORDER BY s.created_at, s.rowid",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([session_id], |row| {
            Ok(ChildSession {
                id: row.get(0)?,
                title: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                created_at: epoch_to_iso(row.get::<_, Option<i64>>(2)?.unwrap_or(0)),
                updated_at: epoch_to_iso(row.get::<_, Option<i64>>(3)?.unwrap_or(0)),
                message_count: crate::providers::sql_unsigned(row, 4)?,
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}

/// The tool call that spawned a sub-agent session. Crush names that session
/// `<assistant message id>$$<tool call id>`
/// (`internal/session/session.go` `CreateAgentToolSessionID` /
/// `ParseAgentToolSessionID`, exactly two `$$`-separated parts).
pub fn spawning_tool_call_id(child_session_id: &str) -> Option<&str> {
    let (message_id, tool_call_id) = child_session_id.split_once("$$")?;
    (!message_id.is_empty() && !tool_call_id.is_empty() && !tool_call_id.contains("$$"))
        .then_some(tool_call_id)
}

/// Search across every discovered Crush DB.
pub fn search(query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    Ok(search_with_filters(
        query,
        limit,
        &SearchFilters::parse(&Value::Null),
    ))
}

pub(crate) fn search_with_filters(
    query: &str,
    limit: usize,
    filters: &SearchFilters<'_>,
) -> Vec<ClaudeMessage> {
    if query.is_empty() || limit == 0 {
        return Vec::new();
    }
    let pattern = format!("%{query}%");
    let query_lower = query.to_lowercase();
    let mut results = Vec::new();

    for db in discover_projects(MAX_DBS) {
        if results.len() >= limit {
            break;
        }
        let project_dir = db.project_dir;
        let db_path = db.db_path;
        let project_name = project_name_for(&project_dir);
        let project_path = project_path_for(&project_dir);
        if !filters.matches_project(Some(&project_path), Some(&project_name)) {
            continue;
        }
        let Ok(conn) = open_db(&db_path) else {
            continue;
        };
        search_one_db(
            &conn,
            &pattern,
            &query_lower,
            (&project_path, &project_name),
            limit,
            filters,
            &mut results,
        );
    }
    results
}

fn search_one_db(
    conn: &Connection,
    pattern: &str,
    query_lower: &str,
    (project_path, project_name): (&str, &str),
    limit: usize,
    filters: &SearchFilters<'_>,
    results: &mut Vec<ClaudeMessage>,
) {
    // Sub-agent runs are not listed as sessions, so a hit inside one could not
    // be opened from the results.
    let Ok(sub_agent) = sub_agent_condition(conn) else {
        return;
    };
    let hidden = sub_agent
        .map(|cond| format!(" AND session_id NOT IN (SELECT s.id FROM sessions s WHERE {cond})"))
        .unwrap_or_default();
    let Ok(mut stmt) = conn.prepare(&format!(
        "SELECT id, session_id, role, parts, created_at FROM messages WHERE parts LIKE ?1{hidden}"
    )) else {
        return;
    };
    let Ok(rows) = stmt.query_map([pattern], |row| {
        let id: String = row.get(0)?;
        let session_id: String = row.get(1)?;
        let role: String = row.get(2)?;
        let parts: String = row.get(3)?;
        let created: Option<i64> = row.get(4)?;
        Ok((id, session_id, role, parts, created))
    }) else {
        return;
    };
    for row in rows.flatten() {
        if results.len() >= limit {
            return;
        }
        // Match the listed actual_session_id and the loaded message's session id.
        let (id, session_id, role, parts, created) = row;
        let Some(mut msg) = build_message(&session_id, &id, &role, &parts, created.unwrap_or(0))
        else {
            continue;
        };
        let matched = msg
            .content
            .as_ref()
            .map(|c| search_json_value_case_insensitive(c, query_lower))
            .unwrap_or(false);
        if matched && filters.matches_message(&msg) {
            msg.project_name = Some(project_name.to_string());
            msg.project_path = Some(project_path.to_string());
            results.push(msg);
        }
    }
}

// ============================================================================
// Discovery
// ============================================================================

/// Common code roots to scan (mirrors the Aider provider).
fn search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(home) = crate::utils::home_dir() {
        for subdir in ["client", "projects", "code", "src", "dev", "work", "repos"] {
            let d = home.join(subdir);
            if d.is_dir() {
                dirs.push(d);
            }
        }
        dirs.push(home);
    }
    dirs
}

/// A Crush project and the database holding its history.
#[derive(Debug, Clone)]
struct CrushDb {
    /// The project's working directory (the registry's `path`).
    project_dir: String,
    /// `<data_dir>/crush.db`.
    db_path: PathBuf,
}

/// One `projects.json` entry: a project Crush has opened.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RegisteredProject {
    path: String,
    /// Where this project's `crush.db` lives; relative values are relative to
    /// `path`.
    data_dir: String,
}

/// Find up to `max` Crush projects: those in Crush's own `projects.json`
/// registry (most recently opened first), then any other `.crush/crush.db`
/// under the common code roots. A project appears once, and so does a
/// database several registry entries share.
fn discover_projects(max: usize) -> Vec<CrushDb> {
    let registry = read_registry();
    let mut found: Vec<CrushDb> = Vec::new();
    let mut seen_dirs: HashSet<String> = HashSet::new();
    let mut seen_dbs: HashSet<PathBuf> = HashSet::new();
    let mut add = |found: &mut Vec<CrushDb>, project_dir: String| {
        if found.len() >= max || seen_dirs.contains(&project_dir) {
            return;
        }
        let Some(db_path) = resolve_db(&project_dir, &registry) else {
            return;
        };
        let identity = fs::canonicalize(&db_path).unwrap_or_else(|_| db_path.clone());
        if seen_dbs.insert(identity) {
            seen_dirs.insert(project_dir.clone());
            found.push(CrushDb {
                project_dir,
                db_path,
            });
        }
    };

    for project in &registry {
        add(&mut found, project.path.clone());
    }
    if found.len() < max {
        let mut scanned = Vec::new();
        for root in search_dirs() {
            if scanned.len() >= max {
                break;
            }
            find_crush_db(&root, &mut scanned, max, 0);
        }
        scanned.sort();
        scanned.dedup();
        for db in scanned {
            if let Some(project_dir) = project_dir_of(&db) {
                add(&mut found, project_dir);
            }
        }
    }
    found
}

/// Crush's global data directory, which holds `projects.json`: `$CRUSH_GLOBAL_DATA`,
/// else `$XDG_DATA_HOME/crush`, else `%LOCALAPPDATA%\crush` on Windows or
/// `~/.local/share/crush` elsewhere (`internal/config/load.go`
/// `GlobalConfigData`; `internal/projects/projects.go` `projectsFilePath`).
fn global_data_dir() -> Option<PathBuf> {
    // Tests never read the developer's real environment (or registry).
    let env = |key: &str| {
        if cfg!(test) {
            None
        } else {
            std::env::var_os(key).filter(|v| !v.is_empty())
        }
    };
    global_data_dir_with(
        env("CRUSH_GLOBAL_DATA"),
        env("XDG_DATA_HOME"),
        if cfg!(windows) {
            Some(env("LOCALAPPDATA").map_or_else(
                || {
                    PathBuf::from(env("USERPROFILE").unwrap_or_default())
                        .join("AppData")
                        .join("Local")
                },
                PathBuf::from,
            ))
        } else {
            None
        },
        crate::utils::home_dir(),
    )
}

/// [`global_data_dir`] with the environment passed in: `local_app_data` is
/// only given on Windows.
fn global_data_dir_with(
    crush_global_data: Option<std::ffi::OsString>,
    xdg_data_home: Option<std::ffi::OsString>,
    local_app_data: Option<PathBuf>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    if let Some(dir) = crush_global_data {
        return Some(PathBuf::from(dir));
    }
    if let Some(xdg) = xdg_data_home {
        return Some(PathBuf::from(xdg).join("crush"));
    }
    if let Some(local) = local_app_data {
        return Some(local.join("crush"));
    }
    home.map(|h| h.join(".local").join("share").join("crush"))
}

/// Crush's `projects.json` registry, most recently opened first; empty when
/// there is none or it is unreadable.
fn read_registry() -> Vec<RegisteredProject> {
    let Some(file) = global_data_dir().map(|dir| dir.join("projects.json")) else {
        return Vec::new();
    };
    if is_symlink(&file) {
        return Vec::new();
    }
    fs::read_to_string(&file)
        .map(|text| parse_registry(&text))
        .unwrap_or_default()
}

/// `{"projects":[{"path","data_dir","last_accessed"}]}`
/// (`internal/projects/projects.go` `Project`). Entries without a path are
/// skipped; a missing `data_dir` means the default `<path>/.crush`.
fn parse_registry(text: &str) -> Vec<RegisteredProject> {
    let Ok(Value::Object(root)) = serde_json::from_str::<Value>(text) else {
        return Vec::new();
    };
    let Some(Value::Array(projects)) = root.get("projects") else {
        return Vec::new();
    };
    projects
        .iter()
        .filter_map(|p| {
            let path = p.get("path").and_then(Value::as_str)?.trim();
            if path.is_empty() {
                return None;
            }
            let data_dir = p
                .get("data_dir")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|d| !d.is_empty())
                .map_or_else(|| ".crush".to_string(), str::to_string);
            Some(RegisteredProject {
                path: path.to_string(),
                data_dir,
            })
        })
        .collect()
}

/// The database of `project_dir`: the registered `data_dir`'s `crush.db`,
/// else the default `<project_dir>/.crush/crush.db`. Only regular files
/// count, never a symlink.
fn resolve_db(project_dir: &str, registry: &[RegisteredProject]) -> Option<PathBuf> {
    let project = Path::new(project_dir);
    registry
        .iter()
        .filter(|p| p.path == project_dir)
        .map(|p| project.join(&p.data_dir).join("crush.db"))
        .chain(std::iter::once(project.join(".crush").join("crush.db")))
        .find(|db| db.is_file() && !is_symlink(db))
}

fn find_crush_db(dir: &Path, results: &mut Vec<PathBuf>, max: usize, depth: usize) {
    if depth > MAX_DEPTH || results.len() >= max || is_symlink(dir) {
        return;
    }
    let db = dir.join(".crush").join("crush.db");
    if db.is_file() && !is_symlink(&db) {
        results.push(db);
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if results.len() >= max {
            return;
        }
        let path = entry.path();
        if !path.is_dir() || is_symlink(&path) {
            continue;
        }
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if name.starts_with('.')
            || name == "node_modules"
            || name == "target"
            || name == "dist"
            || name == "build"
        {
            continue;
        }
        find_crush_db(&path, results, max, depth + 1);
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn project_dir_of(db_path: &Path) -> Option<String> {
    // <project>/.crush/crush.db -> <project>
    db_path
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_string_lossy().to_string())
}

/// Display name of a project dir, shared by scan and search.
fn project_name_for(project_dir: &str) -> String {
    Path::new(project_dir)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| project_dir.to_string())
}

/// Opaque project identity of a project dir, shared by scan and search.
fn project_path_for(project_dir: &str) -> String {
    format!("{SCHEME}{project_dir}")
}

fn session_path_for(project_dir: &str, session_id: &str) -> String {
    format!("{SCHEME}{project_dir}{SESSION_SEP}{session_id}")
}

fn db_path_for(project_dir: &str) -> Result<PathBuf, String> {
    resolve_db(project_dir, &read_registry())
        .ok_or_else(|| format!("Crush DB not found for project: {project_dir}"))
}

fn parse_session_path(session_path: &str) -> Result<(String, String), String> {
    let rest = session_path.strip_prefix(SCHEME).unwrap_or(session_path);
    rest.rsplit_once(SESSION_SEP)
        .map(|(dir, id)| (dir.to_string(), id.to_string()))
        .ok_or_else(|| format!("Invalid Crush session path: {session_path}"))
}

fn open_db(path: &Path) -> Result<Connection, String> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("Failed to open Crush DB: {e}"))?;
    conn.busy_timeout(std::time::Duration::from_secs(5))
        .map_err(|e| format!("Failed to set busy timeout: {e}"))?;
    Ok(conn)
}

/// A sub-agent run: a session with a parent session. The parent must exist, so
/// a run whose parent was deleted is still listed rather than lost. The
/// condition is over `sessions s`; `None` when the store has no
/// `parent_session_id` column.
fn sub_agent_condition(conn: &Connection) -> Result<Option<&'static str>, String> {
    Ok(has_parent_session_id(conn)?.then_some(
        "COALESCE(s.parent_session_id, '') <> '' \
         AND EXISTS (SELECT 1 FROM sessions p WHERE p.id = s.parent_session_id)",
    ))
}

/// Whether `sessions` has a `parent_session_id` column.
fn has_parent_session_id(conn: &Connection) -> Result<bool, String> {
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM pragma_table_info('sessions') WHERE name = 'parent_session_id')",
        [],
        |row| row.get::<_, bool>(0),
    )
    .map_err(|e| e.to_string())
}

/// Session metadata is authoritative when present. Recover an otherwise
/// unlisted session from its messages, using their time range and count.
/// Sub-agent runs are left out: the `SubAgent` panel lists them under the
/// session that spawned them ([`load_child_sessions`]).
fn session_rows_sql(conn: &Connection) -> Result<String, String> {
    let has_sessions = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sessions')",
            [],
            |row| row.get::<_, bool>(0),
        )
        .map_err(|e| e.to_string())?;
    Ok(if has_sessions {
        let top_level = sub_agent_condition(conn)?
            .map(|cond| format!("WHERE NOT ({cond})"))
            .unwrap_or_default();
        format!(
            "WITH message_sessions AS (
                 SELECT session_id AS id, MIN(created_at) AS created_at,
                        MAX(created_at) AS updated_at, COUNT(*) AS msg_count
                 FROM messages GROUP BY session_id
             )
             SELECT s.id AS id, s.title AS title, s.created_at AS created_at,
                    s.updated_at AS updated_at, COALESCE(m.msg_count, 0) AS msg_count
             FROM sessions s LEFT JOIN message_sessions m ON m.id = s.id
             {top_level}
             UNION ALL
             SELECT m.id, NULL, m.created_at, m.updated_at, m.msg_count
             FROM message_sessions m WHERE NOT EXISTS (SELECT 1 FROM sessions s WHERE s.id = m.id)"
        )
    } else {
        "SELECT session_id AS id, NULL AS title, MIN(created_at) AS created_at,
                MAX(created_at) AS updated_at, COUNT(*) AS msg_count
         FROM messages GROUP BY session_id"
            .to_string()
    })
}

fn project_stats(conn: &Connection) -> Result<(usize, usize, String), String> {
    conn.query_row(
        &format!(
            "SELECT COUNT(*), COALESCE(SUM(msg_count), 0), MAX(updated_at) FROM ({})",
            session_rows_sql(conn)?
        ),
        [],
        |row| {
            let last: Option<i64> = row.get(2)?;
            Ok((
                crate::providers::sql_unsigned(row, 0)?,
                crate::providers::sql_unsigned(row, 1)?,
                epoch_to_iso(last.unwrap_or(0)),
            ))
        },
    )
    .map_err(|e| e.to_string())
}

fn is_symlink(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
}

/// Convert an epoch timestamp (seconds or ms) to RFC3339.
fn epoch_to_iso(n: i64) -> String {
    if n <= 0 {
        return String::new();
    }
    let ms = if n > 1_000_000_000_000 { n } else { n * 1000 };
    ms_to_iso(ms as u64)
}

// ============================================================================
// Message conversion
// ============================================================================

fn build_message(
    session_id: &str,
    msg_id: &str,
    role: &str,
    parts_json: &str,
    created: i64,
) -> Option<ClaudeMessage> {
    // Tool results live on "tool" role rows; surface them in the user lane
    // (Claude convention) so the tool_result renderer picks them up.
    let mapped_role = match role {
        "assistant" => "assistant",
        "system" => "system",
        "user" | "tool" => "user",
        _ => return None,
    };
    let uuid = if msg_id.is_empty() {
        format!("{session_id}-{role}")
    } else {
        msg_id.to_string()
    };
    let blocks = map_parts(parts_json, &uuid);
    if blocks.is_empty() {
        return None;
    }
    Some(build_provider_message(
        PROVIDER,
        uuid,
        session_id,
        epoch_to_iso(created),
        mapped_role,
        Some(mapped_role),
        Some(Value::Array(blocks)),
        None,
    ))
}

/// Map the Crush `parts` JSON array into Claude-style content blocks.
/// `message_uuid` names the message, for the call ids of parts that carry none.
fn map_parts(parts_json: &str, message_uuid: &str) -> Vec<Value> {
    let Ok(Value::Array(items)) = serde_json::from_str::<Value>(parts_json) else {
        return Vec::new();
    };
    let mut blocks = Vec::new();
    for (index, item) in items.iter().enumerate() {
        if item.get("type").and_then(Value::as_str) == Some("shell_command") {
            blocks.extend(shell_command_blocks(
                item.get("data"),
                &format!("{message_uuid}-shell-{index}"),
            ));
        } else {
            blocks.extend(map_part(item));
        }
    }
    blocks
}

/// A bang-mode command the user ran in the TUI (`internal/message/content.go`
/// `ShellCommand`: `command`, `output`, `exit_code`), shown as a `bash` call
/// and its result. Crush writes no call id for it, so `call_id` is minted. A
/// non-zero `exit_code` is a recorded failure; zero is not asserted as success.
fn shell_command_blocks(data: Option<&Value>, call_id: &str) -> Vec<Value> {
    let Some(d) = data else {
        return Vec::new();
    };
    let command = d.get("command").and_then(Value::as_str).unwrap_or("");
    let output = d.get("output").and_then(Value::as_str).unwrap_or("");
    if command.is_empty() && output.is_empty() {
        return Vec::new();
    }
    let mut result = json!({
        "type": "tool_result", "tool_use_id": call_id, "content": output
    });
    if let Some(exit) = d.get("exit_code").and_then(Value::as_i64) {
        result["toolUseResult"] = json!({ "exit_code": exit });
        if exit != 0 {
            result["is_error"] = json!(true);
            result["is_error_source"] = json!("derived");
        }
    }
    vec![
        json!({
            "type": "tool_use", "id": call_id, "name": "bash", "input": { "command": command }
        }),
        result,
    ]
}

fn map_part(part: &Value) -> Option<Value> {
    let kind = part.get("type").and_then(Value::as_str)?;
    let data = part.get("data");
    match kind {
        "text" => {
            let text = data
                .and_then(|d| d.get("text"))
                .and_then(Value::as_str)
                .unwrap_or("");
            if text.is_empty() {
                None
            } else {
                Some(json!({ "type": "text", "text": text }))
            }
        }
        "reasoning" => {
            let thinking = data
                .and_then(|d| d.get("thinking"))
                .and_then(Value::as_str)
                .unwrap_or("");
            if thinking.is_empty() {
                None
            } else {
                Some(json!({
                    "type": "thinking",
                    "thinking": thinking,
                    "signature": data.and_then(|d| d.get("signature")).and_then(Value::as_str).unwrap_or("")
                }))
            }
        }
        "tool_call" => {
            let d = data?;
            let id = d.get("id").and_then(Value::as_str).unwrap_or("");
            let name = d.get("name").and_then(Value::as_str).unwrap_or("unknown");
            // `input` is a raw JSON string; parse it back to an object when possible.
            let input = d
                .get("input")
                .and_then(Value::as_str)
                .and_then(|s| serde_json::from_str::<Value>(s).ok())
                .or_else(|| d.get("input").cloned())
                .unwrap_or_else(|| json!({}));
            Some(json!({ "type": "tool_use", "id": id, "name": name, "input": input }))
        }
        "tool_result" => {
            let d = data?;
            let id = d.get("tool_call_id").and_then(Value::as_str).unwrap_or("");
            let content = d.get("content").and_then(Value::as_str).unwrap_or("");
            let mut block = json!({
                "type": "tool_result",
                "tool_use_id": id,
                "content": tool_result_content(d, content)
            });
            super::result_evidence::copy_error_flag(&mut block, d, "is_error");
            Some(block)
        }
        // finish / image_url / binary: not rendered.
        _ => None,
    }
}

/// A tool result's content. An image result (e.g. `view` of a PNG) keeps its
/// text in `content` and its base64 payload in `data` with the media type in
/// `mime_type` (`internal/message/content.go` `ToolResult`); it becomes a text
/// block plus an image block. Other payloads stay text only.
fn tool_result_content(result: &Value, text: &str) -> Value {
    let data = result.get("data").and_then(Value::as_str).unwrap_or("");
    let mime = result
        .get("mime_type")
        .and_then(Value::as_str)
        .unwrap_or("");
    if data.is_empty() || !mime.starts_with("image/") {
        return Value::String(text.to_string());
    }
    let mut blocks = Vec::new();
    if !text.is_empty() {
        blocks.push(json!({ "type": "text", "text": text }));
    }
    blocks.push(json!({
        "type": "image",
        "source": { "type": "base64", "media_type": mime, "data": data }
    }));
    Value::Array(blocks)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::SandboxHome;
    use tempfile::TempDir;

    // Crush 0.102.0 `sessions` / `messages` tables (`rowid` is implicit:
    // `id` is a text primary key).
    const SCHEMA: &str = "
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            parent_session_id TEXT,
            title TEXT NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0 CHECK (message_count >= 0),
            prompt_tokens INTEGER NOT NULL DEFAULT 0 CHECK (prompt_tokens >= 0),
            completion_tokens INTEGER NOT NULL DEFAULT 0 CHECK (completion_tokens >= 0),
            cost REAL NOT NULL DEFAULT 0.0 CHECK (cost >= 0.0),
            updated_at INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            summary_message_id TEXT, todos TEXT, channel TEXT
        );
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            parts TEXT NOT NULL default '[]',
            model TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            finished_at INTEGER,
            provider TEXT,
            is_summary_message INTEGER DEFAULT 0 NOT NULL,
            model_name TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        );";

    // Session ids from a real Crush 0.102.0 store. The sub-agent session id is
    // `<assistant message id>$$<tool call id>`.
    const MAIN: &str = "7d0c796d-bf08-46e3-a88e-3742553407d8";
    const IMAGE: &str = "7ed900e4-805e-4d90-834d-27f9f4c7c0bd";
    const PARENT: &str = "956a8b87-8531-43e9-9ee4-6ff155f9a29b";
    const CHILD: &str = "24ecc5a7-0f0c-4e0e-b40d-7b761f8a3b9a$$call_ff2c87f7";
    const SHELL: &str = "79a2fb52-1e11-4984-9423-b05b43c85a42";
    const PROJECT: &str = "/home/user/demo";

    fn finish(reason: &str, time: i64) -> Value {
        json!({"type":"finish","data":{"reason":reason,"time":time}})
    }

    fn text(text: &str) -> Value {
        json!({"type":"text","data":{"text":text}})
    }

    fn tool_call(id: &str, name: &str, input: &str) -> Value {
        json!({"type":"tool_call","data":{
            "id":id,"name":name,"input":input,"provider_executed":false,"finished":true,
            "started_at_ms":1_795_577_949_633_i64}})
    }

    fn tool_result(
        call_id: &str,
        name: &str,
        content: &str,
        (data, mime): (&str, &str),
        is_error: bool,
    ) -> Value {
        json!({"type":"tool_result","data":{
            "tool_call_id":call_id,"name":name,"content":content,"data":data,"mime_type":mime,
            "metadata":"","is_error":is_error,"finished_at_ms":1_795_577_949_634_i64}})
    }

    fn shell_command(command: &str, output: &str, exit_code: i64) -> Value {
        json!({"type":"shell_command","data":{
            "command":command,"output":output,"exit_code":exit_code}})
    }

    fn add_session(conn: &Connection, id: &str, parent: Option<&str>, title: &str, created: i64) {
        conn.execute(
            "INSERT INTO sessions (id, parent_session_id, title, updated_at, created_at)
             VALUES (?1, ?2, ?3, ?4, ?4)",
            rusqlite::params![id, parent, title, created],
        )
        .unwrap();
    }

    /// Insert in call order: the `rowid` is the insertion order, like Crush's.
    fn add_message(
        conn: &Connection,
        id: &str,
        session: &str,
        role: &str,
        created: i64,
        parts: &[Value],
    ) {
        let mut parts = parts.to_vec();
        parts.push(finish(
            if role == "assistant" {
                "end_turn"
            } else {
                "stop"
            },
            0,
        ));
        conn.execute(
            "INSERT INTO messages (id, session_id, role, parts, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
            rusqlite::params![id, session, role, Value::Array(parts).to_string(), created],
        )
        .unwrap();
    }

    /// Records of a real Crush session store, trimmed to the minimum and with
    /// the sandbox path replaced by `/home/user/demo`. Rows are inserted in the
    /// order Crush wrote them, so a sub-agent's rows sit between its parent's.
    fn fixture_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(SCHEMA).unwrap();

        // Real model session: the user prompt and the reply share one second,
        // and the reply's random id sorts before the prompt's.
        add_session(
            &conn,
            MAIN,
            None,
            "Add multiplication function to math.js",
            1_795_577_637,
        );
        add_message(
            &conn,
            "9cfe4f86-1569-4faf-aa05-16b5e367ae8d",
            MAIN,
            "user",
            1_795_577_637,
            &[text(
                "Read src/math.js and add a function mul(a, b) that returns a*b, exported like add.",
            )],
        );
        add_message(
            &conn,
            "945b8603-34f5-4a9f-9be5-5dfc3eaac296",
            MAIN,
            "assistant",
            1_795_577_637,
            &[text(
                "[uses view tool]\n```javascript\nexport function add(a, b) {\n    return a + b;\n}\n```",
            )],
        );

        // `view` of a PNG.
        add_session(&conn, IMAGE, None, "Demo session", 1_795_577_949);
        add_message(
            &conn,
            "0e7d9248-0f4b-4402-9686-6914a79f281f",
            IMAGE,
            "user",
            1_795_577_949,
            &[text("Show the logo.png file.")],
        );
        add_message(
            &conn,
            "efdbfb25-9976-4c22-9b36-762670931817",
            IMAGE,
            "assistant",
            1_795_577_949,
            &[tool_call(
                "call_17fb6ad7",
                "view",
                r#"{"file_path": "/home/user/demo/logo.png"}"#,
            )],
        );
        add_message(
            &conn,
            "05d433d4-f52d-4026-b43f-48e94b5d0748",
            IMAGE,
            "tool",
            1_795_577_949,
            &[tool_result(
                "call_17fb6ad7",
                "view",
                "Loaded image/png content",
                (
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
                    "image/png",
                ),
                false,
            )],
        );

        // `agent_dispatch`: the sub-agent run is its own session whose rows are
        // interleaved with its parent's.
        add_session(&conn, PARENT, None, "Demo session", 1_795_577_949);
        add_session(&conn, CHILD, Some(PARENT), "Demo session", 1_795_577_949);
        let t = 1_795_577_949;
        add_message(
            &conn,
            "e1f0b416-2c24-4572-91bf-de5bc9278a80",
            PARENT,
            "user",
            t,
            &[text("Delegate to a sub-agent: list src.")],
        );
        add_message(
            &conn,
            "24ecc5a7-0f0c-4e0e-b40d-7b761f8a3b9a",
            PARENT,
            "assistant",
            t,
            &[tool_call(
                "call_ff2c87f7",
                "agent_dispatch",
                r#"{"prompt": "Enumerate the src directory (sub task)", "label": "lister"}"#,
            )],
        );
        add_message(
            &conn,
            "24271001-9f11-47cd-a060-574972fcfd7a",
            PARENT,
            "tool",
            t,
            &[tool_result(
                "call_ff2c87f7",
                "agent_dispatch",
                "Started subagent \"paulo-lister\". The report arrives on its own and wakes you.",
                ("", ""),
                false,
            )],
        );
        add_message(
            &conn,
            "f891c27f-2bc2-43bc-9a9d-d44822446c60",
            CHILD,
            "user",
            t,
            &[text("Enumerate the src directory (sub task)")],
        );
        add_message(
            &conn,
            "2737aee2-8ae0-4907-8e24-ac96b4591904",
            CHILD,
            "assistant",
            t,
            &[tool_call(
                "call_8a4c2e14",
                "ls",
                r#"{"path": "/home/user/demo/src"}"#,
            )],
        );
        add_message(
            &conn,
            "74daa761-1531-4c73-b096-1d6618bf2d02",
            PARENT,
            "assistant",
            t,
            &[tool_call(
                "call_8ca1011d",
                "agent_status",
                r#"{"label": "lister"}"#,
            )],
        );
        add_message(
            &conn,
            "5bd32a77-143e-4b18-a90c-c1118a23df6f",
            CHILD,
            "tool",
            t,
            &[tool_result(
                "call_8a4c2e14",
                "ls",
                "\n- /home/user/demo/src/\n  - math.js\n",
                ("", ""),
                false,
            )],
        );
        add_message(
            &conn,
            "ae59ba77-c038-4f70-a05b-fe390389bcb1",
            CHILD,
            "assistant",
            t,
            &[text("src has math.js")],
        );
        add_message(
            &conn,
            "505b31e5-5654-4c0a-8f90-b0610c9e53fc",
            PARENT,
            "tool",
            t,
            &[tool_result(
                "call_8ca1011d",
                "agent_status",
                "no subagent \"lister\"; known: paulo-lister",
                ("", ""),
                true,
            )],
        );
        add_message(
            &conn,
            "d22d829e-b625-4cac-b23e-274307dbb849",
            PARENT,
            "assistant",
            t,
            &[text("Sub-agent finished.")],
        );

        // Bang-mode shell commands (`!cmd` in the TUI): user rows holding only
        // a `shell_command` part. The second output carries raw ANSI escapes.
        add_session(&conn, SHELL, None, "Demo session", 1_795_577_980);
        add_message(
            &conn,
            "2f4c4a1e-6b1d-4c68-9d0c-8a3f2b8c1a01",
            SHELL,
            "user",
            1_795_578_000,
            &[shell_command(
                "ls src && echo hi; exit 3",
                "math.js\nhi\n",
                3,
            )],
        );
        add_message(
            &conn,
            "6b9d3c2a-0f75-4e0e-8a5e-3c1d7e4f9b02",
            SHELL,
            "user",
            1_795_578_000,
            &[shell_command(
                "npm test",
                "\n> demo@1.0.0 test\n> node test.js\n\n4 !== 5\n\n    at Object.<anonymous> \u{1b}[90m(/home/user/demo/\u{1b}[39mtest.js:3:8\u{1b}[90m)\u{1b}[39m\n",
                1,
            )],
        );
        conn
    }

    fn blocks(msg: &ClaudeMessage) -> &Vec<Value> {
        msg.content.as_ref().unwrap().as_array().unwrap()
    }

    #[test]
    fn load_sessions_lists_top_level_sessions_with_titles_and_counts() {
        let conn = fixture_db();
        let sessions = load_sessions_conn(&conn, PROJECT).unwrap();
        let mut ids: Vec<&str> = sessions
            .iter()
            .map(|s| s.actual_session_id.as_str())
            .collect();
        ids.sort_unstable();
        let mut expected = vec![MAIN, IMAGE, PARENT, SHELL];
        expected.sort_unstable();
        assert_eq!(ids, expected, "the sub-agent run is not listed");

        let main = sessions
            .iter()
            .find(|s| s.actual_session_id == MAIN)
            .unwrap();
        assert_eq!(main.message_count, 2);
        assert_eq!(
            main.summary.as_deref(),
            Some("Add multiplication function to math.js")
        );
        assert_eq!(main.file_path, format!("crush:///home/user/demo#{MAIN}"));
        assert_eq!(main.provider.as_deref(), Some("crush"));
        let parent = sessions
            .iter()
            .find(|s| s.actual_session_id == PARENT)
            .unwrap();
        assert_eq!(
            parent.message_count, 6,
            "the run's own rows are not counted"
        );
    }

    #[test]
    fn project_counts_exclude_sub_agent_runs() {
        let conn = fixture_db();
        let (sessions, messages, _) = project_stats(&conn).unwrap();
        assert_eq!(sessions, 4);
        assert_eq!(messages, 2 + 3 + 6 + 2);
    }

    #[test]
    fn orphaned_sub_agent_run_is_still_listed() {
        let conn = fixture_db();
        conn.execute_batch("PRAGMA foreign_keys = OFF;").unwrap();
        conn.execute("DELETE FROM sessions WHERE id = ?1", [PARENT])
            .unwrap();
        let sessions = load_sessions_conn(&conn, PROJECT).unwrap();
        assert!(sessions.iter().any(|s| s.actual_session_id == CHILD));
    }

    #[test]
    fn messages_follow_insertion_order_not_created_at_or_id() {
        // Both rows are from one second and the assistant's random id sorts
        // first, so `ORDER BY created_at, id` showed the reply before its prompt.
        let conn = fixture_db();
        let msgs = load_messages_conn(&conn, MAIN).unwrap();
        let order: Vec<(&str, &str)> = msgs
            .iter()
            .map(|m| (m.role.as_deref().unwrap(), m.uuid.as_str()))
            .collect();
        assert_eq!(
            order,
            [
                ("user", "9cfe4f86-1569-4faf-aa05-16b5e367ae8d"),
                ("assistant", "945b8603-34f5-4a9f-9be5-5dfc3eaac296"),
            ]
        );
    }

    #[test]
    fn tool_turns_inside_one_second_keep_their_order() {
        let conn = fixture_db();
        let msgs = load_messages_conn(&conn, PARENT).unwrap();
        let shape: Vec<String> = msgs
            .iter()
            .map(|m| {
                let kinds: Vec<&str> = blocks(m)
                    .iter()
                    .map(|b| b["type"].as_str().unwrap())
                    .collect();
                format!("{}:{}", m.role.as_deref().unwrap(), kinds.join("+"))
            })
            .collect();
        assert_eq!(
            shape,
            [
                "user:text",
                "assistant:tool_use",
                "user:tool_result",
                "assistant:tool_use",
                "user:tool_result",
                "assistant:text",
            ]
        );
    }

    #[test]
    fn tool_call_and_result_map_to_claude_blocks() {
        let conn = fixture_db();
        let msgs = load_messages_conn(&conn, PARENT).unwrap();

        let call = &blocks(&msgs[1])[0];
        assert_eq!(call["type"], "tool_use");
        assert_eq!(call["id"], "call_ff2c87f7");
        assert_eq!(call["name"], "agent_dispatch");
        // `input` is a JSON string and is parsed back into an object.
        assert_eq!(call["input"]["label"], "lister");

        // "tool" role -> user lane, tool_result block; the producer's own flag
        // is copied as written.
        assert_eq!(msgs[2].role.as_deref(), Some("user"));
        let started = &blocks(&msgs[2])[0];
        assert_eq!(started["tool_use_id"], "call_ff2c87f7");
        assert_eq!(started["is_error"], false);
        assert_eq!(blocks(&msgs[4])[0]["is_error"], true);
    }

    #[test]
    fn image_tool_result_becomes_an_image_block() {
        let conn = fixture_db();
        let msgs = load_messages_conn(&conn, IMAGE).unwrap();
        assert_eq!(msgs.len(), 3);
        let result = &blocks(&msgs[2])[0];
        assert_eq!(result["type"], "tool_result");
        assert_eq!(result["tool_use_id"], "call_17fb6ad7");
        let content = result["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Loaded image/png content");
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "base64");
        assert_eq!(content[1]["source"]["media_type"], "image/png");
        assert!(content[1]["source"]["data"]
            .as_str()
            .unwrap()
            .starts_with("iVBORw0KGgo"));
    }

    #[test]
    fn tool_result_payload_without_an_image_type_stays_text() {
        let blob = json!({"data":"AAAA","mime_type":"application/pdf","content":"read pdf"});
        assert_eq!(tool_result_content(&blob, "read pdf"), json!("read pdf"));
        let no_data = json!({"data":"","mime_type":"image/png"});
        assert_eq!(tool_result_content(&no_data, "x"), json!("x"));
        let image_only = json!({"data":"AAAA","mime_type":"image/webp"});
        assert_eq!(
            tool_result_content(&image_only, ""),
            json!([{"type":"image","source":{"type":"base64","media_type":"image/webp","data":"AAAA"}}])
        );
    }

    #[test]
    fn shell_command_rows_show_as_bash_calls() {
        let conn = fixture_db();
        let msgs = load_messages_conn(&conn, SHELL).unwrap();
        assert_eq!(msgs.len(), 2, "a shell_command-only row is not dropped");
        assert!(msgs.iter().all(|m| m.role.as_deref() == Some("user")));

        let failed = blocks(&msgs[0]);
        assert_eq!(failed.len(), 2);
        assert_eq!(failed[0]["type"], "tool_use");
        assert_eq!(failed[0]["name"], "bash");
        assert_eq!(failed[0]["input"]["command"], "ls src && echo hi; exit 3");
        assert_eq!(failed[1]["type"], "tool_result");
        assert_eq!(failed[1]["tool_use_id"], failed[0]["id"]);
        assert_eq!(failed[1]["content"], "math.js\nhi\n");
        assert_eq!(failed[1]["toolUseResult"]["exit_code"], 3);
        assert_eq!(failed[1]["is_error"], true);
        assert_eq!(failed[1]["is_error_source"], "derived");

        let second = blocks(&msgs[1]);
        assert_eq!(second[0]["input"]["command"], "npm test");
        assert_ne!(second[0]["id"], failed[0]["id"]);
        assert!(second[1]["content"]
            .as_str()
            .unwrap()
            .contains("\u{1b}[90m"));
        assert_eq!(second[1]["toolUseResult"]["exit_code"], 1);
    }

    #[test]
    fn shell_command_with_exit_zero_asserts_no_verdict() {
        let blocks = shell_command_blocks(
            Some(&json!({"command":"true","output":"","exit_code":0})),
            "c1",
        );
        assert_eq!(blocks[0]["input"]["command"], "true");
        assert_eq!(blocks[1]["toolUseResult"]["exit_code"], 0);
        assert!(blocks[1].get("is_error").is_none());
        // No exit code recorded: nothing about the outcome is invented.
        let blocks = shell_command_blocks(Some(&json!({"command":"x","output":"y"})), "c2");
        assert!(blocks[1].get("is_error").is_none());
        assert!(blocks[1].get("toolUseResult").is_none());
        assert_eq!(
            shell_command_blocks(Some(&json!({})), "c3"),
            Vec::<Value>::new()
        );
    }

    #[test]
    fn child_sessions_list_the_sub_agent_runs_of_a_session() {
        let conn = fixture_db();
        let children = load_child_sessions_conn(&conn, PARENT).unwrap();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id, CHILD);
        assert_eq!(children[0].title, "Demo session");
        assert_eq!(children[0].message_count, 4);
        assert_ne!(children[0].created_at, "");
        assert!(load_child_sessions_conn(&conn, MAIN).unwrap().is_empty());
        assert!(load_child_sessions_conn(&conn, CHILD).unwrap().is_empty());

        assert_eq!(spawning_tool_call_id(CHILD), Some("call_ff2c87f7"));
        assert_eq!(spawning_tool_call_id(PARENT), None);
        assert_eq!(spawning_tool_call_id("a$$b$$c"), None);
    }

    #[test]
    fn sub_agent_run_loads_under_its_own_id() {
        let conn = fixture_db();
        let msgs = load_messages_conn(&conn, CHILD).unwrap();
        let roles: Vec<&str> = msgs.iter().map(|m| m.role.as_deref().unwrap()).collect();
        assert_eq!(roles, ["user", "assistant", "user", "assistant"]);
        assert_eq!(
            blocks(&msgs[0])[0]["text"],
            "Enumerate the src directory (sub task)"
        );
    }

    #[test]
    fn store_without_parent_session_id_has_no_sub_agent_runs() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT, created_at INTEGER, updated_at INTEGER);
             CREATE TABLE messages (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, parts TEXT NOT NULL DEFAULT '[]', created_at INTEGER);
             INSERT INTO sessions VALUES ('a$$b', 't', 1750000000, 1750000000);",
        )
        .unwrap();
        assert!(!has_parent_session_id(&conn).unwrap());
        assert_eq!(load_sessions_conn(&conn, PROJECT).unwrap().len(), 1);
        assert!(load_child_sessions_conn(&conn, "a").unwrap().is_empty());
    }

    #[test]
    fn search_matches_and_tags_project() {
        let conn = fixture_db();
        let mut results = Vec::new();
        search_one_db(
            &conn,
            "%DELEGATE%",
            "delegate",
            ("crush:///home/user/demo", "demo"),
            10,
            &SearchFilters::parse(&Value::Null),
            &mut results,
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].project_name.as_deref(), Some("demo"));
        assert_eq!(
            results[0].project_path.as_deref(),
            Some("crush:///home/user/demo")
        );
        // session id is the real session (matches load_messages), not the project name.
        assert_eq!(results[0].session_id, PARENT);
    }

    #[test]
    fn search_finds_shell_commands_but_not_inside_sub_agent_runs() {
        let conn = fixture_db();
        let search = |needle: &str| {
            let mut results = Vec::new();
            search_one_db(
                &conn,
                &format!("%{needle}%"),
                &needle.to_lowercase(),
                ("crush:///home/user/demo", "demo"),
                10,
                &SearchFilters::parse(&Value::Null),
                &mut results,
            );
            results
        };
        assert_eq!(search("npm test").len(), 1);
        // The run is not a listed session, so a hit in it could not be opened.
        assert!(search("src has math.js").is_empty());
        assert!(search("Enumerate the src")
            .iter()
            .all(|m| m.session_id == PARENT));
    }

    #[test]
    fn parse_session_path_splits_dir_and_id() {
        let (dir, id) = parse_session_path("crush:///home/user/proj#sess1").unwrap();
        assert_eq!(dir, "/home/user/proj");
        assert_eq!(id, "sess1");
        // A sub-agent id has `$$` but no `#`.
        let (_, id) = parse_session_path(&format!("crush:///home/user/proj#{CHILD}")).unwrap();
        assert_eq!(id, CHILD);
        assert!(parse_session_path("crush:///no-separator").is_err());
    }

    #[test]
    fn discover_finds_crush_db() {
        let tmp = TempDir::new().unwrap();
        let proj = tmp.path().join("myproj");
        let crush_dir = proj.join(".crush");
        fs::create_dir_all(&crush_dir).unwrap();
        fs::write(crush_dir.join("crush.db"), b"x").unwrap();

        let mut found = Vec::new();
        find_crush_db(tmp.path(), &mut found, 10, 0);
        assert_eq!(found.len(), 1);
        assert!(found[0].ends_with(".crush/crush.db"));
        assert_eq!(project_dir_of(&found[0]).as_deref(), proj.to_str());
    }

    /// `projects.json` as Crush writes it (`internal/projects/projects.go`).
    const REGISTRY: &str = r#"{
      "projects": [
        {"path": "/home/user/demo3", "data_dir": "/data/custom", "last_accessed": "2026-11-25T03:40:30.779472Z"},
        {"path": "/srv/outside/proj2", "data_dir": "/srv/outside/proj2/.crush", "last_accessed": "2026-11-25T03:40:30.632505Z"},
        {"path": "/home/user/demo", "data_dir": "/home/user/demo/.crush", "last_accessed": "2026-11-25T03:40:00.6293Z"}
      ]
    }"#;

    #[test]
    fn registry_lists_projects_with_their_data_dirs() {
        let projects = parse_registry(REGISTRY);
        assert_eq!(
            projects
                .iter()
                .map(|p| (p.path.as_str(), p.data_dir.as_str()))
                .collect::<Vec<_>>(),
            [
                ("/home/user/demo3", "/data/custom"),
                ("/srv/outside/proj2", "/srv/outside/proj2/.crush"),
                ("/home/user/demo", "/home/user/demo/.crush"),
            ]
        );
        // A missing data dir is the default one; entries without a path, a
        // missing list and garbage are ignored.
        let sparse =
            parse_registry(r#"{"projects":[{"path":"/p"},{"path":""},{"data_dir":"/x"}]}"#);
        assert_eq!(
            sparse,
            [RegisteredProject {
                path: "/p".into(),
                data_dir: ".crush".into()
            }]
        );
        assert_eq!(
            parse_registry(r#"{"projects":null}"#),
            Vec::<RegisteredProject>::new()
        );
        assert_eq!(parse_registry("not json"), Vec::<RegisteredProject>::new());
    }

    #[test]
    fn global_data_dir_follows_crush_precedence() {
        let os = |s: &str| Some(std::ffi::OsString::from(s));
        let home = Some(PathBuf::from("/home/user"));
        // `internal/config/load.go` `GlobalConfigData`.
        assert_eq!(
            global_data_dir_with(os("/g"), os("/x"), None, home.clone()),
            Some(PathBuf::from("/g"))
        );
        assert_eq!(
            global_data_dir_with(None, os("/x"), None, home.clone()),
            Some(PathBuf::from("/x/crush"))
        );
        assert_eq!(
            global_data_dir_with(None, None, Some(PathBuf::from("/local")), home.clone()),
            Some(PathBuf::from("/local/crush"))
        );
        assert_eq!(
            global_data_dir_with(None, None, None, home),
            Some(PathBuf::from("/home/user/.local/share/crush"))
        );
    }

    /// A real-schema DB file with one session and one prompt.
    fn write_project_db(db: &Path, session: &str, prompt: &str) {
        fs::create_dir_all(db.parent().unwrap()).unwrap();
        let conn = Connection::open(db).unwrap();
        conn.execute_batch(SCHEMA).unwrap();
        add_session(&conn, session, None, "Demo session", 1_795_577_949);
        add_message(
            &conn,
            &format!("{session}-m1"),
            session,
            "user",
            1_795_577_949,
            &[text(prompt)],
        );
    }

    #[test]
    #[serial_test::serial]
    fn registry_finds_projects_outside_the_scanned_roots_and_with_a_custom_data_dir() {
        let home = SandboxHome::new();
        let elsewhere = TempDir::new().unwrap();

        // Found by the scan, and also registered (listed once).
        let demo = home.path().join("projects").join("demo");
        write_project_db(
            &demo.join(".crush").join("crush.db"),
            "s-demo",
            "scanned needle",
        );
        // Outside every scanned root.
        let proj2 = elsewhere.path().join("proj2");
        write_project_db(
            &proj2.join(".crush").join("crush.db"),
            "s-proj2",
            "outside needle",
        );
        // A custom `--data-dir`.
        let demo3 = home.path().join("projects").join("demo3");
        fs::create_dir_all(&demo3).unwrap();
        let custom = elsewhere.path().join("customdata");
        write_project_db(&custom.join("crush.db"), "s-demo3", "custom needle");
        // Registered, but its database is gone.
        let gone = elsewhere.path().join("gone");

        let registry_dir = home.path().join(".local").join("share").join("crush");
        fs::create_dir_all(&registry_dir).unwrap();
        let entry = |path: &Path, data_dir: &Path| json!({"path": path, "data_dir": data_dir, "last_accessed": "2026-11-25T03:40:30Z"});
        fs::write(
            registry_dir.join("projects.json"),
            json!({"projects": [
                entry(&demo3, &custom),
                entry(&proj2, &proj2.join(".crush")),
                entry(&gone, &gone.join(".crush")),
                entry(&demo, &demo.join(".crush")),
            ]})
            .to_string(),
        )
        .unwrap();

        let projects = scan_projects().unwrap();
        let mut names: Vec<&str> = projects.iter().map(|p| p.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, ["demo", "demo3", "proj2"]);

        for (name, session) in [
            ("demo", "s-demo"),
            ("demo3", "s-demo3"),
            ("proj2", "s-proj2"),
        ] {
            let project = projects.iter().find(|p| p.name == name).unwrap();
            let sessions = load_sessions(&project.path, false).unwrap();
            assert_eq!(sessions.len(), 1, "{name}");
            assert_eq!(sessions[0].actual_session_id, session);
            let messages = load_messages(&sessions[0].file_path).unwrap();
            assert_eq!(messages.len(), 1, "{name}");
        }

        let hits = search("needle", 10).unwrap();
        assert_eq!(hits.len(), 3);
        assert!(detect().is_some());
    }

    #[test]
    #[serial_test::serial]
    fn two_registry_entries_sharing_a_data_dir_are_one_project() {
        let home = SandboxHome::new();
        let elsewhere = TempDir::new().unwrap();
        let shared = elsewhere.path().join("shared");
        write_project_db(&shared.join("crush.db"), "s-shared", "shared needle");
        let registry_dir = home.path().join(".local").join("share").join("crush");
        fs::create_dir_all(&registry_dir).unwrap();
        let (a, b) = (elsewhere.path().join("a"), elsewhere.path().join("b"));
        fs::create_dir_all(&a).unwrap();
        fs::create_dir_all(&b).unwrap();
        fs::write(
            registry_dir.join("projects.json"),
            json!({"projects": [
                {"path": a, "data_dir": shared, "last_accessed": "2026-11-25T03:40:30Z"},
                {"path": b, "data_dir": shared, "last_accessed": "2026-11-25T03:40:00Z"},
            ]})
            .to_string(),
        )
        .unwrap();

        let projects = scan_projects().unwrap();
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].name, "a");
        assert_eq!(search("needle", 10).unwrap().len(), 1);
    }

    #[test]
    #[serial_test::serial]
    fn search_hits_carry_scanned_project_path_for_same_named_projects() {
        let home = SandboxHome::new();
        for (dir, session) in [
            ("client/client-a/app", "sess-a"),
            ("work/client-b/app", "sess-b"),
        ] {
            write_project_db(
                &home.path().join(dir).join(".crush").join("crush.db"),
                session,
                "shared needle",
            );
        }

        let projects = scan_projects().unwrap();
        assert_eq!(projects.len(), 2);
        assert!(projects.iter().all(|p| p.name == "app"));
        assert_ne!(projects[0].path, projects[1].path);

        let hits = search("needle", 10).unwrap();
        assert_eq!(hits.len(), 2);
        for hit in &hits {
            let project = projects
                .iter()
                .find(|p| {
                    load_sessions(&p.path, false)
                        .unwrap()
                        .iter()
                        .any(|s| s.actual_session_id == hit.session_id)
                })
                .expect("hit session belongs to a scanned project");
            assert_eq!(hit.project_path.as_deref(), Some(project.path.as_str()));
        }
    }

    #[test]
    fn map_part_handles_kinds() {
        assert!(map_part(&json!({"type":"finish","data":{"reason":"end_turn"}})).is_none());
        let t = map_part(&json!({"type":"text","data":{"text":"hi"}})).unwrap();
        assert_eq!(t["text"], "hi");
        // tool_call input that isn't valid JSON falls back to the raw value.
        let tc =
            map_part(&json!({"type":"tool_call","data":{"id":"x","name":"n","input":"not json"}}))
                .unwrap();
        assert_eq!(tc["type"], "tool_use");
        assert_eq!(tc["input"], "not json");
    }

    #[test]
    fn epoch_handles_seconds_and_ms() {
        assert_ne!(epoch_to_iso(1_750_000_000), "");
        assert_ne!(epoch_to_iso(1_750_000_000_000), "");
        assert_eq!(epoch_to_iso(0), "");
    }
}

#[cfg(test)]
mod outcome_preservation_tests {
    use super::*;
    #[test]
    fn producer_outcome_evidence_is_not_fabricated_or_dropped() {
        let block=map_part(&serde_json::json!({"type":"tool_result","data":{"tool_call_id":"c","content":"synthetic"}})).unwrap();
        assert_ne!(block["is_error"], serde_json::json!(false));
    }
}
