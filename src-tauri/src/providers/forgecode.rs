use super::scan_report::ScanReport;
use super::ProviderInfo;
use crate::commands::session::NativeRenameResult;
use crate::models::{ClaudeMessage, ClaudeProject, ClaudeSession, TokenUsage};
use crate::utils::{build_provider_message, search_json_value_case_insensitive};
use chrono::{DateTime, NaiveDateTime, Utc};
use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

const PROVIDER_ID: &str = "forgecode";
const STORAGE_TYPE: &str = "sqlite";

#[derive(Debug, Clone)]
struct ConversationColumns {
    id: String,
    workspace_id: String,
    title: Option<String>,
    context: String,
    metrics: Option<String>,
    created_at: Option<String>,
    updated_at: Option<String>,
}

#[derive(Debug, Default)]
struct ProjectAccumulator {
    session_count: usize,
    message_count: usize,
    last_modified: String,
    display_name_votes: BTreeMap<String, usize>,
    cwd_votes: BTreeMap<String, usize>,
}

#[derive(Debug)]
struct ConversationRow {
    conversation_id: String,
    workspace_id: String,
    title: Option<String>,
    context_json: String,
    metrics_json: Option<String>,
    created_at: String,
    updated_at: String,
}

/// Detect ``ForgeCode`` installation.
pub fn detect() -> Option<ProviderInfo> {
    let base_path = get_base_path()?;
    let base = Path::new(&base_path);
    let db_path = base.join(".forge.db");
    let logs_path = base.join("logs");
    let history_path = base.join(".forge_history");

    Some(ProviderInfo {
        id: PROVIDER_ID.to_string(),
        display_name: "ForgeCode".to_string(),
        base_path: base_path.clone(),
        is_available: db_path.is_file() || logs_path.is_dir() || history_path.is_file(),
    })
}

/// Resolve the ``ForgeCode`` base path.
///
/// Lookup precedence for v1:
/// 1. `$FORGE_CONFIG`
/// 2. `~/.forge`
///
/// `.forge.db` remains the authoritative transcript source, while `logs/` and
/// `.forge_history` are treated as secondary detection artifacts only.
pub fn get_base_path() -> Option<String> {
    if let Ok(config_dir) = std::env::var("FORGE_CONFIG") {
        let path = PathBuf::from(&config_dir);
        if path.exists() {
            return Some(path.to_string_lossy().to_string());
        }
    }

    let home = crate::utils::home_dir()?;
    let default_path = home.join(".forge");
    if default_path.exists() {
        Some(default_path.to_string_lossy().to_string())
    } else {
        None
    }
}

/// Scan `ForgeCode` projects from the detected base path.
pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    scan_projects_reporting(&ScanReport::default())
}

/// [`scan_projects`], recording in `report` a database that is there but
/// cannot be read.
pub(crate) fn scan_projects_reporting(report: &ScanReport) -> Result<Vec<ClaudeProject>, String> {
    let base_path = get_base_path().ok_or_else(|| "ForgeCode base path not found".to_string())?;
    scan_projects_from_path_reporting(&base_path, report)
}

/// Scan `ForgeCode` projects from an explicit base path.
pub fn scan_projects_from_path(base_path: &str) -> Result<Vec<ClaudeProject>, String> {
    scan_projects_from_path_reporting(base_path, &ScanReport::default())
}

fn scan_projects_from_path_reporting(
    base_path: &str,
    report: &ScanReport,
) -> Result<Vec<ClaudeProject>, String> {
    crate::utils::require_absolute_path(base_path, "ForgeCode base path")?;
    if let Some(projects) = scan_projects_from_db(base_path, report) {
        return Ok(projects);
    }
    log::debug!("ForgeCode: no projects found or DB unavailable at {base_path}");
    Ok(Vec::new())
}

/// Load `ForgeCode` sessions for a virtual workspace path.
pub fn load_sessions(
    project_path: &str,
    _exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    load_sessions_reporting(project_path, &ScanReport::default())
}

/// [`load_sessions`], recording in `report` a database that is there but
/// cannot be read.
pub(crate) fn load_sessions_reporting(
    project_path: &str,
    report: &ScanReport,
) -> Result<Vec<ClaudeSession>, String> {
    let base_path = get_base_path().ok_or_else(|| "ForgeCode not found".to_string())?;
    let workspace_id = parse_workspace_project_path(project_path)
        .ok_or_else(|| format!("Invalid ForgeCode project path: {project_path}"))?;

    if let Some(sessions) = load_sessions_from_db(&base_path, &workspace_id, report) {
        return Ok(sessions);
    }
    log::debug!("ForgeCode: no sessions found for workspace {workspace_id}");
    Ok(Vec::new())
}

/// Load `ForgeCode` messages for a virtual conversation path.
pub fn load_messages(session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    let base_path = get_base_path().ok_or_else(|| "ForgeCode not found".to_string())?;
    let (workspace_id, conversation_id) = parse_conversation_path(session_path)
        .ok_or_else(|| format!("Invalid ForgeCode session path: {session_path}"))?;

    if let Some(messages) = load_messages_from_db(&base_path, &workspace_id, &conversation_id) {
        return Ok(messages);
    }
    log::debug!(
        "ForgeCode: no messages found for workspace {workspace_id} conversation {conversation_id}"
    );
    Ok(Vec::new())
}

/// Search `ForgeCode` messages using the detected database path.
pub fn search(query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    let base_path = get_base_path().ok_or_else(|| "ForgeCode not found".to_string())?;
    Ok(search_from_path(&base_path, query, limit))
}

/// Search `ForgeCode` messages from an explicit base path.
fn search_from_path(base_path: &str, query: &str, limit: usize) -> Vec<ClaudeMessage> {
    let query_lower = query.to_lowercase();
    let Some(conn) = open_db(base_path) else {
        return Vec::new();
    };
    let Some(columns) = resolve_conversation_columns(&conn) else {
        return Vec::new();
    };
    let Some(rows) = load_search_rows(&conn, &columns) else {
        return Vec::new();
    };

    let mut results = Vec::new();

    for row in rows {
        let created_at = normalize_timestamp_text(&row.created_at);
        let updated_at = latest_timestamp(&row.created_at, &row.updated_at);
        let project_path = project_virtual_path(&row.workspace_id);
        let messages = map_context_messages(
            &row.workspace_id,
            &row.conversation_id,
            &parse_context_entries(&row.context_json),
            &created_at,
            &updated_at,
            row.metrics_json.as_deref(),
        );

        for mut message in messages {
            if results.len() >= limit {
                return results;
            }

            if let Some(content) = &message.content {
                if search_json_value_case_insensitive(content, &query_lower) {
                    message.project_path = Some(project_path.clone());
                    results.push(message);
                }
            }
        }
    }

    results
}

/// Load conversation rows used by `ForgeCode` full-text search.
fn load_search_rows(
    conn: &Connection,
    columns: &ConversationColumns,
) -> Option<Vec<ConversationRow>> {
    let query = format!(
        "SELECT {conversation_id}, {workspace_id}, {title}, {context}, {metrics}, {created_at}, {updated_at}
         FROM conversations
         WHERE {workspace_id_raw} IS NOT NULL AND {context_raw} IS NOT NULL
         ORDER BY {updated_at_raw} DESC, {created_at_raw} DESC",
        conversation_id = cast_text_expr(&columns.id),
        workspace_id = cast_text_expr(&columns.workspace_id),
        title = optional_cast_text_expr(columns.title.as_deref()),
        context = cast_text_expr(&columns.context),
        metrics = optional_cast_text_expr(columns.metrics.as_deref()),
        created_at = optional_cast_text_expr(columns.created_at.as_deref()),
        updated_at = optional_cast_text_expr(columns.updated_at.as_deref()),
        workspace_id_raw = quote_ident(&columns.workspace_id),
        context_raw = quote_ident(&columns.context),
        updated_at_raw = optional_order_expr(columns.updated_at.as_deref()),
        created_at_raw = optional_order_expr(columns.created_at.as_deref()),
    );

    let mut stmt = conn.prepare(&query).ok()?;
    let rows = stmt
        .query_map([], |row| {
            Ok(ConversationRow {
                conversation_id: row.get(0)?,
                workspace_id: row.get(1)?,
                title: empty_to_none(row.get::<_, String>(2)?),
                context_json: row.get(3)?,
                metrics_json: empty_to_none(row.get::<_, String>(4)?),
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
            })
        })
        .ok()?;

    Some(rows.filter_map(std::result::Result::ok).collect())
}

/// Rename a `ForgeCode` conversation title in the local database.
pub fn rename_session_title(
    session_path: &str,
    new_title: &str,
) -> Result<NativeRenameResult, String> {
    let base_path = get_base_path().ok_or_else(|| "ForgeCode not found".to_string())?;
    rename_session_title_from_path(&base_path, session_path, new_title)
}

/// Rename a `ForgeCode` conversation title from an explicit base path.
fn rename_session_title_from_path(
    base_path: &str,
    session_path: &str,
    new_title: &str,
) -> Result<NativeRenameResult, String> {
    let (workspace_id, conversation_id) = parse_conversation_path(session_path)
        .ok_or_else(|| format!("Invalid ForgeCode session path: {session_path}"))?;
    let conn = open_db_read_write(base_path)?;
    let columns = resolve_conversation_columns(&conn)
        .ok_or_else(|| "ForgeCode conversations schema not found".to_string())?;
    let title_column = columns
        .title
        .as_deref()
        .ok_or_else(|| "ForgeCode database does not expose a writable title column".to_string())?;
    let existing = load_conversation_row(&conn, &columns, &workspace_id, &conversation_id)
        .ok_or_else(|| format!("ForgeCode conversation not found: {session_path}"))?;
    let trimmed_title = new_title.trim();
    let next_title = if trimmed_title.is_empty() {
        None
    } else {
        Some(trimmed_title.to_string())
    };
    let previous_title = existing.title.unwrap_or_default();

    let affected_rows = if let Some(updated_at_column) = columns.updated_at.as_deref() {
        let query = format!(
            "UPDATE conversations
             SET {title_column} = ?1, {updated_at_column} = ?2
             WHERE CAST({workspace_id_column} AS TEXT) = ?3
               AND CAST({conversation_id_column} AS TEXT) = ?4",
            title_column = quote_ident(title_column),
            updated_at_column = quote_ident(updated_at_column),
            workspace_id_column = quote_ident(&columns.workspace_id),
            conversation_id_column = quote_ident(&columns.id),
        );
        conn.execute(
            &query,
            rusqlite::params![
                next_title.clone(),
                Utc::now().to_rfc3339(),
                workspace_id,
                conversation_id
            ],
        )
        .map_err(|e| format!("Failed to rename ForgeCode conversation: {e}"))?
    } else {
        let query = format!(
            "UPDATE conversations
             SET {title_column} = ?1
             WHERE CAST({workspace_id_column} AS TEXT) = ?2
               AND CAST({conversation_id_column} AS TEXT) = ?3",
            title_column = quote_ident(title_column),
            workspace_id_column = quote_ident(&columns.workspace_id),
            conversation_id_column = quote_ident(&columns.id),
        );
        conn.execute(
            &query,
            rusqlite::params![next_title.clone(), workspace_id, conversation_id],
        )
        .map_err(|e| format!("Failed to rename ForgeCode conversation: {e}"))?
    };

    if affected_rows == 0 {
        return Err(format!("ForgeCode conversation not found: {session_path}"));
    }

    Ok(NativeRenameResult {
        success: true,
        previous_title,
        new_title: next_title.unwrap_or_default(),
        file_path: session_path.to_string(),
    })
}

/// Delete a `ForgeCode` conversation from the detected database.
pub fn delete_conversation(session_path: &str) -> Result<(), String> {
    let base_path = get_base_path().ok_or_else(|| "ForgeCode not found".to_string())?;
    delete_conversation_from_path(&base_path, session_path)
}

/// Delete a `ForgeCode` conversation from an explicit base path.
fn delete_conversation_from_path(base_path: &str, session_path: &str) -> Result<(), String> {
    let (workspace_id, conversation_id) = parse_conversation_path(session_path)
        .ok_or_else(|| format!("Invalid ForgeCode session path: {session_path}"))?;
    let conn = open_db_read_write(base_path)?;
    let columns = resolve_conversation_columns(&conn)
        .ok_or_else(|| "ForgeCode conversations schema not found".to_string())?;

    let query = format!(
        "DELETE FROM conversations
         WHERE CAST({workspace_id_column} AS TEXT) = ?1
           AND CAST({conversation_id_column} AS TEXT) = ?2",
        workspace_id_column = quote_ident(&columns.workspace_id),
        conversation_id_column = quote_ident(&columns.id),
    );

    let affected_rows = conn
        .execute(&query, rusqlite::params![workspace_id, conversation_id])
        .map_err(|e| format!("Failed to delete ForgeCode conversation: {e}"))?;

    if affected_rows == 0 {
        return Err(format!("ForgeCode conversation not found: {session_path}"));
    }

    Ok(())
}

/// Open the `ForgeCode` `SQLite` database in read-only mode.
fn open_db(base_path: &str) -> Option<Connection> {
    open_db_reporting(base_path, &ScanReport::default())
}

/// [`open_db`], recording in `report` a database that is there but cannot
/// be opened or read (its header is read here, not at the first query).
fn open_db_reporting(base_path: &str, report: &ScanReport) -> Option<Connection> {
    let db_path = Path::new(base_path).join(".forge.db");
    if !report.is_file(&db_path) {
        return None;
    }

    let conn = report.db(
        &db_path,
        Connection::open_with_flags(
            &db_path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        ),
    )?;
    report.db(
        &db_path,
        conn.query_row("PRAGMA schema_version", [], |row| row.get::<_, i64>(0)),
    )?;
    Some(conn)
}

/// Open the `ForgeCode` `SQLite` database in read-write mode.
fn open_db_read_write(base_path: &str) -> Result<Connection, String> {
    let db_path = Path::new(base_path).join(".forge.db");
    if !db_path.is_file() {
        return Err(format!(
            "ForgeCode database not found: {}",
            db_path.display()
        ));
    }

    Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("Failed to open ForgeCode database: {e}"))
}

/// Build project summaries from `ForgeCode` conversation rows.
fn scan_projects_from_db(base_path: &str, report: &ScanReport) -> Option<Vec<ClaudeProject>> {
    let conn = open_db_reporting(base_path, report)?;
    let db_path = Path::new(base_path).join(".forge.db");
    let columns = resolve_conversation_columns(&conn)?;
    let rows = load_project_rows(&conn, &columns, &db_path, report)?;

    let mut workspaces: BTreeMap<String, ProjectAccumulator> = BTreeMap::new();

    for row in rows {
        let message_count = parse_context_entries(&row.context_json).len();
        let last_modified = latest_timestamp(&row.created_at, &row.updated_at);
        let extracted_display_name =
            extract_workspace_display_name_from_context_json(&row.context_json);
        let extracted_cwds = extract_cwds_from_context_json(&row.context_json);

        let entry = workspaces.entry(row.workspace_id).or_default();
        entry.session_count += 1;
        entry.message_count += message_count;
        entry.last_modified = max_timestamp(&entry.last_modified, &last_modified);
        if let Some(display_name) = extracted_display_name {
            *entry.display_name_votes.entry(display_name).or_default() += 1;
        }
        for (cwd, count) in extracted_cwds {
            *entry.cwd_votes.entry(cwd).or_default() += count;
        }
    }

    let mut projects: Vec<ClaudeProject> = workspaces
        .into_iter()
        .map(|(workspace_id, acc)| {
            let actual_path = choose_best_cwd(&acc.cwd_votes)
                .unwrap_or_else(|| project_virtual_path(&workspace_id));
            ClaudeProject {
                name: choose_workspace_display_name(&workspace_id, &acc.display_name_votes),
                path: project_virtual_path(&workspace_id),
                actual_path,
                session_count: acc.session_count,
                message_count: acc.message_count,
                last_modified: acc.last_modified,
                git_info: None,
                provider: Some(PROVIDER_ID.to_string()),
                storage_type: Some(STORAGE_TYPE.to_string()),
                custom_directory_label: None,
            }
        })
        .collect();

    projects.sort_by(|a, b| compare_timestamps(&b.last_modified, &a.last_modified));
    Some(projects)
}

/// Load `ForgeCode` session summaries for a workspace.
fn load_sessions_from_db(
    base_path: &str,
    workspace_id: &str,
    report: &ScanReport,
) -> Option<Vec<ClaudeSession>> {
    let conn = open_db_reporting(base_path, report)?;
    let db_path = Path::new(base_path).join(".forge.db");
    let columns = resolve_conversation_columns(&conn)?;
    let rows = load_workspace_rows(&conn, &columns, workspace_id, &db_path, report)?;
    let project_name = resolve_workspace_display_name_from_rows(workspace_id, &rows);

    let mut sessions: Vec<ClaudeSession> = rows
        .into_iter()
        .filter(|row| !row.context_json.trim().is_empty())
        .map(|row| {
            let entries = parse_context_entries(&row.context_json);
            let has_tool_use = context_entries_have_tool_use(&entries);
            let first_message_time = normalize_timestamp_text(&row.created_at);
            let last_message_time = latest_timestamp(&row.created_at, &row.updated_at);

            ClaudeSession {
                session_id: session_virtual_path(&row.workspace_id, &row.conversation_id),
                actual_session_id: row.conversation_id.clone(),
                file_path: session_file_virtual_path(&row.workspace_id, &row.conversation_id),
                project_name: project_name.clone(),
                message_count: entries.len(),
                first_message_time,
                last_message_time: last_message_time.clone(),
                last_modified: last_message_time,
                has_tool_use,
                has_errors: false,
                summary: row.title.filter(|title| !title.trim().is_empty()),
                is_renamed: false,
                provider: Some(PROVIDER_ID.to_string()),
                storage_type: Some(STORAGE_TYPE.to_string()),
                entrypoint: None,
            }
        })
        .collect();

    sessions.sort_by(|a, b| compare_timestamps(&b.last_modified, &a.last_modified));
    Some(sessions)
}

/// Load `ForgeCode` messages for a workspace conversation pair.
fn load_messages_from_db(
    base_path: &str,
    workspace_id: &str,
    conversation_id: &str,
) -> Option<Vec<ClaudeMessage>> {
    let conn = open_db(base_path)?;
    let columns = resolve_conversation_columns(&conn)?;
    let row = load_conversation_row(&conn, &columns, workspace_id, conversation_id)?;
    let entries = parse_context_entries(&row.context_json);
    let created_at = normalize_timestamp_text(&row.created_at);
    let updated_at = latest_timestamp(&row.created_at, &row.updated_at);

    Some(map_context_messages(
        workspace_id,
        conversation_id,
        &entries,
        &created_at,
        &updated_at,
        row.metrics_json.as_deref(),
    ))
}

/// Resolve the conversation table columns exposed by the `ForgeCode` schema.
fn resolve_conversation_columns(conn: &Connection) -> Option<ConversationColumns> {
    let mut stmt = conn.prepare("PRAGMA table_info(conversations)").ok()?;
    let column_names: HashSet<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .ok()?
        .filter_map(std::result::Result::ok)
        .collect();

    let pick = |names: &[&str]| {
        names
            .iter()
            .find(|name| column_names.contains(**name))
            .map(|name| (*name).to_string())
    };

    Some(ConversationColumns {
        id: pick(&["conversation_id", "id"])?,
        workspace_id: pick(&["workspace_id"])?,
        title: pick(&["title"]),
        context: pick(&["context"])?,
        metrics: pick(&["metrics"]),
        created_at: pick(&["created_at", "createdAt"]),
        updated_at: pick(&["updated_at", "updatedAt"]),
    })
}

/// Load per-workspace aggregation rows from the conversations table; a
/// query or row the database cannot read is recorded in `report`.
fn load_project_rows(
    conn: &Connection,
    columns: &ConversationColumns,
    db_path: &Path,
    report: &ScanReport,
) -> Option<Vec<ConversationRow>> {
    let query = format!(
        "SELECT {workspace_id}, {context}, {created_at}, {updated_at}
         FROM conversations
         WHERE {workspace_id_raw} IS NOT NULL AND {context_raw} IS NOT NULL",
        workspace_id = cast_text_expr(&columns.workspace_id),
        context = cast_text_expr(&columns.context),
        created_at = optional_cast_text_expr(columns.created_at.as_deref()),
        updated_at = optional_cast_text_expr(columns.updated_at.as_deref()),
        workspace_id_raw = quote_ident(&columns.workspace_id),
        context_raw = quote_ident(&columns.context),
    );

    let mut stmt = report.db(db_path, conn.prepare(&query))?;
    let rows = stmt.query_map([], |row| {
        Ok(ConversationRow {
            conversation_id: String::new(),
            workspace_id: row.get(0)?,
            title: None,
            context_json: row.get(1)?,
            metrics_json: None,
            created_at: row.get(2)?,
            updated_at: row.get(3)?,
        })
    });
    let rows = report.db(db_path, rows)?;

    Some(rows.filter_map(|row| report.db(db_path, row)).collect())
}

/// Load conversation rows for a single `ForgeCode` workspace; a query or row
/// the database cannot read is recorded in `report`.
fn load_workspace_rows(
    conn: &Connection,
    columns: &ConversationColumns,
    workspace_id: &str,
    db_path: &Path,
    report: &ScanReport,
) -> Option<Vec<ConversationRow>> {
    let query = format!(
        "SELECT {conversation_id}, {workspace_id}, {title}, {context}, {metrics}, {created_at}, {updated_at}
         FROM conversations
         WHERE CAST({workspace_id_raw} AS TEXT) = ?1 AND {context_raw} IS NOT NULL
         ORDER BY {updated_at_raw} DESC, {created_at_raw} DESC",
        conversation_id = cast_text_expr(&columns.id),
        workspace_id = cast_text_expr(&columns.workspace_id),
        title = optional_cast_text_expr(columns.title.as_deref()),
        context = cast_text_expr(&columns.context),
        metrics = optional_cast_text_expr(columns.metrics.as_deref()),
        created_at = optional_cast_text_expr(columns.created_at.as_deref()),
        updated_at = optional_cast_text_expr(columns.updated_at.as_deref()),
        workspace_id_raw = quote_ident(&columns.workspace_id),
        context_raw = quote_ident(&columns.context),
        updated_at_raw = optional_order_expr(columns.updated_at.as_deref()),
        created_at_raw = optional_order_expr(columns.created_at.as_deref()),
    );

    let mut stmt = report.db(db_path, conn.prepare(&query))?;
    let rows = stmt.query_map([workspace_id], |row| {
        Ok(ConversationRow {
            conversation_id: row.get(0)?,
            workspace_id: row.get(1)?,
            title: empty_to_none(row.get::<_, String>(2)?),
            context_json: row.get(3)?,
            metrics_json: empty_to_none(row.get::<_, String>(4)?),
            created_at: row.get(5)?,
            updated_at: row.get(6)?,
        })
    });
    let rows = report.db(db_path, rows)?;

    Some(rows.filter_map(|row| report.db(db_path, row)).collect())
}

/// Load a single `ForgeCode` conversation row by workspace and conversation id.
fn load_conversation_row(
    conn: &Connection,
    columns: &ConversationColumns,
    workspace_id: &str,
    conversation_id: &str,
) -> Option<ConversationRow> {
    let query = format!(
        "SELECT {conversation_id}, {workspace_id}, {title}, {context}, {metrics}, {created_at}, {updated_at}
         FROM conversations
         WHERE CAST({workspace_id_raw} AS TEXT) = ?1
           AND CAST({conversation_id_raw} AS TEXT) = ?2
         LIMIT 1",
        conversation_id = cast_text_expr(&columns.id),
        workspace_id = cast_text_expr(&columns.workspace_id),
        title = optional_cast_text_expr(columns.title.as_deref()),
        context = cast_text_expr(&columns.context),
        metrics = optional_cast_text_expr(columns.metrics.as_deref()),
        created_at = optional_cast_text_expr(columns.created_at.as_deref()),
        updated_at = optional_cast_text_expr(columns.updated_at.as_deref()),
        workspace_id_raw = quote_ident(&columns.workspace_id),
        conversation_id_raw = quote_ident(&columns.id),
    );

    let mut stmt = conn.prepare(&query).ok()?;
    stmt.query_row([workspace_id, conversation_id], |row| {
        Ok(ConversationRow {
            conversation_id: row.get(0)?,
            workspace_id: row.get(1)?,
            title: empty_to_none(row.get::<_, String>(2)?),
            context_json: row.get(3)?,
            metrics_json: empty_to_none(row.get::<_, String>(4)?),
            created_at: row.get(5)?,
            updated_at: row.get(6)?,
        })
    })
    .ok()
}

/// Parse `ForgeCode` context JSON into a list of entry values.
///
/// `ForgeCode` stores the context as an object with a `messages` array
/// (`ContextRecord`, `forge_repo` `conversation_record.rs`); it never writes a
/// bare array.
fn parse_context_entries(context_json: &str) -> Vec<Value> {
    let Ok(Value::Object(mut context)) = serde_json::from_str::<Value>(context_json) else {
        return Vec::new();
    };

    match context.remove("messages") {
        Some(Value::Array(entries)) => entries,
        _ => Vec::new(),
    }
}

/// Return whether any context entry contains tool usage data.
fn context_entries_have_tool_use(entries: &[Value]) -> bool {
    entries.iter().any(entry_contains_tool_use)
}

/// Return whether a context entry contains tool usage data.
fn entry_contains_tool_use(entry: &Value) -> bool {
    let (kind, payload) = extract_context_variant(entry);
    if kind == "tool" {
        return true;
    }

    extract_embedded_tool_use(payload).is_some()
}

/// Map `ForgeCode` context entries into normalized messages.
fn map_context_messages(
    workspace_id: &str,
    conversation_id: &str,
    entries: &[Value],
    created_at: &str,
    updated_at: &str,
    metrics_json: Option<&str>,
) -> Vec<ClaudeMessage> {
    let mut messages: Vec<ClaudeMessage> = entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            map_context_message(
                workspace_id,
                conversation_id,
                index,
                entry,
                created_at,
                updated_at,
            )
        })
        .collect();

    if let Some(metrics) = extract_metrics_metadata(metrics_json) {
        if let Some(last) = messages.last_mut() {
            attach_message_metadata(last, "forgecodeMetrics", metrics);
        }
    }

    messages
}

/// Map a single `ForgeCode` context entry into a normalized message.
fn map_context_message(
    workspace_id: &str,
    conversation_id: &str,
    index: usize,
    entry: &Value,
    created_at: &str,
    updated_at: &str,
) -> ClaudeMessage {
    let (kind, payload) = extract_context_variant(entry);
    let normalized_payload = merge_context_payload(entry, payload);
    let timestamp = extract_timestamp(&normalized_payload)
        .or_else(|| extract_timestamp(entry))
        .unwrap_or_else(|| fallback_message_timestamp(index, created_at, updated_at));
    let model = extract_model(&normalized_payload).or_else(|| extract_model(entry));

    match kind {
        "tool" => map_tool_message(
            workspace_id,
            conversation_id,
            index,
            &normalized_payload,
            timestamp,
            model,
        ),
        "image" => map_image_message(
            workspace_id,
            conversation_id,
            index,
            &normalized_payload,
            timestamp,
            model,
        ),
        _ => map_text_message(
            workspace_id,
            conversation_id,
            index,
            &normalized_payload,
            timestamp,
            model,
        ),
    }
}

/// Map a text-style `ForgeCode` context entry into a message.
fn map_text_message(
    workspace_id: &str,
    conversation_id: &str,
    index: usize,
    payload: &Value,
    timestamp: String,
    model: Option<String>,
) -> ClaudeMessage {
    let role = normalize_role(extract_role(payload).as_deref(), "user");
    let message_type = message_type_for_role(&role).to_string();
    let content =
        user_prompt_content(&role, payload).or_else(|| build_text_message_content(payload));

    let mut message = build_provider_message(
        PROVIDER_ID,
        message_uuid(workspace_id, conversation_id, index),
        conversation_id,
        timestamp,
        &message_type,
        Some(&role),
        content.clone(),
        model,
    );

    message.usage = extract_usage(payload);
    message.cost_usd = extract_cost_usd(payload);
    message.stop_reason = extract_string(payload, &["stop_reason", "stopReason"]);
    message.tool_use = content.as_ref().and_then(extract_embedded_tool_use);
    if let Some(metadata) = extract_safe_message_metadata(payload) {
        attach_message_metadata(&mut message, "forgecodeMessageMetadata", metadata);
    }

    message
}

/// Return the prompt the user typed for a `ForgeCode` user message.
///
/// `content` is the prompt wrapped for the model (`<task>…</task>` or
/// `<feedback>…</feedback>` plus `<system_date>`); the text as typed is
/// `raw_content: {"Text": "…"}` (`EventValue::Text(UserPrompt)`, `forge_domain`
/// `event.rs`). A `Command` raw content has no typed text, so the wrapped
/// content is kept.
fn user_prompt_content(role: &str, payload: &Value) -> Option<Value> {
    if role != "user" {
        return None;
    }
    payload
        .get("raw_content")?
        .get("Text")?
        .as_str()
        .filter(|text| !text.trim().is_empty())
        .map(|text| Value::String(text.to_string()))
}

/// Map an image-style `ForgeCode` context entry into a message.
fn map_image_message(
    workspace_id: &str,
    conversation_id: &str,
    index: usize,
    payload: &Value,
    timestamp: String,
    model: Option<String>,
) -> ClaudeMessage {
    let role = normalize_role(extract_role(payload).as_deref(), "user");
    let message_type = message_type_for_role(&role).to_string();
    let content = Value::Array(vec![json!({
        "type": "image",
        "source": payload.clone(),
    })]);

    let mut message = build_provider_message(
        PROVIDER_ID,
        message_uuid(workspace_id, conversation_id, index),
        conversation_id,
        timestamp,
        &message_type,
        Some(&role),
        Some(content),
        model,
    );

    message.usage = extract_usage(payload);
    message.cost_usd = extract_cost_usd(payload);
    if let Some(metadata) = extract_safe_message_metadata(payload) {
        attach_message_metadata(&mut message, "forgecodeMessageMetadata", metadata);
    }

    message
}

/// Map a tool-result `ForgeCode` context entry into a user message holding one
/// `tool_result` block.
///
/// `ForgeCode` records `{name, call_id, output:{is_error, values:[...]}}`
/// (`ToolResultRecord`, `ToolOutputRecord`, `ToolValueRecord`) and never writes
/// a tool call in a tool entry: calls live in the assistant text entry's
/// `tool_calls`. The result is paired with its call through `call_id`.
fn map_tool_message(
    workspace_id: &str,
    conversation_id: &str,
    index: usize,
    payload: &Value,
    timestamp: String,
    model: Option<String>,
) -> ClaudeMessage {
    let tool_name = extract_string(payload, &["name"]).unwrap_or_else(|| "tool".to_string());
    let tool_use_id = extract_string(payload, &["call_id"])
        .unwrap_or_else(|| message_uuid(workspace_id, conversation_id, index));
    let output = payload.get("output").unwrap_or(&Value::Null);
    let content = tool_output_content(output);
    let producer_flag = output.get("is_error").and_then(Value::as_bool);
    let failed_shell_exit = shell_failure_exit_code(&content).is_some();
    let is_error_source =
        if producer_flag == Some(true) || (producer_flag.is_some() && !failed_shell_exit) {
            "producer"
        } else {
            "derived"
        };

    let mut message = build_provider_message(
        PROVIDER_ID,
        message_uuid(workspace_id, conversation_id, index),
        conversation_id,
        timestamp,
        "user",
        Some("user"),
        Some(Value::Array(vec![json!({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
            "name": tool_name,
            // A nonzero shell exit is still `is_error:false` in the record (the tool
            // call itself succeeded); only the recorded exit marker reveals it.
            "is_error": producer_flag.unwrap_or(false) || failed_shell_exit,
            "is_error_source": is_error_source,
        })])),
        model,
    );

    if let Some(metadata) = extract_safe_message_metadata(payload) {
        attach_message_metadata(&mut message, "forgecodeMessageMetadata", metadata);
    }
    message
}

/// Join the `values` of a `ForgeCode` tool output into `tool_result` content.
///
/// Text-only output becomes one string (an error result carries two text
/// values); an output holding an image becomes a block array.
fn tool_output_content(output: &Value) -> Value {
    let blocks: Vec<Value> = output
        .get("values")
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(tool_value_block).collect())
        .unwrap_or_default();

    let texts: Option<Vec<&str>> = blocks
        .iter()
        .map(|block| block.get("text").and_then(Value::as_str))
        .collect();
    match texts {
        Some(texts) => Value::String(texts.join("\n")),
        None => Value::Array(blocks),
    }
}

/// Convert one `ToolValueRecord` into a content block; `Empty` yields none.
///
/// The record enum is externally tagged with `camelCase` variant names, so
/// `AI` serializes as `aI`; `Markdown`, `FileDiff` and `Pair` are legacy
/// variants the producer still reads from old rows and folds into text.
fn tool_value_block(value: &Value) -> Option<Value> {
    let text_block = |text: &str| Some(json!({ "type": "text", "text": text }));
    let object = value.as_object()?;

    if let Some(text) = object.get("text").and_then(Value::as_str) {
        return text_block(text);
    }
    if let Some(text) = object.get("markdown").and_then(Value::as_str) {
        return text_block(text);
    }
    if let Some(text) = object
        .get("aI")
        .or_else(|| object.get("ai"))
        .and_then(|ai| ai.get("value"))
        .and_then(Value::as_str)
    {
        return text_block(text);
    }
    if let Some(image) = object.get("image") {
        return Some(json!({ "type": "image", "source": image.clone() }));
    }
    if let Some(path) = object
        .get("fileDiff")
        .and_then(|diff| diff.get("path"))
        .and_then(Value::as_str)
    {
        return text_block(&format!("[File diff: {path}]"));
    }
    object
        .get("pair")
        .and_then(Value::as_array)
        .and_then(|pair| pair.first())
        .and_then(tool_value_block)
}

/// Return the nonzero exit code a `ForgeCode` shell result records.
///
/// The shell tool renders `<shell_output command=".." shell=".." exit_code="N">`
/// with every attribute on its own line (`forge_app` `operation.rs` 597-598 writes
/// `exit_code`; `forge_template` `element.rs` 77-92 renders the attributes) while
/// the record's `is_error` stays false for a nonzero exit. The attribute is the
/// last one before the `<stdout>` child, so it is read from the final line of
/// the opening tag only, never from the command text or the captured output.
fn shell_failure_exit_code(content: &Value) -> Option<i32> {
    let text = content.as_str()?;
    if !text.starts_with("<shell_output\n") {
        return None;
    }
    let opening_tag = &text[..text.find("\n>\n<stdout")?];
    let code = opening_tag
        .lines()
        .next_back()?
        .strip_prefix("  exit_code=\"")?
        .strip_suffix('"')?
        .parse::<i32>()
        .ok()?;
    (code != 0).then_some(code)
}

/// Extract the `ForgeCode` context entry variant name.
///
/// Entries are `{"message":{"text"|"tool"|"image":{..}},"usage":{..}}`; the
/// producer also reads its older unwrapped `{"text"|"tool"|"image":{..}}` form
/// (`ContextMessageRecord` deserializer in `forge_repo` `conversation_record.rs`).
fn extract_context_variant(entry: &Value) -> (&'static str, &Value) {
    let message = entry.get("message").unwrap_or(entry);
    if let Some(text) = message.get("text") {
        return ("text", text);
    }
    if let Some(tool) = message.get("tool") {
        return ("tool", tool);
    }
    if let Some(image) = message.get("image") {
        return ("image", image);
    }

    ("text", entry)
}

/// Merge nested `ForgeCode` payload objects into a single value.
fn merge_context_payload(entry: &Value, payload: &Value) -> Value {
    let mut merged = payload.clone();
    let Some(merged_object) = merged.as_object_mut() else {
        return merged;
    };

    let Some(entry_object) = entry.as_object() else {
        return merged;
    };

    for key in [
        "usage",
        "timestamp",
        "created_at",
        "createdAt",
        "time",
        "cost",
        "cost_usd",
        "costUSD",
    ] {
        if !merged_object.contains_key(key) {
            if let Some(value) = entry_object.get(key) {
                merged_object.insert(key.to_string(), value.clone());
            }
        }
    }

    merged
}

/// Extract a message role from a `ForgeCode` payload.
fn extract_role(value: &Value) -> Option<String> {
    extract_string(value, &["role", "speaker", "author"])
}

/// Normalize `ForgeCode` role names to the shared role set.
fn normalize_role(role: Option<&str>, default_role: &str) -> String {
    let normalized = role.unwrap_or(default_role).trim().to_ascii_lowercase();

    match normalized.as_str() {
        "tool" => "user".to_string(),
        "" => default_role.to_string(),
        _ => normalized,
    }
}

/// Translate a normalized role into the shared message type.
fn message_type_for_role(role: &str) -> &'static str {
    match role {
        "assistant" => "assistant",
        "system" => "system",
        _ => "user",
    }
}

/// Extract the best content value from a `ForgeCode` payload.
fn extract_content_value(payload: &Value) -> Option<Value> {
    if let Some(content) = extract_json(payload, &["content", "message", "body", "value"]) {
        if !value_is_effectively_empty(&content) {
            return Some(content);
        }
    }

    extract_json(payload, &["raw_content", "rawContent"])
        .or_else(|| extract_text(payload).map(Value::String))
}

/// Return whether a JSON value is empty for message rendering.
fn value_is_effectively_empty(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(text) => text.trim().is_empty(),
        Value::Array(items) => items.is_empty(),
        Value::Object(object) => object.is_empty(),
        _ => false,
    }
}

/// Build normalized message content for a text payload.
fn build_text_message_content(payload: &Value) -> Option<Value> {
    let tool_calls = extract_tool_call_blocks(payload);
    if !tool_calls.is_empty() {
        let mut blocks = Vec::new();
        if let Some(text) = extract_text(payload).filter(|text| !text.trim().is_empty()) {
            blocks.push(json!({
                "type": "text",
                "text": text,
            }));
        }
        blocks.extend(tool_calls);
        return Some(Value::Array(blocks));
    }

    extract_content_value(payload).or_else(|| Some(payload.clone()))
}

/// Extract tool-call content blocks from a `ForgeCode` payload.
fn extract_tool_call_blocks(payload: &Value) -> Vec<Value> {
    payload
        .get("tool_calls")
        .or_else(|| payload.get("toolCalls"))
        .and_then(Value::as_array)
        .map(|tool_calls| {
            tool_calls
                .iter()
                .filter_map(normalize_tool_call_block)
                .collect()
        })
        .unwrap_or_default()
}

/// Normalize a `ForgeCode` tool-call block into shared message content.
fn normalize_tool_call_block(tool_call: &Value) -> Option<Value> {
    let name = extract_string(tool_call, &["name"])?;
    let id = extract_string(tool_call, &["call_id", "callId", "id"])
        .unwrap_or_else(|| format!("tool-{}", name.to_ascii_lowercase()));
    let input = extract_json(tool_call, &["arguments", "input", "args", "params"])
        .map(|value| normalize_tool_call_input(&value))
        .unwrap_or(Value::Object(Map::new()));

    Some(json!({
        "type": "tool_use",
        "id": id,
        "name": name,
        "input": input,
    }))
}

/// Normalize `ForgeCode` tool input into a JSON object.
fn normalize_tool_call_input(value: &Value) -> Value {
    match value {
        Value::String(text) => {
            serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.clone()))
        }
        _ => value.clone(),
    }
}

/// Extract embedded tool usage metadata from a `ForgeCode` payload.
fn extract_embedded_tool_use(content: &Value) -> Option<Value> {
    match content {
        Value::Array(items) => items.iter().find_map(|item| {
            if item.get("type").and_then(Value::as_str) == Some("tool_use") {
                Some(item.clone())
            } else {
                None
            }
        }),
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("tool_use") {
                return Some(content.clone());
            }
            if let Some(tool_calls) = object
                .get("tool_calls")
                .or_else(|| object.get("toolCalls"))
                .and_then(Value::as_array)
            {
                return tool_calls.iter().find_map(normalize_tool_call_block);
            }
            object
                .get("tool_use")
                .or_else(|| object.get("toolUse"))
                .cloned()
        }
        _ => None,
    }
}

/// Extract token usage information from `ForgeCode` message payloads.
fn extract_usage(payload: &Value) -> Option<TokenUsage> {
    let usage = payload.get("usage").unwrap_or(payload);
    let direct_input_tokens = extract_u32(usage, &["input_tokens", "inputTokens"]);
    let prompt_tokens = extract_u32(usage, &["prompt_tokens", "promptTokens"]);
    let output_tokens = extract_u32(
        usage,
        &[
            "output_tokens",
            "outputTokens",
            "completion_tokens",
            "completionTokens",
        ],
    );
    let cache_creation_input_tokens = extract_u32(
        usage,
        &["cache_creation_input_tokens", "cacheCreationInputTokens"],
    );
    let cache_read_input_tokens = extract_u32(
        usage,
        &[
            "cache_read_input_tokens",
            "cacheReadInputTokens",
            "cached_tokens",
            "cachedTokens",
        ],
    );
    let input_tokens = direct_input_tokens.or_else(|| {
        prompt_tokens.map(|prompt| prompt.saturating_sub(cache_read_input_tokens.unwrap_or(0)))
    });
    let reasoning_tokens = extract_u32(usage, &["reasoning_tokens", "reasoningTokens"]);
    let service_tier = extract_string(usage, &["service_tier", "serviceTier"]);

    if input_tokens.is_none()
        && output_tokens.is_none()
        && cache_creation_input_tokens.is_none()
        && cache_read_input_tokens.is_none()
        && reasoning_tokens.is_none()
        && service_tier.is_none()
    {
        None
    } else {
        Some(TokenUsage {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens,
            cache_read_input_tokens,
            reasoning_tokens,
            service_tier,
            ..Default::default()
        })
    }
}

/// Extract a USD cost value from `ForgeCode` metrics.
fn extract_cost_usd(payload: &Value) -> Option<f64> {
    extract_f64(payload, &["cost_usd", "costUSD", "cost"]).or_else(|| {
        payload
            .get("usage")
            .and_then(|usage| extract_f64(usage, &["cost"]))
    })
}

/// Extract the best timestamp for a `ForgeCode` context entry.
fn extract_timestamp(value: &Value) -> Option<String> {
    extract_json(value, &["timestamp", "created_at", "createdAt", "time"]).and_then(|raw| {
        normalize_timestamp_value(&raw)
            .and_then(|normalized| normalized.as_str().map(ToString::to_string))
    })
}

/// Build a fallback timestamp for a message missing its own timestamp.
fn fallback_message_timestamp(index: usize, created_at: &str, updated_at: &str) -> String {
    if index == 0 && !created_at.is_empty() {
        created_at.to_string()
    } else if !updated_at.is_empty() {
        updated_at.to_string()
    } else {
        created_at.to_string()
    }
}

/// Extract the model identifier from a `ForgeCode` payload.
fn extract_model(value: &Value) -> Option<String> {
    extract_string(value, &["model", "model_id", "modelId"])
}

/// Extract `ForgeCode` metadata fields that are safe to expose on messages.
fn extract_safe_message_metadata(payload: &Value) -> Option<Value> {
    let mut metadata = Map::new();

    if let Some(file_operations) = extract_json(payload, &["file_operations", "fileOperations"]) {
        metadata.insert("fileOperations".to_string(), file_operations);
    }
    if let Some(files_accessed) = extract_json(payload, &["files_accessed", "filesAccessed"]) {
        metadata.insert("filesAccessed".to_string(), files_accessed);
    }
    if let Some(reasoning_details) =
        extract_json(payload, &["reasoning_details", "reasoningDetails"])
    {
        metadata.insert("reasoningDetails".to_string(), reasoning_details);
    }
    if let Some(raw_content) = extract_json(payload, &["raw_content", "rawContent"]) {
        metadata.insert("rawContent".to_string(), raw_content);
    }

    let related_ids = extract_related_conversation_ids(payload);
    if !related_ids.is_empty() {
        metadata.insert(
            "relatedConversationIds".to_string(),
            Value::Array(related_ids.into_iter().map(Value::String).collect()),
        );
    }

    if metadata.is_empty() {
        None
    } else {
        Some(Value::Object(metadata))
    }
}

/// Extract metrics metadata that should be attached to normalized messages.
fn extract_metrics_metadata(metrics_json: Option<&str>) -> Option<Value> {
    let metrics_json = metrics_json?;
    let Ok(metrics) = serde_json::from_str::<Value>(metrics_json) else {
        return None;
    };

    let mut metadata = Map::new();

    if let Some(session_start_time) = extract_json(
        &metrics,
        &[
            "session_start_time",
            "sessionStartTime",
            "start_time",
            "startTime",
            "started_at",
            "startedAt",
        ],
    ) {
        metadata.insert(
            "sessionStartTime".to_string(),
            normalize_timestamp_value(&session_start_time).unwrap_or(session_start_time),
        );
    }

    if let Some(file_operations) = extract_json(&metrics, &["file_operations", "fileOperations"]) {
        metadata.insert("fileOperations".to_string(), file_operations);
    }
    if let Some(files_accessed) = extract_json(&metrics, &["files_accessed", "filesAccessed"]) {
        metadata.insert("filesAccessed".to_string(), files_accessed);
    }
    if let Some(files_changed) = extract_json(&metrics, &["files_changed", "filesChanged"]) {
        if !metadata.contains_key("fileOperations") {
            if let Some(count) = files_changed.as_object().map(|object| object.len() as u64) {
                metadata.insert("fileOperations".to_string(), Value::from(count));
            }
        }
        if !metadata.contains_key("filesAccessed") {
            if let Some(paths) = files_changed.as_object().map(|object| {
                object
                    .keys()
                    .cloned()
                    .map(Value::String)
                    .collect::<Vec<_>>()
            }) {
                metadata.insert("filesAccessed".to_string(), Value::Array(paths));
            }
        }
    }

    let related_ids = extract_related_conversation_ids(&metrics);
    if !related_ids.is_empty() {
        metadata.insert(
            "relatedConversationIds".to_string(),
            Value::Array(related_ids.into_iter().map(Value::String).collect()),
        );
    }

    if metadata.is_empty() {
        None
    } else {
        Some(Value::Object(metadata))
    }
}

/// Attach provider-specific metadata to a normalized message.
fn attach_message_metadata(message: &mut ClaudeMessage, key: &str, value: Value) {
    if let Some(Value::Object(object)) = &mut message.data {
        object.insert(key.to_string(), value);
    } else {
        let mut object = Map::new();
        object.insert(key.to_string(), value);
        message.data = Some(Value::Object(object));
    }
}

/// Extract related `ForgeCode` conversation ids from metadata.
fn extract_related_conversation_ids(value: &Value) -> Vec<String> {
    let mut ids = BTreeSet::new();

    for key in [
        "related_conversation_ids",
        "relatedConversationIds",
        "related_conversations",
        "relatedConversations",
        "related_conversation_id",
        "relatedConversationId",
    ] {
        if let Some(raw) = value.get(key) {
            collect_related_conversation_ids(raw, &mut ids);
        }
    }

    ids.into_iter().collect()
}

/// Collect related `ForgeCode` conversation ids from nested values.
fn collect_related_conversation_ids(value: &Value, ids: &mut BTreeSet<String>) {
    match value {
        Value::String(id) if !id.trim().is_empty() => {
            ids.insert(id.clone());
        }
        Value::Number(number) => {
            ids.insert(number.to_string());
        }
        Value::Array(items) => {
            for item in items {
                collect_related_conversation_ids(item, ids);
            }
        }
        Value::Object(object) => {
            for key in ["id", "conversation_id", "conversationId"] {
                if let Some(inner) = object.get(key) {
                    collect_related_conversation_ids(inner, ids);
                }
            }
        }
        _ => {}
    }
}

/// Extract a string-like text value from JSON content.
fn extract_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Object(object) => {
            if let Some(text) = object.get("text").and_then(Value::as_str) {
                return Some(text.to_string());
            }
            if let Some(content) = object.get("content") {
                match content {
                    Value::String(text) => Some(text.clone()),
                    Value::Array(items) => items.iter().find_map(|item| {
                        item.get("text")
                            .and_then(Value::as_str)
                            .map(ToString::to_string)
                    }),
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Extract a string value from a JSON object key.
fn extract_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        value.get(key).and_then(|inner| match inner {
            Value::String(text) => Some(text.clone()),
            Value::Number(number) => Some(number.to_string()),
            _ => None,
        })
    })
}

/// Clone a JSON value from a JSON object key.
fn extract_json(value: &Value, keys: &[&str]) -> Option<Value> {
    keys.iter().find_map(|key| value.get(key).cloned())
}

/// Extract a u32 value from a JSON object key.
fn extract_u32(value: &Value, keys: &[&str]) -> Option<u32> {
    keys.iter()
        .filter_map(|key| value.get(key))
        .find_map(value_to_u32)
}

/// Convert a JSON value into a u32 when possible.
fn value_to_u32(value: &Value) -> Option<u32> {
    match value {
        Value::Number(number) => number.as_u64().and_then(|v| u32::try_from(v).ok()),
        Value::String(text) => text.parse::<u32>().ok(),
        Value::Object(object) => object
            .get("actual")
            .or_else(|| object.get("value"))
            .or_else(|| object.get("count"))
            .and_then(value_to_u32),
        _ => None,
    }
}

/// Extract an f64 value from a JSON object key.
fn extract_f64(value: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .filter_map(|key| value.get(key))
        .find_map(value_to_f64)
}

/// Convert a JSON value into an f64 when possible.
fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.parse::<f64>().ok(),
        Value::Object(object) => object
            .get("actual")
            .or_else(|| object.get("value"))
            .or_else(|| object.get("amount"))
            .and_then(value_to_f64),
        _ => None,
    }
}

/// Resolve the preferred `ForgeCode` workspace display name from aggregated rows.
fn resolve_workspace_display_name_from_rows(
    workspace_id: &str,
    rows: &[ConversationRow],
) -> String {
    let mut display_name_votes: BTreeMap<String, usize> = BTreeMap::new();

    for row in rows {
        if let Some(display_name) =
            extract_workspace_display_name_from_context_json(&row.context_json)
        {
            *display_name_votes.entry(display_name).or_default() += 1;
        }
    }

    choose_workspace_display_name(workspace_id, &display_name_votes)
}

/// Choose the best workspace display name from vote counts.
fn choose_workspace_display_name(
    workspace_id: &str,
    display_name_votes: &BTreeMap<String, usize>,
) -> String {
    display_name_votes
        .iter()
        .max_by(|(left_name, left_votes), (right_name, right_votes)| {
            left_votes
                .cmp(right_votes)
                .then_with(|| right_name.len().cmp(&left_name.len()))
                .then_with(|| left_name.cmp(right_name))
        })
        .map(|(display_name, _)| display_name.clone())
        .unwrap_or_else(|| project_display_name(workspace_id))
}

/// Extract a workspace display name from `ForgeCode` context JSON.
fn extract_workspace_display_name_from_context_json(context_json: &str) -> Option<String> {
    let parsed = serde_json::from_str::<Value>(context_json).ok()?;
    extract_workspace_display_name_from_value(&parsed)
}

/// Extract a workspace display name from a JSON value.
fn extract_workspace_display_name_from_value(value: &Value) -> Option<String> {
    let home_dir = crate::utils::home_dir();
    let home_dir = home_dir.as_deref();
    let mut cwd_votes: BTreeMap<String, usize> = BTreeMap::new();
    collect_workspace_display_name_votes(value, home_dir, &mut cwd_votes);

    cwd_votes
        .into_iter()
        .max_by(|(left_name, left_votes), (right_name, right_votes)| {
            left_votes
                .cmp(right_votes)
                .then_with(|| right_name.len().cmp(&left_name.len()))
                .then_with(|| left_name.cmp(right_name))
        })
        .map(|(display_name, _)| display_name)
}

/// Collect workspace display-name votes from conversation rows.
fn collect_workspace_display_name_votes(
    value: &Value,
    home_dir: Option<&Path>,
    cwd_votes: &mut BTreeMap<String, usize>,
) {
    match value {
        Value::Object(map) => {
            if let Some(display_name) = map
                .get("cwd")
                .and_then(Value::as_str)
                .and_then(|cwd| display_name_from_cwd(cwd, home_dir))
            {
                *cwd_votes.entry(display_name).or_default() += 1;
            }

            for nested in map.values() {
                collect_workspace_display_name_votes(nested, home_dir, cwd_votes);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_workspace_display_name_votes(item, home_dir, cwd_votes);
            }
        }
        _ => {}
    }
}

/// Derive a workspace display name from a cwd path.
fn display_name_from_cwd(cwd: &str, home_dir: Option<&Path>) -> Option<String> {
    let trimmed = cwd.trim();
    if trimmed.is_empty() {
        return None;
    }

    let cwd_path = Path::new(trimmed);
    if home_dir.is_some_and(|home| cwd_path == home) {
        return None;
    }

    cwd_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToString::to_string)
}

/// Extract cwd values from `ForgeCode` context JSON.
fn extract_cwds_from_context_json(context_json: &str) -> BTreeMap<String, usize> {
    let mut cwd_votes = BTreeMap::new();
    if let Ok(parsed) = serde_json::from_str::<Value>(context_json) {
        collect_cwd_votes(&parsed, &mut cwd_votes);
    }
    cwd_votes
}

/// Collect cwd vote counts from `ForgeCode` conversation rows.
fn collect_cwd_votes(value: &Value, cwd_votes: &mut BTreeMap<String, usize>) {
    match value {
        Value::Object(map) => {
            if let Some(cwd) = map.get("cwd").and_then(Value::as_str) {
                let trimmed = cwd.trim();
                if !trimmed.is_empty() {
                    *cwd_votes.entry(trimmed.to_string()).or_default() += 1;
                }
            }
            for nested in map.values() {
                collect_cwd_votes(nested, cwd_votes);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_cwd_votes(item, cwd_votes);
            }
        }
        _ => {}
    }
}

/// Picks the most-voted cwd, filtering out home directories.
fn choose_best_cwd(cwd_votes: &BTreeMap<String, usize>) -> Option<String> {
    let home_dir = crate::utils::home_dir();
    cwd_votes
        .iter()
        .filter(|(path, _)| {
            // Exclude bare home directories — they don't represent a real project
            let p = Path::new(path.as_str());
            home_dir.as_deref().is_none_or(|home| p != home)
        })
        .max_by(|(left_path, left_votes), (right_path, right_votes)| {
            left_votes
                .cmp(right_votes)
                .then_with(|| right_path.len().cmp(&left_path.len()))
                .then_with(|| left_path.cmp(right_path))
        })
        .map(|(path, _)| path.clone())
}

/// Build the `ForgeCode` project display name for a workspace.
fn project_display_name(workspace_id: &str) -> String {
    format!("Workspace {workspace_id}")
}

/// Build the virtual project path for a `ForgeCode` workspace.
fn project_virtual_path(workspace_id: &str) -> String {
    format!("forgecode://workspace/{workspace_id}")
}

/// Build the virtual session path for a `ForgeCode` conversation.
fn session_virtual_path(workspace_id: &str, conversation_id: &str) -> String {
    format!("forgecode://workspace/{workspace_id}/conversation/{conversation_id}")
}

/// Build the virtual file path alias for a `ForgeCode` conversation.
fn session_file_virtual_path(workspace_id: &str, conversation_id: &str) -> String {
    format!("forgecode-db://workspace/{workspace_id}/conversation/{conversation_id}")
}

/// Build a stable message UUID for a `ForgeCode` conversation entry.
fn message_uuid(workspace_id: &str, conversation_id: &str, index: usize) -> String {
    format!("forgecode://workspace/{workspace_id}/conversation/{conversation_id}/message/{index}")
}

/// Parse a `ForgeCode` virtual project path into a workspace id.
fn parse_workspace_project_path(project_path: &str) -> Option<String> {
    let workspace_id = project_path
        .strip_prefix("forgecode://workspace/")
        .unwrap_or(project_path);

    if is_valid_virtual_component(workspace_id) {
        Some(workspace_id.to_string())
    } else {
        None
    }
}

/// Parse a `ForgeCode` virtual conversation path.
fn parse_conversation_path(session_path: &str) -> Option<(String, String)> {
    let raw_path = session_path
        .strip_prefix("forgecode-db://workspace/")
        .or_else(|| session_path.strip_prefix("forgecode://workspace/"))
        .unwrap_or(session_path);
    let (workspace_id, conversation_id) = raw_path.split_once("/conversation/")?;

    if !is_valid_virtual_component(workspace_id) || !is_valid_virtual_component(conversation_id) {
        return None;
    }

    Some((workspace_id.to_string(), conversation_id.to_string()))
}

/// Validate a single `ForgeCode` virtual path component.
///
/// IDs sit at the boundary of rename/delete/load routing, so we apply
/// the standard `^[A-Za-z0-9_-]+$` allowlist used elsewhere for path
/// components rather than the looser "no slashes / dots" check.
fn is_valid_virtual_component(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Quote an `SQLite` identifier for generated `ForgeCode` queries.
fn quote_ident(identifier: &str) -> String {
    format!("\"{}\"", identifier.replace('"', "\"\""))
}

/// Build a text-cast SQL expression for a required column.
fn cast_text_expr(identifier: &str) -> String {
    format!("COALESCE(CAST({} AS TEXT), '')", quote_ident(identifier))
}

/// Build a text-cast SQL expression for an optional column.
fn optional_cast_text_expr(identifier: Option<&str>) -> String {
    identifier.map_or_else(|| "''".to_string(), cast_text_expr)
}

/// Build an ORDER BY expression for an optional timestamp column.
fn optional_order_expr(identifier: Option<&str>) -> String {
    identifier
        .map(quote_ident)
        .unwrap_or_else(|| "rowid".to_string())
}

/// Convert blank strings into None.
fn empty_to_none(value: String) -> Option<String> {
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
}

/// Return the newer of two timestamp strings.
fn latest_timestamp(created_at: &str, updated_at: &str) -> String {
    let updated = normalize_timestamp_text(updated_at);
    if updated.is_empty() {
        normalize_timestamp_text(created_at)
    } else {
        updated
    }
}

/// Return the latest timestamp from a set of candidates.
fn max_timestamp(current: &str, candidate: &str) -> String {
    if current.is_empty() {
        return candidate.to_string();
    }
    if candidate.is_empty() {
        return current.to_string();
    }

    if compare_timestamps(candidate, current).is_gt() {
        candidate.to_string()
    } else {
        current.to_string()
    }
}

/// Compare two timestamps after normalizing their formats.
fn compare_timestamps(left: &str, right: &str) -> std::cmp::Ordering {
    match (timestamp_sort_key(left), timestamp_sort_key(right)) {
        (Some(left_ts), Some(right_ts)) => left_ts.cmp(&right_ts),
        (Some(_), None) => std::cmp::Ordering::Greater,
        (None, Some(_)) => std::cmp::Ordering::Less,
        (None, None) => left.cmp(right),
    }
}

/// Build a sortable key for timestamp comparisons.
fn timestamp_sort_key(value: &str) -> Option<i64> {
    let normalized = normalize_timestamp_text(value);
    if normalized.is_empty() {
        return None;
    }

    DateTime::parse_from_rfc3339(&normalized)
        .ok()
        .map(|timestamp| timestamp.timestamp_millis())
}

/// Normalize a timestamp string into RFC3339 form when possible.
fn normalize_timestamp_text(value: &str) -> String {
    normalize_timestamp_value(&Value::String(value.to_string()))
        .and_then(|normalized| normalized.as_str().map(ToString::to_string))
        .unwrap_or_default()
}

/// Normalize a timestamp JSON value into text.
fn normalize_timestamp_value(value: &Value) -> Option<Value> {
    match value {
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return None;
            }

            if let Ok(raw_number) = trimmed.parse::<i64>() {
                return normalize_timestamp_number(raw_number).map(Value::String);
            }

            if let Ok(timestamp) = DateTime::parse_from_rfc3339(trimmed) {
                return Some(Value::String(timestamp.with_timezone(&Utc).to_rfc3339()));
            }

            if let Ok(timestamp) = NaiveDateTime::parse_from_str(trimmed, "%Y-%m-%d %H:%M:%S%.f") {
                return Some(Value::String(
                    timestamp
                        .and_utc()
                        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                ));
            }

            if let Ok(timestamp) = NaiveDateTime::parse_from_str(trimmed, "%Y-%m-%dT%H:%M:%S%.f") {
                return Some(Value::String(
                    timestamp
                        .and_utc()
                        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                ));
            }

            Some(Value::String(trimmed.to_string()))
        }
        Value::Number(number) => number
            .as_i64()
            .and_then(normalize_timestamp_number)
            .map(Value::String),
        _ => None,
    }
}

/// Normalize a numeric timestamp into RFC3339 text.
fn normalize_timestamp_number(value: i64) -> Option<String> {
    let millis = if value.abs() >= 10_000_000_000 {
        value
    } else {
        value.saturating_mul(1000)
    };

    DateTime::from_timestamp_millis(millis).map(|timestamp| timestamp.to_rfc3339())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use rusqlite::params;
    use serial_test::serial;
    use tempfile::TempDir;

    /// RAII guard that restores a process-wide env var when dropped.
    ///
    /// Use this instead of manual save / set / restore blocks so that the
    /// original value is put back even if an assertion in the test panics.
    /// Tests in this module rely on `--test-threads=1` (see `CLAUDE.md`
    /// "Phase 1: Quality Gate") because env vars are global to the process.
    ///
    /// `original` is stored as `OsString` rather than `String` so that
    /// non-UTF-8 values (legitimate on macOS / Linux) are restored
    /// losslessly on drop instead of being silently dropped.
    struct EnvGuard {
        key: &'static str,
        original: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            let original = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, original }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    /// Workspace id as `ForgeCode` stores it: a 64-bit hash in a `BIGINT` column.
    const WORKSPACE: &str = "8945904052846126217";
    /// Second workspace for grouping tests.
    const OTHER_WORKSPACE: &str = "1844674407370955161";
    /// Conversation id of the tool session captured from `@antinomyhq/forge` 2.13.21.
    const TOOL_SESSION: &str = "22703e47-7aae-42e9-8b45-30626ad9e2f6";
    /// Conversation id of the second captured session (text only).
    const TEXT_SESSION: &str = "315d1b00-062b-46fc-beb4-e64a7164e0cd";

    /// Context entries copied from a session written by `@antinomyhq/forge`
    /// 2.13.21 (one record per line): system information, a `<task>` prompt, a
    /// `read` call and result, a `<feedback>` prompt, a `shell` call whose result
    /// exits 1 with `is_error:false`, a final reply, and a `read` of a missing
    /// file whose result has `is_error:true` and two values. Paths are rewritten
    /// to `/home/user/demo`, long system text and stderr are trimmed.
    const TOOL_SESSION_ENTRIES: &str = r#"{"message":{"text":{"role":"System","content":"<system_information>\n<operating_system>macos</operating_system>\n<current_working_directory>/home/user/demo</current_working_directory>\n<default_shell>/bin/sh</default_shell>\n<home_directory>/home/user</home_directory>\n<file_list>\n - src/\n - package.json\n - test.js\n</file_list>\n<workspace_extensions command=\"git ls-files\" files=\"3\" extensions=\"2\">\n - .js: 2 files (67%)\n - .json: 1 files (33%)\n</workspace_extensions>\n</system_information>","model":"qwen2.5-coder:7b"}}}
{"message":{"text":{"role":"User","content":"<task>Read src/math.js and add a function mul(a, b) that returns a*b, exported like add.</task>\n<system_date>2026-11-25</system_date>\n","raw_content":{"Text":"Read src/math.js and add a function mul(a, b) that returns a*b, exported like add."},"model":"qwen2.5-coder:7b"}}}
{"message":{"text":{"role":"Assistant","content":"","tool_calls":[{"name":"read","call_id":"call_168106ba","arguments":{"file_path":"/home/user/demo/src/math.js"}}],"model":"qwen2.5-coder:7b"}},"usage":{"prompt_tokens":{"actual":100},"completion_tokens":{"actual":20},"total_tokens":{"actual":120},"cached_tokens":{"actual":0}}}
{"message":{"tool":{"name":"read","call_id":"call_168106ba","output":{"is_error":false,"values":[{"text":"<file\n  path=\"/home/user/demo/src/math.js\"\n  display_lines=\"1-2000\"\n  total_lines=\"4\"\n><![CDATA[1:function add(a, b) {\n2:  return a + b;\n3:}\n4:module.exports = { add };]]>\n</file>"}]}}}}
{"message":{"text":{"role":"User","content":"<feedback>Run npm test and tell me why it fails.</feedback>\n<system_date>2026-11-25</system_date>\n","raw_content":{"Text":"Run npm test and tell me why it fails."},"model":"qwen2.5-coder:7b"}}}
{"message":{"text":{"role":"Assistant","content":"","tool_calls":[{"name":"shell","call_id":"call_d338cce3","arguments":{"command":"npm test","cwd":"/home/user/demo"}}],"model":"qwen2.5-coder:7b"}},"usage":{"prompt_tokens":{"actual":100},"completion_tokens":{"actual":20},"total_tokens":{"actual":120},"cached_tokens":{"actual":0}}}
{"message":{"tool":{"name":"shell","call_id":"call_d338cce3","output":{"is_error":false,"values":[{"text":"<shell_output\n  command=\"npm test\"\n  shell=\"/bin/sh\"\n  exit_code=\"1\"\n>\n<stdout\n  total_lines=\"4\"\n><![CDATA[\n> demo@1.0.0 test\n> node test.js\n]]>\n</stdout>\n<stderr\n  total_lines=\"25\"\n><![CDATA[node:assert:95\n  throw new AssertionError(obj);\n  ^\n\nAssertionError [ERR_ASSERTION]: Expected values to be strictly equal:\n\n4 !== 5\n]]>\n</stderr>\n</shell_output>"}]}}}}
{"message":{"text":{"role":"Assistant","content":"`npm test` fails because test.js asserts add(2, 2) === 5, but add(2, 2) returns 4.","model":"qwen2.5-coder:7b"}},"usage":{"prompt_tokens":{"actual":110},"completion_tokens":{"actual":20},"total_tokens":{"actual":130},"cached_tokens":{"actual":0}}}
{"message":{"text":{"role":"User","content":"<feedback>Read the file missing.txt</feedback>\n<system_date>2026-11-25</system_date>\n","raw_content":{"Text":"Read the file missing.txt"},"model":"qwen2.5-coder:7b"}}}
{"message":{"text":{"role":"Assistant","content":"","tool_calls":[{"name":"read","call_id":"call_46263e5d","arguments":{"file_path":"/home/user/demo/missing.txt"}}],"model":"qwen2.5-coder:7b"}},"usage":{"prompt_tokens":{"actual":100},"completion_tokens":{"actual":20},"total_tokens":{"actual":120},"cached_tokens":{"actual":0}}}
{"message":{"tool":{"name":"read","call_id":"call_46263e5d","output":{"is_error":true,"values":[{"text":"<tool_call_error>\n<cause><![CDATA[Failed to get metadata for file /home/user/demo/missing.txt\n\nCaused by:\n    0: No such file or directory (os error 2)]]></cause>\n<reflection>You must now deeply reflect on the error above:\n1. Pinpoint exactly what was wrong with the tool call — was it the wrong tool, incorrect or missing parameters, or malformed structure?\n2. Explain why that mistake happened. Did you misunderstand the tool's schema? Miss a required field? Misread the context?\n3. Make the correct tool call as it should have been made.\n\nDo NOT skip this reflection.</reflection>\n</tool_call_error>"},{"text":"<retry>Tool call failed\n- **Attempts remaining:** 2\n- **Next steps:** Analyze the error, identify the root cause, and adjust your approach before retrying.</retry>"}]}}}}
{"message":{"text":{"role":"Assistant","content":"The file missing.txt does not exist.","model":"qwen2.5-coder:7b"}},"usage":{"prompt_tokens":{"actual":110},"completion_tokens":{"actual":20},"total_tokens":{"actual":130},"cached_tokens":{"actual":0}}}"#;

    /// The `metrics` column of the captured tool session.
    const TOOL_SESSION_METRICS: &str = r#"{"started_at":"2026-11-25T03:41:14.757220Z","files_changed":{"/home/user/demo/src/math.js":{"lines_added":4,"lines_removed":1,"content_hash":"07a88a323565db77cfb6b5cf7035565aef7451b5fe9209f9666d3e22b714c766","tool":"patch"},"/home/user/demo/test.js":{"lines_added":0,"lines_removed":0,"content_hash":"e2fe857781414a1d0b3dd9d86383499df001b05c3feb3fca3ec31388c17d683e","tool":"read"}},"files_accessed":["/home/user/demo/test.js","/home/user/demo/src/math.js"]}"#;

    /// Parse the captured tool-session entries.
    fn tool_session_entries() -> Vec<Value> {
        TOOL_SESSION_ENTRIES
            .lines()
            .map(|line| serde_json::from_str(line).expect("fixture record is JSON"))
            .collect()
    }

    /// Wrap entries in the context object `ForgeCode` stores.
    fn context_json(conversation_id: &str, entries: &[Value]) -> String {
        json!({ "conversation_id": conversation_id, "messages": entries }).to_string()
    }

    /// Entries of a minimal session: one prompt, one `shell` call run in `cwd`.
    fn shell_session_entries(prompt: &str, cwd: &str) -> Vec<Value> {
        vec![
            json!({"message": {"text": {
                "role": "User",
                "content": format!("<task>{prompt}</task>\n<system_date>2026-11-25</system_date>\n"),
                "raw_content": {"Text": prompt},
            }}}),
            json!({"message": {"text": {
                "role": "Assistant",
                "content": "",
                "tool_calls": [{
                    "name": "shell",
                    "call_id": "call_0001",
                    "arguments": {"command": "ls", "cwd": cwd},
                }],
            }}}),
        ]
    }

    /// Create a temporary `ForgeCode` test database with the real schema
    /// (`conversations` table from the 2.13.21 migrations).
    fn create_test_db(tmp: &TempDir) -> Connection {
        let db_path = tmp.path().join(".forge.db");
        let conn = Connection::open(db_path).expect("create forgecode test db");
        conn.execute_batch(
            "CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY NOT NULL,
                title TEXT,
                workspace_id BIGINT NOT NULL,
                context TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP,
                metrics TEXT
            );",
        )
        .expect("create conversations table");
        conn
    }

    /// Insert one conversation row.
    #[allow(clippy::too_many_arguments)]
    fn insert_conversation(
        conn: &Connection,
        id: &str,
        workspace_id: &str,
        title: Option<&str>,
        context: Option<&str>,
        metrics: Option<&str>,
        created_at: &str,
        updated_at: &str,
    ) {
        conn.execute(
            "INSERT INTO conversations (conversation_id, workspace_id, title, context, metrics, created_at, updated_at)
             VALUES (?1, CAST(?2 AS BIGINT), ?3, ?4, ?5, ?6, ?7)",
            params![id, workspace_id, title, context, metrics, created_at, updated_at],
        )
        .unwrap();
    }

    /// Seed the temporary `ForgeCode` test database with fixture rows.
    fn seed_test_data(conn: &Connection) {
        let entries = tool_session_entries();
        insert_conversation(
            conn,
            TOOL_SESSION,
            WORKSPACE,
            Some("Mock tool session"),
            Some(&context_json(TOOL_SESSION, &entries)),
            Some(TOOL_SESSION_METRICS),
            "2026-11-25 03:38:50.911065",
            "2026-11-25 03:41:14.905489",
        );

        // Text-only session: system information, the prompt, a plain reply.
        let text_entries = [entries[0].clone(), entries[1].clone(), entries[7].clone()];
        insert_conversation(
            conn,
            TEXT_SESSION,
            WORKSPACE,
            Some("Math Library: Implement a multiply function"),
            Some(&context_json(TEXT_SESSION, &text_entries)),
            Some(r#"{"started_at":"2026-11-25T03:37:13.405476Z","files_changed":{}}"#),
            "2026-11-25 03:33:13.465471",
            "2026-11-25 03:37:37.174669",
        );

        // A second workspace.
        insert_conversation(
            conn,
            "0b3f6c1e-6f43-4c5e-9d57-0d4a8f5b2c11",
            OTHER_WORKSPACE,
            Some("Lab session"),
            Some(&context_json(
                "0b3f6c1e-6f43-4c5e-9d57-0d4a8f5b2c11",
                &shell_session_entries("list the files", "/home/user/lab"),
            )),
            None,
            "2026-11-24 09:15:00.000000",
            "2026-11-24 09:15:00.000000",
        );

        insert_conversation(
            conn,
            "conv-null",
            WORKSPACE,
            Some("Ignored null context"),
            None,
            None,
            "2026-11-23 08:00:00.000000",
            "2026-11-23 08:00:00.000000",
        );

        insert_conversation(
            conn,
            "conv-bad",
            WORKSPACE,
            Some("Malformed context"),
            Some("{not-json}"),
            None,
            "2026-11-25 10:00:00.000000",
            "2026-11-25 10:05:00.000000",
        );
    }

    #[test]
    /// Extract usage derives consumed input from prompt and cached tokens.
    fn extract_usage_derives_consumed_input_from_prompt_and_cached_tokens() {
        let usage = extract_usage(&json!({
            "usage": {
                "prompt_tokens": { "actual": 120 },
                "completion_tokens": { "actual": 45 },
                "cached_tokens": { "actual": 30 },
                "cost": 0.125
            }
        }))
        .expect("usage should be extracted");

        assert_eq!(usage.input_tokens, Some(90));
        assert_eq!(usage.output_tokens, Some(45));
        assert_eq!(usage.cache_read_input_tokens, Some(30));
        assert_eq!(usage.cache_creation_input_tokens, None);
    }

    #[test]
    /// Extract usage preserves explicit input tokens.
    fn extract_usage_preserves_explicit_input_tokens() {
        let usage = extract_usage(&json!({
            "usage": {
                "input_tokens": 120,
                "output_tokens": 45,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 10
            }
        }))
        .expect("usage should be extracted");

        assert_eq!(usage.input_tokens, Some(120));
        assert_eq!(usage.output_tokens, Some(45));
        assert_eq!(usage.cache_read_input_tokens, Some(30));
        assert_eq!(usage.cache_creation_input_tokens, Some(10));
    }

    /// Load the messages of the captured tool session from a seeded database.
    fn tool_session_messages() -> Vec<ClaudeMessage> {
        let tmp = tempfile::tempdir().unwrap();
        let conn = create_test_db(&tmp);
        seed_test_data(&conn);
        drop(conn);

        load_messages_from_db(&tmp.path().to_string_lossy(), WORKSPACE, TOOL_SESSION).unwrap()
    }

    /// The first content block of `kind` in a message.
    fn first_block<'a>(message: &'a ClaudeMessage, kind: &str) -> &'a Value {
        message
            .content
            .as_ref()
            .and_then(Value::as_array)
            .and_then(|blocks| blocks.iter().find(|block| block["type"] == kind))
            .unwrap_or_else(|| panic!("message has no {kind} block"))
    }

    /// Map one context entry the way a session load does.
    fn map_entry(entry: &Value) -> ClaudeMessage {
        map_context_message("w", "c", 0, entry, "", "")
    }

    #[test]
    /// Verify `SQLite` scan projects groups rows by workspace.
    #[serial]
    fn sqlite_scan_projects_groups_rows_by_workspace() {
        let _home = crate::test_utils::SandboxHome::new();
        let tmp = tempfile::tempdir().unwrap();
        let conn = create_test_db(&tmp);
        seed_test_data(&conn);
        drop(conn);

        let projects =
            scan_projects_from_db(&tmp.path().to_string_lossy(), &ScanReport::default()).unwrap();
        assert_eq!(projects.len(), 2);
        assert_eq!(projects[0].name, "demo");
        assert_eq!(
            projects[0].path,
            format!("forgecode://workspace/{WORKSPACE}")
        );
        assert_eq!(projects[0].actual_path, "/home/user/demo");
        assert_eq!(projects[0].session_count, 3);
        assert_eq!(projects[0].message_count, 15);
        assert_eq!(projects[0].storage_type, Some("sqlite".to_string()));
        assert_eq!(projects[1].name, "lab");
        assert_eq!(
            projects[1].path,
            format!("forgecode://workspace/{OTHER_WORKSPACE}")
        );
        assert_eq!(projects[1].actual_path, "/home/user/lab");
        assert_eq!(projects[1].session_count, 1);
        assert_eq!(projects[1].message_count, 2);
    }

    #[test]
    /// Verify `SQLite` load sessions filters null context and preserves virtual ids.
    #[serial]
    fn sqlite_load_sessions_filters_null_context_and_preserves_virtual_ids() {
        let _home = crate::test_utils::SandboxHome::new();
        let tmp = tempfile::tempdir().unwrap();
        let conn = create_test_db(&tmp);
        seed_test_data(&conn);
        drop(conn);

        let sessions = load_sessions_from_db(
            &tmp.path().to_string_lossy(),
            WORKSPACE,
            &ScanReport::default(),
        )
        .unwrap();
        assert_eq!(sessions.len(), 3);
        assert_eq!(
            sessions[0].session_id,
            format!("forgecode://workspace/{WORKSPACE}/conversation/conv-bad")
        );
        assert_eq!(
            sessions[0].file_path,
            format!("forgecode-db://workspace/{WORKSPACE}/conversation/conv-bad")
        );
        assert_eq!(sessions[0].message_count, 0);
        assert_eq!(sessions[0].project_name, "demo");
        assert!(!sessions[0].has_tool_use);

        assert_eq!(sessions[1].actual_session_id, TOOL_SESSION);
        assert_eq!(sessions[1].project_name, "demo");
        assert_eq!(sessions[1].message_count, 12);
        assert!(sessions[1].has_tool_use);
        assert_eq!(sessions[1].summary, Some("Mock tool session".to_string()));
        assert_eq!(sessions[1].storage_type, Some("sqlite".to_string()));

        assert_eq!(sessions[2].actual_session_id, TEXT_SESSION);
        assert_eq!(sessions[2].message_count, 3);
        assert!(!sessions[2].has_tool_use);
    }

    #[test]
    /// A tool result carries only `call_id`; it must pair with the assistant
    /// `tool_calls[].call_id` it answers (issue #863 F2).
    fn tool_results_pair_with_their_calls_by_call_id() {
        let messages = tool_session_messages();

        for (call_index, result_index, call_id, name) in [
            (2, 3, "call_168106ba", "read"),
            (5, 6, "call_d338cce3", "shell"),
            (9, 10, "call_46263e5d", "read"),
        ] {
            let call = first_block(&messages[call_index], "tool_use");
            assert_eq!(call["id"], call_id);
            assert_eq!(call["name"], name);
            let result = first_block(&messages[result_index], "tool_result");
            assert_eq!(result["tool_use_id"], call_id);
            assert_eq!(result["name"], name);
        }
        assert_eq!(
            messages[2].tool_use.as_ref().unwrap()["id"],
            "call_168106ba"
        );
    }

    #[test]
    /// Result text is `output.values[].text`, not the whole `output` object, and
    /// `output.is_error` is the producer flag (issue #863 F3).
    fn tool_result_content_is_the_output_values_text() {
        let messages = tool_session_messages();
        let entries = tool_session_entries();

        let read = first_block(&messages[3], "tool_result");
        assert_eq!(
            read["content"],
            entries[3]["message"]["tool"]["output"]["values"][0]["text"]
        );
        assert!(read["content"]
            .as_str()
            .unwrap()
            .starts_with("<file\n  path="));
        assert_eq!(read["is_error"], false);
        assert_eq!(read["is_error_source"], "producer");

        // An error result carries two text values, joined in order.
        let error = first_block(&messages[10], "tool_result");
        let text = error["content"].as_str().unwrap();
        assert!(text.starts_with("<tool_call_error>\n<cause>"));
        assert!(text.ends_with("</retry>"));
        assert!(text.contains("</tool_call_error>\n<retry>Tool call failed"));
        assert_eq!(error["is_error"], true);
        assert_eq!(error["is_error_source"], "producer");
    }

    #[test]
    /// A failing `npm test` is `is_error:false` in the record; the shell result's
    /// `exit_code="1"` attribute (`forge_app` `operation.rs` 597-598) marks it.
    fn shell_result_with_nonzero_exit_is_an_error() {
        let messages = tool_session_messages();

        let shell = first_block(&messages[6], "tool_result");
        assert_eq!(
            tool_session_entries()[6]["message"]["tool"]["output"]["is_error"],
            false
        );
        assert_eq!(shell["is_error"], true);
        assert_eq!(shell["is_error_source"], "derived");
        assert!(shell["content"]
            .as_str()
            .unwrap()
            .contains("exit_code=\"1\""));
    }

    #[test]
    /// Only the exit attribute in the opening `<shell_output>` tag counts, and a
    /// zero exit stays a success.
    fn shell_exit_marker_is_read_from_the_opening_tag_only() {
        let entries = tool_session_entries();
        let failing = entries[6]["message"]["tool"]["output"]["values"][0]["text"]
            .as_str()
            .unwrap();
        assert_eq!(
            shell_failure_exit_code(&Value::String(failing.to_string())),
            Some(1)
        );

        let passing = failing.replace("exit_code=\"1\"", "exit_code=\"0\"");
        assert_eq!(
            shell_failure_exit_code(&Value::String(passing.clone())),
            None
        );

        // Captured output that prints the marker does not turn a success into a failure.
        let echoed = passing.replace("> node test.js", "> node test.js\n  exit_code=\"1\"");
        assert_eq!(
            shell_failure_exit_code(&Value::String(echoed.clone())),
            None
        );
        let message = map_entry(&json!({"message": {"tool": {
            "name": "shell",
            "call_id": "call_echo",
            "output": {"is_error": false, "values": [{"text": echoed}]},
        }}}));
        let block = first_block(&message, "tool_result");
        assert_eq!(block["is_error"], false);
        assert_eq!(block["is_error_source"], "producer");

        // Other tools are never read for an exit code.
        assert_eq!(
            shell_failure_exit_code(&Value::String(
                "<file>\n  exit_code=\"1\"\n>\n<stdout".into()
            )),
            None
        );
    }

    #[test]
    /// The prompt is the typed text from `raw_content.Text`, not the
    /// `<task>`/`<feedback>` + `<system_date>` wrapper (issue #863 New).
    fn user_prompt_is_the_typed_text_not_the_wrapped_content() {
        let messages = tool_session_messages();

        assert_eq!(messages[0].message_type, "system");
        for (index, prompt) in [
            (
                1,
                "Read src/math.js and add a function mul(a, b) that returns a*b, exported like add.",
            ),
            (4, "Run npm test and tell me why it fails."),
            (8, "Read the file missing.txt"),
        ] {
            assert_eq!(messages[index].message_type, "user");
            assert_eq!(messages[index].content, Some(Value::String(prompt.to_string())));
        }
    }

    #[test]
    /// Roles, model, usage and metrics of the captured entries.
    fn sqlite_load_messages_maps_roles_usage_and_metrics() {
        let messages = tool_session_messages();

        assert_eq!(messages.len(), 12);
        let types: Vec<&str> = messages.iter().map(|m| m.message_type.as_str()).collect();
        assert_eq!(
            types,
            [
                "system",
                "user",
                "assistant",
                "user",
                "user",
                "assistant",
                "user",
                "assistant",
                "user",
                "assistant",
                "user",
                "assistant"
            ]
        );
        assert_eq!(messages[2].model, Some("qwen2.5-coder:7b".to_string()));
        let usage = messages[2].usage.as_ref().unwrap();
        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(20));
        assert_eq!(
            messages[7].content,
            Some(Value::String(
                "`npm test` fails because test.js asserts add(2, 2) === 5, but add(2, 2) returns 4."
                    .to_string()
            ))
        );

        let metrics = messages[11]
            .data
            .as_ref()
            .and_then(|data| data.get("forgecodeMetrics"))
            .unwrap();
        assert_eq!(metrics["fileOperations"], 2);
        assert_eq!(
            metrics["sessionStartTime"],
            "2026-11-25T03:41:14.757220+00:00"
        );
        assert_eq!(metrics["filesAccessed"].as_array().map(Vec::len), Some(2));
    }

    #[test]
    /// Tool output values the producer can write besides `text`
    /// (`ToolValueRecord` in `forge_repo` `conversation_record.rs`): `Empty` is
    /// dropped, `AI` (serialized `aI`) contributes its value, an image keeps its
    /// record in a block array.
    fn tool_output_values_cover_the_producer_variants() {
        let message = map_entry(&json!({"message": {"tool": {
            "name": "task",
            "call_id": "call_task",
            "output": {"is_error": false, "values": [
                {"text": "first"},
                "empty",
                {"aI": {"value": "sub-agent reply", "conversation_id": "0b3f6c1e-6f43-4c5e-9d57-0d4a8f5b2c11"}},
            ]},
        }}}));
        assert_eq!(
            first_block(&message, "tool_result")["content"],
            "first\nsub-agent reply"
        );

        let image = json!({"url": "data:image/png;base64,iVBORw0KGgo=", "mime_type": "image/png"});
        let message = map_entry(&json!({"message": {"tool": {
            "name": "read",
            "call_id": "call_img",
            "output": {"is_error": false, "values": [{"text": "screenshot"}, {"image": image}]},
        }}}));
        assert_eq!(
            first_block(&message, "tool_result")["content"],
            json!([
                {"type": "text", "text": "screenshot"},
                {"type": "image", "source": image},
            ])
        );
    }

    #[test]
    /// A context image entry is a user message holding the image record.
    fn image_entry_maps_to_an_image_block() {
        let image = json!({"url": "data:image/png;base64,iVBORw0KGgo=", "mime_type": "image/png"});
        let message = map_entry(&json!({"message": {"image": image}}));

        assert_eq!(message.message_type, "user");
        assert_eq!(first_block(&message, "image")["source"], image);
    }

    #[test]
    /// The producer also reads its older unwrapped `{"text"|"tool": ..}` entries
    /// (`ContextMessageRecord` deserializer in `forge_repo` `conversation_record.rs`).
    fn unwrapped_entries_map_like_wrapped_ones() {
        let entries = tool_session_entries();
        for index in [1, 2, 3] {
            let wrapped = map_entry(&entries[index]);
            let unwrapped = map_entry(&entries[index]["message"]);
            assert_eq!(unwrapped.message_type, wrapped.message_type);
            assert_eq!(unwrapped.content, wrapped.content);
        }
    }

    #[test]
    /// A command prompt has no typed text, so its wrapped content is kept.
    fn command_prompt_keeps_its_content() {
        let message = map_entry(&json!({"message": {"text": {
            "role": "User",
            "content": "<command>fixme</command>",
            "raw_content": {"Command": {"name": "fixme", "template": "fix it", "parameters": []}},
        }}}));

        assert_eq!(
            message.content,
            Some(Value::String("<command>fixme</command>".to_string()))
        );
    }

    #[test]
    /// The context is an object with a `messages` array, never a bare array.
    fn parse_context_entries_reads_only_the_messages_array() {
        assert_eq!(
            parse_context_entries(r#"{"messages":[{"message":{}}]}"#).len(),
            1
        );
        assert_eq!(parse_context_entries(r#"[{"message":{}}]"#).len(), 0);
        assert_eq!(parse_context_entries(r#"{"conversation_id":"c"}"#).len(), 0);
    }

    #[test]
    /// Verify `SQLite` load messages returns empty for malformed context without panicking.
    fn sqlite_load_messages_returns_empty_for_malformed_context_without_panicking() {
        let tmp = tempfile::tempdir().unwrap();
        let conn = create_test_db(&tmp);
        seed_test_data(&conn);
        drop(conn);

        let messages =
            load_messages_from_db(&tmp.path().to_string_lossy(), WORKSPACE, "conv-bad").unwrap();

        assert!(messages.is_empty());
    }

    #[test]
    /// `FORGE_CONFIG=<dir>` makes `<dir>/.forge.db` the transcript store
    /// (`forge_config` `reader.rs` `resolve_base_path`, `forge_domain` `env.rs`
    /// `database_path`).
    #[serial]
    fn forge_config_directory_holds_the_database() {
        let tmp = tempfile::tempdir().unwrap();
        let conn = create_test_db(&tmp);
        seed_test_data(&conn);
        drop(conn);
        let _guard = EnvGuard::set("FORGE_CONFIG", tmp.path());

        assert_eq!(
            get_base_path(),
            Some(tmp.path().to_string_lossy().to_string())
        );
        let messages = load_messages(&format!(
            "forgecode-db://workspace/{WORKSPACE}/conversation/{TOOL_SESSION}"
        ))
        .unwrap();
        assert_eq!(messages.len(), 12);
    }

    #[test]
    /// Extract workspace display name prefers cwd basename and ignores home dir.
    #[serial]
    fn extract_workspace_display_name_prefers_cwd_basename_and_ignores_home_dir() {
        // The "home" entry has to be the home the code under test resolves, so
        // it comes from the guard. `dirs::home_dir()` was the real one on
        // Windows, which is not the home being ignored (#551).
        let home = crate::test_utils::SandboxHome::new();
        let context = json!({
            "messages": [
                {"message": {"text": {"role": "Assistant", "content": "", "tool_calls": [{
                    "name": "shell",
                    "call_id": "call_home",
                    "arguments": {"command": "ls", "cwd": home.path().to_string_lossy().to_string()},
                }]}}},
                {"message": {"text": {"role": "Assistant", "content": "", "tool_calls": [{
                    "name": "shell",
                    "call_id": "call_demo",
                    "arguments": {"command": "ls", "cwd": crate::test_utils::abs("home/user/demo")},
                }]}}},
            ]
        });

        assert_eq!(
            extract_workspace_display_name_from_value(&context),
            Some("demo".to_string())
        );
    }

    #[test]
    /// Verify detection prefers forge config and checks artifacts.
    fn detection_prefers_forge_config_and_checks_artifacts() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(".forge_history"), "history").unwrap();
        let _guard = EnvGuard::set("FORGE_CONFIG", tmp.path());

        let detected = detect().unwrap();

        assert_eq!(detected.id, "forgecode");
        assert_eq!(detected.display_name, "ForgeCode");
        assert_eq!(detected.base_path, tmp.path().to_string_lossy());
        assert!(detected.is_available);
    }

    #[test]
    /// Verify `SQLite` search matches message content.
    fn sqlite_search_matches_message_content() {
        let tmp = tempfile::tempdir().unwrap();
        let conn = create_test_db(&tmp);
        seed_test_data(&conn);
        drop(conn);

        let base = tmp.path().to_string_lossy().to_string();
        let results = search_from_path(&base, "tell me why", 10);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].provider.as_deref(), Some("forgecode"));
        assert_eq!(results[0].session_id, TOOL_SESSION);
        assert_eq!(
            results[0].uuid,
            format!("forgecode://workspace/{WORKSPACE}/conversation/{TOOL_SESSION}/message/4")
        );
        assert_eq!(
            results[0].content,
            Some(Value::String(
                "Run npm test and tell me why it fails.".to_string()
            ))
        );
        assert!(results.iter().all(|message| {
            message
                .content
                .as_ref()
                .is_some_and(|content| search_json_value_case_insensitive(content, "tell me why"))
        }));

        // The model-facing wrapper is not part of what the user typed.
        assert!(search_from_path(&base, "system_date", 10).is_empty());
    }

    #[test]
    /// #672: hits carry the scanned workspace path, so two workspaces whose
    /// cwd leaf is `app` stay apart in search.
    #[serial]
    fn sqlite_search_hits_carry_scanned_project_path() {
        let _home = crate::test_utils::SandboxHome::new();
        let tmp = tempfile::tempdir().unwrap();
        let conn = create_test_db(&tmp);
        for (workspace, conversation, cwd) in [
            (
                "1001",
                "a1b2c3d4-0000-4000-8000-000000000001",
                "/work/client-a/app",
            ),
            (
                "1002",
                "a1b2c3d4-0000-4000-8000-000000000002",
                "/work/client-b/app",
            ),
        ] {
            insert_conversation(
                &conn,
                conversation,
                workspace,
                Some("t"),
                Some(&context_json(
                    conversation,
                    &shell_session_entries("needle", cwd),
                )),
                None,
                "2026-11-25 08:00:00.000000",
                "2026-11-25 08:00:00.000000",
            );
        }
        drop(conn);
        let base = tmp.path().to_string_lossy().to_string();

        let projects = scan_projects_from_db(&base, &ScanReport::default()).unwrap();
        let results = search_from_path(&base, "needle", 10);
        assert_eq!(results.len(), 2);
        for (conversation, cwd) in [
            ("a1b2c3d4-0000-4000-8000-000000000001", "/work/client-a/app"),
            ("a1b2c3d4-0000-4000-8000-000000000002", "/work/client-b/app"),
        ] {
            let project = projects.iter().find(|p| p.actual_path == cwd).unwrap();
            assert_eq!(project.name, "app");
            let hit = results
                .iter()
                .find(|m| m.session_id == conversation)
                .unwrap();
            assert_eq!(hit.project_path.as_deref(), Some(project.path.as_str()));
        }
        assert_ne!(results[0].project_path, results[1].project_path);
    }

    #[test]
    /// Verify `SQLite` rename session title updates conversation title.
    fn sqlite_rename_session_title_updates_conversation_title() {
        let tmp = tempfile::tempdir().unwrap();
        let conn = create_test_db(&tmp);
        seed_test_data(&conn);
        drop(conn);

        let result = rename_session_title_from_path(
            &tmp.path().to_string_lossy(),
            &format!("forgecode://workspace/{WORKSPACE}/conversation/{TOOL_SESSION}"),
            "Updated Forge Title",
        )
        .unwrap();

        assert_eq!(result.previous_title, "Mock tool session");
        assert_eq!(result.new_title, "Updated Forge Title");

        let conn = open_db_read_write(&tmp.path().to_string_lossy()).unwrap();
        let title: String = conn
            .query_row(
                "SELECT title FROM conversations WHERE conversation_id = ?1",
                [TOOL_SESSION],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(title, "Updated Forge Title");
    }

    #[test]
    /// Verify `SQLite` delete conversation removes row.
    fn sqlite_delete_conversation_removes_row() {
        let tmp = tempfile::tempdir().unwrap();
        let conn = create_test_db(&tmp);
        seed_test_data(&conn);
        drop(conn);

        delete_conversation_from_path(
            &tmp.path().to_string_lossy(),
            &format!("forgecode-db://workspace/{WORKSPACE}/conversation/{TOOL_SESSION}"),
        )
        .unwrap();

        let conn = open_db_read_write(&tmp.path().to_string_lossy()).unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM conversations WHERE conversation_id = ?1",
                [TOOL_SESSION],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
    }
    #[test]
    /// Parse conversation path accepts virtual session and file paths.
    fn parse_conversation_path_accepts_virtual_session_and_file_paths() {
        assert_eq!(
            parse_conversation_path("forgecode://workspace/ws-1/conversation/conv-1"),
            Some(("ws-1".to_string(), "conv-1".to_string()))
        );
        assert_eq!(
            parse_conversation_path("forgecode-db://workspace/ws-1/conversation/conv-1"),
            Some(("ws-1".to_string(), "conv-1".to_string()))
        );
    }

    #[test]
    /// Reject ids that fall outside the `[A-Za-z0-9_-]+` allowlist.
    fn parse_conversation_path_rejects_non_allowlist_components() {
        // path separators and traversal — original constraints
        assert_eq!(parse_conversation_path("forgecode://workspace/.."), None);
        assert_eq!(parse_conversation_path("forgecode://workspace/."), None);
        assert_eq!(
            parse_conversation_path("forgecode://workspace/ws/conversation/../escape"),
            None
        );
        // characters outside the tightened allowlist
        assert_eq!(
            parse_conversation_path("forgecode://workspace/ws 1/conversation/conv-1"),
            None,
            "spaces are rejected"
        );
        assert_eq!(
            parse_conversation_path("forgecode://workspace/ws.1/conversation/conv-1"),
            None,
            "dots are rejected"
        );
        assert_eq!(
            parse_conversation_path("forgecode://workspace/ws-1/conversation/conv:1"),
            None,
            "colons are rejected"
        );
    }
}
