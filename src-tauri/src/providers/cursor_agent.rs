//! Cursor Agent CLI provider.
//!
//! Reads conversation transcripts written by the Cursor Agent CLI under
//! `~/.cursor/projects/<encoded-project>/agent-transcripts/<uuid>/<uuid>.jsonl`
//! (`CURSOR_DATA_DIR` moves `~/.cursor` here), and the older flat
//! `agent-transcripts/<uuid>.jsonl`. This is a different data source from the
//! `cursor` provider (which reads the Cursor IDE's `SQLite` storage) — see
//! issue #304.
//!
//! The transcript is a lossy mirror the CLI regenerates from its chat store
//! (`agent-transcript` in the CLI bundle, 2026.11.22). One JSON object per
//! line:
//! ```json
//! {"role":"user","message":{"content":[{"type":"text","text":"<user_query>\n…\n</user_query>"}]}}
//! {"role":"assistant","message":{"content":[{"type":"text","text":"…"},{"type":"tool_use","name":"Read","input":{…}}]}}
//! {"type":"turn_ended","status":"success"}
//! ```
//! An assistant line's text is the visible text, then the reasoning, with
//! `[REDACTED]` for each redacted reasoning part; each tool call follows as a
//! `tool_use` with no id and no result. User text has the CLI's context tags
//! removed and keeps the `<user_query>` wrapper. There is no per-line
//! `id`/`timestamp`/`sessionId`; the session id is the transcript file's UUID
//! stem and times come from the file mtime.
//!
//! ## Session titles (`/rename`)
//!
//! Transcripts carry no title. The CLI stores the title it shows in its own
//! resume picker (updated by the `/rename` slash command) under
//! `<config root>/chats/<md5(cwd)>/<session-uuid>/meta.json` (`title` field,
//! mirrored as `name` in that directory's `store.db`); the config root is
//! `~/.cursor` unless `CURSOR_CONFIG_DIR` or `XDG_CONFIG_HOME` moves it. The
//! session UUID is shared with the transcript path, so [`load_sessions`]
//! merges that stored title over the first-prompt fallback chain.
//!
//! ## Content-block normalisation (issue #472)
//!
//! | Block type | Behaviour |
//! |---|---|
//! | `{"type":"text","text":"<user_query>…</user_query>"}` | Strip XML wrapper; drop trailing `<context>` blob |
//! | `{"type":"text","text":"…\n\n[REDACTED]"}` | Drop the `[REDACTED]` lines; skip the message when nothing is left |
//! | `{"type":"tool_use","name":…,"input":{…}}` | A tool call (no result is recorded) |
//! | `{"type":"tool_result","content":[…]}` | Recurse into nested content array |
//! | `{"type":"command_output","output":"…"}` | Surface shell stdout/stderr |
//!
//! A `<user_query>` that opens like the notices Cursor adds when a sub-agent
//! or background command finishes is not shown as a prompt.

use super::ProviderInfo;
use crate::models::{ClaudeMessage, ClaudeProject, ClaudeSession};
use crate::utils::{
    build_provider_message, decode_with_filesystem_check, search_json_value_case_insensitive,
};
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

const PROVIDER_ID: &str = "cursor-agent";

/// Max characters of the first user prompt used as a session title.
const SUMMARY_MAX_CHARS: usize = 80;

// ============================================================================
// Public API
// ============================================================================

/// Detect a Cursor Agent CLI installation.
pub fn detect() -> Option<ProviderInfo> {
    let base = get_base_path()?;
    Some(ProviderInfo {
        id: PROVIDER_ID.to_string(),
        display_name: "Cursor Agent".to_string(),
        is_available: has_any_transcript(Path::new(&base)),
        base_path: base,
    })
}

/// Base path for Cursor Agent transcripts: `<data root>/projects`.
pub fn get_base_path() -> Option<String> {
    let projects = data_root()?.join("projects");
    if projects.is_dir() {
        Some(projects.to_string_lossy().to_string())
    } else {
        None
    }
}

/// The value of `name` when set to something other than blanks, as the CLI
/// reads its directory overrides.
fn non_blank_env(name: &str) -> Option<PathBuf> {
    // A test build ignores `XDG_CONFIG_HOME`, as `utils::config_dir` does:
    // CI runners set it, which would move every sandboxed `~/.cursor`.
    if cfg!(test) && name == "XDG_CONFIG_HOME" {
        return None;
    }
    std::env::var_os(name)
        .filter(|value| !value.to_string_lossy().trim().is_empty())
        .map(PathBuf::from)
}

/// The CLI's config root, which holds `chats/` and `acp-sessions/`.
pub(crate) fn config_root() -> Option<PathBuf> {
    config_root_from(non_blank_env, crate::utils::home_dir())
}

/// `CURSOR_CONFIG_DIR`, else `$XDG_CONFIG_HOME/cursor`, else `~/.cursor`
/// (`cursor-config/dist/paths.js` in the CLI bundle).
fn config_root_from(
    env: impl Fn(&str) -> Option<PathBuf>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    env("CURSOR_CONFIG_DIR")
        .or_else(|| env("XDG_CONFIG_HOME").map(|dir| dir.join("cursor")))
        .or_else(|| Some(home?.join(".cursor")))
}

/// The CLI's data root, which holds `projects/` (the transcripts):
/// `CURSOR_DATA_DIR`, else `~/.cursor`.
fn data_root() -> Option<PathBuf> {
    non_blank_env("CURSOR_DATA_DIR").or_else(|| Some(crate::utils::home_dir()?.join(".cursor")))
}

/// Root of the Cursor Agent CLI chat metadata store: `<config root>/chats`.
pub(crate) fn chats_root() -> Option<PathBuf> {
    chats_root_in(&config_root()?)
}

/// `chats` under the config root, or `None` when missing or a symlink.
///
/// `WalkDir` follows a symlinked traversal root by default, so reject it here
/// rather than reading titles from an unintended location.
fn chats_root_in(config_root: &Path) -> Option<PathBuf> {
    let chats = config_root.join("chats");
    if chats.is_dir() && !is_symlink(&chats) {
        Some(chats)
    } else {
        None
    }
}

/// True if at least one `*/agent-transcripts/**/*.jsonl` exists under `base`.
fn has_any_transcript(base: &Path) -> bool {
    project_dirs(base).iter().any(|p| {
        let transcripts = p.join("agent-transcripts");
        transcripts.is_dir() && !transcript_files(&transcripts).is_empty()
    })
}

/// Scan Cursor Agent projects under the default `~/.cursor/projects` root.
pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    let base = get_base_path().ok_or("Cursor projects path not found")?;
    scan_projects_in(Path::new(&base))
}

/// Implementation of [`scan_projects`] parameterised by the projects root so
/// tests can pass an isolated temp dir.
pub fn scan_projects_in(base: &Path) -> Result<Vec<ClaudeProject>, String> {
    let mut projects = Vec::new();

    for project_dir in project_dirs(base) {
        let transcripts_dir = project_dir.join("agent-transcripts");
        if !transcripts_dir.is_dir() {
            continue;
        }

        let mut session_count = 0usize;
        let mut message_count = 0usize;
        let mut last_modified_ts = 0u64;

        for entry in transcript_files(&transcripts_dir) {
            session_count += 1;
            if let Ok(meta) = entry.metadata() {
                message_count += (meta.len() / 400) as usize;
                if let Ok(modified) = meta.modified() {
                    if let Ok(dur) = modified.duration_since(std::time::SystemTime::UNIX_EPOCH) {
                        last_modified_ts = last_modified_ts.max(dur.as_secs());
                    }
                }
            }
        }

        if session_count == 0 {
            continue;
        }

        let (display_name, actual_path) = project_display(&project_dir);

        projects.push(ClaudeProject {
            name: display_name,
            path: project_path(&project_dir),
            actual_path,
            session_count,
            message_count,
            last_modified: ts_to_rfc3339(last_modified_ts),
            git_info: None,
            provider: Some(PROVIDER_ID.to_string()),
            storage_type: Some("json".to_string()),
            custom_directory_label: None,
        });
    }

    projects.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(projects)
}

/// The transcript files [`load_sessions`] lists for a project folder.
pub(crate) fn transcript_paths(project_dir: &Path) -> Vec<PathBuf> {
    transcript_files(&project_dir.join("agent-transcripts"))
        .into_iter()
        .map(DirEntry::into_path)
        .collect()
}

/// Load the sessions (transcripts) for a Cursor Agent project.
pub fn load_sessions(
    project_path: &str,
    _exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    if project_path.trim().is_empty() {
        return Err("project_path is required".to_string());
    }
    let project_dir = Path::new(project_path);
    if !project_dir.is_dir() {
        return Ok(vec![]);
    }
    validate_under_base(project_dir)?;
    if is_symlink(project_dir) {
        return Err(format!(
            "Project path must not be a symlink: {}",
            project_dir.display()
        ));
    }

    let transcripts_dir = project_dir.join("agent-transcripts");
    if !transcripts_dir.is_dir() {
        return Ok(vec![]);
    }

    let project_name = project_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("Unknown")
        .to_string();

    // Titles the CLI stores (and `/rename` updates), keyed by session UUID.
    let title_index = chats_root()
        .map(|root| load_native_title_index(&root))
        .unwrap_or_default();

    let mut sessions = Vec::new();
    for entry in transcript_files(&transcripts_dir) {
        if let Some(session) = extract_session_info(entry.path(), &project_name, &title_index) {
            sessions.push(session);
        }
    }

    sessions.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(sessions)
}

/// Load all messages from a Cursor Agent transcript file.
pub fn load_messages(session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    let path = Path::new(session_path);
    if !path.exists() {
        return Err(format!("Session file not found: {session_path}"));
    }
    validate_under_base(path)?;
    if is_symlink(path) {
        return Err("Session file must not be a symlink".to_string());
    }

    let data = fs::read_to_string(path).map_err(|e| format!("Failed to read session file: {e}"))?;
    let session_id = file_uuid(path);
    let timestamp = file_mtime_rfc3339(path);

    Ok(parse_transcript(&data, &session_id, &timestamp))
}

/// Convert a transcript file's contents into messages. Pure (no filesystem /
/// validation) so it can be unit-tested directly.
///
/// A message's uuid is `<session>-<n>`, `n` counting every user/assistant
/// line, shown or not, so a search hit (also from this function) resolves
/// to the same message in [`load_messages`].
fn parse_transcript(data: &str, session_id: &str, timestamp: &str) -> Vec<ClaudeMessage> {
    let mut messages = Vec::new();
    let mut msg_index = 0u64;
    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if !is_conversation_turn(&value) {
            continue;
        }
        if let Some(msg) = convert_message(&value, session_id, timestamp, msg_index) {
            messages.push(msg);
        }
        msg_index += 1;
    }
    messages
}

/// Search across all Cursor Agent transcripts.
pub fn search(query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    let Some(base) = get_base_path() else {
        return Ok(vec![]);
    };
    Ok(search_in(Path::new(&base), query, limit))
}

/// Implementation of [`search`] parameterised by the projects root. Only
/// what a transcript shows is searched.
fn search_in(base: &Path, query: &str, limit: usize) -> Vec<ClaudeMessage> {
    let query_lower = query.to_lowercase();
    let mut results = Vec::new();

    for project_dir in project_dirs(base) {
        if is_symlink(&project_dir) {
            continue;
        }
        let transcripts_dir = project_dir.join("agent-transcripts");
        if !transcripts_dir.is_dir() {
            continue;
        }
        let (project_name, _) = project_display(&project_dir);
        let project_path = project_path(&project_dir);

        for entry in transcript_files(&transcripts_dir) {
            let path = entry.path();
            let Ok(data) = fs::read_to_string(path) else {
                continue;
            };
            let session_id = file_uuid(path);
            let timestamp = file_mtime_rfc3339(path);

            for mut msg in parse_transcript(&data, &session_id, &timestamp) {
                let hit = msg.content.as_ref().is_some_and(|content| {
                    search_json_value_case_insensitive(content, &query_lower)
                });
                if !hit {
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
    }

    results
}

// ============================================================================
// Content-block normalisation (issue #472)
// ============================================================================

/// Strip Cursor's `<user_query>…</user_query>` XML envelope from a user text
/// block and remove any trailing `<context>` or similar XML blobs.
///
/// If the text does not contain a `<user_query>` tag it is returned unchanged
/// so non-wrapped assistant / tool messages are unaffected.
///
/// Any residual XML-style tags (`<tag>` / `</tag>`) are stripped after
/// extraction so stray inline tags don't leak into the rendered output.
fn clean_user_text(text: &str) -> String {
    // Extract what is inside <user_query>…</user_query>.
    let inner = if let Some(start) = text.find("<user_query>") {
        let after_open = &text[start + "<user_query>".len()..];
        // Truncate at the closing tag — this drops any <context> blob that
        // Cursor appends after </user_query>.
        if let Some(end) = after_open.find("</user_query>") {
            &after_open[..end]
        } else {
            after_open
        }
    } else {
        // No wrapper — pass through (covers assistant messages and plain turns).
        text
    };

    // Strip any remaining XML-style tags to clean up inline markup.
    let cleaned = strip_xml_tags(inner);
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Remove all `<…>` and `</…>` substrings from `s`.
fn strip_xml_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out
}

/// The sentinel the CLI writes for each redacted reasoning part, on its own
/// line after the visible text.
const REDACTED: &str = "[REDACTED]";

/// `text` without its `[REDACTED]` lines.
fn without_redacted(text: &str) -> String {
    if !text.contains(REDACTED) {
        return text.trim().to_string();
    }
    text.lines()
        .filter(|line| line.trim() != REDACTED)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Openings of the `<user_query>` messages Cursor adds when a sub-agent or a
/// background command finishes. The model is told the result; the user did
/// not type them (the `INJECTED_QUERY_PREFIXES` of the `CursorDump` exporter,
/// observed in real transcripts).
const INJECTED_QUERY_PREFIXES: &[&str] = &[
    "The beginning of the above subagent result",
    "The above subagent result",
    "A subagent has completed",
    "The background shell command",
    "Briefly inform the user about the task result",
];

fn is_injected_query(text: &str) -> bool {
    let text = text.trim_start();
    INJECTED_QUERY_PREFIXES
        .iter()
        .any(|prefix| text.starts_with(prefix))
}

/// Recursively collect text from a legacy `tool_result` / `command_output`
/// block, or a text block nested in one, into `out` (issue #472).
fn collect_block_text(block: &Value, role: &str, out: &mut Vec<String>) {
    let ty = block.get("type").and_then(Value::as_str).unwrap_or("");

    match ty {
        "text" => {
            let raw = without_redacted(block.get("text").and_then(Value::as_str).unwrap_or(""));
            let cleaned = if role == "user" {
                clean_user_text(&raw)
            } else {
                raw
            };
            if !cleaned.trim().is_empty() {
                out.push(cleaned);
            }
        }
        "tool_result" => {
            // Cursor nests a content array inside tool_result blocks.
            if let Some(inner) = block.get("content").and_then(Value::as_array) {
                for inner_block in inner {
                    collect_block_text(inner_block, role, out);
                }
            }
            // Some tool_result blocks also carry a top-level "output" field.
            if let Some(output) = block.get("output").and_then(Value::as_str) {
                let trimmed = output.trim();
                if !trimmed.is_empty() {
                    out.push(format!("[tool output]\n{trimmed}"));
                }
            }
        }
        "command_output" => {
            if let Some(output) = block.get("output").and_then(Value::as_str) {
                let trimmed = output.trim();
                if !trimmed.is_empty() {
                    out.push(format!("[shell]\n{trimmed}"));
                }
            }
        }
        // Skip redacted / unknown block types.
        _ => {}
    }
}

// ============================================================================
// Message conversion
// ============================================================================

/// A renderable conversation turn has a `user`/`assistant` role.
fn is_conversation_turn(value: &Value) -> bool {
    matches!(
        value.get("role").and_then(Value::as_str),
        Some("user" | "assistant")
    )
}

/// Convert one transcript line into a `ClaudeMessage`, or `None` when it
/// shows nothing: only redacted reasoning, or a notification Cursor injected
/// as a user message. `msg_index` makes the generated UUID
/// stable/deterministic so global-search navigation can resolve it back
/// inside `load_messages`.
///
/// Text blocks are kept, cleaned. The CLI writes each tool call as
/// `{"type":"tool_use","name","input"}` with no id and no result, so a call
/// gets an id from its line and position. Legacy `tool_result` /
/// `command_output` blocks become text.
fn convert_message(
    value: &Value,
    session_id: &str,
    timestamp: &str,
    msg_index: u64,
) -> Option<ClaudeMessage> {
    let role = value.get("role").and_then(Value::as_str)?;
    let uuid = format!("{session_id}-{msg_index}");
    let blocks = value
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)?;

    let mut content: Vec<Value> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    let flush = |texts: &mut Vec<String>, content: &mut Vec<Value>| {
        if !texts.is_empty() {
            content.push(serde_json::json!({"type": "text", "text": texts.join("\n")}));
            texts.clear();
        }
    };
    for block in blocks {
        if block.get("type").and_then(Value::as_str) == Some("tool_use") {
            flush(&mut texts, &mut content);
            let name = block.get("name").and_then(Value::as_str).unwrap_or("tool");
            let input = block
                .get("input")
                .filter(|input| input.is_object())
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            content.push(serde_json::json!({
                "type": "tool_use",
                "id": format!("{uuid}-tool-{}", content.len()),
                "name": name,
                "input": input,
            }));
        } else {
            collect_block_text(block, role, &mut texts);
        }
    }
    flush(&mut texts, &mut content);

    if content.is_empty() {
        return None;
    }
    if role == "user" {
        let first_text = content[0].get("text").and_then(Value::as_str);
        if first_text.is_some_and(is_injected_query) {
            return None;
        }
    }

    let tool_use = content
        .iter()
        .find(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
        .cloned();
    let mut msg = build_provider_message(
        PROVIDER_ID,
        uuid,
        session_id,
        timestamp.to_string(),
        role,
        Some(role),
        Some(Value::Array(content)),
        None,
    );
    msg.tool_use = tool_use;
    Some(msg)
}

// ============================================================================
// Session helpers
// ============================================================================

/// Build a `session UUID → stored title` map from
/// `~/.cursor/chats/<workspace-hash>/<session-uuid>/meta.json`.
///
/// The CLI writes the title shown in its own resume picker to `meta.json`'s
/// `title` field, and `/rename` updates it. The session UUID (directory name)
/// is the same identifier used for the transcript file stem under
/// `~/.cursor/projects`, so no workspace-hash mapping is needed to join.
fn load_native_title_index(chats_root: &Path) -> HashMap<String, String> {
    WalkDir::new(chats_root)
        .min_depth(3)
        .max_depth(3)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_name() == "meta.json")
        .filter(|e| !is_symlink(e.path()))
        .filter_map(|entry| {
            let uuid = entry.path().parent()?.file_name()?.to_str()?;
            let data = fs::read_to_string(entry.path()).ok()?;
            let value = serde_json::from_str::<Value>(&data).ok()?;
            let title = value.get("title")?.as_str()?.trim();
            if title.is_empty() {
                return None;
            }
            Some((uuid.to_string(), title.to_string()))
        })
        .collect()
}

fn extract_session_info(
    file_path: &Path,
    project_name: &str,
    native_titles: &HashMap<String, String>,
) -> Option<ClaudeSession> {
    let data = fs::read_to_string(file_path).ok()?;
    let uuid = file_uuid(file_path);
    let mtime = file_mtime_rfc3339(file_path);

    // Count and title what the transcript shows.
    let messages = parse_transcript(&data, &uuid, &mtime);
    if messages.is_empty() {
        return None;
    }
    let has_tool_use = messages.iter().any(|m| m.tool_use.is_some());
    let summary = messages
        .iter()
        .filter(|m| m.message_type == "user")
        .find_map(|m| {
            m.content
                .as_ref()?
                .as_array()?
                .iter()
                .find_map(|block| block.get("text")?.as_str().map(summarize))
        });
    let native_title = native_titles.get(&uuid);

    Some(ClaudeSession {
        session_id: file_path.to_string_lossy().to_string(),
        actual_session_id: uuid.clone(),
        file_path: file_path.to_string_lossy().to_string(),
        project_name: project_name.to_string(),
        message_count: messages.len(),
        first_message_time: mtime.clone(),
        last_message_time: mtime.clone(),
        last_modified: mtime,
        has_tool_use,
        has_errors: false,
        // Stored title (the CLI's resume-picker name, updated by `/rename`)
        // wins over the first-prompt fallback, which wins over the raw UUID.
        summary: native_title.cloned().or(summary).or(Some(uuid)),
        is_renamed: native_title.is_some(),
        provider: Some(PROVIDER_ID.to_string()),
        storage_type: Some("json".to_string()),
        entrypoint: None,
    })
}

/// Build a short session title from the first user prompt, stripping the
/// `<user_query>` wrapper Cursor injects.
fn summarize(text: &str) -> String {
    let cleaned = clean_user_text(text);
    if cleaned.chars().count() > SUMMARY_MAX_CHARS {
        let truncated: String = cleaned.chars().take(SUMMARY_MAX_CHARS).collect();
        format!("{truncated}\u{2026}")
    } else {
        cleaned
    }
}

// ============================================================================
// Filesystem helpers
// ============================================================================

/// Immediate child directories of `base`.
fn project_dirs(base: &Path) -> Vec<std::path::PathBuf> {
    WalkDir::new(base)
        .min_depth(1)
        .max_depth(1)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_dir())
        .map(|e| e.path().to_path_buf())
        .collect()
}

/// The `ClaudeProject.path` scan reports for a project directory; search
/// tags hits with the same string.
fn project_path(project_dir: &Path) -> String {
    project_dir.to_string_lossy().to_string()
}

/// `(name, actual_path)` scan reports for a project directory: the decoded
/// workspace leaf and path when the encoded name resolves on disk, else the
/// encoded name and the directory itself.
pub(crate) fn project_display(project_dir: &Path) -> (String, String) {
    let dir_name = project_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("Unknown");

    match decode_with_filesystem_check(dir_name) {
        Some(real_path) => {
            let leaf = Path::new(&real_path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| dir_name.to_string());
            (leaf, real_path)
        }
        None => (dir_name.to_string(), project_path(project_dir)),
    }
}

/// Non-symlinked `*.jsonl` transcripts under an `agent-transcripts`
/// directory: `<uuid>/<uuid>.jsonl`, and the older flat `<uuid>.jsonl` the
/// CLI still reads (`agent-transcript/dist/paths.js`) when that chat has no
/// file in the current layout. A sub-agent's `<uuid>/subagents/*.jsonl` is
/// not a session.
fn transcript_files(transcripts_dir: &Path) -> Vec<DirEntry> {
    let files: Vec<DirEntry> = WalkDir::new(transcripts_dir)
        .min_depth(1)
        .max_depth(2)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("jsonl"))
        .filter(|e| !is_symlink(e.path()))
        .collect();
    let current: std::collections::HashSet<String> = files
        .iter()
        .filter(|e| e.depth() == 2)
        .map(|e| file_uuid(e.path()))
        .collect();
    files
        .into_iter()
        .filter(|e| e.depth() == 2 || !current.contains(&file_uuid(e.path())))
        .collect()
}

fn file_uuid(path: &Path) -> String {
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

/// Confine `path` to the `~/.cursor/projects` root.
fn validate_under_base(path: &Path) -> Result<(), String> {
    let base = get_base_path().ok_or("Cursor projects path not found")?;
    let canon_base = Path::new(&base)
        .canonicalize()
        .map_err(|e| format!("Failed to resolve Cursor base: {e}"))?;
    let canon_path = path
        .canonicalize()
        .map_err(|e| format!("Failed to resolve session path: {e}"))?;
    if canon_path.starts_with(&canon_base) {
        Ok(())
    } else {
        Err(format!(
            "Path is outside the Cursor projects root: {}",
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
    use std::fs;
    use tempfile::TempDir;

    // ---------------------------------------------------------------------------
    // Fixtures
    // ---------------------------------------------------------------------------

    const SAMPLE: &str = concat!(
        r#"{"role":"user","message":{"content":[{"type":"text","text":"<user_query>fix the LOGIN bug</user_query>"}]}}"#,
        "\n",
        r#"{"role":"assistant","message":{"content":[{"type":"text","text":"Looking into ZmagicToken now"}]}}"#,
        "\n",
    );

    fn write_transcript(base: &Path, project: &str, uuid: &str, body: &str) {
        let dir = base.join(project).join("agent-transcripts").join(uuid);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(format!("{uuid}.jsonl")), body).unwrap();
    }

    // ---------------------------------------------------------------------------
    // Existing tests (all must still pass)
    // ---------------------------------------------------------------------------

    /// #672: hits carry the scanned project's path (and its decoded name), so
    /// two workspaces whose leaf is `app` stay apart in search.
    #[cfg(unix)]
    #[test]
    fn search_hits_carry_scanned_project_path() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let workspaces = root.join("ws");
        let projects_root = root.join("projects");
        let encoded_ws = workspaces
            .to_string_lossy()
            .trim_start_matches('/')
            .replace('/', "-");
        for client in ["client-a", "client-b"] {
            fs::create_dir_all(workspaces.join(client).join("app")).unwrap();
            let encoded = format!("{encoded_ws}-{client}-app");
            write_transcript(&projects_root, &encoded, &format!("uuid-{client}"), SAMPLE);
        }

        let projects = scan_projects_in(&projects_root).unwrap();
        let results = search_in(&projects_root, "zmagictoken", 10);
        assert_eq!(results.len(), 2);
        for client in ["client-a", "client-b"] {
            let project = projects
                .iter()
                .find(|p| p.actual_path.ends_with(&format!("{client}/app")))
                .unwrap();
            assert_eq!(project.name, "app");
            let hit = results
                .iter()
                .find(|m| m.session_id == format!("uuid-{client}"))
                .unwrap();
            assert_eq!(hit.project_path.as_deref(), Some(project.path.as_str()));
            assert_eq!(hit.project_name.as_deref(), Some("app"));
        }
        assert_ne!(results[0].project_path, results[1].project_path);
    }

    #[test]
    fn scan_lists_only_projects_with_transcripts() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        write_transcript(base, "Users-jack-client-foo", "uuid-1", SAMPLE);
        fs::create_dir_all(base.join("Users-jack-client-bar").join("terminals")).unwrap();

        let projects = scan_projects_in(base).unwrap();
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].session_count, 1);
        assert_eq!(projects[0].provider.as_deref(), Some("cursor-agent"));
    }

    #[test]
    fn extract_session_info_derives_id_count_and_title() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        write_transcript(base, "Users-jack-client-foo", "uuid-1", SAMPLE);
        let file = base
            .join("Users-jack-client-foo")
            .join("agent-transcripts")
            .join("uuid-1")
            .join("uuid-1.jsonl");

        let session =
            extract_session_info(&file, "Users-jack-client-foo", &HashMap::new()).unwrap();
        assert_eq!(session.actual_session_id, "uuid-1");
        assert_eq!(session.message_count, 2);
        assert_eq!(session.provider.as_deref(), Some("cursor-agent"));
        // Title must be the clean user query, not the raw XML wrapper.
        assert_eq!(session.summary.as_deref(), Some("fix the LOGIN bug"));
    }

    #[test]
    fn parse_transcript_maps_roles_with_deterministic_uuids() {
        let messages = parse_transcript(SAMPLE, "uuid-1", "2026-06-20T00:00:00Z");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role.as_deref(), Some("user"));
        assert_eq!(messages[0].message_type, "user");
        assert_eq!(messages[0].provider.as_deref(), Some("cursor-agent"));
        assert_eq!(messages[1].role.as_deref(), Some("assistant"));
        assert_eq!(messages[0].uuid, "uuid-1-0");
        assert_eq!(messages[1].uuid, "uuid-1-1");
        // User text is cleaned: XML wrapper stripped.
        let user_content = messages[0].content.as_ref().unwrap().to_string();
        assert!(
            !user_content.contains("<user_query>"),
            "raw XML tag leaked into user message"
        );
        assert!(
            user_content.contains("fix the LOGIN bug"),
            "user query text missing"
        );
    }

    #[test]
    fn parse_transcript_skips_blank_and_non_turn_lines() {
        let data = format!(
            "\n  \n{SAMPLE}{}",
            r#"{"role":"system","message":{"content":[]}}"#
        );
        let messages = parse_transcript(&data, "s", "");
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn summarize_strips_wrapper_and_truncates() {
        assert_eq!(
            summarize("<user_query>hello world</user_query>"),
            "hello world"
        );
        let long = "x".repeat(200);
        let s = summarize(&long);
        assert!(s.chars().count() <= SUMMARY_MAX_CHARS + 1);
        assert!(s.ends_with('\u{2026}'));
    }

    // ---------------------------------------------------------------------------
    // New tests for issue #472
    // ---------------------------------------------------------------------------

    /// `clean_user_text` strips the `<user_query>` wrapper and drops the trailing `<context>` blob.
    #[test]
    fn clean_user_text_strips_wrapper_and_context() {
        let input = "<user_query>do something useful</user_query><context>lots of verbose context here</context>";
        let result = clean_user_text(input);
        assert_eq!(result, "do something useful");
        assert!(!result.contains("context"), "context blob leaked");
        assert!(!result.contains('<'), "XML tags leaked");
    }

    /// `clean_user_text` passes non-wrapped assistant text through unchanged.
    #[test]
    fn clean_user_text_no_wrapper_passthrough() {
        let input = "Here is the fix for your bug.";
        assert_eq!(clean_user_text(input), input);
    }

    /// A message consisting entirely of redacted blocks must be skipped.
    #[test]
    fn redacted_only_message_is_skipped() {
        // Explicit redaction type.
        let explicit =
            r#"{"role":"user","message":{"content":[{"type":"redacted","data":"[REDACTED]"}]}}"#;
        // Literal [REDACTED] text sentinel.
        let sentinel =
            r#"{"role":"user","message":{"content":[{"type":"text","text":"[REDACTED]"}]}}"#;
        // Mixed: both forms in one message.
        let mixed_redacted = r#"{"role":"assistant","message":{"content":[{"type":"redacted","data":"[REDACTED]"},{"type":"text","text":"[REDACTED]"}]}}"#;

        let data = format!("{explicit}\n{sentinel}\n{mixed_redacted}\n");
        let messages = parse_transcript(&data, "s", "");
        assert_eq!(messages.len(), 0, "all-redacted messages must be dropped");
    }

    /// A message where only *some* blocks are redacted must still be shown.
    #[test]
    fn mixed_redacted_message_is_kept() {
        let line = r#"{"role":"assistant","message":{"content":[{"type":"redacted","data":"[REDACTED]"},{"type":"text","text":"Here is what I found."}]}}"#;
        let messages = parse_transcript(line, "s", "");
        assert_eq!(messages.len(), 1, "partially-redacted message must be kept");
        let content = messages[0].content.as_ref().unwrap().to_string();
        assert!(content.contains("Here is what I found."));
        assert!(
            !content.contains("[REDACTED]"),
            "redacted placeholder leaked into output"
        );
    }

    /// `tool_result` and `command_output` blocks must appear in the rendered message.
    #[test]
    fn tool_result_and_command_output_extracted() {
        // JSONL: one compact JSON object per line (parse_transcript splits on
        // newlines, so the transcript line must not be pretty-printed).
        let line = r#"{"role":"assistant","message":{"content":[{"type":"tool_result","content":[{"type":"text","text":"file contents here"}]},{"type":"command_output","output":"$ cargo build\nCompiling foo v0.1.0"}]}}"#;
        let messages = parse_transcript(line, "s", "");
        assert_eq!(messages.len(), 1);
        let content = messages[0].content.as_ref().unwrap().to_string();
        assert!(
            content.contains("file contents here"),
            "tool_result text missing"
        );
        assert!(content.contains("cargo build"), "command_output missing");
    }

    // ---------------------------------------------------------------------------
    // Native title index (`~/.cursor/chats/<hash>/<uuid>/meta.json`)
    // ---------------------------------------------------------------------------

    fn write_meta(chats_root: &Path, workspace_hash: &str, uuid: &str, body: &str) {
        let dir = chats_root.join(workspace_hash).join(uuid);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("meta.json"), body).unwrap();
    }

    /// The title the CLI shows in its resume picker (and that `/rename` writes)
    /// lives in `meta.json`'s `title` field, keyed by the session UUID that is
    /// also the transcript file stem.
    #[test]
    fn native_title_index_reads_titles_from_meta_json() {
        let tmp = TempDir::new().unwrap();
        let chats = tmp.path();
        write_meta(
            chats,
            "hash-a",
            "uuid-1",
            r#"{"schemaVersion":1,"createdAtMs":1,"hasConversation":true,"title":"my rename","updatedAtMs":2}"#,
        );
        write_meta(
            chats,
            "hash-b",
            "uuid-2",
            r#"{"schemaVersion":1,"hasConversation":true,"title":"  padded  "}"#,
        );

        let index = load_native_title_index(chats);
        assert_eq!(index.get("uuid-1").map(String::as_str), Some("my rename"));
        // Surrounding whitespace is stripped.
        assert_eq!(index.get("uuid-2").map(String::as_str), Some("padded"));
    }

    /// Missing/empty/malformed titles must not poison the index; non-meta files
    /// (store.db, `prompt_history.json`) and stray layouts must be ignored.
    #[test]
    fn native_title_index_skips_unusable_entries() {
        let tmp = TempDir::new().unwrap();
        let chats = tmp.path();
        // No `title` field (e.g. subagent sessions).
        write_meta(
            chats,
            "h1",
            "uuid-1",
            r#"{"schemaVersion":1,"isSubagent":true}"#,
        );
        // Empty / whitespace-only title.
        write_meta(chats, "h1", "uuid-2", r#"{"schemaVersion":1,"title":"  "}"#);
        // Malformed JSON.
        write_meta(chats, "h1", "uuid-3", "{not json");
        // Non-string title.
        write_meta(chats, "h1", "uuid-4", r#"{"title":42}"#);
        // A meta.json at the wrong depth must not be picked up.
        fs::write(chats.join("meta.json"), r#"{"title":"top-level"}"#).unwrap();
        // store.db / prompt_history.json next to a valid title must be ignored.
        let dir = chats.join("h1").join("uuid-5");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("store.db"), b"sqlite").unwrap();
        fs::write(dir.join("prompt_history.json"), "[]").unwrap();
        fs::write(dir.join("meta.json"), r#"{"title":"real title"}"#).unwrap();

        let index = load_native_title_index(chats);
        assert_eq!(index.len(), 1, "only the valid title may be indexed");
        assert_eq!(index.get("uuid-5").map(String::as_str), Some("real title"));
    }

    /// A stored native title wins over the trimmed first prompt and marks the
    /// session as renamed — this is the `/rename` fix.
    #[test]
    fn extract_session_info_prefers_native_title_over_first_prompt() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        write_transcript(base, "Users-jack-client-foo", "uuid-1", SAMPLE);
        let file = base
            .join("Users-jack-client-foo")
            .join("agent-transcripts")
            .join("uuid-1")
            .join("uuid-1.jsonl");

        let mut index = HashMap::new();
        index.insert("uuid-1".to_string(), "my-renamed-session".to_string());

        let session = extract_session_info(&file, "Users-jack-client-foo", &index).unwrap();
        assert_eq!(session.summary.as_deref(), Some("my-renamed-session"));
        assert!(session.is_renamed, "native title must set is_renamed");
    }

    /// Without a stored title the current behaviour is unchanged: trimmed first
    /// prompt, `is_renamed` false.
    #[test]
    fn extract_session_info_falls_back_to_first_prompt_without_native_title() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        write_transcript(base, "Users-jack-client-foo", "uuid-1", SAMPLE);
        let file = base
            .join("Users-jack-client-foo")
            .join("agent-transcripts")
            .join("uuid-1")
            .join("uuid-1.jsonl");

        // Index present but holding only an unrelated session.
        let mut index = HashMap::new();
        index.insert("other-uuid".to_string(), "other title".to_string());

        let session = extract_session_info(&file, "Users-jack-client-foo", &index).unwrap();
        assert_eq!(session.summary.as_deref(), Some("fix the LOGIN bug"));
        assert!(!session.is_renamed);
    }

    /// `WalkDir` follows a symlinked traversal root by default, so a symlinked
    /// `~/.cursor/chats` must be rejected outright. A real directory is accepted.
    #[test]
    fn chats_root_rejects_symlinked_chats_dir() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();

        // A real chats directory is accepted.
        let good = base.join("good");
        fs::create_dir_all(good.join(".cursor").join("chats")).unwrap();
        assert!(chats_root_in(&good.join(".cursor")).is_some());

        // A symlinked chats directory is rejected.
        let target = base.join("elsewhere");
        fs::create_dir_all(&target).unwrap();
        let bad = base.join("bad");
        fs::create_dir_all(bad.join(".cursor")).unwrap();
        let link = bad.join(".cursor").join("chats");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target, &link).unwrap();
        #[cfg(windows)]
        std::os::windows::fs::symlink_dir(&target, &link).unwrap();
        assert!(
            chats_root_in(&bad.join(".cursor")).is_none(),
            "symlinked chats root must be rejected"
        );
    }

    /// Lines as the CLI's transcript writer (`agent-transcript`, 2026.11.22)
    /// produces them.
    const TOOL_SESSION: &str = concat!(
        r#"{"role":"user","message":{"content":[{"type":"text","text":"<user_query>\nRead the config\nand fix it\n</user_query>"}]}}"#,
        "\n",
        r#"{"role":"assistant","message":{"content":[{"type":"text","text":"[REDACTED]"},{"type":"tool_use","name":"Read","input":{"path":"src/config.ts"}},{"type":"tool_use","name":"Shell","input":{"command":"npm test"}}]}}"#,
        "\n",
        r#"{"role":"user","message":{"content":[{"type":"text","text":"<user_query>\nThe above subagent result is already visible to the user. Briefly summarise it.\n</user_query>"}]}}"#,
        "\n",
        r#"{"role":"assistant","message":{"content":[{"type":"text","text":"Fixed the port.\n\n[REDACTED]\n[REDACTED]"}]}}"#,
        "\n",
        r#"{"type":"turn_ended","status":"success"}"#,
        "\n",
    );

    #[test]
    fn tool_only_turns_show_their_calls_and_injected_notices_are_not_prompts() {
        let messages = parse_transcript(TOOL_SESSION, "s", "");
        let types: Vec<&str> = messages.iter().map(|m| m.message_type.as_str()).collect();
        assert_eq!(types, ["user", "assistant", "assistant"]);

        let calls = messages[1].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(calls.len(), 2, "the redacted reasoning is not shown");
        assert_eq!(calls[0]["type"], "tool_use");
        assert_eq!(calls[0]["name"], "Read");
        assert_eq!(
            calls[0]["input"],
            serde_json::json!({"path": "src/config.ts"})
        );
        assert_eq!(
            calls[1]["input"],
            serde_json::json!({"command": "npm test"})
        );
        assert_ne!(calls[0]["id"], calls[1]["id"]);
        assert!(messages[1].tool_use.is_some());

        let reply = messages[2].content.as_ref().unwrap();
        assert_eq!(reply[0]["text"], "Fixed the port.");

        // The skipped notice still takes its index, so ids match search hits.
        assert_eq!(messages[2].uuid, "s-3");
    }

    #[test]
    fn session_counts_and_titles_what_is_shown_and_search_finds_tool_input() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        write_transcript(base, "Users-jack-client-foo", "uuid-1", TOOL_SESSION);
        let file = base.join("Users-jack-client-foo/agent-transcripts/uuid-1/uuid-1.jsonl");
        let session = extract_session_info(&file, "foo", &HashMap::new()).unwrap();
        assert_eq!(session.message_count, 3);
        assert!(session.has_tool_use);
        assert_eq!(
            session.summary.as_deref(),
            Some("Read the config and fix it")
        );

        let hits = search_in(base, "npm test", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].uuid, "uuid-1-1");
        // A notice is not a hit, and a wrapper tag is not searched.
        assert!(search_in(base, "subagent result", 10).is_empty());
        assert!(search_in(base, "user_query", 10).is_empty());
    }

    #[test]
    fn legacy_flat_transcripts_are_sessions_unless_the_current_file_exists() {
        let tmp = TempDir::new().unwrap();
        let transcripts = tmp.path().join("proj").join("agent-transcripts");
        fs::create_dir_all(transcripts.join("both/subagents")).unwrap();
        fs::write(transcripts.join("flat.jsonl"), SAMPLE).unwrap();
        fs::write(transcripts.join("both.jsonl"), SAMPLE).unwrap();
        fs::write(transcripts.join("both/both.jsonl"), SAMPLE).unwrap();
        fs::write(transcripts.join("both/subagents/sub.jsonl"), SAMPLE).unwrap();

        let mut files: Vec<String> = transcript_files(&transcripts)
            .iter()
            .map(|e| {
                e.path()
                    .strip_prefix(&transcripts)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();
        files.sort();
        assert_eq!(files, ["both/both.jsonl", "flat.jsonl"]);
    }

    #[test]
    fn config_root_follows_the_cli_order() {
        let home = Some(PathBuf::from("/home/me"));
        let env = |vars: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                vars.iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| PathBuf::from(value))
            }
        };
        assert_eq!(
            config_root_from(env(&[]), home.clone()),
            Some(PathBuf::from("/home/me/.cursor"))
        );
        assert_eq!(
            config_root_from(env(&[("XDG_CONFIG_HOME", "/xdg")]), home.clone()),
            Some(PathBuf::from("/xdg/cursor"))
        );
        assert_eq!(
            config_root_from(
                env(&[("XDG_CONFIG_HOME", "/xdg"), ("CURSOR_CONFIG_DIR", "/cfg")]),
                home
            ),
            Some(PathBuf::from("/cfg"))
        );
    }

    #[test]
    #[serial_test::serial]
    fn cursor_data_dir_moves_the_transcripts() {
        let _home = crate::test_utils::SandboxHome::new();
        let data = TempDir::new().unwrap();
        let data_dir = data.path().canonicalize().unwrap();
        let projects = data_dir.join("projects");
        write_transcript(&projects, "Users-me-app", "uuid-1", SAMPLE);
        struct Restore(Option<std::ffi::OsString>);
        impl Drop for Restore {
            fn drop(&mut self) {
                match self.0.take() {
                    Some(value) => std::env::set_var("CURSOR_DATA_DIR", value),
                    None => std::env::remove_var("CURSOR_DATA_DIR"),
                }
            }
        }
        let _restore = Restore(std::env::var_os("CURSOR_DATA_DIR"));
        std::env::set_var("CURSOR_DATA_DIR", &data_dir);

        assert_eq!(get_base_path().map(PathBuf::from), Some(projects.clone()));
        let project = scan_projects().unwrap().remove(0);
        let sessions = load_sessions(&project.path, false).unwrap();
        assert_eq!(load_messages(&sessions[0].file_path).unwrap().len(), 2);
    }
}
