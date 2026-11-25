use crate::commands::session::SearchFilters;
use crate::models::{ClaudeMessage, ClaudeProject, ClaudeSession};
use crate::providers::ProviderInfo;
use crate::utils::{build_provider_message, is_symlink, search_json_value_case_insensitive};
use chrono::{Local, NaiveDateTime, SecondsFormat, TimeZone, Utc};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const HISTORY_FILE: &str = ".aider.chat.history.md";
const SESSION_HEADER_PREFIX: &str = "# aider chat started at ";

/// Detect Aider by checking top-level project directories for history files.
/// Only does a shallow (depth-1) check to avoid slow recursive scans at startup.
pub fn detect() -> Option<ProviderInfo> {
    let dirs = get_search_dirs();
    // Shallow check: look for .aider.chat.history.md directly in search dirs
    // and their immediate children (depth 1 only, no recursive scan)
    let has_history = dirs.iter().any(|d| {
        if d.join(HISTORY_FILE).is_file() {
            return true;
        }
        // Check one level of subdirectories
        fs::read_dir(d)
            .into_iter()
            .flatten()
            .flatten()
            .any(|entry| entry.path().join(HISTORY_FILE).is_file())
    });

    Some(ProviderInfo {
        id: "aider".to_string(),
        display_name: "Aider".to_string(),
        base_path: dirs
            .first()
            .map(|d| d.to_string_lossy().to_string())
            .unwrap_or_default(),
        is_available: has_history,
    })
}

/// Scan for all Aider projects
pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    let mut projects = Vec::new();

    for history_path in history_files() {
        let project_dir = history_path.parent().ok_or("Invalid history file path")?;
        let content = match fs::read_to_string(&history_path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let sessions: Vec<_> = split_sessions(&content)
            .into_iter()
            .filter(|s| !s.is_blank_preamble())
            .collect();
        if sessions.is_empty() {
            continue;
        }

        projects.push(ClaudeProject {
            name: project_name_for(project_dir),
            path: project_path_for(project_dir),
            actual_path: project_dir.to_string_lossy().into_owned(),
            session_count: sessions.len(),
            message_count: sessions
                .iter()
                .map(|s| parse_blocks(&s.content).len())
                .sum(),
            last_modified: sessions
                .last()
                .and_then(|s| s.timestamp.clone())
                .unwrap_or_default(),
            git_info: None,
            provider: Some("aider".to_string()),
            storage_type: Some("markdown".to_string()),
            custom_directory_label: None,
        });
    }

    Ok(projects)
}

/// Load sessions for an Aider project
pub fn load_sessions(
    project_path: &str,
    _exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    let dir = project_path
        .strip_prefix("aider://")
        .unwrap_or(project_path);

    // Validate path is absolute and points to a real directory
    let dir_path = PathBuf::from(dir);
    if !dir_path.is_absolute() {
        return Err("Aider project path must be absolute".to_string());
    }

    let history_path = dir_path.join(HISTORY_FILE);
    if !history_path.is_file() {
        return Ok(Vec::new());
    }
    validate_history_file(&history_path)?;

    let content =
        fs::read_to_string(&history_path).map_err(|e| format!("Failed to read history: {e}"))?;

    let project_name = project_name_for(&dir_path);
    let history_file = history_path.to_string_lossy();

    let sessions_data = split_sessions(&content);
    let mut sessions = Vec::with_capacity(sessions_data.len());

    for (index, session) in sessions_data.iter().enumerate().rev() {
        if session.is_blank_preamble() {
            continue;
        }
        let blocks = parse_blocks(&session.content);
        let message_count = blocks.len();
        let summary = blocks
            .iter()
            .find(|b| b.role == BlockRole::User)
            .and_then(|b| b.text.lines().find(|l| !l.trim().is_empty()))
            .map(|l| l.trim().to_string());
        let timestamp = session.timestamp.clone().unwrap_or_default();
        let session_path = session_id_for(&history_file, index);

        sessions.push(ClaudeSession {
            session_id: session_path.clone(),
            actual_session_id: format!("session-{index}"),
            file_path: session_path,
            project_name: project_name.clone(),
            message_count,
            first_message_time: timestamp.clone(),
            last_message_time: timestamp.clone(),
            last_modified: timestamp,
            has_tool_use: session.content.contains("\n> "),
            has_errors: false,
            summary,
            is_renamed: false,
            provider: Some("aider".to_string()),
            storage_type: Some("markdown".to_string()),
            entrypoint: None,
        });
    }

    Ok(sessions)
}

/// Load messages from a specific Aider session
pub fn load_messages(session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    // session_path format: aider://<history_file_path>#<session_index>
    let (file_path, session_index) = parse_session_path(session_path)?;

    // Validate: must be absolute, must end with the expected filename
    let path = Path::new(file_path);
    if !path.is_absolute() {
        return Err("Session path must be absolute".to_string());
    }
    validate_history_file(path)?;

    let content = fs::read_to_string(path).map_err(|e| format!("Failed to read history: {e}"))?;

    let sessions = split_sessions(&content);
    let session = sessions
        .get(session_index)
        .ok_or_else(|| format!("Session index {session_index} out of range"))?;

    let base_timestamp = session.timestamp.as_deref().unwrap_or_default();
    let session_id = session_id_for(file_path, session_index);

    Ok(parse_messages(
        &session.content,
        &session_id,
        base_timestamp,
    ))
}

/// Search across all Aider sessions
pub fn search(query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    Ok(search_with_filters(
        query,
        limit,
        &SearchFilters::parse(&serde_json::Value::Null),
    ))
}

pub(crate) fn search_with_filters(
    query: &str,
    limit: usize,
    filters: &SearchFilters<'_>,
) -> Vec<ClaudeMessage> {
    if limit == 0 {
        return Vec::new();
    }
    let query_lower = query.to_lowercase();
    let mut results = Vec::new();

    for history_path in history_files() {
        let Some(project_dir) = history_path.parent() else {
            continue;
        };
        let project_name = project_name_for(project_dir);
        let project_path = project_path_for(project_dir);
        if !filters.matches_project(Some(&project_path), Some(&project_name)) {
            continue;
        }
        let content = match fs::read_to_string(&history_path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let file_path = history_path.to_string_lossy();

        for (index, session) in split_sessions(&content).iter().enumerate() {
            let base_ts = session.timestamp.as_deref().unwrap_or_default();
            let session_id = session_id_for(&file_path, index);

            for mut msg in parse_messages(&session.content, &session_id, base_ts) {
                if !filters.matches_message(&msg) {
                    continue;
                }
                if msg.content.as_ref().is_some_and(|content| {
                    search_json_value_case_insensitive(content, &query_lower)
                }) {
                    msg.project_name = Some(project_name.clone());
                    msg.project_path = Some(project_path.clone());
                    results.push(msg);
                    if results.len() >= limit {
                        return results;
                    }
                }
            }
        }
    }

    results
}

// ============================================================================
// Private helpers
// ============================================================================

fn project_name_for(project_dir: &Path) -> String {
    project_dir
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

/// Opaque project identity shared by `scan_projects` and `search`.
fn project_path_for(project_dir: &Path) -> String {
    format!("aider://{}", project_dir.to_string_lossy())
}

fn session_id_for(history_path: &str, index: usize) -> String {
    format!("aider://{history_path}#{index}")
}

/// Visit each discovered history only once, even when search roots overlap.
fn history_files() -> impl Iterator<Item = PathBuf> {
    let mut seen = HashSet::new();
    get_search_dirs()
        .into_iter()
        .flat_map(|dir| find_history_files(&dir, 100))
        .filter(move |path| seen.insert(path.canonicalize().unwrap_or_else(|_| path.clone())))
}

fn get_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(home) = crate::utils::home_dir() {
        for subdir in ["client", "projects", "code", "src", "dev", "work", "repos"] {
            let d = home.join(subdir);
            if d.is_dir() {
                dirs.push(d);
            }
        }
        // Also check home dir itself
        dirs.push(home);
    }
    dirs
}

/// Deepest project directory the scan descends to, below a search dir.
const MAX_SCAN_DEPTH: usize = 4;

/// Directories the scan never descends into.
fn is_scanned_dir_name(name: &str) -> bool {
    !name.starts_with('.')
        && name != "node_modules"
        && name != "target"
        && name != "dist"
        && name != "build"
}

/// Confine a caller-supplied history file to what `scan_projects` can find:
/// below a search dir, within the scan depth, and never inside a directory
/// the scan skips. Both sides are canonicalised, so a symlink is judged by
/// where it points.
fn validate_history_file(path: &Path) -> Result<(), String> {
    if path.file_name().and_then(|n| n.to_str()) != Some(HISTORY_FILE) {
        return Err(format!("Invalid Aider history file: {}", path.display()));
    }
    let outside = || {
        format!(
            "Aider history file is outside the scanned directories: {}",
            path.display()
        )
    };
    let canonical = path.canonicalize().map_err(|_| outside())?;
    let project = canonical.parent().ok_or_else(outside)?;
    // Within scope are also history files a client saved itself through the
    // export allowlist (e.g. ~/Downloads): it only reads back its own bytes.
    let scanned = get_search_dirs()
        .iter()
        // The scan refuses to walk a symlinked search dir; so does this.
        .filter(|dir| !is_symlink(dir))
        .filter_map(|dir| dir.canonicalize().ok())
        .any(|root| {
            project.strip_prefix(&root).is_ok_and(|rel| {
                rel.components().count() <= MAX_SCAN_DEPTH
                    && rel
                        .components()
                        .all(|c| is_scanned_dir_name(&c.as_os_str().to_string_lossy()))
            })
        });
    if scanned {
        Ok(())
    } else {
        Err(outside())
    }
}

fn find_history_files(dir: &Path, max: usize) -> Vec<PathBuf> {
    let mut files = Vec::new();
    find_history_recursive(dir, &mut files, max, 0, MAX_SCAN_DEPTH);
    files
}

fn find_history_recursive(
    dir: &Path,
    results: &mut Vec<PathBuf>,
    max: usize,
    depth: usize,
    max_depth: usize,
) {
    if depth > max_depth || results.len() >= max {
        return;
    }
    if is_symlink(dir) {
        return;
    }

    let history = dir.join(HISTORY_FILE);
    if history.is_file() && !is_symlink(&history) {
        results.push(history);
    }

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if results.len() >= max {
                return;
            }
            let path = entry.path();
            if path.is_dir() && !is_symlink(&path) {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if is_scanned_dir_name(&name) {
                    find_history_recursive(&path, results, max, depth + 1, max_depth);
                }
            }
        }
    }
}

struct SessionData {
    timestamp: Option<String>,
    content: String,
}

impl SessionData {
    /// The blank line Aider writes before the first header. It is never
    /// listed, but it keeps its slot so every real session keeps the index
    /// (`aider://<file>#<index>`) that favorites, names and other saved
    /// references to it are keyed by.
    fn is_blank_preamble(&self) -> bool {
        self.timestamp.is_none() && self.content.trim().is_empty()
    }
}

/// Split the history into sessions on `# aider chat started at` headers.
///
/// Aider writes the header with `datetime.now().strftime("%Y-%m-%d %H:%M:%S")`
/// (`aider/io.py`, `InputOutput.__init__`), i.e. the machine's local wall
/// clock with no zone information, so it is read in the local zone.
///
/// The text before the first header is the entry at index 0, which is
/// [`SessionData::is_blank_preamble`] in a file Aider wrote; callers skip it.
fn split_sessions(content: &str) -> Vec<SessionData> {
    split_sessions_in(content, &Local)
}

fn split_sessions_in<Tz: TimeZone>(content: &str, tz: &Tz) -> Vec<SessionData> {
    let mut sessions = Vec::new();
    let mut current_lines: Vec<&str> = Vec::new();
    let mut current_timestamp: Option<String> = None;

    for line in content.lines() {
        if let Some(ts_str) = line.strip_prefix(SESSION_HEADER_PREFIX) {
            // Save previous session
            if !current_lines.is_empty() {
                sessions.push(SessionData {
                    timestamp: current_timestamp.take(),
                    content: current_lines.join("\n"),
                });
                current_lines.clear();
            }
            current_timestamp = Some(header_timestamp(ts_str, tz));
        } else {
            current_lines.push(line);
        }
    }

    // Last session
    if !current_lines.is_empty() {
        sessions.push(SessionData {
            timestamp: current_timestamp,
            content: current_lines.join("\n"),
        });
    }

    sessions
}

/// Convert the header's local wall-clock time to an RFC 3339 UTC instant.
/// Unparseable text is passed through unchanged.
fn header_timestamp<Tz: TimeZone>(ts_str: &str, tz: &Tz) -> String {
    let trimmed = ts_str.trim();
    let Some(naive) = trimmed
        .get(..19)
        .and_then(|s| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").ok())
    else {
        return trimmed.to_string();
    };
    // An ambiguous (DST fold) time resolves to its first occurrence; a time
    // inside a DST gap cannot have been written by a local clock, so it falls
    // back to being read as UTC.
    let utc = tz
        .from_local_datetime(&naive)
        .earliest()
        .map_or_else(|| naive.and_utc(), |dt| dt.with_timezone(&Utc));
    utc.to_rfc3339_opts(SecondsFormat::Secs, true)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockRole {
    User,
    Assistant,
    Tool,
}

struct Block {
    role: BlockRole,
    text: String,
}

/// Aider ends every `####` / `>` line with two spaces (a markdown hard break).
fn strip_hard_break(text: &str) -> &str {
    text.strip_suffix("  ").unwrap_or(text)
}

/// A prompt keeps its first line's indentation; only leading blank lines
/// (which render nothing) and trailing whitespace are dropped.
fn user_body(text: &str) -> &str {
    let body = text.trim_end();
    let start = body
        .find(|c: char| !c.is_whitespace())
        .map_or(0, |first| body[..first].rfind('\n').map_or(0, |nl| nl + 1));
    &body[start..]
}

/// Group a session's lines into messages.
///
/// Producer format (`aider/io.py`):
/// - `user_input` writes one `#### ` line per input line (a blank input line
///   is `####` plus spaces), so adjacent `#### ` lines are one prompt.
/// - `ai_output` writes `"\n" + reply + "\n\n"`: a reply is always set off by
///   blank lines, so a `> ` line glued to reply text is a markdown quote inside
///   the reply, not a tool message.
/// - `tool_output` writes `"> " + text + "  \n"` once, so a multi-line message
///   only has its first line prefixed and only its last line ends in the
///   two-space hard break (`append_chat_history`); the lines in between have
///   no marker and no blank line before them.
fn parse_blocks(content: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut role: Option<BlockRole> = None;
    let mut text = String::new();
    // The last tool line lacked the hard break, so the message continues.
    let mut tool_open = false;

    fn flush(blocks: &mut Vec<Block>, role: &mut Option<BlockRole>, text: &mut String) {
        if let Some(role) = role.take() {
            let body = if role == BlockRole::User {
                user_body(text)
            } else {
                text.trim()
            };
            if !body.trim().is_empty() {
                blocks.push(Block {
                    role,
                    text: body.to_string(),
                });
            }
        }
        text.clear();
    }

    for line in content.lines() {
        if let Some(user_text) = line.strip_prefix("#### ") {
            let user_text = strip_hard_break(user_text);
            if role == Some(BlockRole::User) {
                text.push('\n');
                text.push_str(user_text);
            } else {
                flush(&mut blocks, &mut role, &mut text);
                role = Some(BlockRole::User);
                text.push_str(user_text);
            }
        } else if let Some(tool_text) = line.strip_prefix("> ") {
            if role == Some(BlockRole::Assistant) && !text.ends_with('\n') {
                // Markdown quote glued to reply text.
                text.push('\n');
                text.push_str(line);
                continue;
            }
            if role == Some(BlockRole::Tool) {
                text.push('\n');
            } else {
                flush(&mut blocks, &mut role, &mut text);
                role = Some(BlockRole::Tool);
            }
            text.push_str(strip_hard_break(tool_text));
            tool_open = !line.ends_with("  ");
        } else if line.trim().is_empty() {
            if role == Some(BlockRole::Assistant) {
                text.push('\n');
            } else {
                // A blank line ends a prompt or a tool message.
                flush(&mut blocks, &mut role, &mut text);
            }
        } else if role == Some(BlockRole::Tool) && tool_open {
            // Unprefixed continuation of a multi-line tool message.
            text.push('\n');
            text.push_str(strip_hard_break(line));
            tool_open = !line.ends_with("  ");
        } else if role == Some(BlockRole::Assistant) {
            text.push('\n');
            text.push_str(line);
        } else {
            flush(&mut blocks, &mut role, &mut text);
            role = Some(BlockRole::Assistant);
            text.push_str(line);
        }
    }

    flush(&mut blocks, &mut role, &mut text);
    blocks
}

fn parse_messages(content: &str, session_id: &str, base_timestamp: &str) -> Vec<ClaudeMessage> {
    parse_blocks(content)
        .into_iter()
        .enumerate()
        .map(|(index, block)| {
            let (msg_type, msg_role) = match block.role {
                BlockRole::User => ("user", Some("user")),
                BlockRole::Assistant => ("assistant", Some("assistant")),
                BlockRole::Tool => ("system", None),
            };

            let mut msg = build_provider_message(
                "aider",
                format!("aider-{}", index + 1),
                session_id,
                base_timestamp.to_string(),
                msg_type,
                msg_role,
                Some(serde_json::json!([{"type": "text", "text": block.text}])),
                None,
            );

            if block.role == BlockRole::Tool {
                msg.subtype = Some("tool_output".to_string());
            }
            msg
        })
        .collect()
}

fn parse_session_path(session_path: &str) -> Result<(&str, usize), String> {
    let path = session_path
        .strip_prefix("aider://")
        .unwrap_or(session_path);

    let (file_path, index_str) = path
        .rsplit_once('#')
        .ok_or_else(|| format!("Invalid session path: {session_path}"))?;

    let index: usize = index_str
        .parse()
        .map_err(|_| format!("Invalid session index: {index_str}"))?;

    Ok((file_path, index))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Records from a real Aider 0.86.2 run (`aider --message ...`), trimmed;
    /// the command-line path is replaced with a neutral one. Every `####` and
    /// `>` line ends with the two-space hard break Aider writes, so the lines
    /// are spelled out instead of using a raw string (editors strip trailing
    /// spaces).
    const REAL_HISTORY_LINES: &[&str] = &[
        "",
        "# aider chat started at 2026-11-25 12:35:26",
        "",
        "> Detected dumb terminal, disabling fancy input and pretty output.  ",
        "> /home/user/aider --model ollama_chat/qwen2.5-coder:7b --message Read src/math.js and add mul. src/math.js  ",
        "> Aider v0.86.2  ",
        "> Added src/math.js to the chat.  ",
        "",
        "#### Read src/math.js and add a function mul(a, b) that returns a*b, exported like add.  ",
        "",
        "To add the `mul` function to `src/math.js`, I will:",
        "",
        "1. Define the `mul` function.",
        "",
        "> Tokens: 764 sent, 92 received.  ",
        "> Applied edit to src/math.js  ",
        "",
        "# aider chat started at 2026-11-25 12:41:39",
        "",
        "> Aider v0.86.2  ",
        "> Added test.js to the chat.  ",
        "",
        "#### /run npm test  ",
        "> Add 0.3k tokens of command output to the chat? (Y)es/(N)o [Yes]: y  ",
        "> Added 28 lines of output to the chat.  ",
        "",
        "# aider chat started at 2026-11-25 12:41:41",
        "",
        "> Detected dumb terminal, disabling fancy input and pretty output.  ",
        "> /home/user/aider --model ollama_chat/qwen2.5-coder:7b --message Please do two things:",
        "- list the files in src",
        "- summarize what test.js checks test.js  ",
        "> Aider v0.86.2  ",
        "> Added test.js to the chat.  ",
        "",
        "#### Please do two things:  ",
        "#### - list the files in src  ",
        "#### - summarize what test.js checks  ",
        "",
        "1. List the files in `src`:",
        "   - `math.js`",
        "",
        "> Tokens: 818 sent, 84 received.  ",
        "",
        "# aider chat started at 2026-11-25 12:45:25",
        "",
        "> Aider v0.86.2  ",
        "",
        "#### Line one of file prompt  ",
        "#### Line two  ",
        "####   ",
        "#### Line four after blank  ",
        "",
        "Understood. Please let me know which files need changes.",
        "",
        "> Tokens: 815 sent, 36 received.  ",
    ];

    fn real_history() -> String {
        let mut s = REAL_HISTORY_LINES.join("\n");
        s.push('\n');
        s
    }

    fn texts(messages: &[ClaudeMessage]) -> Vec<(String, String)> {
        messages
            .iter()
            .map(|m| {
                let text = m.content.as_ref().unwrap().as_array().unwrap()[0]["text"]
                    .as_str()
                    .unwrap()
                    .to_string();
                (m.message_type.clone(), text)
            })
            .collect()
    }

    fn kst() -> chrono::FixedOffset {
        chrono::FixedOffset::east_opt(9 * 3600).unwrap()
    }

    /// The sessions the app lists: everything but the blank preamble.
    fn listed_sessions() -> Vec<SessionData> {
        split_sessions_in(&real_history(), &kst())
            .into_iter()
            .filter(|s| !s.is_blank_preamble())
            .collect()
    }

    #[test]
    fn leading_blank_line_is_not_a_listed_session_but_keeps_its_slot() {
        let all = split_sessions_in(&real_history(), &kst());
        assert_eq!(all.len(), 5, "preamble + one session per header");
        assert!(all[0].is_blank_preamble());
        assert!(all[1..].iter().all(|s| s.timestamp.is_some()));
        assert_eq!(listed_sessions().len(), 4);
    }

    #[test]
    fn header_time_is_local_wall_clock_converted_to_utc() {
        let sessions = listed_sessions();
        assert_eq!(
            sessions[0].timestamp.as_deref(),
            Some("2026-11-25T03:35:26Z")
        );
        assert_eq!(
            sessions[3].timestamp.as_deref(),
            Some("2026-11-25T03:45:25Z")
        );

        let west = chrono::FixedOffset::west_opt(5 * 3600).unwrap();
        assert_eq!(
            header_timestamp("2026-11-25 12:35:26", &west),
            "2026-11-25T17:35:26Z"
        );
        assert_eq!(header_timestamp("not a time", &west), "not a time");
    }

    #[test]
    fn header_time_uses_the_machine_zone() {
        let expected = Local
            .with_ymd_and_hms(2026, 11, 25, 12, 35, 26)
            .unwrap()
            .with_timezone(&Utc)
            .to_rfc3339_opts(SecondsFormat::Secs, true);
        let sessions: Vec<_> = split_sessions(&real_history())
            .into_iter()
            .filter(|s| !s.is_blank_preamble())
            .collect();
        assert_eq!(sessions[0].timestamp.as_deref(), Some(expected.as_str()));
    }

    #[test]
    fn adjacent_prompt_lines_are_one_user_message() {
        let sessions = listed_sessions();
        let messages = parse_messages(&sessions[2].content, "s", "t");
        let users: Vec<_> = texts(&messages)
            .into_iter()
            .filter(|(t, _)| t == "user")
            .collect();
        assert_eq!(users.len(), 1);
        assert_eq!(
            users[0].1,
            "Please do two things:\n- list the files in src\n- summarize what test.js checks"
        );
    }

    #[test]
    fn blank_prompt_line_is_a_paragraph_break() {
        let sessions = listed_sessions();
        let messages = parse_messages(&sessions[3].content, "s", "t");
        let users: Vec<_> = texts(&messages)
            .into_iter()
            .filter(|(t, _)| t == "user")
            .collect();
        assert_eq!(users.len(), 1);
        assert_eq!(
            users[0].1,
            "Line one of file prompt\nLine two\n\nLine four after blank"
        );
    }

    #[test]
    fn multi_line_banner_is_not_assistant_text() {
        let sessions = listed_sessions();
        let messages = texts(&parse_messages(&sessions[2].content, "s", "t"));
        let kinds: Vec<_> = messages.iter().map(|(t, _)| t.as_str()).collect();
        // banner, prompt, reply, footer
        assert_eq!(kinds, ["system", "user", "assistant", "system"]);
        assert!(messages[0].1.contains(
            "--message Please do two things:\n- list the files in src\n- summarize what test.js checks test.js\nAider v0.86.2"
        ));
        assert!(!messages[2]
            .1
            .contains("summarize what test.js checks test.js"));
    }

    #[test]
    fn run_command_without_output_has_no_assistant_message() {
        let sessions = listed_sessions();
        let kinds: Vec<_> = texts(&parse_messages(&sessions[1].content, "s", "t"))
            .into_iter()
            .map(|(t, _)| t)
            .collect();
        assert_eq!(kinds, ["system", "user", "system"]);
    }

    #[test]
    #[serial_test::serial]
    fn message_count_equals_emitted_messages() {
        let home = crate::test_utils::SandboxHome::new();
        let project = home.path().join("client").join("proj");
        write_history(&project);

        let sessions = load_sessions(&format!("aider://{}", project.display()), false).unwrap();
        assert_eq!(sessions.len(), 4);
        // Index 0 is the blank line before the first header; real sessions
        // keep the ids (indices) they always had.
        let indices: Vec<_> = sessions
            .iter()
            .map(|s| s.session_id.rsplit('#').next().unwrap().to_string())
            .collect();
        assert_eq!(indices, ["4", "3", "2", "1"]);
        // load_sessions lists newest first.
        let counts: Vec<_> = sessions.iter().map(|s| s.message_count).collect();
        assert_eq!(counts, [4, 4, 3, 4]);
        for session in &sessions {
            let emitted = load_messages(&session.file_path).unwrap().len();
            assert_eq!(session.message_count, emitted, "{}", session.file_path);
        }
        let titles: Vec<_> = sessions.iter().map(|s| s.summary.as_deref()).collect();
        assert_eq!(
            titles,
            [
                Some("Line one of file prompt"),
                Some("Please do two things:"),
                Some("/run npm test"),
                Some(
                    "Read src/math.js and add a function mul(a, b) that returns a*b, exported like add."
                ),
            ]
        );
    }

    #[test]
    fn test_parse_messages() {
        let sessions = listed_sessions();
        let messages = parse_messages(&sessions[0].content, "test-session", "t");

        // banner, user, assistant, footer
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].message_type, "system");
        assert_eq!(messages[1].message_type, "user");
        assert_eq!(messages[2].message_type, "assistant");
        assert_eq!(messages[3].message_type, "system");
        assert_eq!(messages[0].subtype.as_deref(), Some("tool_output"));
        assert_eq!(messages[0].provider, Some("aider".to_string()));
    }

    #[test]
    fn user_prompt_keeps_first_line_indentation_but_not_leading_blank_lines() {
        let content = "####   \n####     print(1)  \n####   \n\nok";
        let messages = texts(&parse_messages(content, "s", "t"));
        assert_eq!(
            messages[0],
            ("user".to_string(), "    print(1)".to_string())
        );
        let blank = parse_messages("####   \n####   \n\nok", "s", "t");
        assert_eq!(blank.len(), 1, "a blank prompt is not a message");
    }

    #[test]
    fn markdown_quote_glued_to_a_reply_stays_in_the_reply() {
        let content =
            "#### why\n\nExplain:\n> quoted\ncontinued answer\n\n> Tokens: 1 sent, 2 received.  ";
        let messages = texts(&parse_messages(content, "s", "t"));
        let kinds: Vec<_> = messages.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(kinds, ["user", "assistant", "system"]);
        assert_eq!(messages[1].1, "Explain:\n> quoted\ncontinued answer");
    }

    #[test]
    fn tool_message_ends_at_its_hard_break() {
        // The first line lacks the two spaces, so the next line continues it;
        // once a line carries them the message is complete.
        let content = "> first\nsecond  \nreply text";
        let messages = texts(&parse_messages(content, "s", "t"));
        assert_eq!(
            messages,
            [
                ("system".to_string(), "first\nsecond".to_string()),
                ("assistant".to_string(), "reply text".to_string()),
            ]
        );
    }
    #[test]
    fn test_parse_user_message() {
        let content = "#### Hello world\n\nResponse here";
        let messages = parse_messages(content, "s1", "2025-01-01T00:00:00Z");
        assert_eq!(messages[0].message_type, "user");
        let text = messages[0].content.as_ref().unwrap().as_array().unwrap()[0]["text"]
            .as_str()
            .unwrap();
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_parse_session_path() {
        let (file, idx) =
            parse_session_path("aider:///home/user/project/.aider.chat.history.md#3").unwrap();
        assert_eq!(file, "/home/user/project/.aider.chat.history.md");
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_empty_content() {
        let messages = parse_messages("", "s1", "2025-01-01T00:00:00Z");
        assert!(messages.is_empty());
    }

    #[test]
    fn test_tool_output_grouping() {
        let content = "> Line 1\n> Line 2\n> Line 3";
        let messages = parse_messages(content, "s1", "2025-01-01T00:00:00Z");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message_type, "system");
        let text = messages[0].content.as_ref().unwrap().as_array().unwrap()[0]["text"]
            .as_str()
            .unwrap();
        assert!(text.contains("Line 1"));
        assert!(text.contains("Line 3"));
    }

    /// Write `real_history()` as `<dir>/.aider.chat.history.md`.
    fn write_history(dir: &Path) -> PathBuf {
        fs::create_dir_all(dir).unwrap();
        let file = dir.join(HISTORY_FILE);
        fs::write(&file, real_history()).unwrap();
        file
    }

    #[test]
    #[serial_test::serial]
    fn load_messages_rejects_path_outside_root() {
        let _home = crate::test_utils::SandboxHome::new();
        let outside = tempfile::TempDir::new().unwrap();
        let file = write_history(&outside.path().join("proj"));

        let res = load_messages(&format!("aider://{}#0", file.display()));
        assert!(res.is_err(), "history outside the scanned dirs was read");
        let res = load_sessions(
            &format!("aider://{}", outside.path().join("proj").display()),
            false,
        );
        assert!(res.is_err(), "project outside the scanned dirs was listed");
    }

    #[test]
    #[serial_test::serial]
    fn load_messages_rejects_history_the_scan_skips() {
        let home = crate::test_utils::SandboxHome::new();
        // Hidden directories are never scanned.
        let file = write_history(&home.path().join(".private").join("proj"));

        assert!(load_messages(&format!("aider://{}#0", file.display())).is_err());
    }

    #[test]
    #[serial_test::serial]
    fn load_messages_accepts_history_under_a_scanned_dir() {
        let home = crate::test_utils::SandboxHome::new();
        let project = home.path().join("client").join("proj");
        let file = write_history(&project);

        let messages = load_messages(&format!("aider://{}#1", file.display())).unwrap();
        assert!(!messages.is_empty());
        let sessions = load_sessions(&format!("aider://{}", project.display()), false).unwrap();
        assert!(!sessions.is_empty());
    }

    /// The scan never walks a symlinked search dir, so an id through one is
    /// outside what the scan can find.
    #[cfg(unix)]
    #[test]
    #[serial_test::serial]
    fn load_messages_rejects_history_under_a_symlinked_search_dir() {
        let home = crate::test_utils::SandboxHome::new();
        let elsewhere = tempfile::TempDir::new().unwrap();
        let file = write_history(&elsewhere.path().join("proj"));
        std::os::unix::fs::symlink(elsewhere.path(), home.path().join("client")).unwrap();
        let via_link = home.path().join("client").join("proj").join(HISTORY_FILE);
        assert!(via_link.is_file(), "fixture: {} missing", file.display());

        assert!(load_messages(&format!("aider://{}#0", via_link.display())).is_err());
    }

    #[test]
    #[serial_test::serial]
    fn search_hits_carry_scanned_project_path_for_same_named_projects() {
        let home = crate::test_utils::SandboxHome::new();
        write_history(&home.path().join("client").join("client-a").join("app"));
        write_history(&home.path().join("work").join("client-b").join("app"));

        let projects = scan_projects().unwrap();
        let paths: std::collections::HashSet<String> =
            projects.iter().map(|p| p.path.clone()).collect();
        assert_eq!(paths.len(), 2, "two distinct app projects: {paths:?}");
        assert!(projects.iter().all(|p| p.name == "app"));

        let hits = search("mul", 100).unwrap();
        let hit_paths: std::collections::HashSet<String> = hits
            .iter()
            .map(|m| m.project_path.clone().expect("hit has project_path"))
            .collect();
        assert_eq!(hit_paths, paths);
    }
}
