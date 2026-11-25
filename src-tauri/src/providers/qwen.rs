//! Qwen Code provider (Alibaba's Gemini-CLI-derived terminal agent).
//!
//! Qwen Code auto-saves the full transcript (user + assistant + tool results)
//! as per-session JSONL at
//! `<runtime-base>/projects/<sanitizedCwd>/chats/<sessionId>.jsonl`, where the
//! runtime base is `$QWEN_RUNTIME_DIR` / `$QWEN_HOME` / `~/.qwen` and
//! `sanitizedCwd` is the project path with every non-alphanumeric char replaced
//! by `-`. Each line is a `ChatRecord`:
//! ```json
//! { "uuid": "...", "parentUuid": null, "sessionId": "...", "timestamp": "...",
//!   "type": "user"|"assistant"|"tool_result"|"system", "cwd": "/abs/path",
//!   "model": "qwen3-coder-plus", "usageMetadata": {...},
//!   "message": { "role": "user"|"model", "parts": [Part, ...] } }
//! ```
//! `Part` is the `@google/genai` shape: `{text}` / `{text,thought:true}` /
//! `{functionCall:{name,args,id}}` / `{functionResponse:{name,response,id}}`.
//! We map these to the viewer's Claude-style content blocks. Projects group by
//! the real `cwd` carried on each record.
//!
//! Session titles: `/rename` appends `{"type":"system","subtype":"custom_title",
//! "systemPayload":{"customTitle","titleSource"?}}` to the transcript (last one
//! wins; no `titleSource` means manual) and every session also has a
//! `<sessionId>.title.json` sidecar `{version:1,displayName,source}` with
//! `source` `manual`|`auto`|`prompt`. A `prompt` sidecar only mirrors the first
//! prompt and is not a title. Transcript title first, then a manual/auto
//! sidecar, then the first real prompt (`sessionService.ts` of Qwen Code).
//!
//! User-role records the CLI writes on its own behalf (`provenance:"system"`
//! notifications and cron prompts, `external_agent`, `mod_prompt`) are not
//! typed by the user and are dropped, except `<task-notification>` text which
//! the viewer already renders as an agent update.

use crate::models::{ClaudeMessage, ClaudeProject, ClaudeSession, TokenUsage};
use crate::providers::scan_report::ScanReport;
use crate::providers::ProviderInfo;
use crate::utils::{build_provider_message, search_json_value_case_insensitive};
use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const PROVIDER: &str = "qwen";
const SCHEME: &str = "qwen://";
const SUMMARY_MAX_CHARS: usize = 80;

/// Runtime base dir: `$QWEN_RUNTIME_DIR` / `$QWEN_HOME` / `~/.qwen`.
fn runtime_base() -> Option<PathBuf> {
    for env in ["QWEN_RUNTIME_DIR", "QWEN_HOME"] {
        if let Ok(v) = std::env::var(env) {
            let v = v.trim();
            if !v.is_empty() {
                return Some(PathBuf::from(v));
            }
        }
    }
    Some(crate::utils::home_dir()?.join(".qwen"))
}

fn projects_dir() -> Option<PathBuf> {
    projects_dir_reporting(&ScanReport::default())
}

/// [`projects_dir`], recording in `report` a Qwen home that may not be
/// looked into.
fn projects_dir_reporting(report: &ScanReport) -> Option<PathBuf> {
    let dir = runtime_base()?.join("projects");
    if report.is_dir(&dir) {
        Some(dir)
    } else {
        None
    }
}

/// Detect a Qwen Code installation.
pub fn detect() -> Option<ProviderInfo> {
    let base = projects_dir()?;
    Some(ProviderInfo {
        id: PROVIDER.to_string(),
        display_name: "Qwen Code".to_string(),
        is_available: !session_files(&base, &ScanReport::default()).is_empty(),
        base_path: base.to_string_lossy().to_string(),
    })
}

/// Base path (`<runtime-base>/projects`), for the file watcher.
pub fn get_base_path() -> Option<String> {
    projects_dir().map(|p| p.to_string_lossy().to_string())
}

/// All `<projects>/<dir>/chats/*.jsonl` session files; what the listing
/// cannot read is recorded in `report`.
fn session_files(projects: &Path, report: &ScanReport) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for project in report
        .walk(WalkDir::new(projects).min_depth(1).max_depth(1))
        .filter(|e| !e.path_is_symlink())
        .filter(|e| e.file_type().is_dir())
    {
        let chats = project.path().join("chats");
        if !report.is_dir(&chats) {
            continue;
        }
        for entry in report
            .walk(WalkDir::new(&chats).min_depth(1).max_depth(1))
            .filter(|e| e.file_type().is_file())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("jsonl"))
            .filter(|e| !is_symlink(e.path()))
        {
            files.push(entry.path().to_path_buf());
        }
    }
    files
}

/// Project identity for a session `cwd` — shared by scan and search so
/// search hits group under the scanned project (#672).
fn project_path_for(cwd: &str) -> String {
    format!("{SCHEME}{cwd}")
}

/// Display name for a session `cwd` (its folder leaf, else the cwd itself).
fn project_name_for(cwd: &str) -> String {
    Path::new(cwd)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| cwd.to_string())
}

/// Scan Qwen projects (sessions grouped by the real `cwd`).
pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    scan_projects_reporting(&ScanReport::default())
}

/// [`scan_projects`], recording in `report` the project and chat
/// directories and session files it could not read.
#[allow(clippy::unnecessary_wraps)] // Result kept to match the provider interface.
pub(crate) fn scan_projects_reporting(report: &ScanReport) -> Result<Vec<ClaudeProject>, String> {
    let Some(base) = projects_dir_reporting(report) else {
        return Ok(vec![]);
    };
    struct Agg {
        session_count: usize,
        message_count: usize,
        last_modified: String,
    }
    let mut by_cwd: HashMap<String, Agg> = HashMap::new();

    for file in session_files(&base, report) {
        let Some(data) = report.read_to_string(&file) else {
            continue;
        };
        let Some(meta) = session_meta(&data) else {
            continue;
        };
        if meta.message_count == 0 {
            continue;
        }
        let cwd = meta.cwd.unwrap_or_else(|| "unknown".to_string());
        let mtime = file_mtime_rfc3339(&file);
        let entry = by_cwd.entry(cwd).or_insert_with(|| Agg {
            session_count: 0,
            message_count: 0,
            last_modified: String::new(),
        });
        entry.session_count += 1;
        entry.message_count += meta.message_count;
        let last = meta.last_ts.clone().unwrap_or(mtime);
        if last > entry.last_modified {
            entry.last_modified = last;
        }
    }

    let mut projects: Vec<ClaudeProject> = by_cwd
        .into_iter()
        .map(|(cwd, agg)| {
            let name = project_name_for(&cwd);
            ClaudeProject {
                name,
                path: project_path_for(&cwd),
                actual_path: cwd,
                session_count: agg.session_count,
                message_count: agg.message_count,
                last_modified: agg.last_modified,
                git_info: None,
                provider: Some(PROVIDER.to_string()),
                storage_type: None,
                custom_directory_label: None,
            }
        })
        .collect();
    projects.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(projects)
}

/// Load the sessions for one Qwen project (filtered by `cwd`).
pub fn load_sessions(
    project_path: &str,
    _exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    load_sessions_reporting(project_path, &ScanReport::default())
}

/// [`load_sessions`], recording in `report` what it could not read: a
/// session file that cannot be read may belong to the project.
#[allow(clippy::unnecessary_wraps)] // Result kept to match the provider interface.
pub(crate) fn load_sessions_reporting(
    project_path: &str,
    report: &ScanReport,
) -> Result<Vec<ClaudeSession>, String> {
    let Some(base) = projects_dir_reporting(report) else {
        return Ok(vec![]);
    };
    let target_cwd = project_path.strip_prefix(SCHEME).unwrap_or(project_path);
    let mut sessions = Vec::new();

    for file in session_files(&base, report) {
        let Some(data) = report.read_to_string(&file) else {
            continue;
        };
        let Some(meta) = session_meta(&data) else {
            continue;
        };
        if meta.message_count == 0 {
            continue;
        }
        if meta.cwd.as_deref().unwrap_or("unknown") != target_cwd {
            continue;
        }
        let (summary, is_renamed) =
            resolve_summary(meta.custom_title, read_sidecar(&file, report), meta.summary);
        let mtime = file_mtime_rfc3339(&file);
        let first = meta.first_ts.clone().unwrap_or_else(|| mtime.clone());
        let last = meta.last_ts.clone().unwrap_or(mtime);
        sessions.push(ClaudeSession {
            session_id: file.to_string_lossy().to_string(),
            actual_session_id: meta.session_id,
            file_path: file.to_string_lossy().to_string(),
            project_name: Path::new(target_cwd)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
            message_count: meta.message_count,
            first_message_time: first,
            last_message_time: last.clone(),
            last_modified: last,
            has_tool_use: meta.has_tool_use,
            has_errors: false,
            summary,
            is_renamed,
            provider: Some(PROVIDER.to_string()),
            storage_type: None,
            entrypoint: None,
        });
    }

    sessions.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(sessions)
}

/// Load all messages from one Qwen session file.
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
    Ok(parse_messages(&data))
}

/// Search across all Qwen sessions.
pub fn search(query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    let Some(base) = projects_dir() else {
        return Ok(vec![]);
    };
    let query_lower = query.to_lowercase();
    let mut results = Vec::new();
    for file in session_files(&base, &ScanReport::default()) {
        let Ok(data) = fs::read_to_string(&file) else {
            continue;
        };
        // Same cwd source (and "unknown" fallback) as the scan groups by.
        let cwd = session_meta(&data)
            .and_then(|m| m.cwd)
            .unwrap_or_else(|| "unknown".to_string());
        let project_name = project_name_for(&cwd);
        let project_path = project_path_for(&cwd);
        for mut msg in parse_messages(&data) {
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
// Pure parsing (unit-testable)
// ============================================================================

struct SessionMeta {
    session_id: String,
    cwd: Option<String>,
    message_count: usize,
    summary: Option<String>,
    first_ts: Option<String>,
    last_ts: Option<String>,
    has_tool_use: bool,
    /// Last `system/custom_title` record of the transcript.
    custom_title: Option<NativeTitle>,
}

/// A title the producer persisted for a session.
#[derive(Debug, PartialEq, Eq)]
struct NativeTitle {
    text: String,
    /// `true` for a name the user chose (`/rename`), `false` for a generated one.
    manual: bool,
}

/// `systemPayload` of a `custom_title` record. A missing `titleSource` is a
/// legacy manual title (`chatRecordingService.ts` `currentTitleSource`).
fn custom_title_of(rec: &Value) -> Option<NativeTitle> {
    let payload = rec.get("systemPayload")?;
    let text = payload.get("customTitle")?.as_str()?.trim();
    if text.is_empty() {
        return None;
    }
    let source = payload.get("titleSource").and_then(Value::as_str);
    Some(NativeTitle {
        text: text.to_string(),
        manual: source != Some("auto"),
    })
}

/// Longest `displayName` the producer's sidecar reader accepts, in chars.
const SIDECAR_NAME_MAX_CHARS: usize = 256;

/// What a `<sessionId>.title.json` sidecar holds, validated like the producer's
/// `readSessionTitleSidecar`.
#[derive(Debug, PartialEq, Eq)]
enum Sidecar {
    /// A title (`source` `manual` or `auto`).
    Title(NativeTitle),
    /// The first prompt's label (`source` `prompt`); not a title.
    Prompt(String),
}

fn parse_sidecar(data: &str) -> Option<Sidecar> {
    let value: Value = serde_json::from_str(data).ok()?;
    if value.get("version").and_then(Value::as_u64) != Some(1) {
        return None;
    }
    let name = value.get("displayName")?.as_str()?.trim();
    if name.is_empty() || name.chars().count() > SIDECAR_NAME_MAX_CHARS {
        return None;
    }
    let title = |manual| {
        Some(Sidecar::Title(NativeTitle {
            text: name.to_string(),
            manual,
        }))
    };
    match value.get("source").and_then(Value::as_str)? {
        "manual" => title(true),
        "auto" => title(false),
        "prompt" => Some(Sidecar::Prompt(name.to_string())),
        _ => None,
    }
}

/// The session's list title and whether the user named it. The producer's
/// order: transcript `custom_title`, then a non-prompt sidecar; otherwise the
/// first real prompt (or, for a transcript that lost its head, the sidecar's
/// prompt label).
fn resolve_summary(
    transcript_title: Option<NativeTitle>,
    sidecar: Option<Sidecar>,
    first_prompt: Option<String>,
) -> (Option<String>, bool) {
    let (title, prompt_label) = match sidecar {
        Some(Sidecar::Title(t)) => (transcript_title.or(Some(t)), None),
        Some(Sidecar::Prompt(p)) => (transcript_title, Some(p)),
        None => (transcript_title, None),
    };
    if let Some(t) = title {
        return (Some(t.text), t.manual);
    }
    (
        first_prompt.or_else(|| prompt_label.map(|p| summarize(&p))),
        false,
    )
}

/// A user-role record the CLI wrote on its own behalf rather than the person
/// typing it, which the viewer must not show as a prompt. `<task-notification>`
/// text is kept: the frontend renders it as an agent update.
fn is_injected_user_record(rec: &Value) -> bool {
    let provenance = rec.get("provenance").and_then(Value::as_str);
    let subtype = rec.get("subtype").and_then(Value::as_str);
    let injected = matches!(provenance, Some("system" | "external_agent" | "mod_prompt"))
        || matches!(subtype, Some("notification" | "cron" | "agent_message"));
    injected && !carries_task_notification(rec)
}

fn carries_task_notification(rec: &Value) -> bool {
    rec.get("message")
        .and_then(|m| m.get("parts"))
        .and_then(Value::as_array)
        .is_some_and(|parts| {
            parts.iter().any(|p| {
                p.get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|t| t.trim_start().starts_with("<task-notification>"))
            })
        })
}

/// Lightweight per-session metadata from a chat JSONL (one parse).
fn session_meta(data: &str) -> Option<SessionMeta> {
    let mut session_id = String::new();
    let mut cwd = None;
    let mut message_count = 0usize;
    let mut summary = None;
    let mut first_ts = None;
    let mut last_ts = None;
    let mut has_tool_use = false;
    let mut custom_title = None;

    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if session_id.is_empty() {
            if let Some(s) = rec.get("sessionId").and_then(Value::as_str) {
                session_id = s.to_string();
            }
        }
        if cwd.is_none() {
            cwd = rec.get("cwd").and_then(Value::as_str).map(str::to_string);
        }
        let rec_type = rec.get("type").and_then(Value::as_str).unwrap_or("");
        if rec_type == "system"
            && rec.get("subtype").and_then(Value::as_str) == Some("custom_title")
        {
            // Re-appended at EOF on exit/re-anchor: the last record is current.
            if let Some(title) = custom_title_of(&rec) {
                custom_title = Some(title);
            }
            continue;
        }
        if !matches!(rec_type, "user" | "assistant" | "tool_result") {
            continue;
        }
        if rec_type == "user" && is_injected_user_record(&rec) {
            continue;
        }
        message_count += 1;
        if let Some(ts) = rec.get("timestamp").and_then(Value::as_str) {
            if first_ts.is_none() {
                first_ts = Some(ts.to_string());
            }
            last_ts = Some(ts.to_string());
        }
        if summary.is_none() && rec_type == "user" {
            summary = first_text(&rec).map(|t| summarize(&t));
        }
        if rec_type == "tool_result" || record_has_function_call(&rec) {
            has_tool_use = true;
        }
    }

    if session_id.is_empty() && cwd.is_none() && message_count == 0 {
        return None;
    }
    Some(SessionMeta {
        session_id,
        cwd,
        message_count,
        summary,
        first_ts,
        last_ts,
        has_tool_use,
        custom_title,
    })
}

/// Parse a chat JSONL into messages.
fn parse_messages(data: &str) -> Vec<ClaudeMessage> {
    let mut messages = Vec::new();
    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if let Some(msg) = convert_record(&rec) {
            messages.push(msg);
        }
    }
    messages
}

fn convert_record(rec: &Value) -> Option<ClaudeMessage> {
    let rec_type = rec.get("type").and_then(Value::as_str)?;
    // tool_result records carry role 'user' in genai but are their own record
    // type; surface them in the user lane like Claude tool results.
    let (message_type, role) = match rec_type {
        "user" if is_injected_user_record(rec) => return None,
        "user" | "tool_result" => ("user", "user"),
        "assistant" => ("assistant", "assistant"),
        _ => return None, // skip system / UI records
    };

    let parts = rec
        .get("message")
        .and_then(|m| m.get("parts"))
        .and_then(Value::as_array);
    let mut blocks: Vec<Value> = parts
        .map(|ps| ps.iter().filter_map(convert_part).collect())
        .unwrap_or_default();
    if rec_type == "tool_result" {
        mark_tool_outcome(&mut blocks, rec.get("toolCallResult"));
    }
    if blocks.is_empty() {
        return None;
    }

    let session_id = rec.get("sessionId").and_then(Value::as_str).unwrap_or("");
    let uuid = rec
        .get("uuid")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or(session_id)
        .to_string();
    let timestamp = rec
        .get("timestamp")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let model = rec.get("model").and_then(Value::as_str).map(str::to_string);

    let mut msg = build_provider_message(
        PROVIDER,
        uuid,
        session_id,
        timestamp,
        message_type,
        Some(role),
        Some(Value::Array(blocks)),
        model,
    );
    msg.parent_uuid = rec
        .get("parentUuid")
        .and_then(Value::as_str)
        .map(str::to_string);
    if let Some(usage) = rec.get("usageMetadata") {
        msg.usage = Some(convert_usage(usage));
    }
    Some(msg)
}

/// Map a `@google/genai` `Part` to a Claude-style content block.
fn convert_part(part: &Value) -> Option<Value> {
    if let Some(call) = part.get("functionCall") {
        let name = call
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let id = call.get("id").and_then(Value::as_str).unwrap_or("");
        let input = call.get("args").cloned().unwrap_or_else(|| json!({}));
        return Some(json!({ "type": "tool_use", "id": id, "name": name, "input": input }));
    }
    if let Some(resp) = part.get("functionResponse") {
        let id = resp.get("id").and_then(Value::as_str).unwrap_or("");
        // `response` is `{ output }` on success and `{ error }` for a failed
        // or cancelled call (`createErrorResponse` / `createCancelledResponse`
        // in coreToolScheduler.ts); show that text, not the JSON envelope.
        let response = resp.get("response");
        let content = response
            .and_then(|r| r.get("output").or_else(|| r.get("error")))
            .map(stringify_value)
            .or_else(|| response.map(stringify_value))
            .unwrap_or_default();
        return Some(json!({ "type": "tool_result", "tool_use_id": id, "content": content }));
    }
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        if text.is_empty() {
            return None;
        }
        // A part flagged thought:true is reasoning, not user-visible text.
        if part
            .get("thought")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Some(json!({ "type": "thinking", "thinking": text, "signature": "" }));
        }
        return Some(json!({ "type": "text", "text": text }));
    }
    // inlineData / fileData / executableCode / codeExecutionResult: not rendered
    // in this MVP.
    None
}

fn convert_usage(usage: &Value) -> TokenUsage {
    let g = |k: &str| usage.get(k).and_then(Value::as_u64).map(|n| n as u32);
    let prompt_tokens = g("promptTokenCount");
    let cached_tokens = g("cachedContentTokenCount");
    TokenUsage {
        // Gemini-compatible usage reports promptTokenCount inclusive of the
        // cached subset. Keep only non-cached prompt tokens in input_tokens so
        // totals and billing do not count the same cached tokens twice.
        input_tokens: prompt_tokens.map(|prompt| prompt.saturating_sub(cached_tokens.unwrap_or(0))),
        output_tokens: g("candidatesTokenCount"),
        cache_creation_input_tokens: None,
        cache_read_input_tokens: cached_tokens,
        reasoning_tokens: g("thoughtsTokenCount"),
        service_tier: None,
        ..Default::default()
    }
}

fn record_has_function_call(rec: &Value) -> bool {
    rec.get("message")
        .and_then(|m| m.get("parts"))
        .and_then(Value::as_array)
        .is_some_and(|ps| ps.iter().any(|p| p.get("functionCall").is_some()))
}

/// Producer outcome of a `tool_result` record: its `toolCallResult.status` is
/// `success`/`error` (or `cancelled`, which is neither) for the call named by
/// `toolCallResult.callId`. Nothing is inferred from the result text.
fn mark_tool_outcome(blocks: &mut [Value], call_result: Option<&Value>) {
    let Some(call_result) = call_result else {
        return;
    };
    let is_error = match call_result.get("status").and_then(Value::as_str) {
        Some("error") => true,
        Some("success") => false,
        _ => return,
    };
    let Some(call_id) = call_result.get("callId").and_then(Value::as_str) else {
        return;
    };
    for block in blocks.iter_mut().filter(|b| {
        b.get("type").and_then(Value::as_str) == Some("tool_result")
            && b.get("tool_use_id").and_then(Value::as_str) == Some(call_id)
    }) {
        block["is_error"] = Value::Bool(is_error);
    }
}

/// The `<sessionId>.title.json` sidecar next to a chat file, if readable.
fn read_sidecar(chat_file: &Path, report: &ScanReport) -> Option<Sidecar> {
    let path = chat_file.with_extension("title.json");
    if is_symlink(&path) {
        return None;
    }
    parse_sidecar(&report.read_to_string(&path)?)
}

/// First non-thought text of a record's message (for the session summary).
fn first_text(rec: &Value) -> Option<String> {
    let parts = rec
        .get("message")
        .and_then(|m| m.get("parts"))
        .and_then(Value::as_array)?;
    parts.iter().find_map(|p| {
        if p.get("thought").and_then(Value::as_bool).unwrap_or(false) {
            return None;
        }
        p.get("text")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    })
}

fn stringify_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn summarize(text: &str) -> String {
    let cleaned = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.chars().count() > SUMMARY_MAX_CHARS {
        format!(
            "{}…",
            cleaned.chars().take(SUMMARY_MAX_CHARS).collect::<String>()
        )
    } else {
        cleaned
    }
}

fn is_symlink(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
}

fn validate_under_base(path: &Path) -> Result<(), String> {
    let base = projects_dir().ok_or("Qwen projects path not found")?;
    let canon_base = base
        .canonicalize()
        .map_err(|e| format!("Failed to resolve Qwen base: {e}"))?;
    let canon_path = path
        .canonicalize()
        .map_err(|e| format!("Failed to resolve session path: {e}"))?;
    if canon_path.starts_with(&canon_base) {
        Ok(())
    } else {
        Err(format!(
            "Path is outside the Qwen projects root: {}",
            path.display()
        ))
    }
}

fn file_mtime_rfc3339(path: &Path) -> String {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::SystemTime::UNIX_EPOCH).ok())
        .map(|d| {
            #[allow(clippy::cast_possible_wrap)]
            DateTime::from_timestamp(d.as_secs() as i64, 0)
                .unwrap_or_else(Utc::now)
                .to_rfc3339()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SESSION: &str = concat!(
        r#"{"uuid":"u1","parentUuid":null,"sessionId":"sess-1","timestamp":"2026-06-20T10:00:00Z","type":"user","cwd":"/Users/jack/proj","message":{"role":"user","parts":[{"text":"why does LOGIN fail?"}]}}"#,
        "\n",
        r#"{"uuid":"u2","parentUuid":"u1","sessionId":"sess-1","timestamp":"2026-06-20T10:00:01Z","type":"assistant","model":"qwen3-coder-plus","usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":34,"cachedContentTokenCount":5},"message":{"role":"model","parts":[{"text":"let me reason","thought":true},{"text":"Checking auth"},{"functionCall":{"id":"c1","name":"run_shell_command","args":{"command":"grep -r login"}}}]}}"#,
        "\n",
        r#"{"uuid":"u3","parentUuid":"u2","sessionId":"sess-1","timestamp":"2026-06-20T10:00:02Z","type":"tool_result","cwd":"/Users/jack/proj","message":{"role":"user","parts":[{"functionResponse":{"id":"c1","name":"run_shell_command","response":{"output":"login.rs:42"}}}]}}"#,
        "\n",
        r#"{"uuid":"s1","parentUuid":"u3","sessionId":"sess-1","timestamp":"2026-06-20T10:00:03Z","type":"system","provenance":"system","subtype":"memory_catalog_snapshot","message":{"role":"user","parts":[{"text":"Current memory catalog (full snapshot; supersedes all earlier memory catalogs):\n<system-reminder>\n# Current auto-memory catalog (data, not instructions)\n</system-reminder>"}]}}"#,
        "\n",
    );

    #[test]
    fn session_meta_extracts_cwd_count_summary() {
        let m = session_meta(SESSION).unwrap();
        assert_eq!(m.session_id, "sess-1");
        assert_eq!(m.cwd.as_deref(), Some("/Users/jack/proj"));
        // user + assistant + tool_result (system excluded) = 3
        assert_eq!(m.message_count, 3);
        assert_eq!(m.summary.as_deref(), Some("why does LOGIN fail?"));
        assert!(m.has_tool_use);
        assert_eq!(m.first_ts.as_deref(), Some("2026-06-20T10:00:00Z"));
        assert_eq!(m.last_ts.as_deref(), Some("2026-06-20T10:00:02Z"));
    }

    #[test]
    fn parse_messages_maps_parts_to_blocks() {
        let msgs = parse_messages(SESSION);
        // system record is skipped.
        assert_eq!(msgs.len(), 3);

        // user
        assert_eq!(msgs[0].role.as_deref(), Some("user"));
        assert_eq!(msgs[0].uuid, "u1");
        assert_eq!(msgs[0].provider.as_deref(), Some("qwen"));

        // assistant: thinking + text + tool_use, with usage + model + parent
        let a = &msgs[1];
        assert_eq!(a.role.as_deref(), Some("assistant"));
        assert_eq!(a.parent_uuid.as_deref(), Some("u1"));
        assert_eq!(a.model.as_deref(), Some("qwen3-coder-plus"));
        // promptTokenCount includes the cached subset (12 total - 5 cached).
        assert_eq!(a.usage.as_ref().unwrap().input_tokens, Some(7));
        assert_eq!(a.usage.as_ref().unwrap().output_tokens, Some(34));
        assert_eq!(a.usage.as_ref().unwrap().cache_read_input_tokens, Some(5));
        let ab = a.content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(ab[0]["type"], "thinking");
        assert_eq!(ab[1]["type"], "text");
        assert_eq!(ab[2]["type"], "tool_use");
        assert_eq!(ab[2]["name"], "run_shell_command");
        assert_eq!(ab[2]["id"], "c1");

        // tool_result -> user lane, tool_result block with extracted output
        assert_eq!(msgs[2].role.as_deref(), Some("user"));
        let tr = msgs[2].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(tr[0]["type"], "tool_result");
        assert_eq!(tr[0]["tool_use_id"], "c1");
        assert_eq!(tr[0]["content"], "login.rs:42");
    }

    #[test]
    fn parse_messages_preserves_parallel_agent_calls_and_results() {
        let session = concat!(
            r#"{"uuid":"a1","sessionId":"sess-agents","timestamp":"2026-07-07T00:00:00Z","type":"assistant","message":{"role":"model","parts":[{"functionCall":{"id":"c1","name":"agent","args":{"description":"Check API","prompt":"Review API"}}},{"functionCall":{"id":"c2","name":"task","args":{"description":"Check UI","prompt":"Review UI"}}}]}}"#,
            "\n",
            r#"{"uuid":"r1","sessionId":"sess-agents","timestamp":"2026-07-07T00:00:01Z","type":"tool_result","message":{"role":"user","parts":[{"functionResponse":{"id":"c1","name":"agent","response":{"output":"API OK"}}},{"functionResponse":{"id":"c2","name":"task","response":{"output":"UI OK"}}}]}}"#,
            "\n",
        );

        let messages = parse_messages(session);
        assert_eq!(messages.len(), 2);

        let calls = messages[0].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["name"], "agent");
        assert_eq!(calls[0]["id"], "c1");
        assert_eq!(calls[1]["name"], "task");
        assert_eq!(calls[1]["id"], "c2");

        let results = messages[1].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["tool_use_id"], "c1");
        assert_eq!(results[1]["tool_use_id"], "c2");
    }

    #[test]
    fn convert_part_kinds() {
        assert!(convert_part(&json!({"text":""})).is_none());
        assert_eq!(convert_part(&json!({"text":"hi"})).unwrap()["type"], "text");
        assert_eq!(
            convert_part(&json!({"text":"r","thought":true})).unwrap()["type"],
            "thinking"
        );
        // functionResponse with object output is stringified.
        let tr =
            convert_part(&json!({"functionResponse":{"id":"x","response":{"output":{"k":1}}}}))
                .unwrap();
        assert_eq!(tr["type"], "tool_result");
        assert!(tr["content"].as_str().unwrap().contains("\"k\""));
    }

    #[test]
    #[serial_test::serial]
    fn load_messages_rejects_path_outside_root() {
        let _home = crate::test_utils::SandboxHome::new();
        let qwen_home = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(qwen_home.path().join("projects")).unwrap();
        let previous = std::env::var_os("QWEN_HOME");
        std::env::set_var("QWEN_HOME", qwen_home.path());

        let outside = tempfile::TempDir::new().unwrap();
        let file = outside.path().join("chat.jsonl");
        fs::write(&file, SESSION).unwrap();
        let res = load_messages(&file.to_string_lossy());

        match previous {
            Some(v) => std::env::set_var("QWEN_HOME", v),
            None => std::env::remove_var("QWEN_HOME"),
        }
        assert!(res
            .expect_err("path should be rejected")
            .contains("outside"));
    }

    #[test]
    #[serial_test::serial]
    fn search_hits_carry_scanned_project_path_for_same_named_cwds() {
        let _home = crate::test_utils::SandboxHome::new();
        let qwen_home = tempfile::TempDir::new().unwrap();
        for (dir, cwd) in [("a", "/work/client-a/app"), ("b", "/work/client-b/app")] {
            let chats = qwen_home.path().join("projects").join(dir).join("chats");
            fs::create_dir_all(&chats).unwrap();
            let line = json!({ "uuid": format!("u-{dir}"), "sessionId": dir,
                "timestamp": "2026-06-20T10:00:00Z", "type": "user", "cwd": cwd,
                "message": { "role": "user", "parts": [{ "text": "needle" }] } });
            fs::write(chats.join(format!("{dir}.jsonl")), format!("{line}\n")).unwrap();
        }
        let previous_runtime = std::env::var_os("QWEN_RUNTIME_DIR");
        let previous = std::env::var_os("QWEN_HOME");
        std::env::remove_var("QWEN_RUNTIME_DIR");
        std::env::set_var("QWEN_HOME", qwen_home.path());

        let projects = scan_projects().unwrap();
        let hits = search("needle", 10).unwrap();

        if let Some(v) = previous_runtime {
            std::env::set_var("QWEN_RUNTIME_DIR", v);
        }
        match previous {
            Some(v) => std::env::set_var("QWEN_HOME", v),
            None => std::env::remove_var("QWEN_HOME"),
        }

        let path_of = |cwd: &str| {
            projects
                .iter()
                .find(|p| p.actual_path == cwd)
                .map(|p| p.path.clone())
                .unwrap()
        };
        let a = path_of("/work/client-a/app");
        let b = path_of("/work/client-b/app");
        assert_ne!(a, b);
        assert_eq!(hits.len(), 2);
        for hit in &hits {
            let expected = if hit.uuid == "u-a" { &a } else { &b };
            assert_eq!(hit.project_path.as_ref(), Some(expected));
            assert_eq!(hit.project_name.as_deref(), Some("app"));
        }
    }

    // Records below are real Qwen Code 0.28.4 output (qwen2.5-coder:7b via
    // Ollama), trimmed, with the sandbox path replaced by /home/user/demo.
    // Producer: QwenLM/qwen-code@e8424c88ddd463c56602063efde57c328b917cd4
    // (v0.28.4) packages/core/src/services/chatRecordingService.ts
    // L320-331 (`ChatRecordProvenance`), L3171-3195 (`createNotificationRecord`:
    // `subtype` + `provenance:"system"`), L4194-4210 (`custom_title` record);
    // session-title-sidecar.ts L16-25, L64-102; sessionService.ts L1130-1153
    // (transcript title, then a non-`prompt` sidecar); coreToolScheduler.ts
    // L1310-1342 (`response: { error }`; L1344-1380 for cancelled calls).

    const PROMPT_TEXT: &str =
        "Read src/math.js and add a function mul(a, b) that returns a*b, exported like add.";
    const REAL_PROMPT: &str = r#"{"uuid":"7f67b65a-269d-4922-94f2-9a641a7c9448","parentUuid":null,"sessionId":"66666666-2222-4333-8444-555555555555","timestamp":"2026-11-25T03:56:57.997Z","type":"user","provenance":"real_user","cwd":"/home/user/demo","version":"0.28.4","gitBranch":"main","message":{"role":"user","parts":[{"text":"Read src/math.js and add a function mul(a, b) that returns a*b, exported like add."}]},"promptId":"66666666-2222-4333-8444-555555555555########0"}"#;
    /// `/rename Demo math session`, typed in the interactive TUI.
    const REAL_CUSTOM_TITLE: &str = r#"{"uuid":"0f6653e4-abc7-4f9c-98f6-50819523e3af","parentUuid":"faafc68d-10a5-411f-a614-a8c2b528067b","sessionId":"66666666-2222-4333-8444-555555555555","timestamp":"2026-11-25T04:01:07.509Z","type":"system","provenance":"system","cwd":"/home/user/demo","version":"0.28.4","gitBranch":"main","subtype":"custom_title","systemPayload":{"customTitle":"Demo math session","titleSource":"manual"}}"#;
    const REAL_SIDECAR_PROMPT: &str = r#"{"version":1,"displayName":"Read src/math.js and add a function mul(a, b) that returns a*b, exported like add.","source":"prompt","updatedAt":"2026-11-25T03:54:11.392Z"}"#;
    const REAL_SIDECAR_MANUAL: &str = r#"{"version":1,"displayName":"Demo math session","source":"manual","updatedAt":"2026-11-25T04:01:07.536Z"}"#;
    /// Background-shell completion injected by the CLI (interactive TUI).
    const REAL_NOTIFICATION: &str = r#"{"uuid":"89a6bf40-f30d-4f54-99ab-96e8969619fb","parentUuid":"296aa97c-45ba-4ea4-a49c-beaabd397dee","sessionId":"88888888-2222-4333-8444-555555555555","timestamp":"2026-11-25T04:04:19.392Z","type":"user","provenance":"system","cwd":"/home/user/demo","version":"0.28.4","gitBranch":"main","subtype":"notification","deliveredTurn":true,"message":{"role":"user","parts":[{"text":"<task-notification>\n<task-id>bg_e3a6c4c9</task-id>\n<kind>shell</kind>\n<status>completed</status>\n<summary>Shell command \"sleep 3; echo background-done\" completed.</summary>\n</task-notification>"}]}}"#;
    /// A failing `npm test`: `response.error`, `toolCallResult.status:"error"`.
    const REAL_FAILED_SHELL: &str = r#"{"uuid":"32a34072-347c-4355-9cc5-3ac1c714d5a0","parentUuid":"ad90f6df-bdbf-4163-816e-a5bdc660330e","sessionId":"55555555-2222-4333-8444-555555555555","timestamp":"2026-11-25T03:55:10.219Z","type":"tool_result","provenance":"tool_result","cwd":"/home/user/demo","version":"0.28.4","gitBranch":"main","message":{"role":"user","parts":[{"functionResponse":{"id":"call_nhkp98dw","name":"run_shell_command","response":{"error":"Command: npm test\nDirectory: /home/user/demo\nOutput (stdout and stderr combined): > demo@1.0.0 test\n> node test.js\n\nAssertionError [ERR_ASSERTION]: Expected values to be strictly equal:\n\n4 !== 5"}}}]},"toolCallResult":{"callId":"call_nhkp98dw","status":"error","resultDisplay":{"type":"shell_result","version":1,"directory":"/home/user/demo","exitCode":1,"outcome":"failed"}}}"#;
    const REAL_OK_SHELL: &str = r#"{"uuid":"a0f1c0de-0000-4000-8000-000000000001","parentUuid":"32a34072-347c-4355-9cc5-3ac1c714d5a0","sessionId":"55555555-2222-4333-8444-555555555555","timestamp":"2026-11-25T03:55:11.000Z","type":"tool_result","provenance":"tool_result","cwd":"/home/user/demo","version":"0.28.4","gitBranch":"main","message":{"role":"user","parts":[{"functionResponse":{"id":"call_2xph16yq","name":"run_shell_command","response":{"output":"ok"}}}]},"toolCallResult":{"callId":"call_2xph16yq","status":"success"}}"#;

    fn jsonl(lines: &[&str]) -> String {
        lines.join("\n") + "\n"
    }

    /// Write `<QWEN_HOME>/projects/demo/chats/<id>.jsonl` (+ optional sidecar),
    /// list the project's sessions and restore the env.
    fn list_demo_sessions(id: &str, transcript: &str, sidecar: Option<&str>) -> Vec<ClaudeSession> {
        let _home = crate::test_utils::SandboxHome::new();
        let qwen_home = tempfile::TempDir::new().unwrap();
        let chats = qwen_home.path().join("projects").join("demo").join("chats");
        fs::create_dir_all(&chats).unwrap();
        fs::write(chats.join(format!("{id}.jsonl")), transcript).unwrap();
        if let Some(sidecar) = sidecar {
            fs::write(chats.join(format!("{id}.title.json")), sidecar).unwrap();
        }
        let previous_runtime = std::env::var_os("QWEN_RUNTIME_DIR");
        let previous = std::env::var_os("QWEN_HOME");
        std::env::remove_var("QWEN_RUNTIME_DIR");
        std::env::set_var("QWEN_HOME", qwen_home.path());

        let sessions = load_sessions("qwen:///home/user/demo", false).unwrap();

        if let Some(v) = previous_runtime {
            std::env::set_var("QWEN_RUNTIME_DIR", v);
        }
        match previous {
            Some(v) => std::env::set_var("QWEN_HOME", v),
            None => std::env::remove_var("QWEN_HOME"),
        }
        sessions
    }

    #[test]
    #[serial_test::serial]
    fn rename_record_in_transcript_becomes_the_session_title() {
        let transcript = jsonl(&[REAL_PROMPT, REAL_CUSTOM_TITLE]);
        let sessions = list_demo_sessions("66666666", &transcript, Some(REAL_SIDECAR_PROMPT));
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].summary.as_deref(), Some("Demo math session"));
        assert!(sessions[0].is_renamed);
        // The title record is metadata, not a message.
        assert_eq!(sessions[0].message_count, 1);
    }

    #[test]
    #[serial_test::serial]
    fn manual_sidecar_titles_a_transcript_without_the_rename_record() {
        let sessions = list_demo_sessions(
            "66666666",
            &jsonl(&[REAL_PROMPT]),
            Some(REAL_SIDECAR_MANUAL),
        );
        assert_eq!(sessions[0].summary.as_deref(), Some("Demo math session"));
        assert!(sessions[0].is_renamed);
    }

    #[test]
    #[serial_test::serial]
    fn prompt_sidecar_is_not_a_title() {
        let sessions = list_demo_sessions(
            "66666666",
            &jsonl(&[REAL_PROMPT]),
            Some(REAL_SIDECAR_PROMPT),
        );
        assert_eq!(
            sessions[0].summary.as_deref(),
            Some(summarize(PROMPT_TEXT).as_str())
        );
        assert!(!sessions[0].is_renamed);
    }

    #[test]
    fn title_resolution_follows_the_producer_order() {
        let manual = |t: &str| NativeTitle {
            text: t.to_string(),
            manual: true,
        };
        let side = parse_sidecar;
        // Transcript record beats the sidecar; the sidecar beats the prompt.
        assert_eq!(
            resolve_summary(
                Some(manual("A")),
                side(REAL_SIDECAR_MANUAL),
                Some("p".into())
            ),
            (Some("A".to_string()), true)
        );
        assert_eq!(
            resolve_summary(None, side(REAL_SIDECAR_MANUAL), Some("p".into())),
            (Some("Demo math session".to_string()), true)
        );
        // A generated title is shown but is not a user rename.
        let auto = r#"{"version":1,"displayName":"Fix math","source":"auto","updatedAt":""}"#;
        assert_eq!(
            resolve_summary(None, side(auto), Some("p".into())),
            (Some("Fix math".to_string()), false)
        );
        // A transcript that lost its head falls back to the prompt label.
        assert_eq!(
            resolve_summary(None, side(REAL_SIDECAR_PROMPT), None),
            (Some(summarize(PROMPT_TEXT)), false)
        );
        // Invalid sidecars are ignored like the producer's reader does.
        assert!(side(r#"{"version":2,"displayName":"x","source":"manual"}"#).is_none());
        assert!(side(r#"{"version":1,"displayName":" ","source":"manual"}"#).is_none());
        assert!(side(r#"{"version":1,"displayName":"x","source":"other"}"#).is_none());
        // No `titleSource` is a legacy manual title; the last record wins.
        let data = jsonl(&[
            REAL_PROMPT,
            &REAL_CUSTOM_TITLE.replace("Demo math session", "First"),
            &REAL_CUSTOM_TITLE.replace(r#","titleSource":"manual""#, ""),
        ]);
        assert_eq!(
            session_meta(&data).unwrap().custom_title,
            Some(manual("Demo math session"))
        );
    }

    #[test]
    fn injected_user_records_are_not_prompts() {
        // Producer `createNotificationRecord` also writes `subtype:"cron"`
        // with `provenance:"system"` (not triggered in the sample).
        let cron = REAL_NOTIFICATION
            .replace(r#""subtype":"notification""#, r#""subtype":"cron""#)
            .replace(
                "<task-notification>\\n<task-id>bg_e3a6c4c9</task-id>\\n<kind>shell</kind>\\n<status>completed</status>\\n<summary>Shell command \\\"sleep 3; echo background-done\\\" completed.</summary>\\n</task-notification>",
                "run the nightly check",
            );
        assert!(cron.contains("run the nightly check"));
        let data = jsonl(&[&cron, REAL_PROMPT]);
        let msgs = parse_messages(&data);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].uuid, "7f67b65a-269d-4922-94f2-9a641a7c9448");
        let meta = session_meta(&data).unwrap();
        assert_eq!(meta.message_count, 1);
        // The injected prompt must not become the session summary.
        assert_eq!(
            meta.summary.as_deref(),
            Some(summarize(PROMPT_TEXT).as_str())
        );
    }

    #[test]
    fn task_notification_stays_for_the_frontend_agent_update_card() {
        let msgs = parse_messages(&jsonl(&[REAL_NOTIFICATION]));
        assert_eq!(msgs.len(), 1);
        let blocks = msgs[0].content.as_ref().unwrap().as_array().unwrap();
        assert!(blocks[0]["text"]
            .as_str()
            .unwrap()
            .starts_with("<task-notification>"));
    }

    #[test]
    fn failed_shell_result_is_an_error_with_its_text() {
        let msgs = parse_messages(&jsonl(&[REAL_FAILED_SHELL, REAL_OK_SHELL]));
        let failed = msgs[0].content.as_ref().unwrap()[0].clone();
        assert_eq!(failed["tool_use_id"], "call_nhkp98dw");
        assert_eq!(failed["is_error"], true);
        let text = failed["content"].as_str().unwrap();
        assert!(text.starts_with("Command: npm test\nDirectory: /home/user/demo"));
        let ok = msgs[1].content.as_ref().unwrap()[0].clone();
        assert_eq!(ok["content"], "ok");
        assert_eq!(ok["is_error"], false);
    }

    #[test]
    fn tool_result_without_producer_status_gets_no_error_flag() {
        // Cancelled calls also use `response.error`, but are not failures.
        let cancelled = REAL_FAILED_SHELL.replace(r#""status":"error""#, r#""status":"cancelled""#);
        let msgs = parse_messages(&jsonl(&[&cancelled]));
        let block = &msgs[0].content.as_ref().unwrap()[0];
        assert!(block.get("is_error").is_none());
        assert!(block["content"]
            .as_str()
            .unwrap()
            .starts_with("Command: npm test"));
    }
}
