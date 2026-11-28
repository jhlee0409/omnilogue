use super::cline_sdk;
use crate::commands::session::SearchFilters;
use crate::models::{ClaudeMessage, ClaudeProject, ClaudeSession};
use crate::providers::scan_report::ScanReport;
use crate::providers::ProviderInfo;
use crate::utils::{
    build_provider_message, is_symlink, ms_to_iso, search_json_value_case_insensitive,
};
use rusqlite::{Connection, OpenFlags};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

/// Known Cline-family extension IDs and their display names
const EXTENSIONS: &[(&str, &str)] = &[
    ("saoudrizwan.claude-dev", "Cline"),
    ("rooveterinaryinc.roo-cline", "Roo Code"),
    // Kilo Code is a Cline/Roo fork: per-task files (api_conversation_history.json,
    // ui_messages.json, task_metadata.json) are byte-identical. But unlike Cline
    // (disk state/taskHistory.json) and Roo (disk tasks/_index.json), Kilo keeps its
    // task index ONLY in VS Code globalState — one row in the global state.vscdb keyed
    // by the extension id — so load_task_history falls back to reading that. The cwd
    // field there is `workspace`, not `cwdOnTaskInitialization` (see task_cwd).
    ("kilocode.kilo-code", "Kilo Code"),
    // PearAI's agent is a Roo Code fork (`trypear/PearAI-Roo-Code`, extension
    // `PearAI.pearai-roo-cline`), with Roo's task files and globalState index.
    ("pearai.pearai-roo-cline", "PearAI Agent"),
];

/// Kilo Code's own legacy CLI ran the extension against a mock VS Code whose
/// global storage is `~/.kilocode/cli/global` (`Kilo-Org/kilocode-legacy`
/// `packages/agent-runtime/src/utils/paths.ts` `getGlobalStorageDir`): the
/// same `tasks/<id>/` files, with the extension's globalState in
/// `global-state.json` instead of `state.vscdb`.
const KILO_CLI_GLOBAL: &str = ".kilocode/cli/global";

/// Kilo Code's legacy CLI store, when it is a real directory.
fn kilo_cli_base(report: &ScanReport) -> Option<PathBuf> {
    let home = crate::utils::home_dir()?;
    let kilocode = home.join(".kilocode");
    let cli = kilocode.join("cli");
    let global = cli.join("global");
    let real = [&kilocode, &cli, &global]
        .iter()
        .all(|dir| !is_symlink(dir));
    (real && report.is_dir(&global)).then_some(global)
}

fn is_kilo_cli_base(base_path: &Path) -> bool {
    base_path.ends_with(KILO_CLI_GLOBAL)
}

/// Detect Cline/Roo Code installations
pub fn detect() -> Option<ProviderInfo> {
    let paths = get_all_base_paths();
    let sdk_sessions = cline_sdk::sessions_dir();
    let is_available = !paths.is_empty() || sdk_sessions.is_some();

    Some(ProviderInfo {
        id: "cline".to_string(),
        display_name: "Cline".to_string(),
        base_path: paths
            .first()
            .map(|(p, _)| p.clone())
            .or(sdk_sessions)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default(),
        is_available,
    })
}

/// Scan all Cline/Roo Code projects
pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    scan_projects_reporting(&ScanReport::default())
}

/// [`scan_projects`], recording in `report` the extension directories, task
/// indexes and SDK sessions it could not read.
#[allow(clippy::unnecessary_wraps)] // Result kept to match the provider interface.
pub(crate) fn scan_projects_reporting(report: &ScanReport) -> Result<Vec<ClaudeProject>, String> {
    // Each base path may fall back to opening its editor's global state.vscdb
    // (5s busy_timeout when the editor holds a lock), so the base paths are
    // scanned on a bounded pool instead of stacking those waits sequentially.
    let mut projects: Vec<ClaudeProject> = crate::utils::par_map_bounded(
        get_all_base_paths_reporting(report),
        |(base_path, label)| scan_base_path(&base_path, &label, report),
    )
    .into_iter()
    .flatten()
    .collect();

    // Current Cline releases write to the SDK session store instead (#582).
    if let Some(root) = cline_sdk::sessions_dir_reporting(report) {
        projects.extend(cline_sdk::scan_projects_in(&root, report));
    }

    Ok(projects)
}

/// All projects found under one extension base path (empty when it has no
/// readable task history).
fn scan_base_path(base_path: &Path, label: &str, report: &ScanReport) -> Vec<ClaudeProject> {
    let task_history = load_task_history(base_path, report);
    if task_history.is_empty() {
        return Vec::new();
    }

    // Group tasks by cwd
    let mut by_cwd: HashMap<String, Vec<&Value>> = HashMap::new();
    for item in &task_history {
        let cwd = task_project_cwd(item);
        by_cwd.entry(cwd.to_string()).or_default().push(item);
    }

    let mut projects = Vec::new();
    for (cwd, tasks) in &by_cwd {
        let project_name = extension_project_name(cwd);

        let session_count = tasks.len();
        let message_count: usize = tasks
            .iter()
            .filter_map(|t| t.get("tokensIn").and_then(Value::as_u64))
            .count()
            * 2; // rough estimate

        let last_modified = tasks
            .iter()
            .filter_map(|t| t.get("ts").and_then(Value::as_f64))
            .fold(0.0f64, f64::max);

        let last_modified_str = ms_to_iso(last_modified as u64);

        projects.push(ClaudeProject {
            name: project_name,
            path: extension_path(base_path, cwd),
            actual_path: cwd.clone(),
            session_count,
            message_count,
            last_modified: last_modified_str,
            git_info: None,
            provider: Some("cline".to_string()),
            storage_type: Some("json".to_string()),
            custom_directory_label: Some(label.to_string()),
        });
    }
    projects
}

/// Load sessions for a Cline project
pub fn load_sessions(
    project_path: &str,
    _exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    load_sessions_reporting(project_path, &ScanReport::default())
}

/// [`load_sessions`], recording in `report` the task index or SDK manifests
/// it could not read.
pub(crate) fn load_sessions_reporting(
    project_path: &str,
    report: &ScanReport,
) -> Result<Vec<ClaudeSession>, String> {
    if let Some(cwd) = project_path.strip_prefix(cline_sdk::PREFIX) {
        return Ok(cline_sdk::sessions_dir_reporting(report)
            .map(|root| cline_sdk::load_sessions_in(&root, cwd, report))
            .unwrap_or_default());
    }
    let (base_path, target_cwd) = parse_project_path(project_path)?;
    require_discovered_base(&base_path)?;
    let task_history = load_task_history(&base_path, report);

    let project_name = extension_project_name(&target_cwd);

    let mut sessions: Vec<ClaudeSession> = task_history
        .iter()
        .filter(|item| task_project_cwd(item) == target_cwd)
        .filter_map(|item| {
            let id = item.get("id").and_then(Value::as_str)?;
            let ts = item.get("ts").and_then(Value::as_f64).unwrap_or(0.0) as u64;
            let task = item
                .get("task")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let label = task_label(item);

            let ui_messages_path = task_messages_path(&base_path, id);
            // Use token counts as a proxy for message count to avoid reading full JSON
            let tokens_in = item.get("tokensIn").and_then(Value::as_u64).unwrap_or(0);
            let tokens_out = item.get("tokensOut").and_then(Value::as_u64).unwrap_or(0);
            let message_count = if tokens_in > 0 || tokens_out > 0 {
                2
            } else {
                0
            };

            let timestamp = ms_to_iso(ts);

            Some(ClaudeSession {
                session_id: extension_path(&base_path, id),
                actual_session_id: id.to_string(),
                file_path: ui_messages_path.to_string_lossy().to_string(),
                project_name: project_name.clone(),
                message_count,
                first_message_time: timestamp.clone(),
                last_message_time: timestamp.clone(),
                last_modified: timestamp,
                has_tool_use: true, // Cline is heavily tool-based
                has_errors: false,
                summary: session_summary(&task, label),
                is_renamed: false,
                provider: Some("cline".to_string()),
                storage_type: Some("json".to_string()),
                entrypoint: None,
            })
        })
        .collect();

    sessions.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(sessions)
}

/// Load messages from a Cline session, named by its `cline://` id or by the
/// file it lists as its `file_path` (#707).
pub fn load_messages(session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    match resolve_session(session_path)? {
        SessionRef::Sdk { root, id } => cline_sdk::load_messages_in(&root, &id),
        SessionRef::Task { base, id } => load_task_messages(&base, &id),
    }
}

/// Whether `path` is the file a Cline session lists, in a store discovery
/// finds.
pub fn is_session_file(path: &str) -> bool {
    !path.starts_with("cline://") && resolve_listed_file(Path::new(path)).is_ok()
}

/// The project name of a Cline session (the last component of its working
/// directory), named as [`load_messages`] takes it.
pub fn project_name_for_session(session_path: &str) -> Option<String> {
    match resolve_session(session_path).ok()? {
        SessionRef::Sdk { root, id } => cline_sdk::project_name_of(&root, &id),
        SessionRef::Task { base, id } => load_task_history(&base, &ScanReport::default())
            .iter()
            .find(|item| item.get("id").and_then(Value::as_str) == Some(id.as_str()))
            .map(|item| extension_project_name(task_project_cwd(item))),
    }
}

/// A Cline session resolved to its store.
enum SessionRef {
    /// A task of an extension directory discovery found.
    Task { base: PathBuf, id: String },
    /// A session of the SDK session store.
    Sdk { root: PathBuf, id: String },
}

/// Resolve a session's `cline://` id, or the file a session lists, to its
/// store.
fn resolve_session(session_path: &str) -> Result<SessionRef, String> {
    if let Some(id) = session_path.strip_prefix(cline_sdk::PREFIX) {
        let root =
            cline_sdk::sessions_dir().ok_or_else(|| "Cline session store not found".to_string())?;
        return Ok(SessionRef::Sdk {
            root,
            id: id.to_string(),
        });
    }
    if session_path.starts_with("cline://") {
        let (base, id) = parse_session_path(session_path)?;
        require_discovered_base(&base)?;
        return Ok(SessionRef::Task { base, id });
    }
    resolve_listed_file(Path::new(session_path))
}

/// Resolve the file a session lists back to its store: `tasks/<id>/ui_messages.json`
/// of an extension directory, or `<id>/<id>.messages.json` of the SDK session
/// store. The store must be one discovery finds (the same containment as a
/// `cline://` id), so a file shaped like either anywhere else is refused.
fn resolve_listed_file(file: &Path) -> Result<SessionRef, String> {
    let invalid = || format!("Invalid Cline path: {}", file.display());
    let name = file
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(invalid)?;
    let session_dir = file.parent().ok_or_else(invalid)?;
    let id = session_dir
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(invalid)?;
    let store = session_dir.parent().ok_or_else(invalid)?;
    if !file.is_absolute() || file.components().any(|c| c == Component::ParentDir) {
        return Err(invalid());
    }

    if name == "ui_messages.json" && store.file_name().is_some_and(|n| n == "tasks") {
        let base = store.parent().ok_or_else(invalid)?;
        validate_task_id(id)?;
        require_discovered_base(base)?;
        return Ok(SessionRef::Task {
            base: base.to_path_buf(),
            id: id.to_string(),
        });
    }

    if name.strip_suffix(".messages.json") == Some(id) {
        let root = cline_sdk::sessions_dir().ok_or_else(invalid)?;
        let same_store = matches!(
            (store.canonicalize(), root.canonicalize()),
            (Ok(store), Ok(root)) if store == root
        );
        if !same_store {
            return Err(format!(
                "Cline path is outside the session store: {}",
                file.display()
            ));
        }
        return Ok(SessionRef::Sdk {
            root,
            id: id.to_string(),
        });
    }

    Err(invalid())
}

/// Messages of one task of an extension directory.
fn load_task_messages(base_path: &Path, task_id: &str) -> Result<Vec<ClaudeMessage>, String> {
    let ui_path = task_messages_path(base_path, task_id);
    if !ui_path.is_file() {
        return Err(format!("UI messages file not found for task {task_id}"));
    }

    let data =
        fs::read_to_string(&ui_path).map_err(|e| format!("Failed to read ui_messages: {e}"))?;
    let ui_messages: Vec<Value> =
        serde_json::from_str(&data).map_err(|e| format!("Failed to parse ui_messages: {e}"))?;

    let messages = convert_ui_messages(&ui_messages, task_id);

    Ok(messages)
}

/// Search across all Cline tasks
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
    if limit == 0 {
        return Vec::new();
    }
    let query_lower = query.to_lowercase();
    let mut results = Vec::new();

    for (base_path, _) in get_all_base_paths() {
        let task_history = load_task_history(&base_path, &ScanReport::default());

        for item in &task_history {
            let id = match item.get("id").and_then(Value::as_str) {
                Some(id) => id,
                None => continue,
            };

            let cwd = task_project_cwd(item);
            let project_name = extension_project_name(cwd);
            let project_path = extension_path(&base_path, cwd);
            if !filters.matches_project(Some(&project_path), Some(&project_name)) {
                continue;
            }

            let ui_path = task_messages_path(&base_path, id);
            let data = match fs::read_to_string(&ui_path) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let ui_messages: Vec<Value> = match serde_json::from_str(&data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            for mut claude_msg in convert_ui_messages(&ui_messages, id) {
                if !filters.matches_message(&claude_msg) {
                    continue;
                }
                if let Some(c) = &claude_msg.content {
                    if search_json_value_case_insensitive(c, &query_lower) {
                        claude_msg.project_name = Some(project_name.clone());
                        claude_msg.project_path = Some(project_path.clone());
                        results.push(claude_msg);
                        if results.len() >= limit {
                            return results;
                        }
                    }
                }
            }
        }
    }

    if let Some(root) = cline_sdk::sessions_dir() {
        cline_sdk::search_in(&root, &query_lower, limit, filters, &mut results);
    }

    results
}

// ============================================================================
// Private helpers
// ============================================================================

/// Project bucket for extension tasks that record no working directory.
const UNKNOWN_CWD: &str = "unknown";

/// Display name of an extension task's project.
fn extension_project_name(cwd: &str) -> String {
    Path::new(cwd)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}

/// Opaque path for a project (cwd) or a session (task id) in this store.
fn extension_path(base_path: &Path, cwd_or_id: &str) -> String {
    format!("cline://{}:{cwd_or_id}", base_path.to_string_lossy())
}

fn task_messages_path(base_path: &Path, task_id: &str) -> PathBuf {
    base_path
        .join("tasks")
        .join(task_id)
        .join("ui_messages.json")
}

/// Editor user-data roots holding `<editor>/User/globalStorage`.
///
/// `dirs::config_dir()` is exactly that root on each platform:
/// `~/Library/Application Support` (macOS), `~/.config` (Linux) and
/// `%APPDATA%` (Windows). Windows had no branch at all before #582, so no
/// Cline-family install was ever discoverable there.
fn editor_data_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(config) = crate::utils::config_dir() {
        roots.push(config);
    }
    roots
}

/// Cline-family extension directories under one editor data root; a
/// directory on the way that may not be looked into is recorded in `report`.
fn collect_base_paths_under(root: &Path, report: &ScanReport) -> Vec<(PathBuf, String)> {
    // VSCodium ships as `VSCodium` on Windows and most Linux packages, but as
    // `Codium` in some distro builds — both are checked.
    let editors: &[(&str, &str)] = &[
        ("Code", "VS Code"),
        ("Cursor", "Cursor"),
        ("Code - Insiders", "VS Code Insiders"),
        ("VSCodium", "VSCodium"),
        ("Codium", "VSCodium"),
        // The PearAI editor (`pearai-app` `product.json` `nameShort`).
        ("PearAI", "PearAI"),
    ];

    let mut paths = Vec::new();
    for (editor_dir, editor_label) in editors {
        let editor_root = root.join(editor_dir);
        // `is_dir()` follows symlinks, so every component below `root` is
        // checked: a symlinked `Code`, `User` or `globalStorage` would
        // otherwise walk wherever it points.
        let global_storage = editor_root.join("User").join("globalStorage");
        if [&editor_root, &editor_root.join("User"), &global_storage]
            .iter()
            .any(|dir| is_symlink(dir))
        {
            continue;
        }
        if !report.is_dir(&global_storage) {
            continue;
        }
        for (ext_id, ext_name) in EXTENSIONS {
            let ext_path = global_storage.join(ext_id);
            if report.is_dir(&ext_path) && !is_symlink(&ext_path) {
                paths.push((ext_path, format!("{ext_name} ({editor_label})")));
            }
        }
    }
    paths
}

fn get_all_base_paths() -> Vec<(PathBuf, String)> {
    get_all_base_paths_reporting(&ScanReport::default())
}

/// [`get_all_base_paths`], recording in `report` what it could not look at.
fn get_all_base_paths_reporting(report: &ScanReport) -> Vec<(PathBuf, String)> {
    let mut paths: Vec<(PathBuf, String)> = editor_data_roots()
        .iter()
        .flat_map(|root| collect_base_paths_under(root, report))
        .collect();
    if let Some(base) = kilo_cli_base(report) {
        paths.push((base, "Kilo Code (CLI)".to_string()));
    }
    paths
}

/// A task's working directory. Cline names this `cwdOnTaskInitialization`; the
/// Roo Code / Kilo Code lineage renames it to `workspace`. Try both so projects
/// group correctly across the whole family.
fn task_cwd(item: &Value) -> Option<&str> {
    item.get("cwdOnTaskInitialization")
        .and_then(Value::as_str)
        .or_else(|| item.get("workspace").and_then(Value::as_str))
}

fn task_project_cwd(item: &Value) -> &str {
    task_cwd(item).unwrap_or(UNKNOWN_CWD)
}

/// A short model/profile label for a task, used as the session summary when the
/// task text is empty. Cline stores `modelId`; the Roo/Kilo lineage uses
/// `apiConfigName` (or `mode`) instead.
fn task_label(item: &Value) -> Option<String> {
    ["modelId", "apiConfigName", "mode"]
        .iter()
        .find_map(|k| item.get(*k).and_then(Value::as_str))
        .map(str::to_string)
}

/// Session summary: the task text (truncated), or a model/profile label when the
/// task is empty.
fn session_summary(task: &str, label: Option<String>) -> Option<String> {
    if task.is_empty() {
        label
    } else {
        Some(truncate_chars(task, 100, "..."))
    }
}

/// Truncate `text` to at most `max_chars` **characters** (not bytes), appending
/// `suffix` only when truncation occurs. Char-safe: never panics by splitting a
/// multibyte UTF-8 character — Cline/Roo summaries and tool results are often CJK.
fn truncate_chars(text: &str, max_chars: usize, suffix: &str) -> String {
    match text.char_indices().nth(max_chars) {
        Some((idx, _)) => format!("{}{}", &text[..idx], suffix),
        None => text.to_string(),
    }
}

/// The task index of an extension directory. An index that is there but
/// cannot be read is recorded in `report`; one that does not parse is not
/// an index.
fn load_task_history(base_path: &Path, report: &ScanReport) -> Vec<Value> {
    let history = read_task_history(base_path, report);
    // Kilo's extension copies a legacy task into its own store and keeps the
    // original; the copy is the one Kilo shows, so it is read there
    // (`providers::kilo`), not twice.
    if base_path.ends_with("kilocode.kilo-code") && !history.is_empty() {
        let migrated = crate::providers::kilo::migrated_task_ids();
        if !migrated.is_empty() {
            return history
                .into_iter()
                .filter(|item| {
                    item.get("id")
                        .and_then(Value::as_str)
                        .is_none_or(|id| !migrated.contains(id))
                })
                .collect();
        }
    }
    history
}

/// Every task of an extension directory: the entries of its index, and the
/// tasks on disk that the index does not list (a stale or never-written
/// index; Roo keeps `tasks/_index.json` only as a debounced cache of the
/// per-task `history_item.json` files, `RooCodeInc/Roo-Code@b867ec9`
/// `src/core/task-persistence/TaskHistoryStore.ts`). The index stays the
/// source of an entry it lists, so no task is listed twice.
fn read_task_history(base_path: &Path, report: &ScanReport) -> Vec<Value> {
    let mut history = read_task_index(base_path, report);
    let indexed: HashSet<String> = history
        .iter()
        .filter_map(|item| item.get("id").and_then(Value::as_str).map(str::to_string))
        .collect();
    history.extend(unindexed_task_items(base_path, &indexed, report));
    history
}

/// Task items for the `tasks/<id>/` directories whose id is not in `indexed`.
fn unindexed_task_items(
    base_path: &Path,
    indexed: &HashSet<String>,
    report: &ScanReport,
) -> Vec<Value> {
    let tasks_dir = base_path.join("tasks");
    if is_symlink(&tasks_dir) {
        return Vec::new();
    }
    let mut items: Vec<Value> = report
        .read_dir(&tasks_dir)
        .into_iter()
        // `DirEntry::file_type` does not follow symlinks.
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter_map(|entry| {
            let id = entry.file_name().to_str()?.to_string();
            if indexed.contains(&id) || validate_task_id(&id).is_err() {
                return None;
            }
            task_dir_item(&entry.path(), &id, report)
        })
        .collect();
    items.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    items
}

/// The task item of one task directory, or `None` when it holds no
/// `ui_messages.json`: Roo's own `history_item.json` when it is there, else an
/// item read from the messages (and Cline's `task_metadata.json`).
fn task_dir_item(dir: &Path, id: &str, report: &ScanReport) -> Option<Value> {
    let ui_path = dir.join("ui_messages.json");
    if !report.is_file(&ui_path) {
        return None;
    }
    let history_item = dir.join("history_item.json");
    if report.is_file(&history_item) {
        let parsed = report
            .read_to_string(&history_item)
            .and_then(|data| serde_json::from_str::<Value>(&data).ok());
        if let Some(Value::Object(mut item)) = parsed {
            // The directory name is what the messages are loaded by.
            item.insert("id".to_string(), Value::String(id.to_string()));
            return Some(Value::Object(item));
        }
    }
    let messages: Vec<Value> = serde_json::from_str(&report.read_to_string(&ui_path)?).ok()?;
    let metadata = dir.join("task_metadata.json");
    let model_id = report
        .is_file(&metadata)
        .then(|| report.read_to_string(&metadata))
        .flatten()
        .and_then(|data| serde_json::from_str::<Value>(&data).ok())
        .and_then(|meta| {
            meta.get("model_usage")?
                .as_array()?
                .iter()
                .rev()
                .find_map(|usage| usage.get("model_id")?.as_str().map(str::to_string))
        });
    Some(item_from_messages(id, &messages, model_id))
}

/// A task item rebuilt from the task's own messages, for a task with no
/// `history_item.json`: the start time, the prompt, the token totals of its
/// requests and, for Cline, the working directory its environment details
/// name.
fn item_from_messages(id: &str, messages: &[Value], model_id: Option<String>) -> Value {
    let ts = messages
        .iter()
        .find_map(|msg| msg.get("ts").and_then(Value::as_f64))
        .or_else(|| id.parse::<f64>().ok())
        .unwrap_or(0.0) as u64;
    let prompt = prompt_index(messages)
        .and_then(|index| messages[index].get("text")?.as_str())
        .unwrap_or("");

    let (mut tokens_in, mut tokens_out) = (0u64, 0u64);
    let mut cwd = None;
    for msg in messages {
        if msg.get("say").and_then(Value::as_str) != Some("api_req_started") {
            continue;
        }
        let Some(request) = msg
            .get("text")
            .and_then(Value::as_str)
            .and_then(|text| serde_json::from_str::<Value>(text).ok())
        else {
            continue;
        };
        tokens_in += request.get("tokensIn").and_then(Value::as_u64).unwrap_or(0);
        tokens_out += request
            .get("tokensOut")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if cwd.is_none() {
            cwd = request
                .get("request")
                .and_then(Value::as_str)
                .and_then(request_cwd)
                .map(str::to_string);
        }
    }

    let mut item = serde_json::json!({
        "id": id,
        "ts": ts,
        "task": prompt,
        "tokensIn": tokens_in,
        "tokensOut": tokens_out,
    });
    if let Some(cwd) = cwd {
        item["cwdOnTaskInitialization"] = Value::String(cwd);
    }
    if let Some(model_id) = model_id {
        item["modelId"] = Value::String(model_id);
    }
    item
}

/// The working directory Cline's environment details give a request:
/// `# Current Working Directory (<cwd>) Files` (`cline/cline@e1bdeef`
/// `apps/vscode/src/core/task/index.ts`; a multi-root workspace names a
/// `Primary:` folder there instead, which is not a path).
fn request_cwd(request: &str) -> Option<&str> {
    const MARKER: &str = "# Current Working Directory (";
    let rest = &request[request.find(MARKER)? + MARKER.len()..];
    let cwd = &rest[..rest.find(") Files")?];
    (!cwd.is_empty() && !cwd.starts_with("Primary:")).then_some(cwd)
}

fn read_task_index(base_path: &Path, report: &ScanReport) -> Vec<Value> {
    // Cline format: state/taskHistory.json
    let cline_path = base_path.join("state/taskHistory.json");
    if report.is_file(&cline_path) {
        if let Some(data) = report.read_to_string(&cline_path) {
            if let Ok(items) = serde_json::from_str::<Vec<Value>>(&data) {
                if !items.is_empty() {
                    return items;
                }
            }
        }
    }

    // Roo Code format: tasks/_index.json (entries array)
    let roo_index = base_path.join("tasks/_index.json");
    if report.is_file(&roo_index) {
        if let Some(data) = report.read_to_string(&roo_index) {
            if let Ok(index) = serde_json::from_str::<Value>(&data) {
                if let Some(entries) = index.get("entries").and_then(Value::as_array) {
                    if !entries.is_empty() {
                        return entries.clone();
                    }
                }
            }
        }
    }

    // Kilo's legacy CLI keeps the extension's globalState in a JSON file.
    if is_kilo_cli_base(base_path) {
        let memento = base_path.join("global-state.json");
        return report
            .read_to_string(&memento)
            .and_then(|data| serde_json::from_str::<Value>(&data).ok())
            .and_then(|state| state.get("taskHistory")?.as_array().cloned())
            .unwrap_or_default();
    }

    // Kilo Code (and modern Cline/Roo before they flush to disk) keep the task
    // index only in VS Code globalState. Fall back to reading it from the global
    // state.vscdb that sits beside the extension dir.
    load_task_history_from_global_state(base_path, report).unwrap_or_default()
}

/// Read the task-history index from VS Code's globalState. VS Code persists each
/// extension's globalState as a single `ItemTable` row in the GLOBAL `state.vscdb`
/// (the sibling of the per-extension dir), keyed by the extension id, whose value
/// is a flat JSON object; the Cline/Roo/Kilo lineage stores its task list under
/// `taskHistory`. This is the only place Kilo Code's history can be found.
fn load_task_history_from_global_state(
    base_path: &Path,
    report: &ScanReport,
) -> Option<Vec<Value>> {
    let ext_id = base_path.file_name()?.to_str()?;
    let db_path = base_path.parent()?.join("state.vscdb");
    if !report.is_file(&db_path) {
        return None;
    }

    let conn = report.db(
        &db_path,
        Connection::open_with_flags(&db_path, OpenFlags::SQLITE_OPEN_READ_ONLY),
    )?;
    report.db(
        &db_path,
        conn.busy_timeout(std::time::Duration::from_secs(5)),
    )?;
    let raw: String = report.db(
        &db_path,
        conn.query_row(
            // VS Code keys globalState by the extension id as published
            // (`PearAI.pearai-roo-cline`); the storage dir is lowercased.
            "SELECT value FROM ItemTable WHERE key = ?1 COLLATE NOCASE",
            [ext_id],
            |row| {
                // VS Code stores the value as TEXT; tolerate BLOB-stored bytes too.
                row.get::<_, String>(0).or_else(|_| {
                    row.get::<_, Vec<u8>>(0)
                        .map(|b| String::from_utf8_lossy(&b).into_owned())
                })
            },
        ),
    )?;

    let value: Value = serde_json::from_str(&raw).ok()?;
    let history = value.get("taskHistory")?.as_array()?;
    if history.is_empty() {
        None
    } else {
        Some(history.clone())
    }
}

/// Split `cline://<extension dir>:<tail>` into the extension directory and
/// the tail (a cwd for projects, a task id for sessions).
///
/// The separator is the `:` right after the extension id that ends the base,
/// not the first colon: on Windows both the base (`C:\Users\…`) and a cwd
/// (`D:\work`) carry a drive-letter colon, and splitting at the first one cut
/// the base down to `C` (#582).
fn parse_base_and_tail(path: &str) -> Result<(PathBuf, String), String> {
    let rest = path.strip_prefix("cline://").unwrap_or(path);
    // The separator is the first colon that is not a Windows drive colon
    // (`C:` at the very start). Bases built by `collect_base_paths_under`
    // always end in an extension id there; only then is it trusted, so an
    // extension id that merely appears later, inside the tail, is ignored.
    let is_drive_colon = |i: usize| i == 1 && rest.as_bytes()[0].is_ascii_alphabetic();
    let at_extension = rest
        .match_indices(':')
        .map(|(i, _)| i)
        .find(|&i| !is_drive_colon(i))
        .filter(|&i| {
            EXTENSIONS
                .iter()
                .any(|(ext_id, _)| rest[..i].ends_with(ext_id))
                || is_kilo_cli_base(Path::new(&rest[..i]))
        })
        .map(|i| (PathBuf::from(&rest[..i]), rest[i + 1..].to_string()));
    at_extension
        .or_else(|| {
            rest.split_once(':')
                .map(|(base, tail)| (PathBuf::from(base), tail.to_string()))
        })
        .ok_or_else(|| format!("Invalid Cline path: {path}"))
}

/// A base parsed from a caller-supplied id must be one of the extension
/// directories discovery found (Cline, Roo Code and Kilo Code under every
/// supported editor); compared canonically so a symlink cannot stand in.
fn require_discovered_base(base_path: &Path) -> Result<(), String> {
    let outside = || {
        format!(
            "Cline path is outside the discovered extension directories: {}",
            base_path.display()
        )
    };
    let canonical = base_path.canonicalize().map_err(|_| outside())?;
    let discovered = get_all_base_paths()
        .iter()
        .filter_map(|(base, _)| base.canonicalize().ok())
        .any(|base| base == canonical);
    if discovered {
        Ok(())
    } else {
        Err(outside())
    }
}

fn parse_project_path(project_path: &str) -> Result<(PathBuf, String), String> {
    parse_base_and_tail(project_path)
}

fn parse_session_path(session_path: &str) -> Result<(PathBuf, String), String> {
    let (base_path, task_id) = parse_base_and_tail(session_path)?;
    if !base_path.is_absolute() {
        return Err("Cline base path must be absolute".to_string());
    }

    validate_task_id(&task_id)?;
    Ok((base_path, task_id))
}

/// Reject path traversal in a task id.
fn validate_task_id(task_id: &str) -> Result<(), String> {
    if task_id.contains("..") || task_id.contains('/') || task_id.contains('\\') {
        return Err(format!("Invalid task ID: {task_id}"));
    }
    Ok(())
}

/// Convert a `ClineMessage` to `ClaudeMessage`
fn convert_cline_message(
    msg: &Value,
    session_id: &str,
    counter: &mut u64,
    open_call: Option<&str>,
) -> Option<ClaudeMessage> {
    let msg_type = msg.get("type").and_then(Value::as_str)?;
    let ts = msg.get("ts").and_then(Value::as_f64).unwrap_or(0.0) as u64;
    let timestamp = ms_to_iso(ts);
    let text = msg.get("text").and_then(Value::as_str).unwrap_or("");

    *counter += 1;
    let uuid = format!("cline-{counter}");

    match msg_type {
        "say" => {
            let say = msg.get("say").and_then(Value::as_str).unwrap_or("");
            convert_say_message(say, text, msg, &uuid, session_id, &timestamp, open_call)
        }
        "ask" => {
            let ask = msg.get("ask").and_then(Value::as_str).unwrap_or("");
            convert_ask_message(ask, text, &uuid, session_id, &timestamp)
        }
        _ => None,
    }
}

/// Convert a task's `ui_messages.json` records, in order.
///
/// The user's first prompt is `say:"task"` in Cline (`cline/cline@e1bdeef`
/// `apps/vscode/src/core/task/index.ts`: `this.say("task", task, images,
/// files)`) but `say:"text"` in classic Roo Code, which has no `task` say
/// (`RooCodeInc/Roo-Code@b867ec9` `src/core/task/Task.ts` `startTask`:
/// `this.say("text", task, images)`); that record is a user message.
fn convert_ui_messages(ui_messages: &[Value], session_id: &str) -> Vec<ClaudeMessage> {
    let prompt = prompt_index(ui_messages);
    let mut counter = 0u64;
    let mut messages = Vec::new();
    // The id of the tool call whose result record has not come yet.
    let mut open_call: Option<String> = None;
    for (index, msg) in ui_messages.iter().enumerate() {
        let say = msg.get("say").and_then(Value::as_str).unwrap_or("");
        let converted = convert_cline_message(msg, session_id, &mut counter, open_call.as_deref());
        match say {
            "use_mcp_server" | "browser_action_launch" | "browser_action" | "use_subagents" => {
                open_call = converted
                    .as_ref()
                    .and_then(|m| m.content.as_ref()?.get(0)?.get("id")?.as_str())
                    .map(str::to_string);
            }
            "mcp_server_response" | "browser_action_result" | "subagent" => open_call = None,
            _ => {}
        }
        if let Some(mut converted) = converted {
            if prompt == Some(index) {
                converted.message_type = "user".to_string();
                converted.role = Some("user".to_string());
            }
            messages.push(converted);
        }
    }
    messages
}

/// Index of the record that holds the user's first prompt: a `say:"task"`, or
/// else the first `say:"text"` before any `api_req_started` (Roo says the
/// prompt before it makes its first request; every later `text` is the
/// model's and comes after one).
fn prompt_index(ui_messages: &[Value]) -> Option<usize> {
    ui_messages
        .iter()
        .position(|msg| {
            msg.get("type").and_then(Value::as_str) == Some("say")
                && matches!(
                    msg.get("say").and_then(Value::as_str),
                    Some("task" | "text")
                )
        })
        .filter(|&index| {
            // A `text` that follows a request is the model's, not the prompt.
            !ui_messages[..index]
                .iter()
                .any(|msg| msg.get("say").and_then(Value::as_str) == Some("api_req_started"))
        })
}

fn convert_say_message(
    say: &str,
    text: &str,
    msg: &Value,
    uuid: &str,
    session_id: &str,
    timestamp: &str,
    open_call: Option<&str>,
) -> Option<ClaudeMessage> {
    match say {
        "text" | "completion_result" => {
            if text.is_empty() {
                return None;
            }
            let content = serde_json::json!([{"type": "text", "text": text}]);
            Some(build_provider_message(
                "cline",
                uuid.to_string(),
                session_id,
                timestamp.to_string(),
                "assistant",
                Some("assistant"),
                Some(content),
                None,
            ))
        }
        "reasoning" => {
            let reasoning_text = msg.get("reasoning").and_then(Value::as_str).unwrap_or(text);
            if reasoning_text.is_empty() {
                return None;
            }
            let content = serde_json::json!([{"type": "thinking", "thinking": reasoning_text}]);
            Some(build_provider_message(
                "cline",
                uuid.to_string(),
                session_id,
                timestamp.to_string(),
                "assistant",
                Some("assistant"),
                Some(content),
                None,
            ))
        }
        "tool" => {
            let tool_data: Value = serde_json::from_str(text).unwrap_or(Value::Null);
            let tool_name = tool_data
                .get("tool")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let mapped = map_cline_tool_name(tool_name);
            let path = tool_data.get("path").and_then(Value::as_str).unwrap_or("");

            let mut blocks = vec![serde_json::json!({
                "type": "tool_use",
                "id": format!("cline_tool_{uuid}"),
                "name": mapped,
                "input": {"path": path}
            })];

            // If there's content (result), add tool_result
            if let Some(result_text) = tool_data.get("content").and_then(Value::as_str) {
                blocks.push(serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": format!("cline_tool_{uuid}"),
                    "content": truncate_chars(result_text, 2000, "...(truncated)")
                }));
            }

            // If there's a diff, add it
            if let Some(diff) = tool_data.get("diff").and_then(Value::as_str) {
                blocks.push(serde_json::json!({
                    "type": "text",
                    "text": format!("```diff\n{diff}\n```")
                }));
            }

            Some(build_provider_message(
                "cline",
                uuid.to_string(),
                session_id,
                timestamp.to_string(),
                "assistant",
                Some("assistant"),
                Some(Value::Array(blocks)),
                None,
            ))
        }
        "command" => {
            if text.is_empty() {
                return None;
            }
            let content = serde_json::json!([{
                "type": "tool_use",
                "id": format!("cline_cmd_{uuid}"),
                "name": "Bash",
                "input": {"command": text}
            }]);
            Some(build_provider_message(
                "cline",
                uuid.to_string(),
                session_id,
                timestamp.to_string(),
                "assistant",
                Some("assistant"),
                Some(content),
                None,
            ))
        }
        "command_output" => {
            if text.is_empty() {
                return None;
            }
            // Render as a text block with terminal styling (the command tool_use
            // is in the preceding "command" message — they are paired by ordering)
            let content = serde_json::json!([{
                "type": "text",
                "text": format!("```\n{text}\n```")
            }]);
            Some(build_provider_message(
                "cline",
                uuid.to_string(),
                session_id,
                timestamp.to_string(),
                "assistant",
                Some("assistant"),
                Some(content),
                None,
            ))
        }
        "error" | "diff_error" | "condense_context_error" | "command_permission_denied" => {
            system_notice(text, "error", "error", uuid, session_id, timestamp)
        }
        // Roo / Cline ignore-file blocks: the text is the blocked path.
        "rooignore_error" | "clineignore_error" => {
            if text.is_empty() {
                return None;
            }
            let file = if say == "rooignore_error" {
                ".rooignore"
            } else {
                ".clineignore"
            };
            system_notice(
                &format!("Blocked by {file}: {text}"),
                "error",
                "error",
                uuid,
                session_id,
                timestamp,
            )
        }
        "mcp_notification" => system_notice(text, "notice", "info", uuid, session_id, timestamp),
        // The user's own prompt: Cline writes it as `say:"task"`; classic Roo
        // writes it as the first `say:"text"` (see `prompt_index`, which makes
        // `convert_ui_messages` a user message).
        "task" | "user_feedback" | "user_feedback_diff" => {
            if text.is_empty() {
                return None;
            }
            let content = serde_json::json!([{"type": "text", "text": text}]);
            Some(build_provider_message(
                "cline",
                uuid.to_string(),
                session_id,
                timestamp.to_string(),
                "user",
                Some("user"),
                Some(content),
                None,
            ))
        }
        // Cline's request to use an MCP tool or resource, and the response
        // that follows it (Roo records the request as an ask, so its
        // response stands alone as pretty-printed text).
        "use_mcp_server" => convert_mcp_request(text, uuid, session_id, timestamp),
        "mcp_server_response" => {
            if text.is_empty() {
                return None;
            }
            if open_call.is_some() {
                return call_result(
                    open_call,
                    vec![text_block(text)],
                    uuid,
                    session_id,
                    timestamp,
                );
            }
            Some(assistant_text(
                format!("```\n{text}\n```"),
                uuid,
                session_id,
                timestamp,
            ))
        }
        // Cline's browser tool: the launch URL, each action, and the page it
        // leaves (logs, URL, screenshot).
        "browser_action_launch" => {
            if text.is_empty() {
                return None;
            }
            let input = serde_json::json!({"action": "launch", "url": text});
            Some(tool_call(
                "browser_action",
                input,
                uuid,
                session_id,
                timestamp,
            ))
        }
        "browser_action" => {
            if text.is_empty() {
                return None;
            }
            Some(tool_call(
                "browser_action",
                json_input(text),
                uuid,
                session_id,
                timestamp,
            ))
        }
        "browser_action_result" => call_result(
            open_call,
            browser_result_blocks(text),
            uuid,
            session_id,
            timestamp,
        ),
        // Cline's sub-agents: the prompts it started them with, and their
        // final status.
        "use_subagents" => {
            if text.is_empty() {
                return None;
            }
            Some(tool_call(
                "use_subagents",
                json_input(text),
                uuid,
                session_id,
                timestamp,
            ))
        }
        "subagent" => {
            if text.is_empty() {
                return None;
            }
            let blocks = vec![text_block(&subagent_summary(text))];
            call_result(open_call, blocks, uuid, session_id, timestamp)
        }
        "image" => convert_image(text, uuid, session_id, timestamp),
        // Roo `CodebaseSearchTool`: JSON `{tool, content: {query, results:
        // [{filePath, startLine, endLine, codeChunk, ..}]}}`.
        "codebase_search_result" => convert_codebase_search(text, uuid, session_id, timestamp),
        // The result a finished subtask hands back to its parent task.
        "subtask_result" => {
            if text.is_empty() {
                return None;
            }
            Some(assistant_text(
                format!("Subtask result:\n\n{text}"),
                uuid,
                session_id,
                timestamp,
            ))
        }
        // Skip internal/metadata messages: request bookkeeping and retry
        // timers, checkpoints, context condensing/truncation markers, MCP
        // request starts and payloads, and JSON progress records.
        "api_req_started"
        | "api_req_finished"
        | "api_req_retried"
        | "api_req_retry_delayed"
        | "api_req_rate_limit_wait"
        | "api_req_deleted"
        | "deleted_api_reqs"
        | "shell_integration_warning"
        | "shell_integration_warning_with_suggestion"
        | "checkpoint_created"
        | "checkpoint_saved"
        | "condense_context"
        | "sliding_window_truncation"
        | "compaction"
        | "mcp_server_request_started"
        | "too_many_tools_warning"
        | "user_edit_todos"
        | "subagent_usage"
        | "cloud_provisioning"
        | "load_mcp_documentation"
        | "info"
        | "task_progress"
        | "hook_status"
        | "hook_output_stream"
        | "conditional_rules_applied" => None,
        _ => {
            // For any other say type with text, show as assistant text
            if text.is_empty() {
                return None;
            }
            Some(assistant_text(
                text.to_string(),
                uuid,
                session_id,
                timestamp,
            ))
        }
    }
}

/// An assistant message of one text block.
fn assistant_text(text: String, uuid: &str, session_id: &str, timestamp: &str) -> ClaudeMessage {
    build_provider_message(
        "cline",
        uuid.to_string(),
        session_id,
        timestamp.to_string(),
        "assistant",
        Some("assistant"),
        Some(serde_json::json!([{"type": "text", "text": text}])),
        None,
    )
}

/// An assistant message that calls tool `name`; its `tool_use` id is
/// `cline_tool_<uuid>`, which the result record that follows is paired by.
fn tool_call(
    name: &str,
    input: Value,
    uuid: &str,
    session_id: &str,
    timestamp: &str,
) -> ClaudeMessage {
    build_provider_message(
        "cline",
        uuid.to_string(),
        session_id,
        timestamp.to_string(),
        "assistant",
        Some("assistant"),
        Some(serde_json::json!([{
            "type": "tool_use",
            "id": format!("cline_tool_{uuid}"),
            "name": name,
            "input": input
        }])),
        None,
    )
}

/// The result record of the open call `call_id`: a user message holding a
/// `tool_result`. A result with no open call (Cline records the request as an
/// ask, which is not read, when it was not auto-approved) is an assistant
/// message of the blocks themselves. `None` when there is nothing to show.
fn call_result(
    call_id: Option<&str>,
    blocks: Vec<Value>,
    uuid: &str,
    session_id: &str,
    timestamp: &str,
) -> Option<ClaudeMessage> {
    if blocks.is_empty() {
        return None;
    }
    let (message_type, content) = match call_id {
        Some(id) => {
            let result = match blocks.as_slice() {
                [only] if only["type"] == "text" => only["text"].clone(),
                _ => Value::Array(blocks),
            };
            (
                "user",
                serde_json::json!([{
                    "type": "tool_result",
                    "tool_use_id": id,
                    "content": result
                }]),
            )
        }
        None => ("assistant", Value::Array(blocks)),
    };
    Some(build_provider_message(
        "cline",
        uuid.to_string(),
        session_id,
        timestamp.to_string(),
        message_type,
        Some(message_type),
        Some(content),
        None,
    ))
}

fn text_block(text: &str) -> Value {
    serde_json::json!({"type": "text", "text": text})
}

/// The input of a tool call recorded as a JSON object; the raw text when it
/// is anything else.
fn json_input(text: &str) -> Value {
    serde_json::from_str::<Value>(text)
        .ok()
        .filter(Value::is_object)
        .unwrap_or_else(|| serde_json::json!({"text": text}))
}

/// An image block for a `data:image/…;base64,…` URL.
fn image_block(data_url: &str) -> Option<Value> {
    let (media_type, data) = data_url.strip_prefix("data:")?.split_once(";base64,")?;
    if !media_type.starts_with("image/") || data.is_empty() {
        return None;
    }
    Some(serde_json::json!({
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data}
    }))
}

/// Cline's `say:"use_mcp_server"`: JSON `{serverName, type, toolName?,
/// arguments?, uri?}` (`ClineAskUseMcpServer`, `cline/cline@e1bdeef`
/// `apps/vscode/src/shared/ExtensionMessage.ts`; `arguments` is a JSON
/// string). A tool call is named `mcp__<server>__<tool>`, as MCP tools are
/// in Claude's transcripts; a resource read is `access_mcp_resource`.
fn convert_mcp_request(
    text: &str,
    uuid: &str,
    session_id: &str,
    timestamp: &str,
) -> Option<ClaudeMessage> {
    let Ok(Value::Object(request)) = serde_json::from_str::<Value>(text) else {
        return (!text.is_empty())
            .then(|| assistant_text(text.to_string(), uuid, session_id, timestamp));
    };
    let field = |key: &str| request.get(key).and_then(Value::as_str).unwrap_or("");
    if field("type") == "access_mcp_resource" {
        let input = serde_json::json!({"server_name": field("serverName"), "uri": field("uri")});
        return Some(tool_call(
            "access_mcp_resource",
            input,
            uuid,
            session_id,
            timestamp,
        ));
    }
    let input = match request.get("arguments") {
        Some(Value::String(args)) if args.is_empty() => serde_json::json!({}),
        Some(Value::String(args)) => serde_json::from_str::<Value>(args)
            .unwrap_or_else(|_| serde_json::json!({"arguments": args})),
        Some(other) if !other.is_null() => other.clone(),
        _ => serde_json::json!({}),
    };
    let name = format!("mcp__{}__{}", field("serverName"), field("toolName"));
    Some(tool_call(&name, input, uuid, session_id, timestamp))
}

/// Cline's `say:"browser_action_result"`: JSON `BrowserActionResult
/// {screenshot?, logs?, currentUrl?, currentMousePosition?}`, the screenshot
/// a `data:image/…;base64,…` URL. Empty (a close) when there is nothing.
fn browser_result_blocks(text: &str) -> Vec<Value> {
    let Ok(Value::Object(result)) = serde_json::from_str::<Value>(text) else {
        return if text.is_empty() {
            Vec::new()
        } else {
            vec![text_block(text)]
        };
    };
    let field = |key: &str| {
        result
            .get(key)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
    };
    let mut lines = Vec::new();
    if let Some(logs) = field("logs") {
        lines.push(logs.to_string());
    }
    if let Some(url) = field("currentUrl") {
        lines.push(format!("URL: {url}"));
    }
    let mut blocks = Vec::new();
    if !lines.is_empty() {
        blocks.push(text_block(&lines.join("\n")));
    }
    blocks.extend(field("screenshot").and_then(image_block));
    blocks
}

/// Cline's `say:"subagent"`: JSON `ClineSaySubagentStatus {status, total,
/// completed, items: [{prompt, status, result?, error?, ..}], ..}`, the final
/// state of a `use_subagents` call.
fn subagent_summary(text: &str) -> String {
    let Ok(status) = serde_json::from_str::<Value>(text) else {
        return text.to_string();
    };
    let count = |key: &str| status.get(key).and_then(Value::as_u64).unwrap_or(0);
    let mut lines = vec![format!(
        "Subagents {}: {} of {} done",
        status
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unknown"),
        count("completed"),
        count("total")
    )];
    for (index, item) in status
        .get("items")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let field = |key: &str| item.get(key).and_then(Value::as_str).unwrap_or("");
        lines.push(format!(
            "{}. [{}] {}",
            index + 1,
            field("status"),
            field("prompt")
        ));
        for detail in [field("result"), field("error")] {
            if !detail.is_empty() {
                lines.push(truncate_chars(detail, 2000, "...(truncated)"));
            }
        }
    }
    lines.join("\n")
}

/// Roo's `say:"image"` (`generate_image`): JSON `{imageUri, imagePath}`. The
/// `imageUri` is a webview URI that cannot be opened here, so the file's path
/// is shown; a `data:image/…` URL, if one is recorded, is shown as the image.
fn convert_image(
    text: &str,
    uuid: &str,
    session_id: &str,
    timestamp: &str,
) -> Option<ClaudeMessage> {
    if let Some(image) = image_block(text) {
        return call_result(None, vec![image], uuid, session_id, timestamp);
    }
    let path = serde_json::from_str::<Value>(text)
        .ok()?
        .get("imagePath")?
        .as_str()?
        .to_string();
    system_notice(
        &format!("Generated image: {path}"),
        "notice",
        "info",
        uuid,
        session_id,
        timestamp,
    )
}

/// A system notice of one text block; `None` for empty text.
fn system_notice(
    text: &str,
    subtype: &str,
    level: &str,
    uuid: &str,
    session_id: &str,
    timestamp: &str,
) -> Option<ClaudeMessage> {
    if text.is_empty() {
        return None;
    }
    let mut notice = build_provider_message(
        "cline",
        uuid.to_string(),
        session_id,
        timestamp.to_string(),
        "system",
        None,
        Some(serde_json::json!([{"type": "text", "text": text}])),
        None,
    );
    notice.subtype = Some(subtype.to_string());
    notice.level = Some(level.to_string());
    Some(notice)
}

/// A codebase search as a tool call with its results; the raw text when it
/// is not the JSON Roo writes.
fn convert_codebase_search(
    text: &str,
    uuid: &str,
    session_id: &str,
    timestamp: &str,
) -> Option<ClaudeMessage> {
    if text.is_empty() {
        return None;
    }
    let data: Value = serde_json::from_str(text).unwrap_or(Value::Null);
    let content = &data["content"];
    let Some(results) = content.get("results").and_then(Value::as_array) else {
        return Some(assistant_text(
            text.to_string(),
            uuid,
            session_id,
            timestamp,
        ));
    };
    let query = content.get("query").and_then(Value::as_str).unwrap_or("");
    let listing = results
        .iter()
        .map(|hit| {
            let field = |key: &str| hit.get(key).and_then(Value::as_str).unwrap_or("");
            let line = |key: &str| {
                hit.get(key)
                    .and_then(Value::as_u64)
                    .map_or_else(String::new, |n| n.to_string())
            };
            format!(
                "{}:{}-{}\n{}",
                field("filePath"),
                line("startLine"),
                line("endLine"),
                field("codeChunk")
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let tool_id = format!("cline_tool_{uuid}");
    let blocks = serde_json::json!([
        {
            "type": "tool_use",
            "id": tool_id,
            "name": "codebaseSearch",
            "input": {"query": query}
        },
        {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": truncate_chars(&listing, 2000, "...(truncated)")
        }
    ]);
    Some(build_provider_message(
        "cline",
        uuid.to_string(),
        session_id,
        timestamp.to_string(),
        "assistant",
        Some("assistant"),
        Some(blocks),
        None,
    ))
}

/// Ask types in which the model speaks to the user: a clarifying question
/// (`followup`) or its reply in plan/act mode. The user's answer is recorded
/// separately, as `say:"user_feedback"` (or, when an offered option was
/// picked, as `selected` inside the ask's own JSON).
///
/// Upstream writers, pinned: Cline `AskFollowupQuestionToolHandler` /
/// `PlanModeRespondHandler` (`cline/cline@e1bdeef`,
/// `apps/vscode/src/core/task/tools/handlers/`) write
/// `ask("followup", JSON {question, options})` and
/// `ask("plan_mode_respond", JSON {response, options})`; Roo Code
/// `AskFollowupQuestionTool` (`RooCodeInc/Roo-Code@b867ec9`,
/// `src/core/tools/`) writes `ask("followup", JSON {question, suggest:
/// [{answer, mode}]})` and, while streaming, the bare question as plain text.
fn convert_ask_message(
    ask: &str,
    text: &str,
    uuid: &str,
    session_id: &str,
    timestamp: &str,
) -> Option<ClaudeMessage> {
    match ask {
        "followup" | "act_mode_respond" | "plan_mode_respond" => {
            let reply = render_ask_reply(text)?;
            Some(assistant_text(reply, uuid, session_id, timestamp))
        }
        // Skip permission/confirmation prompts
        _ => None,
    }
}

/// The text of a model-side ask: plain text as is; the JSON Cline and Roo
/// write as the question (or plan reply), its offered options and the option
/// picked, as readable text instead of raw JSON.
fn render_ask_reply(text: &str) -> Option<String> {
    if text.is_empty() {
        return None;
    }
    let Ok(Value::Object(fields)) = serde_json::from_str::<Value>(text) else {
        return Some(text.to_string());
    };
    let body = ["question", "response"]
        .iter()
        .find_map(|key| fields.get(*key).and_then(Value::as_str))
        .unwrap_or("");
    // Cline: `options: [string]`; Roo: `suggest: [{answer, mode}]`.
    let options: Vec<&str> = fields
        .get("options")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .chain(
            fields
                .get("suggest")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|item| item.get("answer").and_then(Value::as_str)),
        )
        .collect();
    let selected = fields.get("selected").and_then(Value::as_str);
    if body.is_empty() && options.is_empty() && selected.is_none() {
        // Not the shape either writer uses: show what was recorded.
        return Some(text.to_string());
    }

    let mut sections = Vec::new();
    if !body.is_empty() {
        sections.push(body.to_string());
    }
    if !options.is_empty() {
        sections.push(
            options
                .iter()
                .map(|option| format!("- {option}"))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
    if let Some(selected) = selected {
        sections.push(format!("Selected: {selected}"));
    }
    Some(sections.join("\n\n"))
}

fn map_cline_tool_name(name: &str) -> &str {
    match name {
        "readFile" => "Read",
        "editedExistingFile" | "newFileCreated" | "fileDeleted" => "Write",
        "listFilesTopLevel" | "listFilesRecursive" | "listCodeDefinitionNames" => "Glob",
        "searchFiles" => "Grep",
        "webFetch" => "WebFetch",
        "webSearch" => "WebSearch",
        _ => name,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn base_paths_are_collected_under_any_editor_data_root() {
        // #582: the collection itself, independent of where the OS keeps its
        // editor data — so it is exercised on every platform, not just macOS.
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let cline = root.join("Code/User/globalStorage/saoudrizwan.claude-dev");
        let roo = root.join("Cursor/User/globalStorage/rooveterinaryinc.roo-cline");
        fs::create_dir_all(&cline).unwrap();
        fs::create_dir_all(&roo).unwrap();
        fs::create_dir_all(root.join("Code/User/globalStorage/unrelated.extension")).unwrap();

        let found = collect_base_paths_under(root, &ScanReport::default());
        let labels: Vec<&str> = found.iter().map(|(_, label)| label.as_str()).collect();
        assert_eq!(found.len(), 2, "only Cline-family extensions: {labels:?}");
        assert!(found.iter().any(|(path, _)| path == &cline));
        assert!(labels.contains(&"Cline (VS Code)"));
        assert!(labels.contains(&"Roo Code (Cursor)"));
    }

    /// The `PearAI` agent (a Roo fork) in the `PearAI` editor: found, and its
    /// task index read from globalState, which VS Code keys by the published
    /// id `PearAI.pearai-roo-cline` while the storage dir is lowercased.
    #[test]
    fn pearai_agent_tasks_are_found_in_the_pearai_editor() {
        let tmp = tempfile::TempDir::new().unwrap();
        let global_storage = tmp.path().join("PearAI/User/globalStorage");
        let agent = global_storage.join("pearai.pearai-roo-cline");
        fs::create_dir_all(&agent).unwrap();
        let found = collect_base_paths_under(tmp.path(), &ScanReport::default());
        assert_eq!(
            found,
            vec![(agent.clone(), "PearAI Agent (PearAI)".to_string())]
        );

        let conn = Connection::open(global_storage.join("state.vscdb")).unwrap();
        conn.execute(
            "CREATE TABLE ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO ItemTable (key, value) VALUES ('PearAI.pearai-roo-cline', ?1)",
            [
                json!({"taskHistory": [{"id": "t1", "ts": 1, "task": "x", "workspace": "/w"}]})
                    .to_string(),
            ],
        )
        .unwrap();
        let history = load_task_history(&agent, &ScanReport::default());
        assert_eq!(history.len(), 1);
        assert_eq!(task_cwd(&history[0]), Some("/w"));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_ancestors_are_not_followed() {
        // Repo rule: directory traversal must not follow symlinks. `is_dir()`
        // does, so an editor/User/globalStorage symlink has to be rejected
        // before it is walked.
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let elsewhere = root.join("elsewhere/User/globalStorage/saoudrizwan.claude-dev");
        fs::create_dir_all(&elsewhere).unwrap();
        std::os::unix::fs::symlink(root.join("elsewhere"), root.join("Code")).unwrap();

        assert!(
            collect_base_paths_under(root, &ScanReport::default()).is_empty(),
            "a symlinked editor directory must not be traversed"
        );

        // The same tree reached without the symlink is still discovered.
        fs::rename(root.join("elsewhere"), root.join("Cursor")).unwrap();
        fs::remove_file(root.join("Code")).unwrap();
        assert_eq!(
            collect_base_paths_under(root, &ScanReport::default()).len(),
            1
        );
    }

    #[test]
    #[serial_test::serial]
    fn editor_data_roots_cover_this_platform() {
        // #582: before this, only macOS (unconditional `Library/Application
        // Support`) and Linux had a branch, so Cline was undiscoverable on
        // Windows. Every platform must contribute its editor data root.
        let _home = crate::test_utils::SandboxHome::new();
        let roots = editor_data_roots();
        let config = crate::utils::config_dir().expect("platform config dir");
        assert!(
            roots.contains(&config),
            "config dir {config:?} missing from {roots:?}"
        );
    }

    #[test]
    fn windows_project_and_session_paths_round_trip() {
        // Both the base (`C:\Users\…`) and the cwd (`D:\work\app`) carry a
        // drive-letter colon, so splitting at the first `:` cut the base down
        // to `C` and every Windows session failed to load.
        let base = r"C:\Users\nikos\AppData\Roaming\Code\User\globalStorage\saoudrizwan.claude-dev";
        let cwd = r"D:\work\app";

        let (parsed_base, parsed_cwd) =
            parse_project_path(&format!("cline://{base}:{cwd}")).unwrap();
        assert_eq!(parsed_base, PathBuf::from(base));
        assert_eq!(parsed_cwd, cwd);

        let (parsed_base, task_id) =
            parse_base_and_tail(&format!("cline://{base}:1789505165522")).unwrap();
        assert_eq!(parsed_base, PathBuf::from(base));
        assert_eq!(task_id, "1789505165522");
    }

    #[test]
    fn extension_id_inside_the_tail_is_not_a_separator() {
        // A base that doesn't end in an extension id keeps the first-colon
        // split even when its cwd happens to contain one.
        let (base, cwd) =
            parse_project_path("cline:///path/to/globalStorage:/work/saoudrizwan.claude-dev:x")
                .unwrap();
        assert_eq!(base, PathBuf::from("/path/to/globalStorage"));
        assert_eq!(cwd, "/work/saoudrizwan.claude-dev:x");
    }

    #[test]
    fn test_convert_say_text() {
        let msg = json!({
            "ts": 1700000000000u64,
            "type": "say",
            "say": "text",
            "text": "Hello from Cline"
        });
        let mut counter = 0;
        let result = convert_cline_message(&msg, "test", &mut counter, None).unwrap();
        assert_eq!(result.message_type, "assistant");
        assert_eq!(result.provider, Some("cline".to_string()));
    }

    #[test]
    fn test_convert_say_tool() {
        let msg = json!({
            "ts": 1700000000000u64,
            "type": "say",
            "say": "tool",
            "text": "{\"tool\":\"readFile\",\"path\":\"/test.txt\",\"content\":\"file contents\"}"
        });
        let mut counter = 0;
        let result = convert_cline_message(&msg, "test", &mut counter, None).unwrap();
        let content = result.content.unwrap();
        let arr = content.as_array().unwrap();
        assert_eq!(arr[0]["type"], "tool_use");
        assert_eq!(arr[0]["name"], "Read");
        assert_eq!(arr[1]["type"], "tool_result");
    }

    #[test]
    fn test_convert_say_command() {
        let msg = json!({
            "ts": 1700000000000u64,
            "type": "say",
            "say": "command",
            "text": "npm test"
        });
        let mut counter = 0;
        let result = convert_cline_message(&msg, "test", &mut counter, None).unwrap();
        let content = result.content.unwrap();
        let arr = content.as_array().unwrap();
        assert_eq!(arr[0]["name"], "Bash");
    }

    #[test]
    fn test_convert_say_reasoning() {
        let msg = json!({
            "ts": 1700000000000u64,
            "type": "say",
            "say": "reasoning",
            "text": "thinking about this..."
        });
        let mut counter = 0;
        let result = convert_cline_message(&msg, "test", &mut counter, None).unwrap();
        let content = result.content.unwrap();
        let arr = content.as_array().unwrap();
        assert_eq!(arr[0]["type"], "thinking");
    }

    #[test]
    fn test_convert_say_error() {
        let msg = json!({
            "ts": 1700000000000u64,
            "type": "say",
            "say": "error",
            "text": "Something went wrong"
        });
        let mut counter = 0;
        let result = convert_cline_message(&msg, "test", &mut counter, None).unwrap();
        assert_eq!(result.message_type, "system");
        assert_eq!(result.level, Some("error".to_string()));
    }

    fn say(kind: &str, text: &str) -> Value {
        json!({"ts": 1700000000000u64, "type": "say", "say": kind, "text": text})
    }

    fn ask(kind: &str, text: &str) -> Value {
        json!({"ts": 1700000000000u64, "type": "ask", "ask": kind, "text": text})
    }

    /// The text of a message's first block.
    fn first_text(message: &ClaudeMessage) -> String {
        message.content.as_ref().unwrap()[0]["text"]
            .as_str()
            .unwrap()
            .to_string()
    }

    /// `(type, text)` of every message converted from `records`.
    fn converted(records: &[Value]) -> Vec<(String, String)> {
        convert_ui_messages(records, "test")
            .iter()
            .map(|m| (m.message_type.clone(), first_text(m)))
            .collect()
    }

    #[test]
    fn test_convert_ask_followup() {
        // #849 C2: the model's question is an assistant message, not the user's.
        let msg = ask("followup", "Can you explain more?");
        let mut counter = 0;
        let result = convert_cline_message(&msg, "test", &mut counter, None).unwrap();
        assert_eq!(result.message_type, "assistant");
        assert_eq!(result.role.as_deref(), Some("assistant"));
        assert_eq!(first_text(&result), "Can you explain more?");
    }

    /// #849 C1: Cline writes the user's prompt as `say:"task"`.
    #[test]
    fn say_task_is_the_users_prompt() {
        let records = [
            say("task", "fix the build"),
            say("api_req_started", "{}"),
            say("text", "Looking at it."),
        ];
        assert_eq!(
            converted(&records),
            vec![
                ("user".to_string(), "fix the build".to_string()),
                ("assistant".to_string(), "Looking at it.".to_string()),
            ]
        );
    }

    /// #849 U3 / C1: classic Roo has no `task` say; its prompt is the first
    /// `say:"text"`, written before the first request. Later `text` is the
    /// model's.
    #[test]
    fn roo_first_text_before_any_request_is_the_users_prompt() {
        let records = [
            say("text", "add a README"),
            say("api_req_started", "{}"),
            say("text", "I will add one."),
            say("api_req_started", "{}"),
            say("text", "Done."),
        ];
        assert_eq!(
            converted(&records),
            vec![
                ("user".to_string(), "add a README".to_string()),
                ("assistant".to_string(), "I will add one.".to_string()),
                ("assistant".to_string(), "Done.".to_string()),
            ]
        );
    }

    #[test]
    fn text_after_a_request_is_never_the_users_prompt() {
        let records = [say("api_req_started", "{}"), say("text", "hello")];
        assert_eq!(
            converted(&records),
            vec![("assistant".to_string(), "hello".to_string())]
        );
        assert_eq!(prompt_index(&records), None);
    }

    /// #849 C2 / U2: Cline's followup is JSON `{question, options}`.
    #[test]
    fn cline_followup_json_shows_its_question_and_options() {
        let text = json!({"question": "Which database?", "options": ["Postgres", "SQLite"]});
        let out = converted(&[ask("followup", &text.to_string())]);
        assert_eq!(
            out,
            vec![(
                "assistant".to_string(),
                "Which database?\n\n- Postgres\n- SQLite".to_string()
            )]
        );
    }

    /// #849 U2: Roo's followup is JSON `{question, suggest: [{answer, mode}]}`.
    #[test]
    fn roo_followup_json_shows_its_question_and_suggestions() {
        let text = json!({"question": "Which mode?",
                          "suggest": [{"answer": "Use code", "mode": "code"}, {"answer": "Stay"}]});
        let out = converted(&[ask("followup", &text.to_string())]);
        assert_eq!(
            out,
            vec![(
                "assistant".to_string(),
                "Which mode?\n\n- Use code\n- Stay".to_string()
            )]
        );
    }

    /// The option the user picked is recorded in the ask, not as feedback.
    #[test]
    fn followup_shows_the_picked_option() {
        let text = json!({"question": "Which?", "options": ["a", "b"], "selected": "b"});
        let out = converted(&[ask("followup", &text.to_string())]);
        assert_eq!(out[0].1, "Which?\n\n- a\n- b\n\nSelected: b");
    }

    /// #849 C2: plan and act mode replies are the model's, and the user's
    /// answer (`user_feedback`) stays the user's.
    #[test]
    fn plan_and_act_mode_replies_are_assistant_and_feedback_is_user() {
        let plan = json!({"response": "Here is the plan.", "options": []});
        let records = [
            ask("plan_mode_respond", &plan.to_string()),
            say("user_feedback", "go ahead"),
            ask("act_mode_respond", "Working on it."),
        ];
        assert_eq!(
            converted(&records),
            vec![
                ("assistant".to_string(), "Here is the plan.".to_string()),
                ("user".to_string(), "go ahead".to_string()),
                ("assistant".to_string(), "Working on it.".to_string()),
            ]
        );
    }

    #[test]
    fn followup_text_that_is_not_the_known_json_is_shown_as_recorded() {
        let out = converted(&[ask("followup", "42"), ask("followup", r#"{"other":1}"#)]);
        assert_eq!(out[0].1, "42");
        assert_eq!(out[1].1, r#"{"other":1}"#);
        assert!(convert_ui_messages(&[ask("followup", "")], "test").is_empty());
    }

    /// #849 C3: bookkeeping says are not shown as assistant prose.
    #[test]
    fn upstream_bookkeeping_says_are_hidden() {
        for kind in [
            "api_req_retry_delayed",
            "api_req_rate_limit_wait",
            "api_req_deleted",
            "checkpoint_saved",
            "condense_context",
            "sliding_window_truncation",
            "mcp_server_request_started",
            "too_many_tools_warning",
            "user_edit_todos",
            "compaction",
            "subagent_usage",
            "cloud_provisioning",
        ] {
            let records = [say(kind, r#"{"some":"payload"}"#)];
            assert!(
                convert_ui_messages(&records, "test").is_empty(),
                "{kind} was shown"
            );
        }
    }

    /// #849 C3: content-bearing says are results and notices, not assistant
    /// prose.
    #[test]
    fn content_bearing_says_render_as_results_and_notices() {
        let out = convert_ui_messages(
            &[
                say("mcp_server_response", "{\"ok\": true}"),
                say("subtask_result", "All tests pass."),
                say("diff_error", "No match found"),
                say("rooignore_error", "secrets.env"),
                say("condense_context_error", "summary failed"),
            ],
            "test",
        );
        assert_eq!(out.len(), 5);
        assert_eq!(first_text(&out[0]), "```\n{\"ok\": true}\n```");
        assert_eq!(out[0].message_type, "assistant");
        assert_eq!(first_text(&out[1]), "Subtask result:\n\nAll tests pass.");
        for (notice, text) in out[2..].iter().zip([
            "No match found",
            "Blocked by .rooignore: secrets.env",
            "summary failed",
        ]) {
            assert_eq!(notice.message_type, "system");
            assert_eq!(notice.level.as_deref(), Some("error"));
            assert_eq!(first_text(notice), text);
        }
    }

    #[test]
    fn codebase_search_result_is_a_tool_call_with_its_hits() {
        let payload = json!({"tool": "codebaseSearch", "content": {
            "query": "auth",
            "results": [{"filePath": "src/auth.rs", "score": 0.9, "startLine": 3,
                         "endLine": 9, "codeChunk": "fn login() {}"}]}});
        let out = convert_ui_messages(&[say("codebase_search_result", &payload.to_string())], "t");
        let blocks = out[0].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[0]["name"], "codebaseSearch");
        assert_eq!(blocks[0]["input"]["query"], "auth");
        assert_eq!(blocks[1]["type"], "tool_result");
        assert_eq!(blocks[1]["tool_use_id"], blocks[0]["id"]);
        assert_eq!(blocks[1]["content"], "src/auth.rs:3-9\nfn login() {}");
    }

    fn blocks(message: &ClaudeMessage) -> &Vec<Value> {
        message.content.as_ref().unwrap().as_array().unwrap()
    }

    /// Cline's `use_mcp_server` request is a tool call named like Claude's MCP
    /// tools, and the `mcp_server_response` after it is that call's result.
    #[test]
    fn mcp_request_and_response_are_a_paired_tool_call() {
        let request = json!({"serverName": "gh", "type": "use_mcp_tool",
                             "toolName": "create_issue", "arguments": "{\"title\":\"x\"}"});
        let out = convert_ui_messages(
            &[
                say("use_mcp_server", &request.to_string()),
                say("mcp_server_request_started", ""),
                say("mcp_server_response", "created #1"),
                say("mcp_server_response", "stray"),
            ],
            "t",
        );
        assert_eq!(out.len(), 3);
        let call = &blocks(&out[0])[0];
        assert_eq!(out[0].message_type, "assistant");
        assert_eq!(call["type"], "tool_use");
        assert_eq!(call["name"], "mcp__gh__create_issue");
        assert_eq!(call["input"], json!({"title": "x"}));
        assert_eq!(out[1].message_type, "user");
        let result = &blocks(&out[1])[0];
        assert_eq!(result["type"], "tool_result");
        assert_eq!(result["tool_use_id"], call["id"]);
        assert_eq!(result["content"], "created #1");
        // A response after the call was answered stands alone.
        assert_eq!(out[2].message_type, "assistant");
        assert_eq!(first_text(&out[2]), "```\nstray\n```");
    }

    #[test]
    fn mcp_resource_request_and_unparsable_arguments() {
        let resource =
            json!({"serverName": "fs", "type": "access_mcp_resource", "uri": "file:///a"});
        let tool = json!({"serverName": "fs", "type": "use_mcp_tool", "toolName": "t",
                          "arguments": "not json"});
        let out = convert_ui_messages(
            &[
                say("use_mcp_server", &resource.to_string()),
                say("use_mcp_server", &tool.to_string()),
            ],
            "t",
        );
        let first = &blocks(&out[0])[0];
        assert_eq!(first["name"], "access_mcp_resource");
        assert_eq!(
            first["input"],
            json!({"server_name": "fs", "uri": "file:///a"})
        );
        let second = &blocks(&out[1])[0];
        assert_eq!(second["name"], "mcp__fs__t");
        assert_eq!(second["input"], json!({"arguments": "not json"}));
    }

    /// `browser_action_launch` text is the URL; `browser_action` is JSON
    /// `{action, coordinate?, text?}`; `browser_action_result` is JSON
    /// `{screenshot?, logs?, currentUrl?, currentMousePosition?}`.
    #[test]
    fn browser_actions_are_tool_calls_with_their_results() {
        let result = json!({"screenshot": "data:image/webp;base64,AAAA", "logs": "console: ok",
                            "currentUrl": "http://localhost:3000/"});
        let out = convert_ui_messages(
            &[
                say("browser_action_launch", "http://localhost:3000"),
                say("browser_action_result", &result.to_string()),
                say(
                    "browser_action",
                    r#"{"action":"click","coordinate":"450,300"}"#,
                ),
                say("browser_action_result", ""),
            ],
            "t",
        );
        // The empty result (a close) shows nothing.
        assert_eq!(out.len(), 3);
        let launch = &blocks(&out[0])[0];
        assert_eq!(launch["name"], "browser_action");
        assert_eq!(
            launch["input"],
            json!({"action": "launch", "url": "http://localhost:3000"})
        );

        assert_eq!(out[1].message_type, "user");
        let paired = &blocks(&out[1])[0];
        assert_eq!(paired["tool_use_id"], launch["id"]);
        assert_eq!(
            paired["content"],
            json!([
                {"type": "text", "text": "console: ok\nURL: http://localhost:3000/"},
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/webp", "data": "AAAA"}}
            ])
        );

        let click = &blocks(&out[2])[0];
        assert_eq!(
            click["input"],
            json!({"action": "click", "coordinate": "450,300"})
        );
    }

    #[test]
    fn browser_result_without_a_call_is_shown_as_the_page_it_left() {
        let result = json!({"logs": "(No new logs)", "currentUrl": "http://x/"});
        let out = convert_ui_messages(&[say("browser_action_result", &result.to_string())], "t");
        assert_eq!(out[0].message_type, "assistant");
        assert_eq!(first_text(&out[0]), "(No new logs)\nURL: http://x/");
    }

    /// `use_subagents` is JSON `{prompts}`; `subagent` is its final status.
    #[test]
    fn subagents_are_a_tool_call_with_their_final_status() {
        let prompts = json!({"prompts": ["find the bug", "write a test"]});
        let status = json!({"status": "completed", "total": 2, "completed": 2, "items": [
            {"index": 0, "prompt": "find the bug", "status": "completed", "result": "in lib.rs"},
            {"index": 1, "prompt": "write a test", "status": "failed", "error": "timed out"}]});
        let out = convert_ui_messages(
            &[
                say("use_subagents", &prompts.to_string()),
                say("subagent", &status.to_string()),
                say("subagent_usage", r#"{"source":"subagents","tokensIn":1}"#),
            ],
            "t",
        );
        assert_eq!(out.len(), 2);
        let call = &blocks(&out[0])[0];
        assert_eq!(call["name"], "use_subagents");
        assert_eq!(call["input"], prompts);
        let result = &blocks(&out[1])[0];
        assert_eq!(result["tool_use_id"], call["id"]);
        assert_eq!(
            result["content"],
            "Subagents completed: 2 of 2 done\n1. [completed] find the bug\nin lib.rs\n\
             2. [failed] write a test\ntimed out"
        );
    }

    /// Roo's `generate_image` says `{imageUri, imagePath}`: the webview URI
    /// cannot be opened, so the path is shown.
    #[test]
    fn roo_image_shows_its_path_and_a_data_url_as_an_image() {
        let payload = json!({"imageUri": "https://file+.vscode-resource.vscode-cdn.net/a.png?t=1",
                             "imagePath": "/w/a.png"});
        let out = convert_ui_messages(
            &[
                say("image", &payload.to_string()),
                say("image", "data:image/png;base64,QUJD"),
                say("image", "not an image"),
            ],
            "t",
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].message_type, "system");
        assert_eq!(first_text(&out[0]), "Generated image: /w/a.png");
        assert_eq!(
            blocks(&out[1])[0],
            json!({"type": "image",
                   "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}})
        );
    }

    fn write_task_files(base: &Path, id: &str, files: &[(&str, String)]) {
        let dir = base.join("tasks").join(id);
        fs::create_dir_all(&dir).unwrap();
        for (name, body) in files {
            fs::write(dir.join(name), body).unwrap();
        }
    }

    /// #849 C4: Roo's per-task `history_item.json` lists a task the index
    /// lacks, and a task the index lists is not listed twice.
    #[test]
    fn tasks_missing_from_the_roo_index_are_listed_from_history_item() {
        let tmp = tempfile::TempDir::new().unwrap();
        let base = tmp.path().join("globalStorage/rooveterinaryinc.roo-cline");
        let ui = r#"[{"type":"say","say":"text","text":"hi","ts":1700000000000}]"#;
        fs::create_dir_all(base.join("tasks")).unwrap();
        fs::write(
            base.join("tasks/_index.json"),
            json!({"version": 1, "entries": [
                {"id": "indexed", "ts": 1, "task": "from index", "workspace": "/w/a"}
            ]})
            .to_string(),
        )
        .unwrap();
        // Listed by the index too: the index entry stays the one listed.
        write_task_files(
            &base,
            "indexed",
            &[
                ("ui_messages.json", ui.to_string()),
                (
                    "history_item.json",
                    json!({"id": "indexed", "ts": 2, "task": "from item", "workspace": "/w/a"})
                        .to_string(),
                ),
            ],
        );
        write_task_files(
            &base,
            "late",
            &[
                ("ui_messages.json", ui.to_string()),
                (
                    "history_item.json",
                    json!({"id": "late", "ts": 3, "task": "from item", "workspace": "/w/b",
                           "tokensIn": 5, "tokensOut": 6})
                    .to_string(),
                ),
            ],
        );
        // No messages: nothing to open.
        write_task_files(
            &base,
            "empty",
            &[(
                "history_item.json",
                json!({"id": "empty", "ts": 4, "task": "x"}).to_string(),
            )],
        );

        let history = load_task_history(&base, &ScanReport::default());
        let ids: Vec<&str> = history
            .iter()
            .map(|item| item["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["indexed", "late"]);
        assert_eq!(history[0]["task"], "from index");
        assert_eq!(task_cwd(&history[1]), Some("/w/b"));
    }

    /// #849 C4: with no index at all, a Cline task is rebuilt from its
    /// `ui_messages.json` and `task_metadata.json`.
    #[test]
    fn tasks_without_any_index_are_rebuilt_from_their_messages() {
        let tmp = tempfile::TempDir::new().unwrap();
        let base = tmp.path().join("globalStorage/saoudrizwan.claude-dev");
        let request = json!({
            "request": "<task>\nfix it\n</task>\n\n# Current Working Directory (/work/app) Files\nsrc/",
            "tokensIn": 100, "tokensOut": 20
        });
        let second = json!({"request": "again", "tokensIn": 7, "tokensOut": 3});
        let messages = json!([
            {"type": "say", "say": "task", "text": "fix it", "ts": 1700000005000u64},
            {"type": "say", "say": "api_req_started", "text": request.to_string(),
             "ts": 1700000006000u64},
            {"type": "say", "say": "api_req_started", "text": second.to_string(),
             "ts": 1700000007000u64},
        ]);
        write_task_files(
            &base,
            "1700000005000",
            &[
                ("ui_messages.json", messages.to_string()),
                (
                    "task_metadata.json",
                    json!({"files_in_context": [], "model_usage": [
                        {"ts": 1, "model_id": "old-model", "model_provider_id": "p", "mode": "act"},
                        {"ts": 2, "model_id": "claude-sonnet", "model_provider_id": "p", "mode": "act"}
                    ]})
                    .to_string(),
                ),
            ],
        );
        // A directory without messages and a stray file are not tasks.
        fs::create_dir_all(base.join("tasks/not-a-task")).unwrap();
        fs::write(base.join("tasks/stray.txt"), "x").unwrap();

        let history = load_task_history(&base, &ScanReport::default());
        assert_eq!(history.len(), 1);
        let item = &history[0];
        assert_eq!(item["id"], "1700000005000");
        assert_eq!(item["ts"], 1_700_000_005_000u64);
        assert_eq!(item["task"], "fix it");
        assert_eq!(item["tokensIn"], 107);
        assert_eq!(item["tokensOut"], 23);
        assert_eq!(item["modelId"], "claude-sonnet");
        assert_eq!(task_cwd(item), Some("/work/app"));
    }

    #[test]
    fn rebuilt_roo_task_takes_its_prompt_from_the_first_text() {
        let messages = [
            say("text", "add a README"),
            say("api_req_started", "{}"),
            say("text", "ok"),
        ];
        let item = item_from_messages("t1", &messages, None);
        assert_eq!(item["task"], "add a README");
        assert!(task_cwd(&item).is_none());
        assert_eq!(
            request_cwd("# Current Working Directory (Primary: app) Files"),
            None
        );
    }

    #[test]
    fn test_skip_api_metadata() {
        let msg = json!({
            "ts": 1700000000000u64,
            "type": "say",
            "say": "api_req_started",
            "text": "{}"
        });
        let mut counter = 0;
        assert!(convert_cline_message(&msg, "test", &mut counter, None).is_none());
    }

    #[test]
    fn test_map_cline_tool_names() {
        assert_eq!(map_cline_tool_name("readFile"), "Read");
        assert_eq!(map_cline_tool_name("editedExistingFile"), "Write");
        assert_eq!(map_cline_tool_name("searchFiles"), "Grep");
        assert_eq!(map_cline_tool_name("webSearch"), "WebSearch");
        assert_eq!(map_cline_tool_name("unknownTool"), "unknownTool");
    }

    #[test]
    fn test_ms_to_iso() {
        let result = ms_to_iso(1700000000000);
        assert!(result.starts_with("2023-11-14T"));
        assert!(result.ends_with('Z'));
    }

    #[test]
    fn test_parse_project_path() {
        let (base, cwd) =
            parse_project_path("cline:///path/to/globalStorage:/Users/jack/project").unwrap();
        assert_eq!(base, PathBuf::from("/path/to/globalStorage"));
        assert_eq!(cwd, "/Users/jack/project");
    }

    #[test]
    fn test_task_cwd_field_fallback() {
        // Cline uses cwdOnTaskInitialization
        assert_eq!(
            task_cwd(&json!({"cwdOnTaskInitialization": "/c"})),
            Some("/c")
        );
        // Roo Code / Kilo Code rename it to `workspace`
        assert_eq!(task_cwd(&json!({"workspace": "/r"})), Some("/r"));
        // Cline name wins when both are present
        assert_eq!(
            task_cwd(&json!({"cwdOnTaskInitialization": "/c", "workspace": "/r"})),
            Some("/c")
        );
        // Neither present
        assert_eq!(task_cwd(&json!({"id": "x"})), None);
    }

    #[test]
    #[serial_test::serial]
    fn test_load_task_history_from_kilo_global_state() {
        use rusqlite::params;

        // Loading a Kilo index also looks for tasks Kilo copied into kilo.db.
        let _home = crate::test_utils::SandboxHome::new();
        let tmp = tempfile::TempDir::new().unwrap();
        let global_storage = tmp.path().join("globalStorage");
        let ext_dir = global_storage.join("kilocode.kilo-code");
        fs::create_dir_all(&ext_dir).unwrap();

        // Mirror how VS Code persists globalState: one ItemTable row per
        // extension id, value = flat JSON of that extension's state.
        let db_path = global_storage.join("state.vscdb");
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute(
                "CREATE TABLE ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)",
                [],
            )
            .unwrap();
            let value = json!({
                "taskHistory": [
                    {
                        "id": "task-1",
                        "ts": 1_700_000_000_000u64,
                        "task": "Implement Kilo",
                        "workspace": "/Users/jack/proj",
                        "tokensIn": 10,
                        "tokensOut": 20
                    }
                ],
                "someOtherKey": 1
            })
            .to_string();
            conn.execute(
                "INSERT INTO ItemTable (key, value) VALUES (?1, ?2)",
                params!["kilocode.kilo-code", value],
            )
            .unwrap();
        }

        // No disk index exists, so this exercises the globalState fallback.
        let history = load_task_history(&ext_dir, &ScanReport::default());
        assert_eq!(history.len(), 1);
        assert_eq!(task_cwd(&history[0]), Some("/Users/jack/proj"));
        assert_eq!(history[0]["task"], "Implement Kilo");
    }

    /// Kilo's extension copies a legacy task into `kilo.db` as
    /// `ses_migrated_…` (slug = task id) and keeps the original: the copy is
    /// Kilo's, so the legacy reader lists only tasks that were not copied.
    #[test]
    #[serial_test::serial]
    fn kilo_tasks_copied_into_kilo_db_are_left_to_the_kilo_reader() {
        let home = crate::test_utils::SandboxHome::new();
        let ext_dir = crate::utils::config_dir()
            .unwrap()
            .join("Code/User/globalStorage/kilocode.kilo-code");
        fs::create_dir_all(&ext_dir).unwrap();
        let conn = Connection::open(ext_dir.parent().unwrap().join("state.vscdb")).unwrap();
        conn.execute(
            "CREATE TABLE ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)",
            [],
        )
        .unwrap();
        let history = json!({"taskHistory": [
            {"id": "copied", "ts": 1, "task": "a", "workspace": "/w"},
            {"id": "kept", "ts": 2, "task": "b", "workspace": "/w"}
        ]});
        conn.execute(
            "INSERT INTO ItemTable (key, value) VALUES ('kilocode.kilo-code', ?1)",
            [history.to_string()],
        )
        .unwrap();

        let kilo = home.path().join(".local/share/kilo");
        fs::create_dir_all(&kilo).unwrap();
        let db = crate::providers::kilo::test_support::create_test_db(&kilo);
        db.execute(
            "INSERT INTO session (id, project_id, slug, title, time_created, time_updated)
             VALUES ('ses_migrated_abc', 'p', 'copied', 'a', 1, 1)",
            [],
        )
        .unwrap();

        let ids = || -> Vec<String> {
            load_task_history(&ext_dir, &ScanReport::default())
                .iter()
                .map(|item| item["id"].as_str().unwrap().to_string())
                .collect()
        };
        assert_eq!(ids(), ["kept"]);

        // A symlinked `storage` makes the Kilo scan list nothing, so the
        // legacy entry must stay visible.
        #[cfg(unix)]
        {
            let elsewhere = home.path().join("elsewhere");
            fs::create_dir_all(&elsewhere).unwrap();
            std::os::unix::fs::symlink(&elsewhere, kilo.join("storage")).unwrap();
            assert_eq!(ids(), ["copied", "kept"]);
        }
    }

    /// Kilo's legacy CLI: `~/.kilocode/cli/global/tasks/<id>/` with the task
    /// index in `global-state.json`.
    #[test]
    #[serial_test::serial]
    fn kilo_legacy_cli_tasks_are_listed_and_loaded() {
        let home = crate::test_utils::SandboxHome::new();
        let base = home.path().join(".kilocode/cli/global");
        write_task(&base, "1700000000000");
        fs::write(
            base.join("global-state.json"),
            json!({"taskHistory": [{
                "id": "1700000000000", "ts": 1_700_000_000_000u64,
                "task": "fix the build", "workspace": "/work/cli-app"
            }]})
            .to_string(),
        )
        .unwrap();

        let project = scan_projects()
            .unwrap()
            .into_iter()
            .find(|p| p.actual_path == "/work/cli-app")
            .expect("the CLI store is scanned");
        let sessions = load_sessions(&project.path, false).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].summary.as_deref(), Some("fix the build"));
        assert!(!load_messages(&sessions[0].session_id).unwrap().is_empty());
    }

    #[test]
    fn test_load_task_history_no_index_is_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let ext_dir = tmp.path().join("globalStorage").join("kilocode.kilo-code");
        fs::create_dir_all(&ext_dir).unwrap();
        // Neither a disk index nor a state.vscdb → empty, no panic.
        assert_eq!(
            load_task_history(&ext_dir, &ScanReport::default()),
            Vec::<serde_json::Value>::new()
        );
    }

    #[test]
    fn test_truncate_chars_is_utf8_safe() {
        // Regression: `&task[..100]` byte-slicing panicked on multibyte text
        // (byte 100 falls mid-character for 3-byte CJK). Char-based never does.
        let cjk = "한".repeat(150);
        let out = truncate_chars(&cjk, 100, "...");
        assert_eq!(out.chars().count(), 103); // 100 chars + "..."
        assert!(out.ends_with("..."));
        // Shorter-than-limit text is returned unchanged (no suffix).
        assert_eq!(truncate_chars("hi", 100, "..."), "hi");
        // Exactly the limit is not truncated.
        let exact = "a".repeat(100);
        assert_eq!(truncate_chars(&exact, 100, "..."), exact);
        // Custom suffix.
        assert_eq!(
            truncate_chars(&"x".repeat(2001), 2000, "...(truncated)"),
            format!("{}...(truncated)", "x".repeat(2000))
        );
    }

    #[test]
    fn test_task_label_field_fallback() {
        assert_eq!(
            task_label(&json!({ "modelId": "claude" })).as_deref(),
            Some("claude")
        );
        // Roo/Kilo have no modelId; they use apiConfigName / mode.
        assert_eq!(
            task_label(&json!({ "apiConfigName": "default" })).as_deref(),
            Some("default")
        );
        assert_eq!(
            task_label(&json!({ "mode": "code" })).as_deref(),
            Some("code")
        );
        assert_eq!(task_label(&json!({ "other": "x" })), None);
    }

    #[test]
    fn test_session_summary() {
        // Non-empty task -> truncated task (char-safe), label ignored.
        assert_eq!(
            session_summary("hello", Some("claude".into())).as_deref(),
            Some("hello")
        );
        // Empty task -> falls back to the label.
        assert_eq!(
            session_summary("", Some("default".into())).as_deref(),
            Some("default")
        );
        // Empty task, no label -> None.
        assert_eq!(session_summary("", None), None);
    }

    fn write_task(base: &Path, task_id: &str) {
        let task = base.join("tasks").join(task_id);
        fs::create_dir_all(&task).unwrap();
        fs::write(
            task.join("ui_messages.json"),
            r#"[{"type":"say","say":"text","text":"hello","ts":1700000000000}]"#,
        )
        .unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn load_messages_rejects_path_outside_root() {
        let _home = crate::test_utils::SandboxHome::new();
        let outside = tempfile::TempDir::new().unwrap();
        let base = outside.path().join("saoudrizwan.claude-dev");
        write_task(&base, "1700000000000");

        let res = load_messages(&format!("cline://{}:1700000000000", base.display()));
        assert!(res.is_err(), "task under an undiscovered base was read");
        let res = load_sessions(&format!("cline://{}:/work", base.display()), false);
        assert!(
            res.is_err(),
            "project under an undiscovered base was listed"
        );
    }

    #[test]
    #[serial_test::serial]
    fn load_messages_accepts_a_discovered_extension_dir() {
        let home = crate::test_utils::SandboxHome::new();
        let config = crate::utils::config_dir().expect("config dir");
        assert!(config.starts_with(home.path()), "config dir not sandboxed");
        let base = config
            .join("Code")
            .join("User")
            .join("globalStorage")
            .join("rooveterinaryinc.roo-cline");
        write_task(&base, "1700000000000");

        let messages = load_messages(&format!("cline://{}:1700000000000", base.display())).unwrap();
        assert!(!messages.is_empty());
    }

    /// Every Cline session opens by the file it lists, in both stores, and
    /// reads what its `cline://` id reads (#707).
    #[test]
    #[serial_test::serial]
    fn sessions_load_by_the_file_they_list() {
        let home = crate::test_utils::SandboxHome::new();
        let task_file = fixtures::write_extension_task(home.path(), "/work/app");
        let sdk_file = fixtures::write_sdk_session(home.path(), "/work/app");

        let mut sessions: Vec<ClaudeSession> = scan_projects()
            .unwrap()
            .iter()
            .flat_map(|project| load_sessions(&project.path, false).unwrap())
            .collect();
        sessions.sort_by(|a, b| a.file_path.cmp(&b.file_path));
        let mut listed = vec![
            task_file.to_string_lossy().to_string(),
            sdk_file.to_string_lossy().to_string(),
        ];
        listed.sort();
        assert_eq!(
            sessions
                .iter()
                .map(|s| s.file_path.clone())
                .collect::<Vec<_>>(),
            listed
        );

        for session in &sessions {
            let uuids = |path: &str| {
                load_messages(path)
                    .unwrap_or_else(|e| panic!("{path}: {e}"))
                    .into_iter()
                    .map(|m| m.uuid)
                    .collect::<Vec<_>>()
            };
            let by_file = uuids(&session.file_path);
            assert!(!by_file.is_empty(), "{}", session.file_path);
            assert_eq!(by_file, uuids(&session.session_id));
            assert!(is_session_file(&session.file_path));
            assert_eq!(
                project_name_for_session(&session.file_path).as_deref(),
                Some("app")
            );
        }
    }

    /// A file shaped like a listed one is read only from a store discovery
    /// finds: anywhere else, or through `..`, it is refused.
    #[test]
    #[serial_test::serial]
    fn listed_files_outside_their_stores_are_refused() {
        let home = crate::test_utils::SandboxHome::new();
        let task_file = fixtures::write_extension_task(home.path(), "/work/app");
        let sdk_file = fixtures::write_sdk_session(home.path(), "/work/app");
        let outside = tempfile::TempDir::new().unwrap();
        let outside_base = outside.path().join("saoudrizwan.claude-dev");
        write_task(&outside_base, "1700000000000");
        let outside_sdk = outside.path().join("sessions/1_x/1_x.messages.json");
        fs::create_dir_all(outside_sdk.parent().unwrap()).unwrap();
        fs::copy(&sdk_file, &outside_sdk).unwrap();
        let task_dir = task_file.parent().unwrap();
        let traversal = task_dir
            .join("..")
            .join(task_dir.file_name().unwrap())
            .join("ui_messages.json");

        for path in [
            outside_base.join("tasks/1700000000000/ui_messages.json"),
            outside_sdk,
            traversal,
            PathBuf::from("tasks/task-1/ui_messages.json"),
        ] {
            let path = path.to_string_lossy().to_string();
            assert!(load_messages(&path).is_err(), "{path} was read");
            assert!(!is_session_file(&path), "{path} was taken for a Cline file");
        }
    }

    /// Search hits carry the scanned project's path, so same-named projects
    /// in different directories and stores stay apart (#672).
    #[test]
    #[serial_test::serial]
    fn search_hits_carry_scanned_project_path_for_same_named_projects() {
        let home = crate::test_utils::SandboxHome::new();
        let ext = crate::utils::config_dir()
            .unwrap()
            .join("Code")
            .join("User")
            .join("globalStorage")
            .join("saoudrizwan.claude-dev");
        let history = json!([
            {"id": "task-a", "ts": fixtures::TS_MS, "task": "hi", "tokensIn": 1,
             "cwdOnTaskInitialization": "/work/client-a/app"},
            {"id": "task-b", "ts": fixtures::TS_MS, "task": "hi", "tokensIn": 1,
             "cwdOnTaskInitialization": "/work/client-b/app"}
        ]);
        fs::create_dir_all(ext.join("state")).unwrap();
        fs::write(ext.join("state/taskHistory.json"), history.to_string()).unwrap();
        for id in ["task-a", "task-b"] {
            let task = ext.join("tasks").join(id);
            fs::create_dir_all(&task).unwrap();
            fs::write(
                task.join("ui_messages.json"),
                json!([{"type": "say", "say": "text", "text": "hi", "ts": fixtures::TS_MS}])
                    .to_string(),
            )
            .unwrap();
        }
        fixtures::write_sdk_session(home.path(), "/work/client-b/app");

        let projects = scan_projects().unwrap();
        assert_eq!(projects.len(), 3);
        assert!(projects.iter().all(|p| p.name == "app"));
        let session_project: HashMap<String, String> = projects
            .iter()
            .flat_map(|project| {
                load_sessions(&project.path, false)
                    .unwrap()
                    .into_iter()
                    .map(|s| (s.actual_session_id, project.path.clone()))
            })
            .collect();
        assert_eq!(session_project.len(), 3);

        let hits = search("hi", 100).unwrap();
        let mut hit_paths: Vec<String> = hits
            .iter()
            .map(|hit| {
                let path = hit.project_path.clone().expect("hit has project_path");
                assert_eq!(Some(&path), session_project.get(&hit.session_id));
                path
            })
            .collect();
        hit_paths.sort();
        hit_paths.dedup();
        assert_eq!(hit_paths.len(), 3, "each project keeps its own path");
    }
}

/// Cline sessions in both stores, for tests across the crate.
#[cfg(test)]
pub(crate) mod fixtures {
    use serde_json::json;
    use std::fs;
    use std::path::{Path, PathBuf};

    /// 2026-10-16T09:00:00Z.
    pub(crate) const TS_MS: u64 = 1_792_141_200_000;

    /// A task of Cline's VS Code extension in `cwd`, under the sandboxed
    /// `home`. Returns the file it lists, `tasks/<id>/ui_messages.json`; its
    /// session id is `cline://<extension dir>:<id>`.
    pub(crate) fn write_extension_task(home: &Path, cwd: &str) -> PathBuf {
        let ext = crate::utils::config_dir()
            .unwrap()
            .join("Code")
            .join("User")
            .join("globalStorage")
            .join("saoudrizwan.claude-dev");
        assert!(ext.starts_with(home), "{}", ext.display());
        let history = json!([{"id": "task-1", "ts": TS_MS, "task": "hi",
                              "tokensIn": 100, "tokensOut": 50,
                              "cwdOnTaskInitialization": cwd}]);
        fs::create_dir_all(ext.join("state")).unwrap();
        fs::write(ext.join("state/taskHistory.json"), history.to_string()).unwrap();
        let messages = json!([
            {"type": "say", "say": "task", "text": "hi", "ts": TS_MS},
            {"type": "say", "say": "text", "text": "hello", "ts": TS_MS + 1_000}
        ]);
        // Joined per component: the listing returns native separators, and a
        // `/`-joined path would not compare equal on Windows.
        let file = ext.join("tasks").join("task-1").join("ui_messages.json");
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, messages.to_string()).unwrap();
        file
    }

    /// A session of Cline's SDK store (`~/.cline/data/sessions/<id>/`) in
    /// `cwd` whose reply carries usage (21 in, 8 out, 3 cache read, 1 cache
    /// write). Returns the file it lists, `<id>.messages.json`; its session
    /// id is `cline://sdk:<id>`.
    pub(crate) fn write_sdk_session(home: &Path, cwd: &str) -> PathBuf {
        let id = "1792141200000_sdk01";
        let dir = home.join(".cline").join("data").join("sessions").join(id);
        fs::create_dir_all(&dir).unwrap();
        let manifest = json!({"version": 1, "session_id": id, "cwd": cwd,
                              "started_at": "2026-10-16T09:00:00.000Z",
                              "ended_at": "2026-10-16T09:00:02.000Z",
                              "metadata": {"title": "Demo"}});
        fs::write(dir.join(format!("{id}.json")), manifest.to_string()).unwrap();
        let transcript = json!({"version": 1, "agent": "lead", "sessionId": id, "messages": [
            {"id": "u1", "role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"id": "a1", "role": "assistant", "ts": TS_MS + 2_000,
             "modelInfo": {"id": "claude-sonnet-4-6", "provider": "anthropic"},
             "metrics": {"inputTokens": 21, "outputTokens": 8, "cacheReadTokens": 3,
                         "cacheWriteTokens": 1, "cost": 0.13},
             "content": [{"type": "text", "text": "ok"}]}
        ]});
        let file = dir.join(format!("{id}.messages.json"));
        fs::write(&file, transcript.to_string()).unwrap();
        file
    }
}
