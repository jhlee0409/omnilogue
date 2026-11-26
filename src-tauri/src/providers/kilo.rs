//! Kilo Code (`~/.local/share/kilo`) — OpenCode-core store.
//!
//! Kilo Code now runs on the OpenCode core: sessions live in a
//! drizzle-managed SQLite database (`kilo.db`) whose schema and
//! `message`/`part` JSON shapes are identical to OpenCode's. The reader
//! therefore delegates to the OpenCode provider's store-parameterized
//! functions and rebrands the `opencode://` scheme to `kilo://` so the two
//! stores can never cross-reference each other's ids.
//!
//! The database is `KILO_DB` when set (an absolute path, or a path under the
//! data directory), else `kilo.db` in `$XDG_DATA_HOME/kilo` or
//! `~/.local/share/kilo` (`Kilo-Org/kilocode` `packages/core/src/global.ts`,
//! `database/database.ts` `path()`).
//!
//! The pre-migration Cline-family stores (VS Code
//! `globalStorage/kilocode.kilo-code`, the old CLI's `~/.kilocode/cli/global`)
//! are read by [`crate::providers::cline`]. Kilo's VS Code extension copies a
//! legacy task into `kilo.db` as session `ses_migrated_…` with the task id as
//! its `slug`, and keeps the original; `migrated_task_ids` lets the Cline
//! reader leave those tasks to this one.

use crate::models::{ClaudeMessage, ClaudeProject, ClaudeSession};
use crate::providers::opencode::{self, ChildSession, SessionTree};
use crate::providers::scan_report::ScanReport;
use crate::providers::ProviderInfo;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// `SQLite` database file inside the Kilo store root, unless `KILO_DB`
/// names another.
pub const DB_FILE: &str = "kilo.db";

const SCHEME: &str = "kilo://";
const OPENCODE_SCHEME: &str = "opencode://";
const PROVIDER_ID: &str = "kilo";
const DISPLAY_NAME: &str = "Kilo Code";

/// Kilo's data directory: `$XDG_DATA_HOME/kilo`, or `~/.local/share/kilo`
/// (on every OS, as `xdg-basedir` resolves it).
fn data_dir() -> Option<PathBuf> {
    let xdg = std::env::var_os("XDG_DATA_HOME")
        .filter(|dir| !dir.is_empty())
        .map(|dir| PathBuf::from(dir).join("kilo"))
        .filter(|dir| dir.exists());
    xdg.or_else(|| {
        Some(
            crate::utils::home_dir()?
                .join(".local")
                .join("share")
                .join("kilo"),
        )
    })
}

/// The database file: `KILO_DB` (absolute, or relative to the data
/// directory), else `<data dir>/kilo.db`. `:memory:` has no file.
fn db_location() -> Option<PathBuf> {
    match std::env::var_os("KILO_DB").filter(|db| !db.is_empty()) {
        Some(db) if db == ":memory:" => None,
        Some(db) if Path::new(&db).is_absolute() => Some(PathBuf::from(db)),
        Some(db) => Some(data_dir()?.join(db)),
        None => Some(data_dir()?.join(DB_FILE)),
    }
}

/// Get the Kilo store root: the directory holding the database.
pub fn get_base_path() -> Option<String> {
    let dir = db_location()?.parent()?.to_path_buf();
    dir.exists().then(|| dir.to_string_lossy().to_string())
}

/// The database's file name inside [`get_base_path`]: `KILO_DB`'s, else
/// `kilo.db`.
pub fn db_file() -> String {
    std::env::var_os("KILO_DB")
        .filter(|db| !db.is_empty() && db != ":memory:")
        .and_then(|db| {
            Path::new(&db)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| DB_FILE.to_string())
}

/// Detect a Kilo Code installation (OpenCode-core store).
///
/// Unlike `OpenCode`, Kilo never had a file-backed `storage/` tree — the
/// database is the only session store — so availability is the database
/// alone.
pub fn detect() -> Option<ProviderInfo> {
    let base_path = get_base_path()?;
    let db_path = Path::new(&base_path).join(db_file());

    Some(ProviderInfo {
        id: PROVIDER_ID.to_string(),
        display_name: DISPLAY_NAME.to_string(),
        base_path: base_path.clone(),
        is_available: db_path.is_file(),
    })
}

/// Legacy task ids Kilo's VS Code extension copied into the database
/// (`legacy-migration/sessions/lib/session.ts`: `id = ses_migrated_<hash>`,
/// `slug = <task id>`). Empty when there is no database or no such session.
pub(crate) fn migrated_task_ids() -> HashSet<String> {
    // The same policy the reader applies (a symlinked database is refused),
    // so a task is never hidden from the Cline reader while this one cannot
    // show its copy.
    let regular = |db: &PathBuf| {
        std::fs::symlink_metadata(db).is_ok_and(|m| m.is_file() && !m.file_type().is_symlink())
    };
    let Some(db) = db_location().filter(regular) else {
        return HashSet::new();
    };
    let Ok(conn) = rusqlite::Connection::open_with_flags(
        &db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) else {
        return HashSet::new();
    };
    let Ok(mut stmt) =
        conn.prepare("SELECT slug FROM session WHERE id LIKE 'ses\\_migrated\\_%' ESCAPE '\\'")
    else {
        return HashSet::new();
    };
    stmt.query_map([], |row| row.get::<_, String>(0))
        .map(|rows| rows.filter_map(Result::ok).collect())
        .unwrap_or_default()
}

/// Rewrite a `kilo://` path into the `opencode://` scheme the shared reader
/// parses. Paths without the prefix pass through unchanged.
fn to_store_scheme(path: &str) -> String {
    path.strip_prefix(SCHEME)
        .map(|rest| format!("{OPENCODE_SCHEME}{rest}"))
        .unwrap_or_else(|| path.to_string())
}

/// Rewrite an `opencode://` path produced by the shared reader into
/// `kilo://`. Paths without the prefix pass through unchanged.
fn to_kilo_scheme(path: &str) -> String {
    path.strip_prefix(OPENCODE_SCHEME)
        .map(|rest| format!("{SCHEME}{rest}"))
        .unwrap_or_else(|| path.to_string())
}

/// The shared reader reports failures in `OpenCode` terms (`"OpenCode base
/// path"`, `opencode://…` paths); restate them for Kilo so users see the store
/// they actually configured.
fn rebrand_error(error: String) -> String {
    error
        .replace("OpenCode", DISPLAY_NAME)
        .replace(OPENCODE_SCHEME, SCHEME)
}

fn rebrand_project(mut project: ClaudeProject) -> ClaudeProject {
    project.path = to_kilo_scheme(&project.path);
    project.provider = Some(PROVIDER_ID.to_string());
    project
}

fn rebrand_session(mut session: ClaudeSession) -> ClaudeSession {
    session.session_id = to_kilo_scheme(&session.session_id);
    session.file_path = to_kilo_scheme(&session.file_path);
    session.provider = Some(PROVIDER_ID.to_string());
    session
}

/// Scan Kilo projects from the default store root.
pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    scan_projects_reporting(&ScanReport::default())
}

/// [`scan_projects`], recording in `report` what it could not read.
pub(crate) fn scan_projects_reporting(report: &ScanReport) -> Result<Vec<ClaudeProject>, String> {
    let base = get_base_path().ok_or("Kilo base path not found")?;
    scan_projects_at_reporting(&base, report)
}

/// [`scan_projects`] against an explicit store root.
pub fn scan_projects_at(base_path: &str) -> Result<Vec<ClaudeProject>, String> {
    scan_projects_at_reporting(base_path, &ScanReport::default())
}

fn scan_projects_at_reporting(
    base_path: &str,
    report: &ScanReport,
) -> Result<Vec<ClaudeProject>, String> {
    opencode::scan_projects_from_store_reporting(base_path, &db_file(), report)
        .map(|projects| projects.into_iter().map(rebrand_project).collect())
        .map_err(rebrand_error)
}

/// Load sessions for a Kilo project (`kilo://<project_id>`).
pub fn load_sessions(
    project_path: &str,
    exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    let base_path = get_base_path().ok_or_else(|| "Kilo not found".to_string())?;
    load_sessions_at(&base_path, project_path, exclude_sidechain)
}

/// [`load_sessions`], recording in `report` what it could not read.
pub(crate) fn load_sessions_reporting(
    project_path: &str,
    report: &ScanReport,
) -> Result<Vec<ClaudeSession>, String> {
    let base_path = get_base_path().ok_or_else(|| "Kilo not found".to_string())?;
    load_sessions_at_reporting(&base_path, project_path, report)
}

/// [`load_sessions`] against an explicit store root.
pub fn load_sessions_at(
    base_path: &str,
    project_path: &str,
    _exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    load_sessions_at_reporting(base_path, project_path, &ScanReport::default())
}

fn load_sessions_at_reporting(
    base_path: &str,
    project_path: &str,
    report: &ScanReport,
) -> Result<Vec<ClaudeSession>, String> {
    opencode::load_sessions_at_reporting(
        base_path,
        &db_file(),
        &to_store_scheme(project_path),
        report,
    )
    .map(|sessions| sessions.into_iter().map(rebrand_session).collect())
    .map_err(rebrand_error)
}

pub(crate) fn load_listed_sessions(
    project_path: &str,
) -> Result<Vec<opencode::sidebar::ListedSession>, String> {
    let base = get_base_path().ok_or_else(|| "Kilo not found".to_string())?;
    load_listed_sessions_at(&base, project_path)
}

fn load_listed_sessions_at(
    base: &str,
    project_path: &str,
) -> Result<Vec<opencode::sidebar::ListedSession>, String> {
    let sessions = opencode::sidebar::load_listed_sessions_at(
        base,
        &db_file(),
        &to_store_scheme(project_path),
    )
    .map_err(rebrand_error)?;
    Ok(sessions
        .into_iter()
        .map(|row| opencode::sidebar::ListedSession {
            session: rebrand_session(row.session),
            parent_session_id: row.parent_session_id.as_deref().map(to_kilo_scheme),
        })
        .collect())
}

/// Load messages for a Kilo session (`kilo://<project_id>/<session_id>` or
/// `kilo://<session_id>`).
pub fn load_messages(session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    let base_path = get_base_path().ok_or_else(|| "Kilo not found".to_string())?;
    load_messages_at(&base_path, session_path)
}

/// [`load_messages`] against an explicit store root.
pub fn load_messages_at(base_path: &str, session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    opencode::load_messages_at(base_path, &db_file(), &to_store_scheme(session_path))
        .map(|messages| {
            messages
                .into_iter()
                .map(|mut message| {
                    message.provider = Some(PROVIDER_ID.to_string());
                    message
                })
                .collect()
        })
        .map_err(rebrand_error)
}

/// Search across all Kilo sessions.
pub fn search(query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    let base_path = get_base_path().ok_or_else(|| "Kilo not found".to_string())?;
    search_at(&base_path, query, limit)
}

/// [`search`] against an explicit store root.
pub fn search_at(base_path: &str, query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    opencode::search_at(base_path, &db_file(), query, limit)
        .map(|messages| {
            messages
                .into_iter()
                .map(|mut message| {
                    message.provider = Some(PROVIDER_ID.to_string());
                    message.project_path = message.project_path.as_deref().map(to_kilo_scheme);
                    message
                })
                .collect()
        })
        .map_err(rebrand_error)
}

/// Child sessions (subagent runs) of a Kilo session, read from `kilo.db`.
pub fn load_child_sessions(project_id: &str, session_id: &str) -> Vec<ChildSession> {
    let Some(base_path) = get_base_path() else {
        return Vec::new();
    };
    load_child_sessions_at(&base_path, project_id, session_id)
}

/// [`load_child_sessions`] against an explicit store root.
pub fn load_child_sessions_at(
    base_path: &str,
    project_id: &str,
    session_id: &str,
) -> Vec<ChildSession> {
    opencode::load_child_sessions_from_db(base_path, &db_file(), project_id, session_id)
}

/// Messages of a Kilo session's descendant sessions (subagent runs), marked
/// as sidechain like the `OpenCode` reader's (#577).
pub fn load_subagent_messages(session_path: &str) -> Vec<ClaudeMessage> {
    let Some(base_path) = get_base_path() else {
        return Vec::new();
    };
    load_subagent_messages_at(&base_path, session_path)
}

/// [`load_subagent_messages`] against an explicit store root.
pub fn load_subagent_messages_at(base_path: &str, session_path: &str) -> Vec<ClaudeMessage> {
    rebrand_messages(opencode::load_subagent_messages_at(
        base_path,
        &db_file(),
        &to_store_scheme(session_path),
    ))
}

/// How a Kilo project's sessions nest (`kilo://<project_id>`), read from
/// `kilo.db` like the `OpenCode` reader's (#577).
pub fn load_session_tree(project_path: &str) -> SessionTree {
    let Some(base_path) = get_base_path() else {
        return SessionTree::default();
    };
    opencode::load_session_tree_at(&base_path, &db_file(), &to_store_scheme(project_path))
}

/// [`load_subagent_messages`] with the project's [`SessionTree`] already
/// loaded.
pub fn load_tree_subagent_messages(tree: &SessionTree, session_path: &str) -> Vec<ClaudeMessage> {
    let Some(base_path) = get_base_path() else {
        return Vec::new();
    };
    rebrand_messages(opencode::load_tree_subagent_messages_at(
        &base_path,
        &db_file(),
        tree,
        &to_store_scheme(session_path),
    ))
}

fn rebrand_messages(messages: Vec<ClaudeMessage>) -> Vec<ClaudeMessage> {
    messages
        .into_iter()
        .map(|mut message| {
            message.provider = Some(PROVIDER_ID.to_string());
            message
        })
        .collect()
}

/// Fixture helpers for a minimal `kilo.db`, shared with stats tests that need
/// a Kilo store behind `KILO_DB`.
#[cfg(test)]
pub(crate) mod test_support {
    use super::DB_FILE;
    use std::path::Path;

    pub(crate) fn create_test_db(dir: &Path) -> rusqlite::Connection {
        create_db(&dir.join(DB_FILE))
    }

    /// An `OpenCode`-core database at `path` (`kilo.db`, or `opencode.db`
    /// for an `OpenCode` store: the schema is the same).
    pub(crate) fn create_db(path: &Path) -> rusqlite::Connection {
        let conn = rusqlite::Connection::open(path).expect("open the OpenCode-core db");
        conn.execute_batch(
            "CREATE TABLE project (
                id TEXT PRIMARY KEY, worktree TEXT NOT NULL, name TEXT,
                time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
                sandboxes TEXT NOT NULL DEFAULT '[]', vcs TEXT, icon_url TEXT,
                icon_color TEXT, time_initialized INTEGER, commands TEXT
            );
            CREATE TABLE session (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, title TEXT NOT NULL,
                slug TEXT NOT NULL DEFAULT '', directory TEXT NOT NULL DEFAULT '',
                version TEXT NOT NULL DEFAULT '1.0', time_created INTEGER NOT NULL,
                time_updated INTEGER NOT NULL, parent_id TEXT, share_url TEXT,
                summary_additions INTEGER, summary_deletions INTEGER, summary_files INTEGER,
                summary_diffs TEXT, revert TEXT, permission TEXT, time_compacting INTEGER,
                time_archived INTEGER, workspace_id TEXT
            );
            CREATE TABLE message (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, time_created INTEGER NOT NULL,
                time_updated INTEGER NOT NULL, data TEXT NOT NULL
            );
            CREATE TABLE part (
                id TEXT PRIMARY KEY, message_id TEXT NOT NULL, session_id TEXT NOT NULL,
                time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL, data TEXT NOT NULL
            );",
        )
        .expect("create kilo tables");
        conn
    }

    /// One project, one session, one user text message.
    pub(crate) fn seed(conn: &rusqlite::Connection) {
        conn.execute(
            "INSERT INTO project (id, worktree, name, time_created, time_updated)
             VALUES ('proj1', '/tmp/kilo-project', 'kilo-project', 1700000000000, 1700000100000)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session (id, project_id, title, directory, time_created, time_updated)
             VALUES ('ses_001', 'proj1', 'Kilo session', '/tmp/kilo-project',
                     1700000000000, 1700000050000)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO message (id, session_id, time_created, time_updated, data)
             VALUES ('msg_001', 'ses_001', 1700000010000, 1700000010000,
                     '{\"role\":\"user\",\"time\":{\"created\":1700000010000}}')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO part (id, message_id, session_id, time_created, time_updated, data)
             VALUES ('prt_001', 'msg_001', 'ses_001', 1700000010000, 1700000010000,
                     '{\"type\":\"text\",\"text\":\"Hello from Kilo\"}')",
            [],
        )
        .unwrap();
    }

    /// Add an assistant reply to `ses_001` carrying token usage, so stats
    /// code has something to aggregate.
    pub(crate) fn seed_assistant_usage(conn: &rusqlite::Connection, input: u32, output: u32) {
        conn.execute(
            "INSERT INTO message (id, session_id, time_created, time_updated, data)
             VALUES ('msg_002', 'ses_001', 1700000020000, 1700000020000, ?1)",
            [format!(
                "{{\"role\":\"assistant\",\"modelID\":\"kilo-test-model\",\"providerID\":\"kilo\",\
                 \"time\":{{\"created\":1700000020000}},\
                 \"tokens\":{{\"input\":{input},\"output\":{output},\"reasoning\":0,\
                 \"cache\":{{\"read\":0,\"write\":0}}}},\"cost\":0.0}}"
            )],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO part (id, message_id, session_id, time_created, time_updated, data)
             VALUES ('prt_002', 'msg_002', 'ses_001', 1700000020000, 1700000020000,
                     '{\"type\":\"text\",\"text\":\"Hi from the Kilo assistant\"}')",
            [],
        )
        .unwrap();
    }

    /// Add a session `id` in `proj1` whose `parent_id` is `parent` (a
    /// subagent run), with one assistant reply carrying token usage.
    pub(crate) fn seed_subagent_run(
        conn: &rusqlite::Connection,
        id: &str,
        parent: &str,
        input: u32,
        output: u32,
    ) {
        conn.execute(
            "INSERT INTO session (id, project_id, title, directory, time_created, time_updated,
                                  parent_id)
             VALUES (?1, 'proj1', ?1, '/tmp/kilo-project', 1700000200000, 1700000200000, ?2)",
            rusqlite::params![id, parent],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO message (id, session_id, time_created, time_updated, data)
             VALUES (?1, ?2, 1700000210000, 1700000210000, ?3)",
            rusqlite::params![
                format!("msg_{id}"),
                id,
                format!(
                    "{{\"role\":\"assistant\",\"modelID\":\"kilo-test-model\",\
                     \"time\":{{\"created\":1700000210000}},\
                     \"tokens\":{{\"input\":{input},\"output\":{output}}}}}"
                )
            ],
        )
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{create_db, create_test_db, seed};
    use super::*;

    // Serial: these read `db_file()`, which `kilo_db_names_the_database`
    // changes through `KILO_DB`.
    #[test]
    #[serial_test::serial]
    fn scan_projects_rebrands_to_kilo_scheme() {
        let tmp = tempfile::tempdir().unwrap();
        let conn = create_test_db(tmp.path());
        seed(&conn);
        drop(conn);

        let projects = scan_projects_at(&tmp.path().to_string_lossy()).unwrap();
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].path, "kilo://proj1");
        assert_eq!(projects[0].provider.as_deref(), Some("kilo"));
        assert_eq!(projects[0].session_count, 1);
        assert_eq!(projects[0].name, "kilo-project");
    }

    #[test]
    #[serial_test::serial]
    fn sidebar_groups_and_pages_kilo_runs_with_kilo_ids() {
        let tmp = tempfile::tempdir().unwrap();
        let conn = create_test_db(tmp.path());
        seed(&conn);
        super::test_support::seed_subagent_run(&conn, "ses_child", "ses_001", 100, 50);
        super::test_support::seed_subagent_run(&conn, "ses_leaf", "ses_child", 30, 10);
        let base = tmp.path().to_str().unwrap();
        let projects = scan_projects_at(base).unwrap();
        let projects = opencode::sidebar::group_projects(base, DB_FILE, projects);
        assert_eq!(
            (
                projects[0].project.session_count,
                projects[0].subagent_count
            ),
            (1, 2)
        );
        let sessions = load_listed_sessions_at(base, "kilo://proj1").unwrap();
        let page = opencode::sidebar::page(sessions, 0, 1, Some("kilo://proj1/ses_leaf"));
        assert_eq!((page.total, page.next_offset, page.has_more), (1, 1, false));
        assert_eq!(page.sessions[1].file_path, "kilo://proj1/ses_child");
        assert_eq!(
            page.sessions[1].parent_session_id.as_deref(),
            Some("kilo://ses_001")
        );
        assert_eq!(
            page.sessions[2].parent_session_id.as_deref(),
            Some("kilo://ses_child")
        );
        assert_eq!(page.sessions[2].provider.as_deref(), Some("kilo"));
    }

    #[test]
    #[serial_test::serial]
    fn load_sessions_and_messages_roundtrip_through_kilo_scheme() {
        let tmp = tempfile::tempdir().unwrap();
        let conn = create_test_db(tmp.path());
        seed(&conn);
        drop(conn);

        let base = tmp.path().to_string_lossy().to_string();
        let sessions = load_sessions_at(&base, "kilo://proj1", false).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, "kilo://ses_001");
        assert_eq!(sessions[0].file_path, "kilo://proj1/ses_001");
        assert_eq!(sessions[0].provider.as_deref(), Some("kilo"));

        let messages = load_messages_at(&base, &sessions[0].file_path).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].provider.as_deref(), Some("kilo"));
        let text = messages[0]
            .content
            .as_ref()
            .and_then(|c| c.get(0))
            .and_then(|item| item.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or_default();
        assert!(
            text.contains("Hello from Kilo"),
            "unexpected content: {text}"
        );
    }

    #[test]
    #[serial_test::serial]
    fn search_finds_text_and_rebrands_provider() {
        let tmp = tempfile::tempdir().unwrap();
        let conn = create_test_db(tmp.path());
        seed(&conn);
        drop(conn);

        let results = search_at(&tmp.path().to_string_lossy(), "hello from kilo", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].provider.as_deref(), Some("kilo"));
    }

    /// #672: hits carry the `kilo://` path scan emits for their project, so
    /// same-named worktrees stay apart.
    #[test]
    #[serial_test::serial]
    fn search_hits_carry_scanned_kilo_project_path() {
        let tmp = tempfile::tempdir().unwrap();
        let conn = create_test_db(tmp.path());
        for (project, worktree, session) in [
            ("proj_a", "/work/client-a/app", "ses_a"),
            ("proj_b", "/work/client-b/app", "ses_b"),
        ] {
            conn.execute(
                "INSERT INTO project (id, worktree, time_created, time_updated)
                 VALUES (?1, ?2, 1700000000000, 1700000000000)",
                rusqlite::params![project, worktree],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO session (id, project_id, title, directory, time_created, time_updated)
                 VALUES (?1, ?2, 't', ?3, 1700000000000, 1700000000000)",
                rusqlite::params![session, project, worktree],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO message (id, session_id, time_created, time_updated, data)
                 VALUES (?1, ?1, 1700000010000, 1700000010000,
                         '{\"role\":\"user\",\"time\":{\"created\":1700000010000}}')",
                [session],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO part (id, message_id, session_id, time_created, time_updated, data)
                 VALUES (?1, ?1, ?1, 1700000010000, 1700000010000,
                         '{\"type\":\"text\",\"text\":\"needle\"}')",
                [session],
            )
            .unwrap();
        }
        drop(conn);
        let base = tmp.path().to_string_lossy().to_string();

        let projects = scan_projects_at(&base).unwrap();
        let results = search_at(&base, "needle", 10).unwrap();
        assert_eq!(results.len(), 2);
        for (worktree, session) in [
            ("/work/client-a/app", "ses_a"),
            ("/work/client-b/app", "ses_b"),
        ] {
            let project = projects.iter().find(|p| p.actual_path == worktree).unwrap();
            assert_eq!(project.name, "app");
            let hit = results.iter().find(|m| m.session_id == session).unwrap();
            assert_eq!(hit.project_path.as_deref(), Some(project.path.as_str()));
        }
        assert_ne!(results[0].project_path, results[1].project_path);
    }

    #[test]
    #[serial_test::serial]
    fn errors_are_reported_in_kilo_terms() {
        let tmp = tempfile::tempdir().unwrap();
        drop(create_test_db(tmp.path()));
        let base = tmp.path().to_string_lossy().to_string();

        let err = load_sessions_at(&base, "kilo://../escape", false).unwrap_err();
        assert_eq!(err, "Invalid Kilo Code project path: kilo://../escape");

        let err = load_messages_at(&base, "kilo://no-session-part").unwrap_err();
        assert_eq!(
            err,
            "Invalid Kilo Code session path: kilo://no-session-part"
        );

        let err = scan_projects_at("relative/kilo").unwrap_err();
        assert!(
            err.contains("Kilo Code base path") && !err.contains("OpenCode"),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[serial_test::serial]
    fn child_sessions_read_from_kilo_db() {
        let tmp = tempfile::tempdir().unwrap();
        let conn = create_test_db(tmp.path());
        seed(&conn);
        conn.execute(
            "INSERT INTO session (id, project_id, title, directory, time_created, time_updated, parent_id)
             VALUES ('ses_child', 'proj1', 'Subagent run', '/tmp/kilo-project',
                     1700000060000, 1700000070000, 'ses_001')",
            [],
        )
        .unwrap();
        drop(conn);

        let children = load_child_sessions_at(&tmp.path().to_string_lossy(), "proj1", "ses_001");
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id, "ses_child");
    }

    #[test]
    #[ignore = "requires a real Kilo store on this machine (KILO_DB or ~/.local/share/kilo)"]
    fn real_store_smoke_scan_and_load() {
        let Some(base) = get_base_path() else {
            eprintln!("no Kilo store on this machine; nothing to verify");
            return;
        };
        let projects = scan_projects_at(&base).expect("scan real store");
        eprintln!("projects: {}", projects.len());
        assert!(!projects.is_empty(), "real Kilo store should have projects");

        let first = &projects[0];
        let sessions = load_sessions_at(&base, &first.path, false).expect("load sessions");
        eprintln!("sessions in {}: {}", first.name, sessions.len());
        assert!(!sessions.is_empty(), "first project should have sessions");

        let messages = load_messages_at(&base, &sessions[0].file_path).expect("load messages");
        eprintln!("messages in first session: {}", messages.len());
        assert!(!messages.is_empty(), "first session should have messages");
        assert_eq!(messages[0].provider.as_deref(), Some("kilo"));
    }

    /// `KILO_DB` names the database file; its directory is the store root
    /// and its own file name is read, not `kilo.db`.
    #[test]
    #[serial_test::serial]
    fn kilo_db_names_the_database() {
        struct EnvVarGuard(&'static str, Option<std::ffi::OsString>);
        impl EnvVarGuard {
            fn set(key: &'static str, value: &std::ffi::OsStr) -> Self {
                let previous = std::env::var_os(key);
                std::env::set_var(key, value);
                Self(key, previous)
            }
        }
        impl Drop for EnvVarGuard {
            fn drop(&mut self) {
                match self.1.take() {
                    Some(value) => std::env::set_var(self.0, value),
                    None => std::env::remove_var(self.0),
                }
            }
        }

        let tmp = tempfile::tempdir().unwrap();
        let db = tmp.path().join("work.db");
        let conn = create_db(&db);
        seed(&conn);
        conn.execute(
            "INSERT INTO session (id, project_id, slug, title, directory, time_created, time_updated)
             VALUES ('ses_migrated_0123', 'proj1', 'legacy-task-1', 'old', '/w', 1, 1)",
            [],
        )
        .unwrap();
        drop(conn);

        let _guard = EnvVarGuard::set("KILO_DB", db.as_os_str());
        let info = detect().expect("the KILO_DB store should be detected");
        assert_eq!(info.base_path, tmp.path().to_string_lossy());
        assert!(info.is_available);
        assert_eq!(db_file(), "work.db");
        assert!(!scan_projects().unwrap().is_empty());
        assert_eq!(
            migrated_task_ids(),
            HashSet::from(["legacy-task-1".to_string()])
        );
    }
}
