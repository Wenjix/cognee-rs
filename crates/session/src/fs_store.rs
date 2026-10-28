//! Filesystem-backed session store — compatible with Python's `FSCacheAdapter`.
//!
//! Each session is stored as a JSON file containing an array of Q&A entries.
//! The file path mirrors the Python key format:
//! `{base_dir}/{user_id}/{session_id}.json`
//!
//! `user_id` and `session_id` are caller-supplied: any identifier that could
//! leave the directory it is joined to (empty, `.`, `..`, a leading `%`, or
//! containing `/` or NUL — plus the Windows-forbidden characters) is stored
//! as `%` + its percent-encoding instead, see [`path_component`]. Safe ids
//! map to themselves, so files written before this encoding keep their names.
//!
//! Python uses `diskcache` which stores values in a SQLite database under
//! `.cognee_fs_cache/sessions_db/`. Our implementation uses plain JSON files
//! for simplicity while keeping the same logical key structure
//! (`agent_sessions:{user_id}:{session_id}`) and identical JSON entry format
//! so data written by one SDK can be read by the other at the API level.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use chrono::Utc;
use tokio::fs;
use uuid::Uuid;

use crate::error::SessionError;
use crate::session_store::{SessionQAUpdate, SessionStore};
use crate::types::{SessionQAEntry, SessionTraceStep, UsedGraphElementIds};

/// JSON layout of a single entry on disk — matches the Python `SessionQAEntry`
/// model fields so the serialised form is cross-SDK compatible.
#[derive(serde::Serialize, serde::Deserialize)]
struct FsQAEntry {
    time: String,
    question: String,
    context: String,
    answer: String,
    qa_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    feedback_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    feedback_score: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    used_graph_element_ids: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    memify_metadata: Option<serde_json::Value>,
}

/// Filesystem-backed session store.
pub struct FsSessionStore {
    base_dir: PathBuf,
    /// How long a session file lives after its last write; `None` never
    /// expires.
    session_ttl: Option<std::time::Duration>,
}

impl FsSessionStore {
    /// Create a new filesystem store rooted at `base_dir`.
    ///
    /// The directory is created on first write if it does not exist.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
            session_ttl: None,
        }
    }

    /// Expire a session's files (Q&A entries, trace steps, graph context)
    /// this many seconds after their last write, as Python's `FSCacheAdapter`
    /// does with `session_ttl_seconds` (its `diskcache` expiry, refreshed on
    /// every write). An expired file is deleted when it is next read. `0`
    /// never expires them, the behaviour of a store built without this call.
    #[must_use]
    pub fn with_session_ttl_seconds(mut self, seconds: u64) -> Self {
        self.session_ttl = (seconds > 0).then(|| std::time::Duration::from_secs(seconds));
        self
    }

    /// Delete `path` if it was last written longer ago than the session TTL,
    /// so the read that follows finds nothing.
    async fn expire(&self, path: &Path) -> Result<(), SessionError> {
        let Some(ttl) = self.session_ttl else {
            return Ok(());
        };
        let modified = match fs::metadata(path).await.and_then(|m| m.modified()) {
            Ok(modified) => modified,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(SessionError::StoreError(format!("fs stat error: {e}"))),
        };
        // A clock that went backwards reads as "just written".
        let age = modified.elapsed().unwrap_or_default();
        if age > ttl {
            match fs::remove_file(path).await {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(SessionError::StoreError(format!("fs delete error: {e}"))),
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// `id` as a single file-name component that cannot leave the directory it is
/// joined to. A safe identifier maps to itself, so session files written before
/// this encoding keep their names. Any other identifier is written as `%`
/// followed by its percent-encoding (`%` and every unsafe character escaped as
/// `%XX` per UTF-8 byte); a raw name never starts with `%`, so the mapping is
/// injective.
///
/// The three on-disk layouts — `{c}.json` (QA), `{c}.traces.json` (trace
/// steps) and `_graph_context_{c}.json` (graph context) — also force the
/// `.traces` suffix and `_graph_context_` prefix to encode: raw names never
/// end with `.traces`, never start with `_graph_context_` or `%`, and an
/// encoded name starts with `%` and contains no `.` or `_` at all, so no QA
/// name can equal a trace or context name (and those two cannot collide
/// either). The checks are ASCII case-insensitive because on
/// case-insensitive filesystems (macOS, Windows) `X.TRACES.json` is the
/// same entry as `x.traces.json`. Without that, session `s`'s trace file
/// would be session `s.traces`'s QA file — and `delete_session("s")` would
/// erase another session's data.
fn path_component(id: &str) -> Cow<'_, str> {
    fn is_unsafe_char(c: char) -> bool {
        if c == '/' || c == '\0' {
            return true;
        }
        if cfg!(windows)
            && (matches!(c, '\\' | ':' | '<' | '>' | '"' | '|' | '?' | '*') || c.is_ascii_control())
        {
            return true;
        }
        false
    }
    let lower = id.to_ascii_lowercase();
    let unsafe_id = id.is_empty()
        || id == "."
        || id == ".."
        || id.starts_with('%')
        || lower.ends_with(".traces")
        || lower.starts_with("_graph_context_")
        || id.chars().any(is_unsafe_char);
    if !unsafe_id {
        return Cow::Borrowed(id);
    }
    let mut out = String::with_capacity(id.len() + 1);
    out.push('%');
    for c in id.chars() {
        // `.` and `_` escape in the encoded form too: that way an encoded
        // name contains neither, so no encoded QA/trace/context name can
        // alias another layout's name (see the doc comment).
        if c == '%' || c == '.' || c == '_' || is_unsafe_char(c) {
            let mut buf = [0u8; 4];
            for b in c.encode_utf8(&mut buf).as_bytes() {
                out.push_str(&format!("%{b:02X}"));
            }
        } else {
            out.push(c);
        }
    }
    Cow::Owned(out)
}

fn session_file(base: &Path, user_id: Option<&str>, session_id: &str) -> PathBuf {
    let uid = user_id.unwrap_or("default");
    base.join(path_component(uid).as_ref())
        .join(format!("{}.json", path_component(session_id)))
}

async fn load_entries(path: &Path) -> Result<Vec<FsQAEntry>, SessionError> {
    match fs::read_to_string(path).await {
        Ok(contents) if !contents.is_empty() => serde_json::from_str(&contents)
            .map_err(|e| SessionError::StoreError(format!("json parse error: {e}"))),
        Ok(_) => Ok(vec![]),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
        Err(e) => Err(SessionError::StoreError(format!("fs read error: {e}"))),
    }
}

async fn save_entries(path: &Path, entries: &[FsQAEntry]) -> Result<(), SessionError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .await
            .map_err(|e| SessionError::StoreError(format!("fs mkdir error: {e}")))?;
    }
    let json = serde_json::to_string_pretty(entries)
        .map_err(|e| SessionError::StoreError(format!("json error: {e}")))?;
    fs::write(path, json)
        .await
        .map_err(|e| SessionError::StoreError(format!("fs write error: {e}")))?;
    Ok(())
}

fn build_entry(qa_id: &str, question: &str, answer: &str, context: Option<&str>) -> FsQAEntry {
    FsQAEntry {
        time: Utc::now().to_rfc3339(),
        question: question.to_string(),
        context: context.unwrap_or("").to_string(),
        answer: answer.to_string(),
        qa_id: qa_id.to_string(),
        feedback_text: None,
        feedback_score: None,
        used_graph_element_ids: None,
        memify_metadata: None,
    }
}

fn fs_entry_to_domain(e: &FsQAEntry, session_id: &str) -> SessionQAEntry {
    let used_graph_element_ids = e
        .used_graph_element_ids
        .as_ref()
        .and_then(|v| serde_json::from_value::<UsedGraphElementIds>(v.clone()).ok());
    let memify_metadata = e
        .memify_metadata
        .as_ref()
        .and_then(|v| serde_json::from_value::<HashMap<String, bool>>(v.clone()).ok());

    SessionQAEntry {
        id: Uuid::parse_str(&e.qa_id).unwrap_or_else(|_| Uuid::new_v4()),
        session_id: session_id.to_string(),
        user_id: None,
        question: e.question.clone(),
        answer: e.answer.clone(),
        context: if e.context.is_empty() {
            None
        } else {
            Some(e.context.clone())
        },
        created_at: chrono::DateTime::parse_from_rfc3339(&e.time)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
        feedback_text: e.feedback_text.clone(),
        feedback_score: e.feedback_score,
        used_graph_element_ids,
        memify_metadata,
    }
}

fn graph_context_file(base: &Path, user_id: Option<&str>, session_id: &str) -> PathBuf {
    let uid = user_id.unwrap_or("default");
    base.join(path_component(uid).as_ref()).join(format!(
        "_graph_context_{}.json",
        path_component(session_id)
    ))
}

fn trace_session_file(base: &Path, user_id: &str, session_id: &str) -> PathBuf {
    base.join(path_component(user_id).as_ref())
        .join(format!("{}.traces.json", path_component(session_id)))
}

/// Remove a per-session record file for `delete_session`. `NotFound` is the
/// only tolerable miss; any other I/O error is returned. Before removing,
/// the file is read once: if it parses as a Q&A list it is another
/// session's QA file, reachable here only through the pre-encoding layout
/// ambiguity (a session literally named `x.traces` or `_graph_context_x`
/// had its QA file at this very path before [`path_component`] existed),
/// so it must be left alone. A file that parses as neither a Q&A list nor
/// the expected record kind cannot be a sibling's data and is removed.
async fn remove_record_file(path: &Path) -> Result<(), SessionError> {
    match fs::read_to_string(path).await {
        Ok(contents) if serde_json::from_str::<Vec<FsQAEntry>>(&contents).is_ok() => {
            tracing::debug!(
                target: "cognee.session",
                path = %path.display(),
                "leaving a legacy Q&A file in place during session deletion"
            );
            return Ok(());
        }
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(SessionError::StoreError(format!(
                "fs read error before delete: {e}"
            )));
        }
    }
    match fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(SessionError::StoreError(format!(
            "fs delete record-file error: {e}"
        ))),
    }
}

async fn load_trace_steps(path: &Path) -> Result<Vec<SessionTraceStep>, SessionError> {
    match fs::read_to_string(path).await {
        Ok(contents) if !contents.is_empty() => serde_json::from_str(&contents)
            .map_err(|e| SessionError::StoreError(format!("json parse error: {e}"))),
        Ok(_) => Ok(vec![]),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
        Err(e) => Err(SessionError::StoreError(format!("fs read error: {e}"))),
    }
}

async fn save_trace_steps(path: &Path, steps: &[SessionTraceStep]) -> Result<(), SessionError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .await
            .map_err(|e| SessionError::StoreError(format!("fs mkdir error: {e}")))?;
    }
    let json = serde_json::to_string_pretty(steps)
        .map_err(|e| SessionError::StoreError(format!("json error: {e}")))?;
    fs::write(path, json)
        .await
        .map_err(|e| SessionError::StoreError(format!("fs write error: {e}")))?;
    Ok(())
}

/// Apply a `SessionQAUpdate` to an `FsQAEntry` in place.
fn apply_update_to_fs_entry(entry: &mut FsQAEntry, updates: &SessionQAUpdate) {
    if let Some(ref q) = updates.question {
        entry.question = q.clone();
    }
    if let Some(ref a) = updates.answer {
        entry.answer = a.clone();
    }
    if let Some(ref ctx) = updates.context {
        entry.context = ctx.as_deref().unwrap_or("").to_string();
    }
    if let Some(ref ft) = updates.feedback_text {
        entry.feedback_text = ft.clone();
    }
    if let Some(ref fs) = updates.feedback_score {
        entry.feedback_score = *fs;
    }
    if let Some(ref ids) = updates.used_graph_element_ids {
        entry.used_graph_element_ids = ids
            .as_ref()
            .map(|v| serde_json::to_value(v).unwrap_or_default());
    }
    if let Some(ref mm) = updates.memify_metadata {
        entry.memify_metadata = mm
            .as_ref()
            .map(|v| serde_json::to_value(v).unwrap_or_default());
    }
}

// ---------------------------------------------------------------------------
// trait impl
// ---------------------------------------------------------------------------

#[async_trait]
impl SessionStore for FsSessionStore {
    async fn create_qa_entry(
        &self,
        session_id: &str,
        user_id: Option<&str>,
        question: &str,
        answer: &str,
        context: Option<&str>,
    ) -> Result<String, SessionError> {
        let path = session_file(&self.base_dir, user_id, session_id);
        let qa_id = Uuid::new_v4().to_string();
        let entry = build_entry(&qa_id, question, answer, context);

        self.expire(&path).await?;
        let mut entries = load_entries(&path).await?;
        entries.push(entry);
        save_entries(&path, &entries).await?;

        Ok(qa_id)
    }

    async fn get_latest_qa_entries(
        &self,
        session_id: &str,
        user_id: Option<&str>,
        last_n: usize,
    ) -> Result<Vec<SessionQAEntry>, SessionError> {
        let path = session_file(&self.base_dir, user_id, session_id);
        self.expire(&path).await?;
        let entries = load_entries(&path).await?;

        let start = entries.len().saturating_sub(last_n);
        Ok(entries[start..]
            .iter()
            .map(|e| fs_entry_to_domain(e, session_id))
            .collect())
    }

    async fn get_all_qa_entries(
        &self,
        session_id: &str,
        user_id: Option<&str>,
    ) -> Result<Vec<SessionQAEntry>, SessionError> {
        let path = session_file(&self.base_dir, user_id, session_id);
        self.expire(&path).await?;
        let entries = load_entries(&path).await?;
        Ok(entries
            .iter()
            .map(|e| fs_entry_to_domain(e, session_id))
            .collect())
    }

    async fn delete_session(
        &self,
        session_id: &str,
        user_id: Option<&str>,
    ) -> Result<bool, SessionError> {
        let path = session_file(&self.base_dir, user_id, session_id);
        let existed = match fs::remove_file(&path).await {
            Ok(()) => true,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
            Err(e) => return Err(SessionError::StoreError(format!("fs delete error: {e}"))),
        };

        // Also remove the graph context file for this session. NotFound is
        // the only tolerable miss — anything else means erasure is
        // incomplete and must not report success.
        let gc_path = graph_context_file(&self.base_dir, user_id, session_id);
        remove_record_file(&gc_path).await?;

        // And the agent-trace file (same NotFound-only semantics).
        let trace_path =
            trace_session_file(&self.base_dir, user_id.unwrap_or("default"), session_id);
        remove_record_file(&trace_path).await?;

        Ok(existed)
    }

    async fn delete_qa_entry(
        &self,
        session_id: &str,
        user_id: Option<&str>,
        qa_id: &str,
    ) -> Result<bool, SessionError> {
        let path = session_file(&self.base_dir, user_id, session_id);
        self.expire(&path).await?;
        let mut entries = load_entries(&path).await?;
        let before = entries.len();
        entries.retain(|e| e.qa_id != qa_id);

        if entries.len() == before {
            return Ok(false);
        }

        if entries.is_empty() {
            // Clean up empty file
            let _ = fs::remove_file(&path).await;
        } else {
            save_entries(&path, &entries).await?;
        }

        Ok(true)
    }

    async fn prune(&self) -> Result<(), SessionError> {
        // Remove the entire base directory and recreate it empty.
        // Errors are ignored when the directory does not exist yet.
        match fs::remove_dir_all(&self.base_dir).await {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(SessionError::StoreError(format!(
                    "fs prune remove_dir_all error: {e}"
                )));
            }
        }
        fs::create_dir_all(&self.base_dir)
            .await
            .map_err(|e| SessionError::StoreError(format!("fs prune create_dir_all error: {e}")))?;
        Ok(())
    }

    async fn update_qa_entry(
        &self,
        session_id: &str,
        user_id: Option<&str>,
        qa_id: &str,
        updates: SessionQAUpdate,
    ) -> Result<bool, SessionError> {
        let path = session_file(&self.base_dir, user_id, session_id);
        self.expire(&path).await?;
        let mut entries = load_entries(&path).await?;

        let Some(entry) = entries.iter_mut().find(|e| e.qa_id == qa_id) else {
            return Ok(false);
        };

        apply_update_to_fs_entry(entry, &updates);
        save_entries(&path, &entries).await?;
        Ok(true)
    }

    async fn get_graph_context(
        &self,
        session_id: &str,
        user_id: Option<&str>,
    ) -> Result<Option<String>, SessionError> {
        let path = graph_context_file(&self.base_dir, user_id, session_id);
        self.expire(&path).await?;
        match fs::read_to_string(&path).await {
            Ok(contents) if !contents.is_empty() => {
                // Stored as a JSON string
                let ctx: String = serde_json::from_str(&contents)
                    .map_err(|e| SessionError::StoreError(format!("json parse error: {e}")))?;
                Ok(Some(ctx))
            }
            Ok(_) => Ok(None),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(SessionError::StoreError(format!("fs read error: {e}"))),
        }
    }

    async fn set_graph_context(
        &self,
        session_id: &str,
        user_id: Option<&str>,
        context: &str,
    ) -> Result<(), SessionError> {
        let path = graph_context_file(&self.base_dir, user_id, session_id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| SessionError::StoreError(format!("fs mkdir error: {e}")))?;
        }
        let json = serde_json::to_string_pretty(context)
            .map_err(|e| SessionError::StoreError(format!("json error: {e}")))?;
        fs::write(&path, json)
            .await
            .map_err(|e| SessionError::StoreError(format!("fs write error: {e}")))?;
        Ok(())
    }

    async fn save_trace_step(
        &self,
        user_id: &str,
        session_id: &str,
        step: SessionTraceStep,
    ) -> Result<String, SessionError> {
        let path = trace_session_file(&self.base_dir, user_id, session_id);
        self.expire(&path).await?;
        let mut steps = load_trace_steps(&path).await?;
        let trace_id = step.trace_id.clone();
        steps.push(step);
        save_trace_steps(&path, &steps).await?;
        Ok(trace_id)
    }

    async fn read_trace_steps(
        &self,
        user_id: &str,
        session_id: &str,
    ) -> Result<Vec<SessionTraceStep>, SessionError> {
        let path = trace_session_file(&self.base_dir, user_id, session_id);
        self.expire(&path).await?;
        load_trace_steps(&path).await
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
mod tests {
    use super::*;

    /// Push `path`'s last write `age` into the past.
    fn age_file(path: &Path, age: std::time::Duration) {
        let file = std::fs::File::options().write(true).open(path).unwrap();
        file.set_modified(std::time::SystemTime::now() - age)
            .unwrap();
    }

    /// With a TTL, a session file not written for longer than it reads as
    /// gone (and is deleted); a fresh one, or any file without a TTL, stays.
    #[tokio::test]
    async fn session_ttl_expires_files_not_written_within_it() {
        let dir = tempfile::tempdir().unwrap();
        let store = FsSessionStore::new(dir.path()).with_session_ttl_seconds(60);
        store
            .create_qa_entry("s", Some("u"), "q", "a", None)
            .await
            .unwrap();
        store
            .set_graph_context("s", Some("u"), "ctx")
            .await
            .unwrap();
        assert_eq!(
            store
                .get_all_qa_entries("s", Some("u"))
                .await
                .unwrap()
                .len(),
            1
        );

        let qa = session_file(dir.path(), Some("u"), "s");
        let gc = graph_context_file(dir.path(), Some("u"), "s");
        age_file(&qa, std::time::Duration::from_secs(120));
        age_file(&gc, std::time::Duration::from_secs(120));

        let untimed = FsSessionStore::new(dir.path());
        assert_eq!(
            untimed
                .get_all_qa_entries("s", Some("u"))
                .await
                .unwrap()
                .len(),
            1,
            "without a TTL nothing expires"
        );

        assert!(
            store
                .get_all_qa_entries("s", Some("u"))
                .await
                .unwrap()
                .is_empty()
        );
        assert!(!qa.exists(), "the expired file is deleted");
        assert_eq!(store.get_graph_context("s", Some("u")).await.unwrap(), None);
    }

    #[tokio::test]
    async fn create_and_read_entry_has_no_feedback() {
        let dir = tempfile::tempdir().expect("tempdir creation must succeed");
        let store = FsSessionStore::new(dir.path());

        let qa_id = store
            .create_qa_entry("sess1", None, "q1?", "a1", None)
            .await
            .expect("create should succeed");

        let entries = store
            .get_all_qa_entries("sess1", None)
            .await
            .expect("get should succeed");

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id.to_string(), qa_id);
        assert!(entries[0].feedback_text.is_none());
        assert!(entries[0].feedback_score.is_none());
        assert!(entries[0].used_graph_element_ids.is_none());
        assert!(entries[0].memify_metadata.is_none());
    }

    #[tokio::test]
    async fn update_qa_entry_sets_feedback_fields() {
        let dir = tempfile::tempdir().expect("tempdir creation must succeed");
        let store = FsSessionStore::new(dir.path());

        let qa_id = store
            .create_qa_entry("sess1", None, "q1?", "a1", None)
            .await
            .expect("create should succeed");

        let updated = store
            .update_qa_entry(
                "sess1",
                None,
                &qa_id,
                SessionQAUpdate {
                    feedback_text: Some(Some("Great answer!".to_string())),
                    feedback_score: Some(Some(5)),
                    ..Default::default()
                },
            )
            .await
            .expect("update should succeed");

        assert!(updated);

        let entries = store
            .get_all_qa_entries("sess1", None)
            .await
            .expect("get should succeed");

        assert_eq!(entries[0].feedback_text.as_deref(), Some("Great answer!"));
        assert_eq!(entries[0].feedback_score, Some(5));
    }

    #[tokio::test]
    async fn update_qa_entry_clears_feedback() {
        let dir = tempfile::tempdir().expect("tempdir creation must succeed");
        let store = FsSessionStore::new(dir.path());

        let qa_id = store
            .create_qa_entry("sess1", None, "q1?", "a1", None)
            .await
            .expect("create should succeed");

        // Set feedback
        store
            .update_qa_entry(
                "sess1",
                None,
                &qa_id,
                SessionQAUpdate {
                    feedback_text: Some(Some("Good".to_string())),
                    feedback_score: Some(Some(4)),
                    ..Default::default()
                },
            )
            .await
            .expect("set should succeed");

        // Clear feedback
        store
            .update_qa_entry(
                "sess1",
                None,
                &qa_id,
                SessionQAUpdate {
                    feedback_text: Some(None),
                    feedback_score: Some(None),
                    ..Default::default()
                },
            )
            .await
            .expect("clear should succeed");

        let entries = store
            .get_all_qa_entries("sess1", None)
            .await
            .expect("get should succeed");

        assert!(entries[0].feedback_text.is_none());
        assert!(entries[0].feedback_score.is_none());
    }

    #[tokio::test]
    async fn update_qa_entry_not_found() {
        let dir = tempfile::tempdir().expect("tempdir creation must succeed");
        let store = FsSessionStore::new(dir.path());

        store
            .create_qa_entry("sess1", None, "q1?", "a1", None)
            .await
            .expect("create should succeed");

        let updated = store
            .update_qa_entry("sess1", None, "nonexistent", SessionQAUpdate::default())
            .await
            .expect("update call should succeed even when not found");

        assert!(!updated);
    }

    #[tokio::test]
    async fn graph_context_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir creation must succeed");
        let store = FsSessionStore::new(dir.path());

        // Initially no context
        let ctx = store
            .get_graph_context("sess1", None)
            .await
            .expect("get should succeed");
        assert!(ctx.is_none());

        // Set context
        store
            .set_graph_context("sess1", None, "Some graph knowledge")
            .await
            .expect("set should succeed");

        // Read it back
        let ctx = store
            .get_graph_context("sess1", None)
            .await
            .expect("get should succeed");
        assert_eq!(ctx.as_deref(), Some("Some graph knowledge"));

        // Overwrite
        store
            .set_graph_context("sess1", None, "Updated graph knowledge")
            .await
            .expect("set should succeed");

        let ctx = store
            .get_graph_context("sess1", None)
            .await
            .expect("get should succeed");
        assert_eq!(ctx.as_deref(), Some("Updated graph knowledge"));
    }

    #[tokio::test]
    async fn update_qa_entry_with_graph_element_ids() {
        let dir = tempfile::tempdir().expect("tempdir creation must succeed");
        let store = FsSessionStore::new(dir.path());

        let qa_id = store
            .create_qa_entry("sess1", None, "q1?", "a1", None)
            .await
            .expect("create should succeed");

        let ids = UsedGraphElementIds {
            node_ids: vec!["node-1".to_string(), "node-2".to_string()],
            edge_ids: vec!["edge-1".to_string()],
        };

        store
            .update_qa_entry(
                "sess1",
                None,
                &qa_id,
                SessionQAUpdate {
                    used_graph_element_ids: Some(Some(ids)),
                    ..Default::default()
                },
            )
            .await
            .expect("update should succeed");

        let entries = store
            .get_all_qa_entries("sess1", None)
            .await
            .expect("get should succeed");

        let graph_ids = entries[0]
            .used_graph_element_ids
            .as_ref()
            .expect("should have graph element ids");
        assert_eq!(graph_ids.node_ids, vec!["node-1", "node-2"]);
        assert_eq!(graph_ids.edge_ids, vec!["edge-1"]);
    }

    #[tokio::test]
    async fn update_qa_entry_with_memify_metadata() {
        let dir = tempfile::tempdir().expect("tempdir creation must succeed");
        let store = FsSessionStore::new(dir.path());

        let qa_id = store
            .create_qa_entry("sess1", None, "q1?", "a1", None)
            .await
            .expect("create should succeed");

        let mut meta = HashMap::new();
        meta.insert("feedback_weights_applied".to_string(), false);

        store
            .update_qa_entry(
                "sess1",
                None,
                &qa_id,
                SessionQAUpdate {
                    memify_metadata: Some(Some(meta)),
                    ..Default::default()
                },
            )
            .await
            .expect("update should succeed");

        let entries = store
            .get_all_qa_entries("sess1", None)
            .await
            .expect("get should succeed");

        let mm = entries[0]
            .memify_metadata
            .as_ref()
            .expect("should have memify metadata");
        assert_eq!(mm.get("feedback_weights_applied"), Some(&false));
    }

    #[tokio::test]
    async fn delete_session_also_removes_graph_context() {
        let dir = tempfile::tempdir().expect("tempdir creation must succeed");
        let store = FsSessionStore::new(dir.path());

        store
            .create_qa_entry("sess1", None, "q1?", "a1", None)
            .await
            .expect("create should succeed");

        store
            .set_graph_context("sess1", None, "some context")
            .await
            .expect("set should succeed");

        let deleted = store
            .delete_session("sess1", None)
            .await
            .expect("delete should succeed");
        assert!(deleted);

        let ctx = store
            .get_graph_context("sess1", None)
            .await
            .expect("get should succeed");
        assert!(ctx.is_none());
    }

    // ── path_component traversal containment (M27–M31) ──────────────────────

    /// `path` must be `base/<one normal dir>/<one normal file name>` — nothing
    /// may escape `base` or add intermediate components.
    fn assert_two_normal_components(base: &Path, path: &Path) {
        let rel = path.strip_prefix(base).expect("path must stay under base");
        let comps: Vec<_> = rel.components().collect();
        assert_eq!(comps.len(), 2, "{rel:?}");
        for c in &comps {
            assert!(matches!(c, std::path::Component::Normal(_)), "{c:?}");
        }
    }

    #[test]
    fn unsafe_session_and_user_ids_cannot_leave_the_base_dir() {
        let base = Path::new("/base");
        for sid in [
            "../escape",
            "../../etc/x",
            "/abs/path",
            "a/b",
            "..",
            ".",
            "",
        ] {
            assert_two_normal_components(base, &session_file(base, Some("u1"), sid));
            assert_two_normal_components(base, &graph_context_file(base, Some("u1"), sid));
            assert_two_normal_components(base, &trace_session_file(base, "u1", sid));
        }
        for uid in ["..", "../other", "/abs", "", "a/b"] {
            assert_two_normal_components(base, &session_file(base, Some(uid), "s1"));
            assert_two_normal_components(base, &graph_context_file(base, Some(uid), "s1"));
            assert_two_normal_components(base, &trace_session_file(base, uid, "s1"));
        }
    }

    #[test]
    fn safe_ids_keep_their_legacy_file_names() {
        let base = Path::new("/base");
        let uuid = Uuid::new_v4().to_string();
        for sid in ["s1", uuid.as_str(), "user%40x"] {
            assert_eq!(
                session_file(base, Some("u"), sid),
                base.join("u").join(format!("{sid}.json"))
            );
            assert_eq!(
                graph_context_file(base, Some("u"), sid),
                base.join("u").join(format!("_graph_context_{sid}.json"))
            );
            assert_eq!(
                trace_session_file(base, "u", sid),
                base.join("u").join(format!("{sid}.traces.json"))
            );
        }
        #[cfg(unix)]
        assert_eq!(
            session_file(base, Some("u"), "chat:42"),
            base.join("u").join("chat:42.json")
        );
    }

    /// Test-only inverse of [`path_component`]: a raw name decodes to
    /// itself; an encoded name loses its leading `%` and each `%XX` escape
    /// decodes back to a UTF-8 byte sequence.
    fn decode(name: &str) -> String {
        let Some(rest) = name.strip_prefix('%') else {
            return name.to_string();
        };
        let bytes = rest.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' && i + 3 <= bytes.len() {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap();
                out.push(u8::from_str_radix(hex, 16).unwrap());
                i += 3;
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        }
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn path_component_encoding_has_a_left_inverse() {
        // A left inverse proves the encoding is injective: distinct inputs
        // can never share one file name.
        for id in [
            "",
            ".",
            "..",
            "%",
            "%25",
            "/2F",
            "//",
            "a/b",
            "%a%2Fb",
            "%a/b",
            "a%2Fb",
            "日本/語",
            "x\0y",
            "user%40x",
            // The layout-ambiguity triggers, plus a safe id with `.`/`_`.
            "s.traces",
            "_graph_context_s",
            "a.b_c",
        ] {
            assert_eq!(decode(&path_component(id)), id, "{id:?}");
        }
    }

    #[test]
    fn encoded_names_are_injective() {
        let names: std::collections::HashSet<_> =
            ["a/b", "%a%2Fb", "%a/b", "..", ".", "", "%x", "/2F", "//"]
                .iter()
                .map(|s| session_file(Path::new("/b"), Some("u"), s))
                .collect();
        assert_eq!(names.len(), 9);
    }

    /// The three file layouts of every id — `{c}.json`, `{c}.traces.json`,
    /// `_graph_context_{c}.json` — must never alias one another, or one
    /// session's trace/context file is another's QA file and erasure of the
    /// one deletes the other's data.
    #[test]
    fn qa_trace_and_context_names_never_collide() {
        let base = Path::new("/b");
        let mut all = std::collections::HashSet::new();
        for sid in ["s", "s.traces", "_graph_context_s", "%s", "%s.traces"] {
            assert!(all.insert(session_file(base, Some("u"), sid)), "QA {sid}");
            assert!(
                all.insert(trace_session_file(base, "u", sid)),
                "trace {sid}"
            );
            assert!(
                all.insert(graph_context_file(base, Some("u"), sid)),
                "context {sid}"
            );
        }
        assert_eq!(all.len(), 15);
    }

    /// `delete_session("s")` removes s's trace file — which without the
    /// `.traces` encoding rule is session `s.traces`'s QA file.
    #[tokio::test]
    async fn delete_session_leaves_traces_suffixed_sessions_qa_intact() {
        let dir = tempfile::tempdir().unwrap();
        let store = FsSessionStore::new(dir.path());
        store
            .create_qa_entry("s", Some("u1"), "q1", "a1", None)
            .await
            .unwrap();
        store
            .create_qa_entry("s.traces", Some("u1"), "q2", "a2", None)
            .await
            .unwrap();
        store
            .save_trace_step(
                "u1",
                "s",
                crate::types::SessionTraceStep {
                    trace_id: "t".to_string(),
                    origin_function: "f".to_string(),
                    status: "success".to_string(),
                    memory_query: String::new(),
                    memory_context: String::new(),
                    method_params: serde_json::json!({}),
                    method_return_value: None,
                    error_message: String::new(),
                    session_feedback: String::new(),
                },
            )
            .await
            .unwrap();

        assert!(store.delete_session("s", Some("u1")).await.unwrap());

        let survivor = store
            .get_all_qa_entries("s.traces", Some("u1"))
            .await
            .unwrap();
        assert_eq!(survivor.len(), 1, "the other session's QA entry must stay");
        assert_eq!(survivor[0].question, "q2");
        assert!(
            store
                .get_all_qa_entries("s", Some("u1"))
                .await
                .unwrap()
                .is_empty()
        );
        assert!(store.read_trace_steps("u1", "s").await.unwrap().is_empty());
    }

    #[test]
    fn layout_triggers_are_ascii_case_insensitive() {
        // Case-insensitive filesystems fold these onto the raw layouts.
        let base = Path::new("/base");
        let raw_qa = session_file(base, Some("u"), "s");
        let raw_trace = trace_session_file(base, "u", "s");
        for sid in ["S.TRACES", "s.Traces", "_GRAPH_CONTEXT_x", "x.TRACES"] {
            let qa = session_file(base, Some("u"), sid);
            assert_ne!(qa, raw_trace, "{sid:?} must not alias a trace file");
            assert_ne!(qa, raw_qa.with_file_name("_graph_context_s.json"));
        }
        assert_eq!(
            decode(&path_component("S.TRACES")),
            "S.TRACES",
            "encoded case-insensitive ids still round-trip"
        );
    }

    #[tokio::test]
    async fn delete_session_keeps_a_legacy_qa_file_at_the_trace_path() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path();
        let store = FsSessionStore::new(base);
        // A pre-encoding store wrote session "s.traces"'s QA file at the raw
        // name `s.traces.json` — the same path that is `s`'s trace file.
        let user_dir = base.join("u1");
        std::fs::create_dir_all(&user_dir).unwrap();
        let legacy_qa = user_dir.join("s.traces.json");
        let entries = vec![FsQAEntry {
            time: "t".to_string(),
            question: "q".to_string(),
            context: "c".to_string(),
            answer: "a".to_string(),
            qa_id: "qa-1".to_string(),
            feedback_text: None,
            feedback_score: None,
            used_graph_element_ids: None,
            memify_metadata: None,
        }];
        std::fs::write(&legacy_qa, serde_json::to_string(&entries).unwrap()).unwrap();

        store
            .create_qa_entry("s", Some("u1"), "q", "a", None)
            .await
            .unwrap();
        assert!(store.delete_session("s", Some("u1")).await.unwrap());

        assert_eq!(
            std::fs::read_to_string(&legacy_qa).unwrap(),
            serde_json::to_string(&entries).unwrap(),
            "the legacy Q&A file at the trace path must survive"
        );
        // A real trace file at the same path is still removed (the legacy
        // file is lifted out of the way first — while it sits there it also
        // blocks `s`'s trace writes, which fail to parse it).
        std::fs::remove_file(&legacy_qa).unwrap();
        store
            .save_trace_step(
                "u1",
                "s",
                SessionTraceStep {
                    trace_id: "t1".to_string(),
                    origin_function: "f".to_string(),
                    status: "success".to_string(),
                    memory_query: String::new(),
                    memory_context: String::new(),
                    method_params: serde_json::json!({}),
                    method_return_value: None,
                    error_message: String::new(),
                    session_feedback: String::new(),
                },
            )
            .await
            .unwrap();
        store
            .create_qa_entry("s", Some("u1"), "q2", "a2", None)
            .await
            .unwrap();
        assert!(store.delete_session("s", Some("u1")).await.unwrap());
        assert!(!trace_session_file(base, "u1", "s").exists());
    }

    #[tokio::test]
    async fn traversal_ids_never_read_or_write_outside_the_owner_dir() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path();
        let victim = base.join("victim.json");
        std::fs::write(&victim, b"precious").unwrap();
        let store = FsSessionStore::new(base);
        store
            .create_qa_entry("../victim", Some("u1"), "q", "a", None)
            .await
            .unwrap();
        assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
        let entries = store
            .get_all_qa_entries("../victim", Some("u1"))
            .await
            .unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].question, "q");
        // Graph context and traces with traversal ids stay under base/u1/ too.
        store
            .set_graph_context("../ctx", Some("u1"), "{}")
            .await
            .unwrap();
        store
            .save_trace_step(
                "u1",
                "../tr",
                SessionTraceStep {
                    trace_id: "t1".to_string(),
                    origin_function: "f".to_string(),
                    status: "success".to_string(),
                    memory_query: String::new(),
                    memory_context: String::new(),
                    method_params: serde_json::json!({}),
                    method_return_value: None,
                    error_message: String::new(),
                    session_feedback: String::new(),
                },
            )
            .await
            .unwrap();
        let mut escaped = Vec::new();
        for entry in std::fs::read_dir(base).unwrap() {
            let e = entry.unwrap();
            if e.file_name() != "u1" && e.file_name() != "victim.json" {
                escaped.push(e.path());
            }
        }
        assert!(
            escaped.is_empty(),
            "files escaped the owner dir: {escaped:?}"
        );
    }
}
