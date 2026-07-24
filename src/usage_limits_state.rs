//! Cross-process coordinator for account-wide OAuth usage data.
//!
//! Every `cship` invocation is a separate short-lived process; multiple
//! Claude Code sessions can invoke it at nearly the same time. This module
//! makes those processes share one on-disk state file and one fetch lock so
//! `/api/oauth/usage` is polled at most once per ~180s, globally, regardless
//! of how many sessions or processes are running.
//!
//! This is the ONLY file that reads/writes the shared cross-process cache
//! directory (`platform::cship_shared_cache_dir()`). It does not make HTTP
//! calls itself — it drives `crate::usage_limits::fetch_usage_limits` and
//! persists its typed outcome.

use crate::usage_limits::UsageLimitsData;
use std::path::{Path, PathBuf};

const STATE_FILE_NAME: &str = "usage-limits-state-v2.json";
const CURRENT_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct SharedUsageState {
    pub schema_version: u32,
    pub token_fingerprint: Option<String>,
    pub usage: Option<UsageLimitsData>,
    pub next_allowed_at_epoch: u64,
    pub rate_limit_until_epoch: u64,
    pub consecutive_errors: u32,
    pub failed_token_fingerprint: Option<String>,
}

impl SharedUsageState {
    fn default_v2() -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            token_fingerprint: None,
            usage: None,
            next_allowed_at_epoch: 0,
            rate_limit_until_epoch: 0,
            consecutive_errors: 0,
            failed_token_fingerprint: None,
        }
    }
}

fn state_file_path_in(dir: &Path) -> PathBuf {
    dir.join(STATE_FILE_NAME)
}

fn load_state_from_dir(dir: &Path) -> SharedUsageState {
    let path = state_file_path_in(dir);
    let raw = match std::fs::read_to_string(&path) {
        Ok(r) => r,
        Err(_) => return SharedUsageState::default_v2(),
    };
    match serde_json::from_str::<SharedUsageState>(&raw) {
        Ok(state) if state.schema_version == CURRENT_SCHEMA_VERSION => state,
        Ok(_) => {
            tracing::debug!("cship.usage_limits_state: unknown schema version, using default");
            SharedUsageState::default_v2()
        }
        Err(e) => {
            tracing::debug!("cship.usage_limits_state: invalid state JSON ({e}), using default");
            SharedUsageState::default_v2()
        }
    }
}

/// Atomically write `state` to `<dir>/usage-limits-state-v2.json`: serialize,
/// write to a sibling temp file in the same directory, flush + fsync, then
/// rename over the target. `fs::rename` is atomic on both POSIX and NTFS
/// when source and destination are on the same volume (guaranteed here since
/// both live in `dir`), so a concurrent reader always sees either the
/// complete previous file or the complete new one, never a partial write.
fn persist_state_to_dir(dir: &Path, state: &SharedUsageState) {
    if std::fs::create_dir_all(dir).is_err() {
        tracing::debug!("cship.usage_limits_state: could not create cache dir");
        return;
    }
    let json = match serde_json::to_string(state) {
        Ok(j) => j,
        Err(e) => {
            tracing::debug!("cship.usage_limits_state: failed to serialize state: {e}");
            return;
        }
    };
    let tmp_path = dir.join(format!("{STATE_FILE_NAME}.tmp.{}", std::process::id()));
    let write_result = (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp_path)?;
        f.write_all(json.as_bytes())?;
        f.sync_all()?;
        Ok(())
    })();
    if let Err(e) = write_result {
        tracing::debug!("cship.usage_limits_state: failed to write temp state file: {e}");
        let _ = std::fs::remove_file(&tmp_path);
        return;
    }
    if let Err(e) = std::fs::rename(&tmp_path, state_file_path_in(dir)) {
        tracing::debug!("cship.usage_limits_state: failed to replace state file: {e}");
        let _ = std::fs::remove_file(&tmp_path);
    }
    set_restrictive_permissions(&state_file_path_in(dir));
}

#[cfg(unix)]
fn set_restrictive_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(meta) = std::fs::metadata(path) {
        let mut perms = meta.permissions();
        perms.set_mode(0o600);
        let _ = std::fs::set_permissions(path, perms);
    }
}

#[cfg(not(unix))]
fn set_restrictive_permissions(_path: &Path) {
    // Windows relies on the current user's directory ACL (spec §5); no
    // equivalent of Unix file-mode bits to set here.
}

// Not yet called from the renderer — Task 3.x (locking) and Task 4.x
// (coordinator) wire these in. Allowed dead code until then so `cargo clippy
// -D warnings` (CI) stays green for this foundational commit.
#[allow(dead_code)]
pub(crate) fn load_state_or_default() -> SharedUsageState {
    match crate::platform::cship_shared_cache_dir() {
        Some(dir) => load_state_from_dir(&dir),
        None => SharedUsageState::default_v2(),
    }
}

#[allow(dead_code)]
pub(crate) fn persist_state(state: &SharedUsageState) {
    if let Some(dir) = crate::platform::cship_shared_cache_dir() {
        persist_state_to_dir(&dir, state);
    }
}

// =============================================================================
// Cross-process fetch lock — non-blocking acquisition + stale recovery
// =============================================================================

// Not yet called from the renderer — Task 3.x (coordinator) wires these in.
// Allowed dead code until then so `cargo clippy -D warnings` (CI) stays green.
#[allow(dead_code)]
const LOCK_FILE_NAME: &str = "usage-limits-fetch.lock";
#[allow(dead_code)]
const LOCK_STALE_AFTER_SECS: u64 = 15;

#[allow(dead_code)]
pub(crate) struct FetchLock {
    path: PathBuf,
    token: String,
}

impl Drop for FetchLock {
    fn drop(&mut self) {
        // Only remove the lock file if it still holds the token *this*
        // instance wrote. If another process reclaimed a stale lock while
        // this instance stalled, the file now holds a different token — in
        // that case the lock is no longer this instance's to delete, and
        // deleting it anyway would let a third process acquire concurrently
        // with the reclaiming owner. A read failure (file already gone, or
        // truly unreadable) is also treated as "not mine to delete".
        if let Ok(contents) = std::fs::read_to_string(&self.path)
            && contents == self.token
        {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Generate a per-acquisition owner token unique enough to distinguish this
/// process/attempt from any other: process ID + current time in nanoseconds.
/// Only needs to be unlikely-to-collide among concurrent local processes, not
/// cryptographically unique — no new dependency required.
fn generate_owner_token() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{}", std::process::id(), nanos)
}

fn try_create_lock_file(path: &Path, token: &str) -> bool {
    use std::io::Write;
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(mut f) => f.write_all(token.as_bytes()).is_ok(),
        Err(_) => false,
    }
}

#[allow(dead_code)]
fn lock_is_stale(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return true; // unreadable lock file — treat as stale, safe to reclaim
    };
    let Ok(modified) = meta.modified() else {
        return true; // platform doesn't support mtime — treat as stale
    };
    match std::time::SystemTime::now().duration_since(modified) {
        Ok(age) => age.as_secs() > LOCK_STALE_AFTER_SECS,
        Err(_) => false, // mtime is in the future (clock skew) — not stale
    }
}

#[allow(dead_code)]
fn try_acquire_fetch_lock_in(dir: &Path) -> Option<FetchLock> {
    if std::fs::create_dir_all(dir).is_err() {
        return None;
    }
    let lock_path = dir.join(LOCK_FILE_NAME);
    let token = generate_owner_token();

    if try_create_lock_file(&lock_path, &token) {
        return Some(FetchLock {
            path: lock_path,
            token,
        });
    }

    // Acquisition failed because the file exists. Recover a stale lock, then
    // attempt exactly once more — never loop.
    if lock_is_stale(&lock_path) {
        let _ = std::fs::remove_file(&lock_path);
        if try_create_lock_file(&lock_path, &token) {
            return Some(FetchLock {
                path: lock_path,
                token,
            });
        }
    }
    None
}

#[allow(dead_code)]
pub(crate) fn try_acquire_fetch_lock() -> Option<FetchLock> {
    let dir = crate::platform::cship_shared_cache_dir()?;
    try_acquire_fetch_lock_in(&dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_state_or_default_when_file_missing_returns_schema_v2_default() {
        let dir = tempfile::tempdir().unwrap();
        let state = load_state_from_dir(dir.path());
        assert_eq!(state.schema_version, 2);
        assert!(state.usage.is_none());
        assert_eq!(state.next_allowed_at_epoch, 0);
    }

    #[test]
    fn test_load_state_or_default_when_json_invalid_returns_default() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("usage-limits-state-v2.json"), "{not json").unwrap();
        let state = load_state_from_dir(dir.path());
        assert_eq!(state.schema_version, 2);
        assert!(state.usage.is_none());
    }

    #[test]
    fn test_load_state_or_default_when_schema_version_unknown_returns_default() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("usage-limits-state-v2.json"),
            r#"{"schema_version":99,"token_fingerprint":null,"usage":null,"next_allowed_at_epoch":0,"rate_limit_until_epoch":0,"consecutive_errors":0,"failed_token_fingerprint":null}"#,
        )
        .unwrap();
        let state = load_state_from_dir(dir.path());
        assert_eq!(state.schema_version, 2);
        assert!(state.usage.is_none());
    }

    #[test]
    fn test_persist_then_load_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let mut state = SharedUsageState::default_v2();
        state.token_fingerprint = Some("fp-abc123".to_string());
        state.next_allowed_at_epoch = 4_000_000;
        persist_state_to_dir(dir.path(), &state);
        let loaded = load_state_from_dir(dir.path());
        assert_eq!(loaded.token_fingerprint, Some("fp-abc123".to_string()));
        assert_eq!(loaded.next_allowed_at_epoch, 4_000_000);
    }

    #[test]
    fn test_persist_state_produces_no_temp_files_left_behind() {
        let dir = tempfile::tempdir().unwrap();
        let state = SharedUsageState::default_v2();
        persist_state_to_dir(dir.path(), &state);
        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(entries, vec!["usage-limits-state-v2.json"]);
    }

    #[test]
    fn test_lock_first_process_acquires() {
        let dir = tempfile::tempdir().unwrap();
        let lock = try_acquire_fetch_lock_in(dir.path());
        assert!(lock.is_some());
    }

    #[test]
    fn test_lock_second_process_fails_immediately() {
        let dir = tempfile::tempdir().unwrap();
        let _first = try_acquire_fetch_lock_in(dir.path()).unwrap();
        let second = try_acquire_fetch_lock_in(dir.path());
        assert!(second.is_none());
    }

    #[test]
    fn test_lock_released_on_drop_can_be_reacquired() {
        let dir = tempfile::tempdir().unwrap();
        {
            let _lock = try_acquire_fetch_lock_in(dir.path()).unwrap();
        } // dropped here
        let second = try_acquire_fetch_lock_in(dir.path());
        assert!(second.is_some());
    }

    #[test]
    fn test_lock_stale_lock_is_removed_and_reacquired() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join("usage-limits-fetch.lock");
        std::fs::write(&lock_path, "").unwrap();
        let old = filetime::FileTime::from_system_time(
            std::time::SystemTime::now() - std::time::Duration::from_secs(20),
        );
        filetime::set_file_mtime(&lock_path, old).unwrap();

        let lock = try_acquire_fetch_lock_in(dir.path());
        assert!(lock.is_some());
    }

    #[test]
    fn test_lock_fresh_lock_is_not_removed() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join("usage-limits-fetch.lock");
        std::fs::write(&lock_path, "").unwrap();
        // mtime defaults to "now" — well under the 15s stale threshold.
        let lock = try_acquire_fetch_lock_in(dir.path());
        assert!(lock.is_none());
    }

    /// Fix 3: the exact stale-lock race. Process A acquires `lock_a`. It
    /// stalls past the staleness window; process B reclaims the lock file
    /// (simulated here by directly overwriting the lock file's contents with
    /// a different owner token, exactly what `try_acquire_fetch_lock_in`'s
    /// stale-recovery path does from a second process). Process A then
    /// resumes and drops `lock_a` — its `Drop` must NOT delete B's
    /// newly-reclaimed lock file, or a third process could acquire
    /// concurrently with B, defeating mutual exclusion.
    #[test]
    fn test_drop_does_not_delete_a_lock_reclaimed_by_another_owner() {
        let dir = tempfile::tempdir().unwrap();
        let lock_a = try_acquire_fetch_lock_in(dir.path()).unwrap();
        let lock_path = dir.path().join(LOCK_FILE_NAME);
        assert!(lock_path.exists(), "lock_a should have created the file");

        // Simulate B's stale-recovery: overwrite with a different owner token.
        std::fs::write(&lock_path, "some-other-owner-token").unwrap();

        drop(lock_a);

        assert!(
            lock_path.exists(),
            "lock_a's drop must not delete a lock file it no longer owns"
        );
        let remaining = std::fs::read_to_string(&lock_path).unwrap();
        assert_eq!(
            remaining, "some-other-owner-token",
            "B's token must be untouched by A's drop"
        );
    }

    #[test]
    fn test_lock_stale_recovery_only_attempted_once() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join("usage-limits-fetch.lock");
        std::fs::write(&lock_path, "").unwrap();
        let old = filetime::FileTime::from_system_time(
            std::time::SystemTime::now() - std::time::Duration::from_secs(20),
        );
        filetime::set_file_mtime(&lock_path, old).unwrap();

        let _lock = try_acquire_fetch_lock_in(dir.path()).unwrap();
        // A second concurrent caller, same instant, must NOT also recover —
        // the first caller already holds the (recreated) lock file.
        let second = try_acquire_fetch_lock_in(dir.path());
        assert!(second.is_none());
    }
}
