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

// =============================================================================
// Fetch eligibility, backoff, and outcome-driven state transitions
// =============================================================================

const DEFAULT_API_INTERVAL_SECS: u64 = 180;
const MIN_API_INTERVAL_SECS: u64 = 180;
const MAX_RATE_LIMIT_BACKOFF_SECS: u64 = 900;
const TRANSIENT_ERROR_RETRY_SECS: u64 = 30;
const RESET_CONFIRMATION_BUFFER_SECS: u64 = 5;

fn fetch_may_be_needed(state: &SharedUsageState, fingerprint: &str, now: u64) -> bool {
    if state.failed_token_fingerprint.as_deref() == Some(fingerprint) {
        return false;
    }
    if now < state.rate_limit_until_epoch {
        return false;
    }
    now >= state.next_allowed_at_epoch
}

fn effective_interval(configured_ttl: Option<u64>) -> u64 {
    configured_ttl
        .unwrap_or(DEFAULT_API_INTERVAL_SECS)
        .max(MIN_API_INTERVAL_SECS)
}

/// Earliest future reset epoch present in a usage snapshot, if any — used to
/// avoid scheduling the next poll for a moment just before a known reset
/// when waiting a few extra seconds would let that same poll confirm the
/// new window instead.
fn earliest_future_reset(data: &UsageLimitsData, now: u64) -> Option<u64> {
    [
        crate::cache::iso8601_to_epoch(&data.five_hour_resets_at),
        crate::cache::iso8601_to_epoch(&data.seven_day_resets_at),
    ]
    .into_iter()
    .flatten()
    .filter(|&e| e > now)
    .min()
}

/// Exponential backoff shared by every "keep failing, wait longer" outcome:
/// `base * 2^(consecutive_errors - 1)`, capped at `cap`. One formula, fed
/// different `base`/`cap` pairs per outcome, instead of two near-identical
/// hand-rolled versions (one for 429s, one for 5xx).
fn backoff_secs(base: u64, consecutive_errors: u32, cap: u64) -> u64 {
    let exponent = consecutive_errors.saturating_sub(1).min(10);
    base.saturating_mul(1_u64 << exponent).min(cap)
}

fn apply_fetch_outcome(
    state: &mut SharedUsageState,
    outcome: crate::usage_limits::UsageFetchOutcome,
    now: u64,
    api_interval: u64,
) {
    use crate::usage_limits::UsageFetchOutcome;
    match outcome {
        UsageFetchOutcome::Success(data) => {
            let reset = earliest_future_reset(&data, now);
            state.usage = Some(data);
            let ordinary_next = now + api_interval;
            state.next_allowed_at_epoch = match reset {
                Some(r)
                    if r + RESET_CONFIRMATION_BUFFER_SECS > now
                        && r + RESET_CONFIRMATION_BUFFER_SECS < ordinary_next + api_interval =>
                {
                    ordinary_next.max(r + RESET_CONFIRMATION_BUFFER_SECS)
                }
                _ => ordinary_next,
            };
            state.rate_limit_until_epoch = 0;
            state.consecutive_errors = 0;
            state.failed_token_fingerprint = None;
        }
        UsageFetchOutcome::RateLimited {
            retry_after_seconds,
        } => {
            state.consecutive_errors = state.consecutive_errors.saturating_add(1);
            let delay = retry_after_seconds
                .map(|retry_after| {
                    retry_after
                        .max(api_interval)
                        .min(MAX_RATE_LIMIT_BACKOFF_SECS)
                })
                .unwrap_or_else(|| {
                    backoff_secs(
                        api_interval,
                        state.consecutive_errors,
                        MAX_RATE_LIMIT_BACKOFF_SECS,
                    )
                });
            state.rate_limit_until_epoch = now + delay;
            state.next_allowed_at_epoch = state.rate_limit_until_epoch;
        }
        UsageFetchOutcome::Unauthorized => {
            if let Some(fp) = state.token_fingerprint.clone() {
                state.failed_token_fingerprint = Some(fp);
            }
            state.next_allowed_at_epoch = now + TRANSIENT_ERROR_RETRY_SECS;
        }
        UsageFetchOutcome::NetworkError | UsageFetchOutcome::InvalidResponse => {
            state.consecutive_errors = state.consecutive_errors.saturating_add(1);
            state.next_allowed_at_epoch = now + TRANSIENT_ERROR_RETRY_SECS;
        }
        UsageFetchOutcome::ServerError { .. } => {
            state.consecutive_errors = state.consecutive_errors.saturating_add(1);
            let delay = backoff_secs(TRANSIENT_ERROR_RETRY_SECS, state.consecutive_errors, 300);
            state.next_allowed_at_epoch = now + delay;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usage_limits::UsageLimitsData;

    fn default_state_at(now: u64) -> SharedUsageState {
        let mut s = SharedUsageState::default_v2();
        s.next_allowed_at_epoch = now; // eligible immediately unless overridden
        s
    }

    #[test]
    fn test_fetch_eligible_when_next_allowed_in_past() {
        let now = 2_000_000;
        let mut state = default_state_at(now);
        state.next_allowed_at_epoch = now - 1;
        assert!(fetch_may_be_needed(&state, "fp-1", now));
    }

    #[test]
    fn test_fetch_not_eligible_before_next_allowed() {
        let now = 2_000_000;
        let mut state = default_state_at(now);
        state.next_allowed_at_epoch = now + 100;
        assert!(!fetch_may_be_needed(&state, "fp-1", now));
    }

    #[test]
    fn test_fetch_not_eligible_during_rate_limit_cooldown() {
        let now = 2_000_000;
        let mut state = default_state_at(now);
        state.next_allowed_at_epoch = now - 1;
        state.rate_limit_until_epoch = now + 500;
        assert!(!fetch_may_be_needed(&state, "fp-1", now));
    }

    #[test]
    fn test_fetch_not_eligible_when_token_is_failed_fingerprint() {
        let now = 2_000_000;
        let mut state = default_state_at(now);
        state.next_allowed_at_epoch = now - 1;
        state.failed_token_fingerprint = Some("fp-1".to_string());
        assert!(!fetch_may_be_needed(&state, "fp-1", now));
    }

    #[test]
    fn test_fetch_eligible_when_failed_fingerprint_is_a_different_token() {
        let now = 2_000_000;
        let mut state = default_state_at(now);
        state.next_allowed_at_epoch = now - 1;
        state.failed_token_fingerprint = Some("fp-old".to_string());
        assert!(fetch_may_be_needed(&state, "fp-new", now));
    }

    #[test]
    fn test_apply_success_outcome_clears_backoff_and_sets_next_allowed() {
        let now = 2_000_000;
        let mut state = default_state_at(now);
        state.consecutive_errors = 3;
        state.rate_limit_until_epoch = now + 900;
        let data = UsageLimitsData::default();
        apply_fetch_outcome(
            &mut state,
            crate::usage_limits::UsageFetchOutcome::Success(data),
            now,
            180,
        );
        assert_eq!(state.consecutive_errors, 0);
        assert_eq!(state.rate_limit_until_epoch, 0);
        assert_eq!(state.next_allowed_at_epoch, now + 180);
        assert!(state.usage.is_some());
    }

    #[test]
    fn test_apply_rate_limited_outcome_honors_retry_after_within_bounds() {
        let now = 2_000_000;
        let mut state = default_state_at(now);
        apply_fetch_outcome(
            &mut state,
            crate::usage_limits::UsageFetchOutcome::RateLimited {
                retry_after_seconds: Some(400),
            },
            now,
            180,
        );
        assert_eq!(state.rate_limit_until_epoch, now + 400);
        assert_eq!(state.next_allowed_at_epoch, now + 400);
    }

    #[test]
    fn test_apply_rate_limited_outcome_clamps_retry_after_below_interval_up_to_interval() {
        let now = 2_000_000;
        let mut state = default_state_at(now);
        apply_fetch_outcome(
            &mut state,
            crate::usage_limits::UsageFetchOutcome::RateLimited {
                retry_after_seconds: Some(30),
            },
            now,
            180,
        );
        assert_eq!(state.rate_limit_until_epoch, now + 180);
    }

    #[test]
    fn test_apply_rate_limited_outcome_caps_retry_after_at_900() {
        let now = 2_000_000;
        let mut state = default_state_at(now);
        apply_fetch_outcome(
            &mut state,
            crate::usage_limits::UsageFetchOutcome::RateLimited {
                retry_after_seconds: Some(5000),
            },
            now,
            180,
        );
        assert_eq!(state.rate_limit_until_epoch, now + 900);
    }

    #[test]
    fn test_apply_rate_limited_outcome_missing_retry_after_uses_exponential_backoff() {
        let now = 2_000_000;
        let mut state = default_state_at(now);
        state.consecutive_errors = 2; // this will become the 3rd consecutive error
        apply_fetch_outcome(
            &mut state,
            crate::usage_limits::UsageFetchOutcome::RateLimited {
                retry_after_seconds: None,
            },
            now,
            180,
        );
        // exponent = consecutive_errors.saturating_sub(1).min(10) using the
        // POST-increment count (3 - 1 = 2) => 180 * 2^2 = 720
        assert_eq!(state.rate_limit_until_epoch, now + 720);
    }

    #[test]
    fn test_apply_network_error_keeps_previous_snapshot_and_retries_in_30s() {
        let now = 2_000_000;
        let mut state = default_state_at(now);
        state.usage = Some(UsageLimitsData {
            five_hour_pct: 42.0,
            ..Default::default()
        });
        apply_fetch_outcome(
            &mut state,
            crate::usage_limits::UsageFetchOutcome::NetworkError,
            now,
            180,
        );
        assert_eq!(state.usage.as_ref().unwrap().five_hour_pct, 42.0);
        assert_eq!(state.next_allowed_at_epoch, now + 30);
    }

    #[test]
    fn test_apply_unauthorized_outcome_sets_failed_token_fingerprint() {
        let now = 2_000_000;
        let mut state = default_state_at(now);
        state.token_fingerprint = Some("fp-current".to_string());
        apply_fetch_outcome(
            &mut state,
            crate::usage_limits::UsageFetchOutcome::Unauthorized,
            now,
            180,
        );
        assert_eq!(
            state.failed_token_fingerprint,
            Some("fp-current".to_string())
        );
    }

    #[test]
    fn test_backoff_secs_doubles_per_error_and_caps() {
        assert_eq!(backoff_secs(180, 1, 900), 180); // 180 * 2^0
        assert_eq!(backoff_secs(180, 2, 900), 360); // 180 * 2^1
        assert_eq!(backoff_secs(180, 3, 900), 720); // 180 * 2^2
        assert_eq!(backoff_secs(180, 4, 900), 900); // 180 * 2^3 = 1440, capped
    }

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
