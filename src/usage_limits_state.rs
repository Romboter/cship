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
use std::time::Duration;

// The filename's `-vN` suffix must be bumped in lockstep with
// `CURRENT_SCHEMA_VERSION` whenever the schema changes. Different-schema
// binaries then use separate files during a version rollout instead of
// endlessly overwriting each other's state (which would defeat the fetch
// throttle for the whole upgrade window). The compiler can't enforce this —
// keep the two in sync by hand.
const STATE_FILE_NAME: &str = "usage-limits-state-v3.json";
const CURRENT_SCHEMA_VERSION: u32 = 3;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct SharedUsageState {
    pub schema_version: u32,
    pub token_fingerprint: Option<String>,
    pub usage: Option<UsageLimitsData>,
    pub next_allowed_at_epoch: u64,
    pub rate_limit_until_epoch: u64,
    pub consecutive_errors: u32,
    pub failed_token_fingerprint: Option<String>,
    /// Fingerprint of an as-yet-uncommitted (`token_fingerprint` still holds
    /// the previous token) token change that has already been given its one
    /// out-of-schedule "bypass and probe immediately" attempt. Additive
    /// field — `#[serde(default)]` so state files written before this field
    /// existed still deserialize (as `None`, i.e. "not yet probed").
    #[serde(default)]
    pub token_change_probe_fingerprint: Option<String>,
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
            token_change_probe_fingerprint: None,
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

/// Atomically write `state` to `<dir>/{STATE_FILE_NAME}`: serialize,
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

pub(crate) fn load_state_or_default() -> SharedUsageState {
    match crate::platform::cship_shared_cache_dir() {
        Some(dir) => load_state_from_dir(&dir),
        None => SharedUsageState::default_v2(),
    }
}

pub(crate) fn persist_state(state: &SharedUsageState) {
    if let Some(dir) = crate::platform::cship_shared_cache_dir() {
        persist_state_to_dir(&dir, state);
    }
}

// =============================================================================
// Cross-process fetch lock — non-blocking OS-level advisory file lock
// =============================================================================

const LOCK_FILE_NAME: &str = "usage-limits-fetch.lock";

/// Holds an OS-level advisory lock (`flock` on Unix, `LockFileEx` on
/// Windows) on the shared lock file for the duration of a fetch. Mutual
/// exclusion is kernel-enforced: only one process can hold the lock at a
/// time, and if the holder crashes or is killed the OS releases the lock
/// the moment its file descriptor closes — no staleness heuristic needed.
/// Dropping the `File` releases the lock; the lock file itself is left in
/// place (it's a permanent rendezvous point, not a token of ownership).
pub(crate) struct FetchLock {
    _file: std::fs::File,
}

fn try_acquire_fetch_lock_in(dir: &Path) -> Option<FetchLock> {
    if std::fs::create_dir_all(dir).is_err() {
        return None;
    }
    let lock_path = dir.join(LOCK_FILE_NAME);
    let file = match std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
    {
        Ok(f) => f,
        Err(e) => {
            tracing::debug!("cship.usage_limits_state: could not open lock file: {e}");
            return None;
        }
    };
    // Non-blocking: if another process holds the lock, give up immediately —
    // callers are a statusline renderer that must never stall.
    match file.try_lock() {
        Ok(()) => Some(FetchLock { _file: file }),
        Err(_) => None,
    }
}

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

/// A token change is "fresh" — eligible for the one-time bypass-and-probe
/// treatment (design points 1/5) — only the first time we see it. Once a
/// probe has been attempted for `fingerprint` (success or failure), further
/// renders for that same still-uncommitted fingerprint fall back to normal
/// `fetch_may_be_needed` scheduling, exactly like a non-token-change render,
/// so a persistently-failing new token doesn't get a live fetch on every
/// render. Success commits `token_fingerprint`, so `fingerprint` no longer
/// reads as "changed" at all — this only governs the failure path.
fn is_fresh_token_change(state: &SharedUsageState, fingerprint: &str) -> bool {
    state.token_fingerprint.as_deref() != Some(fingerprint)
        && state.token_change_probe_fingerprint.as_deref() != Some(fingerprint)
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
    attempted_fingerprint: &str,
) {
    use crate::usage_limits::UsageFetchOutcome;
    match outcome {
        UsageFetchOutcome::Success(data) => {
            let reset = earliest_future_reset(&data, now);
            state.usage = Some(*data);
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
            // Use the fingerprint that was actually attempted in this call,
            // not `state.token_fingerprint` — during a token-change probe
            // that still holds the *previous* (uncommitted) fingerprint, so
            // reading it here would blame the wrong token.
            state.failed_token_fingerprint = Some(attempted_fingerprint.to_string());
            // No retry timer here: `fetch_may_be_needed` unconditionally
            // blocks any fetch while `failed_token_fingerprint` matches the
            // current token, so only a token change (different fingerprint)
            // can unblock — a `next_allowed_at_epoch` bump would be inert.
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

pub(crate) struct SharedUsageResolution {
    pub usage: Option<UsageLimitsData>,
}

/// Usage is only displayable when it was recorded under the fingerprint
/// currently in use (token-change points 1/2) — checked at read time, not
/// wiped destructively, so a stale value on disk simply never surfaces
/// under a different token rather than needing to be actively cleared.
fn displayable_usage(state: &SharedUsageState, fingerprint: &str) -> Option<UsageLimitsData> {
    if state.token_fingerprint.as_deref() == Some(fingerprint) {
        state.usage.clone()
    } else {
        None
    }
}

/// Render invocations must never stall waiting on the network — a slow or
/// unreachable API would otherwise block every eligible statusline render.
const RENDER_FETCH_TIMEOUT: Duration = Duration::from_secs(2);

/// Runs `fetch_fn` on a worker thread and waits up to `RENDER_FETCH_TIMEOUT`
/// for it to reply, bounding how long a render can stall regardless of the
/// HTTP client's own (longer) internal timeout. If the worker hasn't replied
/// in time, its result is abandoned — the thread keeps running in the
/// background, but this call reports `NetworkError` so the caller applies
/// the same backoff as any other transient failure (`apply_fetch_outcome`'s
/// `TRANSIENT_ERROR_RETRY_SECS`), instead of re-attempting the same slow
/// endpoint on the very next render. Generic `F` bound allows tests to
/// inject a fast lambda bypassing real HTTP.
/// `lock` is moved into the spawned thread rather than released by the
/// caller: if the real request outlives the render timeout, the OS-level
/// fetch lock must stay held for as long as the request is actually in
/// flight, not just for as long as the render is willing to wait on it —
/// otherwise another process can acquire the lock and start an overlapping
/// request. The lock drops (and the rendezvous file is freed for the next
/// fetch) whenever the thread actually finishes, bounded by the fetch
/// function's own client-side timeout.
fn fetch_with_render_timeout<F>(
    fetch_fn: F,
    lock: FetchLock,
) -> crate::usage_limits::UsageFetchOutcome
where
    F: FnOnce() -> crate::usage_limits::UsageFetchOutcome + Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _lock = lock;
        let _ = tx.send(fetch_fn());
    });
    match rx.recv_timeout(RENDER_FETCH_TIMEOUT) {
        Ok(outcome) => outcome,
        Err(_) => {
            tracing::warn!(
                "cship.usage_limits_state: API fetch timed out after {}s",
                RENDER_FETCH_TIMEOUT.as_secs()
            );
            crate::usage_limits::UsageFetchOutcome::NetworkError
        }
    }
}

/// Coordinator's single public entry point: resolve account-wide OAuth usage
/// data for the current token, fetching from the Anthropic API at most once
/// per ~180s across all concurrent `cship` processes.
///
/// Non-blocking: a process that can't acquire the fetch lock renders
/// immediately from whatever's already on disk, never waits on another
/// process's in-flight fetch.
pub(crate) fn resolve_shared_usage(
    token: Option<&str>,
    claude_version: Option<&str>,
    configured_ttl: Option<u64>,
    now: u64,
) -> SharedUsageResolution {
    let Some(token) = token else {
        return SharedUsageResolution { usage: None };
    };

    let fingerprint = crate::platform::token_fingerprint(token);
    let interval = effective_interval(configured_ttl);
    let state = load_state_or_default();
    let fresh_token_change = is_fresh_token_change(&state, &fingerprint);

    if !fresh_token_change && !fetch_may_be_needed(&state, &fingerprint, now) {
        return SharedUsageResolution {
            usage: displayable_usage(&state, &fingerprint),
        };
    }

    let Some(_lock) = try_acquire_fetch_lock() else {
        return SharedUsageResolution {
            usage: displayable_usage(&state, &fingerprint),
        };
    };

    // Re-read: another process may have refreshed while we were acquiring.
    let mut state = load_state_or_default();
    let fresh_token_change = is_fresh_token_change(&state, &fingerprint);

    if fresh_token_change {
        // Points 3/4/5: the old failure block no longer applies to this
        // token, the old schedule belonged to the old token, and any
        // existing rate-limit cooldown gets one probe rather than a blind
        // 15-minute wait — `apply_fetch_outcome` re-establishes backoff
        // normally below if the probe itself comes back 429. This bypass is
        // spent exactly once per newly-arrived fingerprint: if the probe
        // fails, `token_change_probe_fingerprint` records that below, so the
        // *next* render for this same fingerprint takes the
        // `fetch_may_be_needed` branch instead, like any ordinary render.
        //
        // `rate_limit_until_epoch` and `consecutive_errors` are cleared too:
        // they describe the *old* token's history. Left alone, a probe that
        // comes back NetworkError/InvalidResponse (rather than another 429)
        // would only bump `next_allowed_at_epoch`, leaving the old token's
        // rate-limit cooldown — up to `MAX_RATE_LIMIT_BACKOFF_SECS` — still
        // in effect and blocking the *new* token via `fetch_may_be_needed`'s
        // separate `rate_limit_until_epoch` check.
        state.failed_token_fingerprint = None;
        state.next_allowed_at_epoch = now;
        state.rate_limit_until_epoch = 0;
        state.consecutive_errors = 0;
    } else if !fetch_may_be_needed(&state, &fingerprint, now) {
        // Someone else already refreshed while we waited for the lock, or
        // this fingerprint already burned its one bypass probe and is now
        // just in normal backoff.
        return SharedUsageResolution {
            usage: displayable_usage(&state, &fingerprint),
        };
    }

    // Recorded unconditionally, even if the state write below fails, so a
    // degraded (but writable-lock-file) cache dir still self-limits fetches
    // to once per `interval` — see `fetch_recently_touched`. Touched now,
    // before the request even starts, since the request may outlive this
    // function's own return (see `fetch_with_render_timeout`).
    _lock.touch();

    let token_owned = token.to_string();
    let claude_version_owned = claude_version.map(str::to_string);
    let outcome = fetch_with_render_timeout(
        move || {
            crate::usage_limits::fetch_usage_limits(&token_owned, claude_version_owned.as_deref())
        },
        _lock,
    );
    let succeeded = matches!(outcome, crate::usage_limits::UsageFetchOutcome::Success(_));
    apply_fetch_outcome(&mut state, outcome, now, interval, &fingerprint);
    if succeeded {
        // Point 6: only a successful fetch commits the new fingerprint —
        // this is the moment the "switch" is actually recorded.
        state.token_fingerprint = Some(fingerprint.clone());
        state.token_change_probe_fingerprint = None;
    } else if fresh_token_change {
        // Spend this fingerprint's one out-of-schedule probe so subsequent
        // renders for the same still-uncommitted fingerprint respect normal
        // backoff instead of bypassing the schedule again.
        state.token_change_probe_fingerprint = Some(fingerprint.clone());
    }
    persist_state(&state);

    SharedUsageResolution {
        usage: displayable_usage(&state, &fingerprint),
    }
    // The OS-level lock itself may still be held by the background thread at
    // this point if the request timed out past the render deadline — it's
    // released whenever that thread actually finishes. The lock file is left
    // in place either way, as a permanent rendezvous point.
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
    fn test_fetch_with_render_timeout_returns_fast_outcome() {
        let dir = tempfile::tempdir().unwrap();
        let lock = try_acquire_fetch_lock_in(dir.path()).unwrap();
        let outcome = fetch_with_render_timeout(
            || {
                crate::usage_limits::UsageFetchOutcome::Success(Box::new(
                    UsageLimitsData::default(),
                ))
            },
            lock,
        );
        assert!(matches!(
            outcome,
            crate::usage_limits::UsageFetchOutcome::Success(_)
        ));
    }

    #[test]
    #[ignore = "slow: blocks for the 2s render timeout"]
    fn test_fetch_with_render_timeout_abandons_slow_fetch() {
        let dir = tempfile::tempdir().unwrap();
        let lock = try_acquire_fetch_lock_in(dir.path()).unwrap();
        let outcome = fetch_with_render_timeout(
            || {
                std::thread::sleep(Duration::from_secs(5));
                crate::usage_limits::UsageFetchOutcome::Success(Box::new(
                    UsageLimitsData::default(),
                ))
            },
            lock,
        );
        assert!(matches!(
            outcome,
            crate::usage_limits::UsageFetchOutcome::NetworkError
        ));
    }

    /// Regression test for the lock-outliving-render-timeout bug: the OS
    /// lock must stay held (blocking a second `try_acquire_fetch_lock_in`)
    /// until the background thread actually finishes, not just until the
    /// render gives up waiting on it.
    #[test]
    #[ignore = "slow: blocks for the 2s render timeout plus a 500ms slow fetch"]
    fn test_fetch_with_render_timeout_keeps_lock_held_past_render_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let lock = try_acquire_fetch_lock_in(dir.path()).unwrap();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let outcome = fetch_with_render_timeout(
                || {
                    // Longer than RENDER_FETCH_TIMEOUT (2s) but short enough
                    // to keep the test fast.
                    std::thread::sleep(RENDER_FETCH_TIMEOUT + Duration::from_millis(500));
                    crate::usage_limits::UsageFetchOutcome::Success(Box::new(
                        UsageLimitsData::default(),
                    ))
                },
                lock,
            );
            let _ = done_tx.send(outcome);
        });

        // Wait past the render timeout — the caller above has already
        // "returned" its NetworkError outcome, but the real fetch (and thus
        // the lock) is still alive.
        std::thread::sleep(RENDER_FETCH_TIMEOUT + Duration::from_millis(100));
        assert!(
            try_acquire_fetch_lock_in(dir.path()).is_none(),
            "lock must still be held while the slow fetch is in flight"
        );

        // Let the slow fetch finish and release the lock. The lock drops
        // just after `tx.send` inside the spawned thread, a few
        // instructions after `done_rx` receives — poll briefly rather than
        // racing a fixed sleep against that window.
        done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let reacquired = (0..50).find_map(|_| {
            let lock = try_acquire_fetch_lock_in(dir.path());
            if lock.is_none() {
                std::thread::sleep(Duration::from_millis(20));
            }
            lock
        });
        assert!(
            reacquired.is_some(),
            "lock must be released once the slow fetch actually completes"
        );
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
            crate::usage_limits::UsageFetchOutcome::Success(Box::new(data)),
            now,
            180,
            "fp-1",
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
            "fp-1",
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
            "fp-1",
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
            "fp-1",
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
            "fp-1",
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
            "fp-1",
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
            "fp-current",
        );
        assert_eq!(
            state.failed_token_fingerprint,
            Some("fp-current".to_string())
        );
    }

    /// Bug 2: during a token-change probe, `state.token_fingerprint` still
    /// holds the *previous* (uncommitted) fingerprint at the point
    /// `apply_fetch_outcome` runs. A 401 for the newly-attempted fingerprint
    /// must blame the fingerprint that was actually attempted, not the
    /// stale previous one, or `failed_token_fingerprint` can never correctly
    /// suppress retries for the actual bad token.
    #[test]
    fn test_apply_unauthorized_outcome_blames_attempted_fingerprint_not_stale_previous() {
        let now = 2_000_000;
        let mut state = default_state_at(now);
        state.token_fingerprint = Some("fp-old-stale".to_string());
        apply_fetch_outcome(
            &mut state,
            crate::usage_limits::UsageFetchOutcome::Unauthorized,
            now,
            180,
            "fp-new-attempted",
        );
        assert_eq!(
            state.failed_token_fingerprint,
            Some("fp-new-attempted".to_string())
        );
        assert_ne!(state.failed_token_fingerprint, state.token_fingerprint);
    }

    // -------------------------------------------------------------------
    // Bug 1: a token-change bypass probe must be spent at most once per
    // newly-arrived fingerprint — a persistently-failing new token must not
    // get a live fetch attempt on every render.
    // -------------------------------------------------------------------

    #[test]
    fn test_is_fresh_token_change_true_for_never_seen_fingerprint() {
        let state = SharedUsageState::default_v2();
        assert!(is_fresh_token_change(&state, "fp-new"));
    }

    #[test]
    fn test_is_fresh_token_change_false_once_committed() {
        let mut state = SharedUsageState::default_v2();
        state.token_fingerprint = Some("fp-new".to_string());
        assert!(!is_fresh_token_change(&state, "fp-new"));
    }

    #[test]
    fn test_is_fresh_token_change_false_after_probe_already_spent() {
        let mut state = SharedUsageState::default_v2();
        state.token_fingerprint = Some("fp-old".to_string());
        state.token_change_probe_fingerprint = Some("fp-new".to_string());
        assert!(!is_fresh_token_change(&state, "fp-new"));
    }

    #[test]
    fn test_is_fresh_token_change_true_for_different_fingerprint_than_spent_probe() {
        let mut state = SharedUsageState::default_v2();
        state.token_fingerprint = Some("fp-old".to_string());
        state.token_change_probe_fingerprint = Some("fp-some-other-new".to_string());
        assert!(is_fresh_token_change(&state, "fp-new"));
    }

    /// End-to-end (network-free) simulation of two consecutive
    /// `resolve_shared_usage`-shaped renders for the *same* failing new
    /// fingerprint, following exactly the state transitions
    /// `resolve_shared_usage` performs (see its `fresh_token_change`
    /// branch): the first render gets the one-time bypass probe; the
    /// second must NOT bypass again and must instead observe the normal
    /// backoff that the first attempt's failure recorded.
    #[test]
    fn test_second_render_for_same_failing_new_fingerprint_does_not_bypass_again() {
        let now = 2_000_000;
        let new_fingerprint = "fp-new-token";
        let mut state = SharedUsageState::default_v2();
        state.token_fingerprint = Some("fp-old-token".to_string());
        state.next_allowed_at_epoch = now + 9_999; // stale schedule from the old token

        // --- Render 1: fresh token change -> bypass and probe once ---
        let fresh = is_fresh_token_change(&state, new_fingerprint);
        assert!(fresh, "first render for a new fingerprint must be fresh");
        state.failed_token_fingerprint = None;
        state.next_allowed_at_epoch = now;

        // The probe fails (e.g. rate-limited) — apply_fetch_outcome records
        // real backoff exactly as it would for an ordinary render.
        apply_fetch_outcome(
            &mut state,
            crate::usage_limits::UsageFetchOutcome::RateLimited {
                retry_after_seconds: Some(400),
            },
            now,
            180,
            new_fingerprint,
        );
        // Not successful, and this was a fresh token change: spend the probe.
        state.token_change_probe_fingerprint = Some(new_fingerprint.to_string());

        assert_eq!(state.rate_limit_until_epoch, now + 400);
        assert_ne!(
            state.token_fingerprint.as_deref(),
            Some(new_fingerprint),
            "failure must never commit the new fingerprint"
        );

        // --- Render 2: same still-uncommitted fingerprint, shortly after ---
        let now2 = now + 10; // well before the 400s cooldown elapses
        let fresh2 = is_fresh_token_change(&state, new_fingerprint);
        assert!(
            !fresh2,
            "second render for the same fingerprint must not bypass again"
        );
        // Falls through to the normal eligibility check instead of forcing
        // a probe — and that check must say "not yet".
        assert!(
            !fetch_may_be_needed(&state, new_fingerprint, now2),
            "second render must respect the recorded rate-limit backoff"
        );

        // --- Render 3: after the cooldown elapses, normal scheduling
        // resumes (still not a bypass — just an ordinary eligible render).
        let now3 = now + 401;
        assert!(!is_fresh_token_change(&state, new_fingerprint));
        assert!(fetch_may_be_needed(&state, new_fingerprint, now3));
    }

    /// Same two-render shape, but the probe fails with 401 instead of 429 —
    /// `failed_token_fingerprint` (now correctly naming the attempted
    /// fingerprint per the bug-2 fix) must also suppress the second render.
    #[test]
    fn test_second_render_after_unauthorized_probe_stays_suppressed() {
        let now = 2_000_000;
        let new_fingerprint = "fp-new-token";
        let mut state = SharedUsageState::default_v2();
        state.token_fingerprint = Some("fp-old-token".to_string());

        assert!(is_fresh_token_change(&state, new_fingerprint));
        state.failed_token_fingerprint = None;
        state.next_allowed_at_epoch = now;

        apply_fetch_outcome(
            &mut state,
            crate::usage_limits::UsageFetchOutcome::Unauthorized,
            now,
            180,
            new_fingerprint,
        );
        state.token_change_probe_fingerprint = Some(new_fingerprint.to_string());

        assert_eq!(
            state.failed_token_fingerprint,
            Some(new_fingerprint.to_string())
        );

        let now2 = now + 10_000;
        assert!(!is_fresh_token_change(&state, new_fingerprint));
        assert!(
            !fetch_may_be_needed(&state, new_fingerprint, now2),
            "a token blamed via failed_token_fingerprint stays suppressed \
             like any ordinary 401'd token, even much later"
        );
    }

    #[test]
    fn test_success_after_fresh_token_change_clears_probe_marker() {
        let now = 2_000_000;
        let new_fingerprint = "fp-new-token";
        let mut state = SharedUsageState::default_v2();
        state.token_fingerprint = Some("fp-old-token".to_string());

        assert!(is_fresh_token_change(&state, new_fingerprint));
        state.failed_token_fingerprint = None;
        state.next_allowed_at_epoch = now;

        let data = UsageLimitsData::default();
        apply_fetch_outcome(
            &mut state,
            crate::usage_limits::UsageFetchOutcome::Success(Box::new(data)),
            now,
            180,
            new_fingerprint,
        );
        // Success commits the new fingerprint and clears any probe marker.
        state.token_fingerprint = Some(new_fingerprint.to_string());
        state.token_change_probe_fingerprint = None;

        assert!(!is_fresh_token_change(&state, new_fingerprint));
        assert_eq!(state.token_change_probe_fingerprint, None);
    }

    /// Regression test: a token switch must not inherit the *old* token's
    /// rate-limit cooldown. Before the fix, the fresh-token branch reset
    /// `next_allowed_at_epoch` but left `rate_limit_until_epoch` untouched;
    /// a probe that failed with a transient (non-429) error then left the
    /// new fingerprint blocked by `fetch_may_be_needed`'s separate
    /// `rate_limit_until_epoch` check for as long as the old token's cooldown
    /// had left to run.
    #[test]
    fn test_fresh_token_change_clears_inherited_rate_limit_on_transient_failure() {
        let now = 2_000_000;
        let new_fingerprint = "fp-new-token";
        let mut state = SharedUsageState::default_v2();
        state.token_fingerprint = Some("fp-old-token".to_string());
        // Old token was rate-limited well past `now`.
        state.rate_limit_until_epoch = now + 900;
        state.consecutive_errors = 3;

        assert!(is_fresh_token_change(&state, new_fingerprint));
        // Mirrors the `fresh_token_change` branch in `resolve_shared_usage`.
        state.failed_token_fingerprint = None;
        state.next_allowed_at_epoch = now;
        state.rate_limit_until_epoch = 0;
        state.consecutive_errors = 0;

        // The probe itself fails transiently (not another 429).
        apply_fetch_outcome(
            &mut state,
            crate::usage_limits::UsageFetchOutcome::NetworkError,
            now,
            60,
            new_fingerprint,
        );
        state.token_change_probe_fingerprint = Some(new_fingerprint.to_string());

        assert_eq!(
            state.rate_limit_until_epoch, 0,
            "the old token's rate-limit cooldown must not survive the switch"
        );
        assert!(
            fetch_may_be_needed(&state, new_fingerprint, now + 60),
            "the new fingerprint must be eligible again once its own transient \
             backoff elapses, not blocked by the old token's cooldown"
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
    fn test_resolve_shared_usage_no_token_returns_no_usage() {
        let resolution = resolve_shared_usage(None, None, None, 1_000_000);
        assert!(resolution.usage.is_none());
    }

    #[test]
    fn test_load_state_or_default_when_file_missing_returns_schema_v3_default() {
        let dir = tempfile::tempdir().unwrap();
        let state = load_state_from_dir(dir.path());
        assert_eq!(state.schema_version, 3);
        assert!(state.usage.is_none());
        assert_eq!(state.next_allowed_at_epoch, 0);
    }

    #[test]
    fn test_load_state_or_default_when_json_invalid_returns_default() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("usage-limits-state-v3.json"), "{not json").unwrap();
        let state = load_state_from_dir(dir.path());
        assert_eq!(state.schema_version, 3);
        assert!(state.usage.is_none());
    }

    #[test]
    fn test_load_state_or_default_when_schema_version_unknown_returns_default() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("usage-limits-state-v3.json"),
            r#"{"schema_version":99,"token_fingerprint":null,"usage":null,"next_allowed_at_epoch":0,"rate_limit_until_epoch":0,"consecutive_errors":0,"failed_token_fingerprint":null}"#,
        )
        .unwrap();
        let state = load_state_from_dir(dir.path());
        assert_eq!(state.schema_version, 3);
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
        assert_eq!(entries, vec!["usage-limits-state-v3.json"]);
    }

    #[test]
    fn test_lock_first_process_acquires() {
        let dir = tempfile::tempdir().unwrap();
        let lock = try_acquire_fetch_lock_in(dir.path());
        assert!(lock.is_some());
    }

    #[test]
    fn test_lock_concurrent_acquire_while_held_fails_immediately() {
        let dir = tempfile::tempdir().unwrap();
        let _first = try_acquire_fetch_lock_in(dir.path()).unwrap();
        let second = try_acquire_fetch_lock_in(dir.path());
        assert!(second.is_none());
    }

    #[test]
    fn test_lock_released_on_drop_can_be_immediately_reacquired() {
        let dir = tempfile::tempdir().unwrap();
        {
            let _lock = try_acquire_fetch_lock_in(dir.path()).unwrap();
        } // dropped here — OS releases the advisory lock
        let second = try_acquire_fetch_lock_in(dir.path());
        assert!(second.is_some());
    }

    /// A pre-existing lock *file* with no live lock on it (e.g. left behind
    /// by a previous run — the file is a permanent rendezvous point, never
    /// deleted) must not block acquisition: only a held OS lock blocks.
    #[test]
    fn test_lock_preexisting_unlocked_file_does_not_block() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(LOCK_FILE_NAME), "leftover").unwrap();
        let lock = try_acquire_fetch_lock_in(dir.path());
        assert!(lock.is_some());
    }

    /// Crash recovery: if the holding process dies without running our
    /// `Drop` (simulated by dropping the raw locked `File` directly, never
    /// constructing a `FetchLock`), the OS releases the lock as soon as the
    /// descriptor closes — a new acquire succeeds with no staleness wait.
    #[test]
    fn test_lock_released_by_os_when_holder_crashes() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join(LOCK_FILE_NAME);
        let crashed_holder = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .unwrap();
        crashed_holder.try_lock().unwrap();

        // While "crashed_holder" is alive and holding the lock, acquisition
        // must fail...
        assert!(try_acquire_fetch_lock_in(dir.path()).is_none());

        // ...and the moment its handle closes, the OS releases the lock.
        drop(crashed_holder);
        assert!(try_acquire_fetch_lock_in(dir.path()).is_some());
    }
}
