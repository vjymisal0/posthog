use anyhow::{bail, Context, Result};
use serde_json::{Map, Value};
use sha1::{Digest, Sha1};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use walkdir::DirEntry;

use crate::{
    api::releases::{Release, ReleaseBuilder},
    sourcemaps::{
        args::{FileSelectionArgs, ReleaseArgs, ReleaseMode},
        constant::CHUNK_ID_NAMESPACE,
        content::SourceMapFile,
        source_pairs::{read_pairs, SourcePair},
    },
    utils::{
        files::{content_hash, FileSelection},
        git::get_git_info,
    },
};

#[derive(clap::Args)]
pub struct InjectArgs {
    #[clap(flatten)]
    pub file_selection: FileSelectionArgs,

    /// If your bundler adds a public path prefix to sourcemap URLs,
    /// we need to ignore it while searching for them
    /// For use alongside e.g. esbuilds "publicPath" config setting.
    #[arg(short, long)]
    pub public_path_prefix: Option<String>,

    #[clap(flatten)]
    pub release: ReleaseArgs,

    /// How the release is associated with exceptions. `symbol-set` (the default) stamps the
    /// release id into the sourcemap so the uploaded symbol set is bound to it: the previous
    /// behavior. EXPERIMENTAL `event` injects the release id into each chunk as
    /// `_posthogReleaseId` so the SDK emits it on every exception, and derives
    /// content-addressed chunk ids that are stable across rebuilds. Also settable via
    /// `POSTHOG_RELEASE_MODE`.
    #[arg(
        long,
        env = "POSTHOG_RELEASE_MODE",
        value_enum,
        default_value = "symbol-set"
    )]
    pub release_mode: ReleaseMode,
}

impl InjectArgs {
    pub fn validate(&self) -> Result<()> {
        self.file_selection.validate()
    }
}

pub fn inject_impl(
    args: &InjectArgs,
    matcher: impl Fn(&DirEntry) -> bool + 'static,
    existing_release: Option<&Release>,
) -> Result<()> {
    let InjectArgs {
        file_selection,
        public_path_prefix,
        release,
        release_mode,
    } = args;

    info!("injecting selection: {}", file_selection);

    // Resolve stdin once. We also need the concrete roots after injection to find and repair
    // Angular service-worker manifests that track the rewritten chunks.
    let file_selection = file_selection.clone().resolve_stdin()?;
    let iterator = FileSelection::try_from(file_selection.clone())?;

    let mut pairs = read_pairs(
        iterator.into_iter().filter(|entry| matcher(entry)),
        public_path_prefix,
    );
    if pairs.is_empty() {
        bail!("no source files found");
    }

    // Hash each chunk before injection so we can report exactly which files injection rewrites.
    let source_hashes_before = source_content_hashes(&pairs);

    match release_mode {
        ReleaseMode::Event => {
            // The release id travels inside each chunk for the SDK to emit, rather than being
            // stamped into the sourcemap, so the release exists but nothing binds a symbol set
            // to it.
            let release_id = resolve_release_id(release.clone(), existing_release)?;
            if release_id.is_none() {
                warn!(
                    "no release could be resolved, injecting chunk ids only — events will carry no release"
                );
            }
            pairs = inject_pairs(pairs, release_id.as_deref())?;
        }
        ReleaseMode::SymbolSet => {
            // Fetch or create a release over the API and stamp its id into the sourcemap,
            // binding the uploaded symbol set to it.
            let created_release_id = if let Some(r) = existing_release {
                Some(r.id.to_string())
            } else {
                let cwd = std::env::current_dir()?;
                get_release_for_maps(&cwd, release.clone(), pairs.iter().map(|p| &p.sourcemap))?
                    .as_ref()
                    .map(|r| r.id.to_string())
            };
            pairs = inject_pairs_legacy(pairs, created_release_id)?;
        }
    }

    let rewritten = rewritten_source_paths(&pairs, &source_hashes_before);

    // Write the source and sourcemaps back to disk.
    for pair in &pairs {
        pair.save()?;
    }

    // Angular generates ngsw.json before this command runs. Keep its SHA-1 entries in sync with
    // the chunks we just changed so existing clients accept the new application version.
    let repaired_manifests =
        update_angular_service_worker_manifests(&file_selection.directory, &rewritten)?;

    info!("injecting done");
    warn_about_rewritten_files(&rewritten, &repaired_manifests);
    Ok(())
}

/// Map each chunk's source path to a hash of its content, for comparing before and after injection.
fn source_content_hashes(pairs: &[SourcePair]) -> HashMap<PathBuf, String> {
    pairs
        .iter()
        .map(|pair| {
            (
                pair.source.inner.path.clone(),
                content_hash([pair.source.inner.content.as_bytes()]),
            )
        })
        .collect()
}

/// Warn when injection changed built files, so pipelines that pin a content hash regenerate it.
///
/// Injection appends a chunk id (and, in event mode, the release id) to each chunk, then writes the
/// files back in place. Any hash computed before injection — a service worker manifest, a
/// Subresource Integrity attribute, a deploy manifest — no longer matches the bytes on disk, which
/// breaks the deploy with no other signal. Report only the chunks whose content actually changed, so
/// re-runs over an already injected build stay quiet.
fn warn_about_rewritten_files(rewritten: &[PathBuf], repaired_manifests: &[PathBuf]) {
    if rewritten.is_empty() {
        return;
    }

    if repaired_manifests.is_empty() {
        warn!(
            "injection rewrote {} built file(s) in place. Any asset hash computed before this step \
             (service worker manifest, Subresource Integrity attribute, deploy manifest) no longer \
             matches and must be regenerated after injecting:",
            rewritten.len()
        );
    } else {
        warn!(
            "injection rewrote {} built file(s) in place and updated {} Angular service-worker \
             manifest(s). Any other asset hash computed before this step (Subresource Integrity \
             attribute, deploy manifest) must still be regenerated:",
            rewritten.len(),
            repaired_manifests.len()
        );
    }
    for path in rewritten {
        warn!("  rewrote {}", path.display());
    }
}

/// Return the source paths whose content differs from `hashes_before`, so a re-run over an already
/// injected build reports nothing.
fn rewritten_source_paths(
    pairs: &[SourcePair],
    hashes_before: &HashMap<PathBuf, String>,
) -> Vec<PathBuf> {
    pairs
        .iter()
        .filter(|pair| {
            let hash_now = content_hash([pair.source.inner.content.as_bytes()]);
            hashes_before
                .get(&pair.source.inner.path)
                .map(String::as_str)
                != Some(&hash_now)
        })
        .map(|pair| pair.source.inner.path.clone())
        .collect()
}

/// Update matching SHA-1 entries in Angular's generated `ngsw.json` manifests.
///
/// Angular writes the manifest as part of `ng build`, before sourcemap injection can run. Without
/// this repair, its service worker rejects every injected chunk because the bytes no longer match
/// the recorded hash, then leaves existing clients on the previous application version.
pub(crate) fn update_angular_service_worker_manifests(
    selection_roots: &[PathBuf],
    rewritten: &[PathBuf],
) -> Result<Vec<PathBuf>> {
    if rewritten.is_empty() {
        return Ok(Vec::new());
    }

    let manifests = find_angular_service_worker_manifests(selection_roots);
    let mut repaired = Vec::new();

    for manifest_path in manifests {
        let bytes = std::fs::read(&manifest_path)
            .with_context(|| format!("Failed to read {}", manifest_path.display()))?;
        let mut manifest: Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("Failed to parse {}", manifest_path.display()))?;
        let hash_table = manifest
            .get_mut("hashTable")
            .and_then(Value::as_object_mut)
            .with_context(|| format!("{} has no hashTable object", manifest_path.display()))?;
        let manifest_root = manifest_path
            .parent()
            .expect("ngsw.json always has a parent directory")
            .canonicalize()
            .with_context(|| {
                format!(
                    "Failed to resolve Angular build directory {}",
                    manifest_path.display()
                )
            })?;

        let updated = update_manifest_hash_table(hash_table, &manifest_root, rewritten)?;
        if updated == 0 {
            continue;
        }

        std::fs::write(&manifest_path, serde_json::to_vec(&manifest)?)
            .with_context(|| format!("Failed to update {}", manifest_path.display()))?;
        info!(
            "updated {updated} asset hash(es) in Angular service-worker manifest {}",
            manifest_path.display()
        );
        repaired.push(manifest_path);
    }

    Ok(repaired)
}

fn find_angular_service_worker_manifests(selection_roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut manifests = HashSet::new();
    for root in selection_roots {
        let scan_root = if root.is_dir() {
            root.as_path()
        } else {
            root.parent().unwrap_or(root.as_path())
        };
        for entry in walkdir::WalkDir::new(scan_root)
            .into_iter()
            .filter_map(|entry| entry.ok())
        {
            if entry.file_type().is_file() && entry.file_name() == "ngsw.json" {
                manifests.insert(entry.into_path());
            }
        }
    }
    manifests.into_iter().collect()
}

fn update_manifest_hash_table(
    hash_table: &mut Map<String, Value>,
    manifest_root: &Path,
    rewritten: &[PathBuf],
) -> Result<usize> {
    let mut updated = 0;

    for source_path in rewritten {
        let canonical_source = source_path.canonicalize().with_context(|| {
            format!("Failed to resolve rewritten file {}", source_path.display())
        })?;
        let Ok(relative_path) = canonical_source.strip_prefix(manifest_root) else {
            continue;
        };
        let relative_url = relative_path.to_string_lossy().replace('\\', "/");
        let Some(key) = manifest_key_for_relative_path(hash_table, &relative_url)? else {
            continue;
        };
        let digest = sha1_hex(&std::fs::read(&canonical_source).with_context(|| {
            format!(
                "Failed to hash rewritten file {}",
                canonical_source.display()
            )
        })?);
        hash_table.insert(key, Value::String(digest));
        updated += 1;
    }

    Ok(updated)
}

fn manifest_key_for_relative_path(
    hash_table: &Map<String, Value>,
    relative_url: &str,
) -> Result<Option<String>> {
    let exact_key = format!("/{relative_url}");
    if hash_table.contains_key(&exact_key) {
        return Ok(Some(exact_key));
    }

    // A non-root Angular base href prefixes every manifest URL (for example,
    // `/my-app/main.js`). The generated manifest does not record that prefix separately, so use
    // a unique path-suffix match. Refuse ambiguous matches instead of changing the wrong asset.
    let suffix = format!("/{relative_url}");
    let matches = hash_table
        .keys()
        .filter(|key| urlencoding::decode(key).is_ok_and(|decoded| decoded.ends_with(&suffix)))
        .cloned()
        .collect::<Vec<_>>();

    match matches.as_slice() {
        [] => Ok(None),
        [key] => Ok(Some(key.clone())),
        _ => bail!(
            "Angular service-worker manifest has multiple entries matching {relative_url}: {}",
            matches.join(", ")
        ),
    }
}

fn sha1_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha1::digest(bytes))
}

/// Event-mode injection (`--release-mode=event`): content-addressed chunk ids plus an optional
/// `_posthogReleaseId` payload. A bundler-emitted debug id, when present, is adopted as the
/// chunk id so one id identifies the chunk across the whole toolchain.
pub fn inject_pairs(
    mut pairs: Vec<SourcePair>,
    release_id: Option<&str>,
) -> Result<Vec<SourcePair>> {
    for pair in &mut pairs {
        let Some(chunk_id) = pair.get_chunk_id() else {
            let chunk_id = adopted_debug_id(pair)
                .unwrap_or_else(|| stable_chunk_id(&pair.source.inner.content));
            pair.add_chunk_id(chunk_id, release_id)?;
            continue;
        };

        // Already injected: the chunk id is content-addressed and the content didn't change,
        // so keep it — but refresh the embedded release id when a different release resolved,
        // or a re-run over an existing dist would keep reporting the old release on every
        // event. When no release resolves, leave the pair untouched: failing to resolve is
        // missing information (e.g. no git context), not evidence the embedded id is stale.
        let Some(release_id) = release_id else {
            continue;
        };
        if pair.get_injected_release_id().as_deref() == Some(release_id) {
            continue;
        }
        pair.remove_chunk_id(chunk_id.clone())?;
        pair.add_chunk_id(chunk_id, Some(release_id))?;
    }

    Ok(pairs)
}

/// A bundler-emitted ECMA-426 debug id is already content-derived, so adopt it as the chunk id
/// instead of deriving our own. Non-UUID values are refused: the id flows into upload rows and
/// SDK events, and a malformed one is worse than a derived one.
fn adopted_debug_id(pair: &SourcePair) -> Option<String> {
    let debug_id = pair.get_debug_id()?;
    if uuid::Uuid::parse_str(&debug_id).is_err() {
        warn!(
            "ignoring malformed debug id {:?} on {} — falling back to a content-derived chunk id",
            debug_id,
            pair.source.inner.path.display()
        );
        return None;
    }
    Some(debug_id)
}

/// Symbol-set-mode injection (the default): a random per-build chunk id and the created release
/// id stamped into the sourcemap. Regenerates the chunk id whenever the release id changes or is
/// missing.
pub fn inject_pairs_legacy(
    mut pairs: Vec<SourcePair>,
    created_release_id: Option<String>,
) -> Result<Vec<SourcePair>> {
    for pair in &mut pairs {
        let current_release_id = pair.get_release_id();
        // We only update release ids and chunk ids when the release id changed or is not present
        if current_release_id != created_release_id || pair.get_chunk_id().is_none() {
            pair.set_release_id(created_release_id.clone());

            let chunk_id = uuid::Uuid::now_v7().to_string();
            if let Some(previous_chunk_id) = pair.get_chunk_id() {
                pair.update_chunk_id(previous_chunk_id, chunk_id)?;
            } else {
                pair.add_chunk_id(chunk_id, None)?;
            }
        }
    }

    Ok(pairs)
}

/// Deterministically derive a chunk id from the pristine minified source (UUIDv5). Identical
/// builds produce identical ids on every machine and rebuild, so uploads dedupe instead of
/// minting a per-build random id.
///
/// The sourcemap is deliberately not part of the identity. It carries `sourcesContent`, so a
/// comment-only edit rewrites the map while the minified code stays byte-identical, and folding
/// the map in would mint a new chunk for code that never changed. The map still reaches the
/// server, because the upload hashes the payload it sends: a map-only change is a content
/// change, and event mode overwrites the stored symbol set with the newer map.
fn stable_chunk_id(source_content: &str) -> String {
    uuid::Uuid::new_v5(&CHUNK_ID_NAMESPACE, source_content.as_bytes()).to_string()
}

/// Resolve the release row whose id gets injected into the chunks. Reuses the release already
/// fetched upstream (the `process` command) when there is one; otherwise resolves name/version
/// from flags and git/CI metadata and fetches or creates the row. Returns `None` only when there
/// isn't enough information to identify a release at all.
fn resolve_release_id(
    release: ReleaseArgs,
    existing_release: Option<&Release>,
) -> Result<Option<String>> {
    if let Some(r) = existing_release {
        return Ok(Some(r.id.to_string()));
    }
    Ok(resolve_release(release)?.map(|r| r.id.to_string()))
}

/// Fetch or create the release identified by `release`, filling in whichever of name and version
/// the flags left out from git and CI metadata. Returns `None` when neither source identifies a
/// release.
///
/// Shared with the `release resolve` command, so a build tool that injects the release id itself
/// lands on the same row `sourcemap inject --release-mode=event` would have injected.
pub fn resolve_release(release: ReleaseArgs) -> Result<Option<Release>> {
    let cwd = std::env::current_dir()?;
    let release = release.resolve_info_plist()?;
    let mut builder: ReleaseBuilder = release.into();
    add_git_info_to_release_builder(&cwd, &mut builder)?;
    if !builder.can_create() {
        return Ok(None);
    }
    Ok(Some(builder.fetch_or_create()?))
}

pub fn get_release_for_maps<'a>(
    directory: &Path,
    release: ReleaseArgs,
    maps: impl IntoIterator<Item = &'a SourceMapFile>,
) -> Result<Option<Release>> {
    // We need to fetch or create a release if: the user specified one, any pair is missing one, or the user
    // forced release overriding
    let release = release.resolve_info_plist()?;
    let needs_release = release.name.is_some()
        || release.version.is_some()
        || release.build.is_some()
        || maps.into_iter().any(|p| !p.has_release_id());

    let mut created_release = None;
    if needs_release {
        let mut builder: ReleaseBuilder = release.into();

        add_git_info_to_release_builder(directory, &mut builder)?;

        if builder.can_create() {
            created_release = Some(builder.fetch_or_create()?);
        }
    }

    Ok(created_release)
}

fn add_git_info_to_release_builder(directory: &Path, builder: &mut ReleaseBuilder) -> Result<()> {
    let needs_git_for_release_fields = !builder.can_create();
    let release_fields_were_provided = builder.has_name() || builder.has_version();

    match get_git_info(Some(directory.to_path_buf())) {
        Ok(Some(info)) => {
            builder.with_git(info);
        }
        Ok(None) if needs_git_for_release_fields && release_fields_were_provided => {
            anyhow::bail!(
                "Release fields are incomplete and git info is unavailable. Provide both --release-name and --release-version, or run from a git repository or supported CI environment."
            );
        }
        Ok(None) => {}
        Err(error) if needs_git_for_release_fields => {
            return Err(error).context("Failed to determine git info for release");
        }
        Err(error) => {
            warn!("Skipping git metadata after failing to determine git info: {error:#}");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::{Mutex, MutexGuard},
    };

    use super::*;
    use crate::sourcemaps::plain::inject::is_javascript_file;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    const XCODE_RELEASE_ENV_VARS: &[&str] = &[
        "PRODUCT_BUNDLE_IDENTIFIER",
        "MARKETING_VERSION",
        "CURRENT_PROJECT_VERSION",
    ];

    const GIT_INFO_ENV_VARS: &[&str] = &[
        "GITHUB_ACTIONS",
        "GITHUB_SHA",
        "GITHUB_REF_NAME",
        "GITHUB_REPOSITORY",
        "GITHUB_SERVER_URL",
        "VERCEL",
        "VERCEL_GIT_PROVIDER",
        "VERCEL_GIT_REPO_OWNER",
        "VERCEL_GIT_REPO_SLUG",
        "VERCEL_GIT_COMMIT_REF",
        "VERCEL_GIT_COMMIT_SHA",
    ];

    fn lock_env() -> MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn remove_env_vars(names: &[&str]) -> Vec<(String, Option<String>)> {
        names
            .iter()
            .map(|name| {
                let value = std::env::var(name).ok();
                std::env::remove_var(name);
                ((*name).to_string(), value)
            })
            .collect()
    }

    struct EnvVarGuard(Vec<(String, Option<String>)>);

    impl EnvVarGuard {
        fn clear(names: &[&str]) -> Self {
            Self(remove_env_vars(names))
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            for (name, value) in self.0.drain(..) {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }

    fn release_args(name: Option<&str>, version: Option<&str>) -> ReleaseArgs {
        release_args_with_build(name, version, None)
    }

    fn release_args_with_build(
        name: Option<&str>,
        version: Option<&str>,
        build: Option<&str>,
    ) -> ReleaseArgs {
        ReleaseArgs {
            name: name.map(String::from),
            version: version.map(String::from),
            build: build.map(String::from),
            info_plist: None,
            skip_release_on_fail: true,
        }
    }

    fn make_git_repo_without_branch_ref() -> tempfile::TempDir {
        let temp_root = tempfile::tempdir().expect("failed to create temporary repo");
        let git_dir = temp_root.path().join(".git");

        fs::create_dir_all(&git_dir).expect("failed to create .git directory");
        fs::write(git_dir.join("HEAD"), "ref: refs/heads/main\n").expect("failed to write HEAD");

        temp_root
    }

    fn chunk_id_for(sourcemap: &str) -> String {
        let dir = tempfile::tempdir().expect("failed to create temporary directory");
        fs::write(
            dir.path().join("app.js"),
            "console.log(1);\n//# sourceMappingURL=app.js.map\n",
        )
        .expect("failed to write source");
        fs::write(dir.path().join("app.js.map"), sourcemap).expect("failed to write sourcemap");

        let selection = FileSelection::from_roots(vec![dir.path().to_path_buf()])
            .include(vec![])
            .expect("failed to build selection")
            .exclude(vec![])
            .expect("failed to build selection");
        let pairs = read_pairs(selection.into_iter().filter(is_javascript_file), &None);

        inject_pairs(pairs, None)
            .expect("failed to inject pairs")
            .first()
            .and_then(SourcePair::get_chunk_id)
            .expect("injected pair carries a chunk id")
    }

    #[test]
    fn rewritten_paths_flag_injection_and_stay_quiet_on_reruns() {
        let dir = tempfile::tempdir().expect("failed to create temporary directory");
        fs::write(
            dir.path().join("app.js"),
            "console.log(1);\n//# sourceMappingURL=app.js.map\n",
        )
        .expect("failed to write source");
        fs::write(
            dir.path().join("app.js.map"),
            r#"{"version":3,"sources":["app.ts"],"sourcesContent":["console.log(1)\n"],"mappings":"AAAA","names":[]}"#,
        )
        .expect("failed to write sourcemap");

        let selection = FileSelection::from_roots(vec![dir.path().to_path_buf()])
            .include(vec![])
            .expect("failed to build selection")
            .exclude(vec![])
            .expect("failed to build selection");
        let pairs = read_pairs(selection.into_iter().filter(is_javascript_file), &None);

        let before = source_content_hashes(&pairs);
        let injected = inject_pairs(pairs, None).expect("failed to inject pairs");

        // First injection appends a chunk id, so the chunk is reported as rewritten.
        assert_eq!(rewritten_source_paths(&injected, &before).len(), 1);

        // A second pass over the already injected build changes nothing, so it stays quiet.
        let before_rerun = source_content_hashes(&injected);
        let reinjected = inject_pairs(injected, None).expect("failed to re-inject pairs");
        assert!(rewritten_source_paths(&reinjected, &before_rerun).is_empty());
    }

    #[test]
    fn angular_service_worker_hashes_follow_rewritten_chunks() {
        let dir = tempfile::tempdir().expect("failed to create temporary directory");
        let chunks = dir.path().join("chunks");
        fs::create_dir(&chunks).expect("failed to create chunks directory");
        let app = dir.path().join("app.js");
        let lazy = chunks.join("lazy chunk.js");
        fs::write(&app, "injected app").expect("failed to write app chunk");
        fs::write(&lazy, "injected lazy chunk").expect("failed to write lazy chunk");
        fs::write(
            dir.path().join("ngsw.json"),
            serde_json::to_vec(&serde_json::json!({
                "configVersion": 1,
                "hashTable": {
                    "/base/app.js": "old-app-hash",
                    "/base/chunks/lazy%20chunk.js": "old-lazy-hash",
                    "/base/unchanged.js": "unchanged-hash"
                }
            }))
            .expect("failed to serialize manifest"),
        )
        .expect("failed to write manifest");

        let repaired = update_angular_service_worker_manifests(
            &[dir.path().to_path_buf()],
            &[app.clone(), lazy.clone()],
        )
        .expect("failed to repair Angular manifest");

        assert_eq!(repaired, vec![dir.path().join("ngsw.json")]);
        let manifest: Value = serde_json::from_slice(
            &fs::read(dir.path().join("ngsw.json")).expect("failed to read repaired manifest"),
        )
        .expect("failed to parse repaired manifest");
        let hash_table = manifest["hashTable"]
            .as_object()
            .expect("manifest hashTable");
        assert_eq!(
            hash_table["/base/app.js"],
            Value::String(sha1_hex(b"injected app"))
        );
        assert_eq!(
            hash_table["/base/chunks/lazy%20chunk.js"],
            Value::String(sha1_hex(b"injected lazy chunk"))
        );
        assert_eq!(hash_table["/base/unchanged.js"], "unchanged-hash");
    }

    #[test]
    fn angular_service_worker_hash_repair_refuses_ambiguous_base_paths() {
        let mut hash_table = serde_json::from_value::<Map<String, Value>>(serde_json::json!({
            "/one/app.js": "one",
            "/two/app.js": "two"
        }))
        .expect("failed to create hash table");

        let error = manifest_key_for_relative_path(&hash_table, "app.js")
            .expect_err("ambiguous paths must not update the wrong asset");

        assert!(error.to_string().contains("multiple entries"));
        hash_table.insert("/app.js".to_string(), Value::String("exact".to_string()));
        assert_eq!(
            manifest_key_for_relative_path(&hash_table, "app.js")
                .expect("exact path should win")
                .as_deref(),
            Some("/app.js")
        );
    }

    #[test]
    fn map_only_changes_keep_the_chunk_id() {
        // Bundlers embed the original file in `sourcesContent`, so editing a comment rewrites
        // the map while the minified code stays byte-identical. Folding the map into the id
        // would mint a new chunk on every such edit and orphan the symbol set already stored.
        let one = chunk_id_for(
            r#"{"version":3,"sources":["app.ts"],"sourcesContent":["// one\nconsole.log(1)\n"],"mappings":"AAAA","names":[]}"#,
        );
        let two = chunk_id_for(
            r#"{"version":3,"sources":["app.ts"],"sourcesContent":["// two\nconsole.log(1)\n"],"mappings":"AAAA","names":[]}"#,
        );

        assert_eq!(one, two);
    }

    #[test]
    fn stable_chunk_id_tracks_the_minified_source() {
        let id = stable_chunk_id("code();");

        assert_eq!(id, stable_chunk_id("code();"));
        assert_ne!(id, stable_chunk_id("other();"));
    }

    #[test]
    fn git_failure_is_not_fatal_when_release_fields_are_explicit() {
        let _env_lock = lock_env();
        let _env_guard = EnvVarGuard::clear(GIT_INFO_ENV_VARS);
        let temp_root = make_git_repo_without_branch_ref();
        let mut builder: ReleaseBuilder = release_args(Some("my-app"), Some("1.0.0")).into();

        let result = add_git_info_to_release_builder(temp_root.path(), &mut builder);

        assert!(result.is_ok());
        assert!(builder.can_create());
    }

    #[test]
    fn git_failure_is_fatal_when_release_fields_need_git() {
        let _env_lock = lock_env();
        let _env_guard = EnvVarGuard::clear(GIT_INFO_ENV_VARS);
        let temp_root = make_git_repo_without_branch_ref();
        let mut builder: ReleaseBuilder = release_args(Some("my-app"), None).into();

        let error = add_git_info_to_release_builder(temp_root.path(), &mut builder)
            .expect_err("git failure should remain fatal when release fields are incomplete");

        assert!(format!("{error:#}").contains("Failed to determine git info for release"));
    }

    #[test]
    fn missing_git_is_not_fatal_for_best_effort_release_creation() {
        let _env_lock = lock_env();
        let _env_guard = EnvVarGuard::clear(GIT_INFO_ENV_VARS);
        let temp_root = tempfile::tempdir().expect("failed to create temporary directory");
        let mut builder: ReleaseBuilder = release_args(None, None).into();

        let result = add_git_info_to_release_builder(temp_root.path(), &mut builder);

        assert!(result.is_ok());
        assert!(!builder.can_create());
    }

    #[test]
    fn unresolved_info_plist_is_not_fatal_without_git_or_xcode_environment() {
        let _env_lock = lock_env();
        let _git_env_guard = EnvVarGuard::clear(GIT_INFO_ENV_VARS);
        let _xcode_env_guard = EnvVarGuard::clear(XCODE_RELEASE_ENV_VARS);
        let temp_root = tempfile::tempdir().expect("failed to create temporary directory");
        let info_plist = temp_root.path().join("Info.plist");
        fs::write(
            &info_plist,
            r#"<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>$(PRODUCT_BUNDLE_IDENTIFIER)</string>
    <key>CFBundleShortVersionString</key>
    <string>$(MARKETING_VERSION)</string>
    <key>CFBundleVersion</key>
    <string>$(CURRENT_PROJECT_VERSION)</string>
</dict>
</plist>"#,
        )
        .expect("failed to write Info.plist");
        let mut args = release_args(None, None);
        args.info_plist = Some(info_plist);
        let resolved = args
            .resolve_info_plist()
            .expect("Info.plist should be readable");
        let mut builder: ReleaseBuilder = resolved.into();

        let result = add_git_info_to_release_builder(temp_root.path(), &mut builder);

        assert!(result.is_ok());
        assert!(!builder.can_create());
    }

    #[test]
    fn missing_git_is_fatal_when_release_args_are_incomplete() {
        let _env_lock = lock_env();
        let _env_guard = EnvVarGuard::clear(GIT_INFO_ENV_VARS);
        let temp_root = tempfile::tempdir().expect("failed to create temporary directory");
        let mut builder: ReleaseBuilder = release_args(Some("my-app"), None).into();

        let error = add_git_info_to_release_builder(temp_root.path(), &mut builder)
            .expect_err("missing git should be fatal when release args are incomplete");

        assert!(format!("{error:#}").contains("Release fields are incomplete"));
    }

    #[test]
    fn build_only_release_args_need_git() {
        let _env_lock = lock_env();
        let _env_guard = EnvVarGuard::clear(GIT_INFO_ENV_VARS);
        let temp_root = tempfile::tempdir().expect("failed to create temporary directory");

        let error = get_release_for_maps(
            temp_root.path(),
            release_args_with_build(None, None, Some("42")),
            std::iter::empty::<&SourceMapFile>(),
        )
        .expect_err("build-only release args should need git to fill the release name");

        assert!(format!("{error:#}").contains("Release fields are incomplete"));
    }
}
