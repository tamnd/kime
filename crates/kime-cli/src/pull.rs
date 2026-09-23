//! `kime pull`: download a model into the Hugging Face cache, laid out the way `huggingface_hub`
//! lays it out (`blobs/<etag>`, `snapshots/<commit>/<path>` linking to the blob, `refs/<branch>`),
//! so Laya and kime share downloads either way.
//!
//! The transfers go through the system `curl`, which every supported OS ships, rather than an HTTP
//! stack in the binary. A token in `HF_TOKEN` is handed to curl on stdin, never on its command
//! line. LFS files are named by their SHA-256 and are checked against it before they are kept.

use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, ExitCode, Stdio};
use std::time::Instant;

use kime::hub::{self, FILES, HubRef};
use serde_json::Value;
use sha2::{Digest, Sha256};

const USAGE: &str = "usage: kime pull <laya | laya-multilingual | org/repo[/subfolder] | hf://org/repo[/subfolder]> [--revision main]
The cache is $HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub. HF_TOKEN is sent when set, and HF_ENDPOINT replaces https://huggingface.co.";

pub(crate) fn run(args: &[String]) -> ExitCode {
    match pull(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kime pull: {e}");
            ExitCode::FAILURE
        }
    }
}

fn pull(args: &[String]) -> Result<(), String> {
    let (mut name, mut revision) = (None, "main".to_string());
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--revision" => revision = it.next().cloned().ok_or("--revision needs a value")?,
            s if s.starts_with('-') || name.is_some() => return Err(USAGE.into()),
            s => name = Some(s.to_string()),
        }
    }
    let name = name.ok_or(USAGE)?;
    let r = HubRef::parse(&name)
        .or_else(|| HubRef::parse(&format!("hf://{name}")))
        .ok_or_else(|| format!("{name:?} is not an alias or org/repo\n{USAGE}"))?;
    let endpoint = std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".into());
    let endpoint = endpoint.trim_end_matches('/');
    let t = Instant::now();

    let info = curl(&format!("{endpoint}/api/models/{}/revision/{revision}", r.repo), &[])
        .map_err(|e| match e.contains("error: 401") || e.contains("error: 404") {
            true => format!(
                "{} at {revision} is not on the hub, or is private and HF_TOKEN is not set",
                r.repo
            ),
            false => e,
        })?;
    let info: Value = serde_json::from_slice(&info).map_err(|e| {
        format!("{}: the hub answered with something that is not JSON: {e}", r.repo)
    })?;
    let commit = info["sha"]
        .as_str()
        .filter(|s| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        .ok_or_else(|| format!("{}: no commit for revision {revision}", r.repo))?
        .to_string();

    let repo = r.repo_dir(&hub::cache_dir());
    let snapshot = repo.join("snapshots").join(&commit);
    std::fs::create_dir_all(repo.join("blobs")).map_err(|e| format!("{}: {e}", repo.display()))?;
    let (mut fetched, mut bytes) = (0, 0u64);
    for file in FILES {
        let path = if r.subfolder.is_empty() {
            file.to_string()
        } else {
            format!("{}/{file}", r.subfolder)
        };
        let dst = snapshot.join(&path);
        if dst.is_file() {
            continue;
        }
        let url = format!("{endpoint}/{}/resolve/{commit}/{path}", r.repo);
        let etag = etag(&url)?;
        let blob = repo.join("blobs").join(&etag);
        if !blob.is_file() {
            let part = blob.with_extension("incomplete");
            download(&url, &part)?;
            // LFS files are named by their SHA-256. Git files by a SHA-1 of the git blob, which the
            // TLS connection to the hub already vouches for.
            if etag.len() == 64 {
                let got = sha256(&part)?;
                if got != etag {
                    let _ = std::fs::remove_file(&part);
                    return Err(format!("{path}: SHA-256 is {got}, the hub says {etag}"));
                }
            }
            std::fs::rename(&part, &blob).map_err(|e| format!("{}: {e}", blob.display()))?;
            fetched += 1;
            bytes += std::fs::metadata(&blob).map(|m| m.len()).unwrap_or(0);
        }
        link(&blob, &dst, path.matches('/').count())?;
    }
    if !revision.bytes().all(|b| b.is_ascii_hexdigit()) || revision.len() != 40 {
        let refs = repo.join("refs").join(&revision);
        std::fs::create_dir_all(refs.parent().unwrap_or(&repo)).map_err(|e| e.to_string())?;
        std::fs::write(&refs, &commit).map_err(|e| format!("{}: {e}", refs.display()))?;
    }
    let dir = if r.subfolder.is_empty() { snapshot } else { snapshot.join(&r.subfolder) };
    println!(
        "{name} at {}: {fetched} files, {:.1} MB in {:.1?}, in {}",
        &commit[..12],
        bytes as f64 / 1e6,
        t.elapsed(),
        dir.display()
    );
    Ok(())
}

/// Runs curl with the URL and, when set, the token in a config on stdin, and returns stdout.
fn curl(url: &str, extra: &[&str]) -> Result<Vec<u8>, String> {
    let mut config = format!("url = \"{}\"\n", url.replace('\\', "\\\\").replace('"', "\\\""));
    if let Ok(token) = std::env::var("HF_TOKEN")
        && !token.is_empty()
    {
        config.push_str(&format!("header = \"Authorization: Bearer {}\"\n", token.trim()));
    }
    let mut child = Command::new("curl")
        .args(["--silent", "--show-error", "--fail", "--proto", "=https,http", "--config", "-"])
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not run curl: {e}"))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(config.as_bytes()).map_err(|e| format!("curl: {e}"))?;
    }
    let out = child.wait_with_output().map_err(|e| format!("curl: {e}"))?;
    if !out.status.success() {
        return Err(format!("{url}: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(out.stdout)
}

/// The blob name of a file: the hub's ETag, or for LFS files the SHA-256 in `X-Linked-Etag`.
fn etag(url: &str) -> Result<String, String> {
    let head = curl(url, &["--head", "--max-redirs", "0"])?;
    let head = String::from_utf8_lossy(&head);
    let header = |name: &str| {
        head.lines().find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim()
                .eq_ignore_ascii_case(name)
                .then(|| v.trim().trim_start_matches("W/").trim_matches('"').to_string())
        })
    };
    let tag = header("x-linked-etag")
        .or_else(|| header("etag"))
        .ok_or_else(|| format!("{url}: no ETag"))?;
    if tag.is_empty() || !tag.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
        return Err(format!("{url}: unexpected ETag {tag:?}"));
    }
    Ok(tag)
}

fn download(url: &str, to: &Path) -> Result<(), String> {
    let to_s = to.to_str().ok_or_else(|| format!("{}: not UTF-8", to.display()))?;
    curl(url, &["--location", "--retry", "3", "--output", to_s]).map(|_| ())
}

fn sha256(path: &Path) -> Result<String, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (mut h, mut buf) = (Sha256::new(), vec![0u8; 1 << 20]);
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Puts the blob at `dst` in the snapshot: a relative symlink where the OS allows one, as
/// `huggingface_hub` does, and a copy elsewhere. `depth` is the number of folders in the file's
/// path inside the repo.
fn link(blob: &Path, dst: &Path, depth: usize) -> Result<(), String> {
    let parent = dst.parent().ok_or("snapshot path has no parent")?;
    std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    let _ = std::fs::remove_file(dst);
    #[cfg(unix)]
    {
        let mut rel = std::path::PathBuf::new();
        for _ in 0..depth + 2 {
            rel.push("..");
        }
        rel.push("blobs");
        rel.push(blob.file_name().ok_or("blob has no name")?);
        std::os::unix::fs::symlink(&rel, dst).map_err(|e| format!("{}: {e}", dst.display()))
    }
    #[cfg(not(unix))]
    {
        let _ = depth;
        std::fs::copy(blob, dst).map(|_| ()).map_err(|e| format!("{}: {e}", dst.display()))
    }
}
