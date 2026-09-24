//! Finding a model by name, in the Hugging Face cache layout so downloads made by Laya or
//! `huggingface_hub` are reused and ours are reused by them.
//!
//! A name is a local path (a checkpoint directory or a `.kime` file), `hf://org/repo` with an
//! optional subfolder after it, or one of the aliases in [`ALIASES`]. Nothing here touches the
//! network. `kime pull` does the downloading, into the same layout.

use std::path::{Path, PathBuf};

/// The published compat models: alias, repo and subfolder.
pub const ALIASES: [(&str, &str, &str); 3] = [
    ("laya", "convaiinnovations/laya", ""),
    ("laya-multilingual", "convaiinnovations/laya", "multilingual"),
    ("laya-typed-decisions", "convaiinnovations/laya", "typed-decisions"),
];

/// The files a compat checkpoint needs, relative to its folder in the repo.
pub const FILES: [&str; 5] = [
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "model.safetensors",
];

/// Where a name points on the hub.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubRef {
    /// `org/repo`.
    pub repo: String,
    /// The folder inside the repo, empty for the root.
    pub subfolder: String,
}

impl HubRef {
    /// Reads an alias or an `hf://org/repo[/subfolder]` reference.
    #[must_use]
    pub fn parse(name: &str) -> Option<HubRef> {
        if let Some((_, repo, sub)) = ALIASES.iter().find(|a| a.0 == name) {
            return Some(HubRef { repo: (*repo).into(), subfolder: (*sub).into() });
        }
        let rest = name.strip_prefix("hf://")?;
        let mut parts = rest.splitn(3, '/');
        let (org, repo) = (parts.next()?, parts.next()?);
        if org.is_empty() || repo.is_empty() {
            return None;
        }
        let subfolder = parts.next().unwrap_or("").trim_matches('/').to_string();
        Some(HubRef { repo: format!("{org}/{repo}"), subfolder })
    }

    /// The repo's folder in the cache.
    #[must_use]
    pub fn repo_dir(&self, cache: &Path) -> PathBuf {
        cache.join(format!("models--{}", self.repo.replace('/', "--")))
    }

    /// The checkpoint folder of the snapshot `refs/main` points at, if there is one.
    #[must_use]
    pub fn local(&self, cache: &Path) -> Option<PathBuf> {
        let repo = self.repo_dir(cache);
        let rev = std::fs::read_to_string(repo.join("refs/main")).ok()?;
        let dir = repo.join("snapshots").join(rev.trim()).join(&self.subfolder);
        dir.join("model.safetensors").is_file().then_some(dir)
    }
}

/// The hub cache: `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`.
#[must_use]
pub fn cache_dir() -> PathBuf {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(d) = var("HF_HUB_CACHE") {
        return d;
    }
    if let Some(d) = var("HF_HOME") {
        return d.join("hub");
    }
    let home = var("HOME").or_else(|| var("USERPROFILE")).unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache/huggingface/hub")
}

/// The path a model name refers to on this machine.
///
/// # Errors
///
/// A message saying what was looked for and how to fetch it.
pub fn resolve(name: &str) -> Result<PathBuf, String> {
    let path = Path::new(name);
    if path.exists() {
        return Ok(path.to_path_buf());
    }
    let Some(r) = HubRef::parse(name) else {
        let known: Vec<&str> = ALIASES.iter().map(|a| a.0).collect();
        return Err(format!(
            "no model {name:?}: not a path, an hf:// reference or one of {}",
            known.join(", ")
        ));
    };
    let cache = cache_dir();
    r.local(&cache)
        .ok_or_else(|| format!("{name} is not in {}, run kime pull {name} first", cache.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let r = HubRef::parse("laya-multilingual").unwrap();
        assert_eq!(
            (r.repo.as_str(), r.subfolder.as_str()),
            ("convaiinnovations/laya", "multilingual")
        );
        let r = HubRef::parse("hf://org/repo/a/b/").unwrap();
        assert_eq!((r.repo.as_str(), r.subfolder.as_str()), ("org/repo", "a/b"));
        assert_eq!(HubRef::parse("hf://org"), None);
        assert_eq!(HubRef::parse("kime-v1-s-en"), None);
        assert_eq!(r.repo_dir(Path::new("/c")), Path::new("/c/models--org--repo"));
    }
}
