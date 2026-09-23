//! The unsafe rule from spec/07-engine.md, checked rather than remembered.
//!
//! `unsafe` is allowed in six crates: the tensor layer, the four backends and the tokenizer. Every
//! other crate has to say `#![forbid(unsafe_code)]` at its root, so that wanting `unsafe` somewhere
//! else is a compile error and a design conversation rather than something a reviewer has to spot.
//! The other half of the rule, a `// SAFETY:` comment on every block in the six, is clippy's
//! `undocumented_unsafe_blocks`.

use std::path::Path;

const ALLOWED: &[&str] =
    &["kime-tensor", "kime-cuda", "kime-metal", "kime-ane", "kime-cpu", "kime-tok"];

pub(crate) fn check(root: &Path) -> Result<(), String> {
    let crates = root.join("crates");
    let mut entries: Vec<_> = std::fs::read_dir(&crates)
        .map_err(|e| format!("could not read {}: {e}", crates.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();

    let mut problems = Vec::new();
    let mut checked = 0;
    for dir in entries {
        let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string();
        if ALLOWED.contains(&name.as_str()) {
            continue;
        }
        for root_file in ["src/lib.rs", "src/main.rs"] {
            let path = dir.join(root_file);
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            checked += 1;
            if !text.lines().any(|l| l.trim() == "#![forbid(unsafe_code)]") {
                problems.push(format!("crates/{name}/{root_file} does not forbid unsafe code"));
            }
        }
    }

    if problems.is_empty() {
        println!("{checked} crate roots forbid unsafe, and {} crates may use it", ALLOWED.len());
        Ok(())
    } else {
        for problem in &problems {
            eprintln!("  {problem}");
        }
        Err(format!("{} crate roots break the unsafe rule", problems.len()))
    }
}
