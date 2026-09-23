//! Build and check tasks.
//!
//! Everything here is a check that has to run somewhere and does not belong in a unit test, either
//! because it reads the whole tree or because it shells out. Running them through cargo rather than
//! a shell script means they work the same on a laptop and on a Windows runner.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

mod style;
mod unsafe_rule;

fn main() -> ExitCode {
    let task = std::env::args().nth(1);
    let result = match task.as_deref() {
        Some("style") => style::check(&root()),
        Some("unsafe") => unsafe_rule::check(&root()),
        Some("msrv") => msrv(),
        Some("ci") => ci(),
        _ => {
            eprintln!("usage: cargo xtask <task>\n");
            eprintln!("  style    prose in every markdown file against the house rules");
            eprintln!("  unsafe   crates outside the six allowed ones forbid unsafe code");
            eprintln!("  msrv     the workspace still builds on the rust-version in Cargo.toml");
            eprintln!("  ci       what the per commit CI gate runs, cheapest first");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives one level down")
        .to_path_buf()
}

fn cargo(args: &[&str]) -> Result<(), String> {
    println!("$ cargo {}", args.join(" "));
    let status = Command::new(env!("CARGO"))
        .args(args)
        .current_dir(root())
        .status()
        .map_err(|e| format!("could not run cargo: {e}"))?;
    if status.success() { Ok(()) } else { Err(format!("cargo {} failed", args.join(" "))) }
}

/// The oldest Rust the manifest claims. If that toolchain is not installed this says so and passes,
/// because CI runs it either way and a fresh clone should not be unbuildable for want of it.
fn msrv() -> Result<(), String> {
    let manifest = std::fs::read_to_string(root().join("Cargo.toml"))
        .map_err(|e| format!("could not read Cargo.toml: {e}"))?;
    let version = manifest
        .lines()
        .find_map(|l| l.strip_prefix("rust-version = \""))
        .and_then(|l| l.strip_suffix('"'))
        .ok_or("no rust-version in Cargo.toml")?;
    let installed = Command::new("rustup")
        .args(["run", version, "rustc", "--version"])
        .output()
        .is_ok_and(|o| o.status.success());
    if !installed {
        println!("rust {version} is not installed, skipping (rustup toolchain install {version})");
        return Ok(());
    }
    println!("$ cargo +{version} check --workspace --all-features");
    let status = Command::new("rustup")
        .args(["run", version, "cargo", "check", "--workspace", "--all-features"])
        .current_dir(root())
        .status()
        .map_err(|e| format!("could not run cargo: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("the workspace does not build on {version}"))
    }
}

fn ci() -> Result<(), String> {
    cargo(&["fmt", "--all", "--check"])?;
    style::check(&root())?;
    unsafe_rule::check(&root())?;
    cargo(&["clippy", "--workspace", "--all-targets", "--all-features", "--", "-D", "warnings"])?;
    cargo(&["test", "--workspace", "--all-features"])?;
    msrv()
}
