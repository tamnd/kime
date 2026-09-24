//! The `kime` binary. One binary with subcommands, per spec/14-sdks-cli.md.
//!
//! Today it answers `--version`, `doctor`, `convert`, `predict`, `pull` and `serve`, and names the milestone that brings each of the other
//! subcommands. A command that exists and says when it will work is more useful than one that is
//! missing, because the help text is where people look first.

#![forbid(unsafe_code)]

use std::process::ExitCode;

mod convert;
mod predict;
mod pull;
mod serve;

/// Every subcommand in spec/14-sdks-cli.md and the milestone in spec/16-roadmap.md that brings it.
const PLANNED: &[(&str, &str, &str)] = &[
    ("train", "fine tune a model on labelled decisions", "M5"),
    ("calibrate", "fit temperatures on a labelled file", "M3"),
    ("eval", "run the quality suites", "M2"),
    ("bench", "run the speed suites", "M0"),
    ("report", "build the scorecard from eval and bench output", "M5"),
    ("models", "list cached and served models", "M1"),
];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version" | "-V" | "version") => {
            println!("kime {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some("doctor") => {
            doctor();
            ExitCode::SUCCESS
        }
        Some("convert") => convert::run(&args[1..]),
        Some("predict") => predict::run(&args[1..]),
        Some("pull") => pull::run(&args[1..]),
        Some("serve") => serve::run(&args[1..]),
        None | Some("--help" | "-h" | "help") => {
            help();
            ExitCode::SUCCESS
        }
        Some(other) => {
            if let Some((_, _, milestone)) = PLANNED.iter().find(|(name, _, _)| *name == other) {
                eprintln!(
                    "kime {other} arrives at {milestone}, see https://github.com/tamnd/kime/milestones"
                );
            } else {
                eprintln!("kime: unknown command {other:?}, try kime --help");
            }
            ExitCode::from(2)
        }
    }
}

fn help() {
    println!("kime {}: typed decisions over text\n", env!("CARGO_PKG_VERSION"));
    println!("usage: kime <command> [options]\n");
    println!("  {:<10} print what this machine offers kime", "doctor");
    println!("  {:<10} pack a Laya checkpoint to .kime, unpack one, or check one", "convert");
    println!("  {:<10} answer questions from the command line", "predict");
    println!("  {:<10} download a model into the cache", "pull");
    println!("  {:<10} run the HTTP server", "serve");
    for (name, what, milestone) in PLANNED {
        println!("  {name:<10} {what} (arrives at {milestone})");
    }
}

/// What a backend would have to work with on this machine. spec/14-sdks-cli.md asks for devices,
/// drivers, CPU features and the cache as well, and those arrive with the backends that read them.
fn doctor() {
    println!("kime {}", env!("CARGO_PKG_VERSION"));
    println!("os        {}", std::env::consts::OS);
    println!("arch      {}", std::env::consts::ARCH);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("threads   {threads}");
    let features = cpu_features();
    println!(
        "cpu       {}",
        if features.is_empty() { "baseline".to_string() } else { features.join(" ") }
    );
}

/// The CPU features the kernels in spec/10-cpu.md dispatch on, as detected at runtime.
fn cpu_features() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut found = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            found.push("avx2");
        }
        if std::arch::is_x86_feature_detected!("fma") {
            found.push("fma");
        }
        if std::arch::is_x86_feature_detected!("f16c") {
            found.push("f16c");
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            found.push("avx512f");
        }
        if std::arch::is_x86_feature_detected!("avx512vnni") {
            found.push("avx512vnni");
        }
        if std::arch::is_x86_feature_detected!("avx512bf16") {
            found.push("avx512bf16");
        }
        // AMX is not listed: std's detection of it is unstable, and reading CPUID ourselves needs
        // unsafe on our minimum Rust. kime-cpu will probe it next to the AMX kernels.
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            found.push("neon");
        }
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            found.push("dotprod");
        }
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            found.push("i8mm");
        }
        if std::arch::is_aarch64_feature_detected!("bf16") {
            found.push("bf16");
        }
        if std::arch::is_aarch64_feature_detected!("fp16") {
            found.push("fp16");
        }
    }
    found
}
