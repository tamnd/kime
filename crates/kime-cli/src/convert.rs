//! `kime convert`: pack a Laya checkpoint directory into one `.kime` file, unpack a `.kime` back
//! into a directory, or check either. Prepacking for a backend and vocabulary trimming arrive with
//! the backends and the native models.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use kime_model::Model;

const USAGE: &str = "usage: kime convert <laya dir> <out.kime> [--force]
       kime convert <in.kime> <out dir> [--force]
       kime convert --check <laya dir or .kime>";

pub(crate) fn run(args: &[String]) -> ExitCode {
    let force = args.iter().any(|a| a == "--force");
    let check = args.iter().any(|a| a == "--check");
    let paths: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let result = match (check, paths.as_slice()) {
        (true, [src]) => open(Path::new(src)).map(|_| ()),
        (false, [src, dst]) => convert(Path::new(src), Path::new(dst), force),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kime convert: {e}");
            ExitCode::FAILURE
        }
    }
}

fn open(src: &Path) -> Result<Model, String> {
    let t = Instant::now();
    let m = Model::open(src).map_err(|e| e.to_string())?;
    let opened = t.elapsed();
    let t = Instant::now();
    m.verify().map_err(|e| e.to_string())?;
    let bytes = m.tensors.data_bytes();
    println!(
        "{}: {} {} tensors, {:.1} MB of weights, {} files, opened in {:.1} ms",
        src.display(),
        m.spec.id,
        m.tensors.entries().len(),
        bytes as f64 / 1e6,
        m.file_names().len(),
        opened.as_secs_f64() * 1e3,
    );
    if let Some(index) = &m.index {
        let s = t.elapsed().as_secs_f64();
        println!(
            "hash {} checked in {:.0} ms ({:.1} GB/s)",
            index.hash,
            s * 1e3,
            bytes as f64 / s / 1e9
        );
    }
    Ok(m)
}

fn convert(src: &Path, dst: &Path, force: bool) -> Result<(), String> {
    if dst.exists() && !force {
        return Err(format!("{} exists, pass --force to replace it", dst.display()));
    }
    let m = open(src)?;
    let t = Instant::now();
    if m.index.is_some() {
        if dst.exists() {
            std::fs::remove_dir_all(dst).map_err(|e| format!("{}: {e}", dst.display()))?;
        }
        m.unpack(dst).map_err(|e| e.to_string())?;
        println!("unpacked to {} in {:.0} ms", dst.display(), t.elapsed().as_secs_f64() * 1e3);
        return Ok(());
    }
    // Write beside the target and rename, so a failed or interrupted pack never leaves a file that
    // looks complete.
    let tmp = PathBuf::from(format!("{}.partial", dst.display()));
    let f = std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
    let hash = m.pack(&mut w).map_err(|e| e.to_string());
    let hash = hash.and_then(|h| {
        let f = w.into_inner().map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
        Ok(h)
    });
    let hash = match hash {
        Ok(h) => h,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    };
    std::fs::rename(&tmp, dst).map_err(|e| format!("{}: {e}", dst.display()))?;
    let size = std::fs::metadata(dst).map(|m| m.len()).unwrap_or(0);
    println!(
        "packed {} ({:.1} MB) in {:.0} ms, hash {hash}",
        dst.display(),
        size as f64 / 1e6,
        t.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}
