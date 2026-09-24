//! `kime serve`: the HTTP server from spec/03-api.md on the models named.

use std::net::{IpAddr, SocketAddr};
use std::process::ExitCode;
use std::time::Instant;

use kime::{Device, Kime, Precision};

use crate::predict::{device, precision};

const USAGE: &str =
    "usage: kime serve [--host 127.0.0.1] [--port 8000] [--models laya,laya-multilingual]
options: --device auto|cpu|cuda[:N]  --threads N  --precision f16|f32|int8
         --max-batch N  --max-body BYTES  --max-queue-ms MS (0 is off)  --io-threads N
         --no-jev-aliases  --api-keys-file PATH  --rpm N  --tps N
The first model answers requests that name no model. API keys also come from KIME_API_KEYS
(comma separated) and from laya-serve's LAYA_API_KEY. With no keys anyone can call the API.";

fn config(args: &[String]) -> Result<kime_serve::Config, String> {
    let (mut host, mut port) = ("127.0.0.1".to_string(), 8000u16);
    let mut names = vec!["laya".to_string()];
    let (mut dev, mut prec, mut threads) = (Device::Auto, Precision::F16, None);
    let (mut max_batch, mut max_body, mut io, mut jev) = (None, None, None, true);
    let (mut max_queue, mut keys_file, mut rpm, mut tps) = (None, None, None, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().ok_or_else(|| format!("{a} needs a value"));
        let num = |v: String| v.parse::<usize>().map_err(|e| format!("{a}: {e}"));
        match a.as_str() {
            "--host" => host = val()?,
            "--port" => port = val()?.parse().map_err(|e| format!("--port: {e}"))?,
            "--models" | "--model" => {
                names = val()?
                    .split(',')
                    .map(|m| m.trim().to_string())
                    .filter(|m| !m.is_empty())
                    .collect();
            }
            "--device" => dev = device(&val()?)?,
            "--precision" => prec = precision(&val()?)?,
            "--threads" => threads = Some(num(val()?)?),
            "--max-batch" => max_batch = Some(num(val()?)?),
            "--max-body" => max_body = Some(num(val()?)?),
            "--max-queue-ms" => max_queue = Some(num(val()?)?),
            "--io-threads" => io = Some(num(val()?)?),
            "--api-keys-file" => keys_file = Some(val()?),
            "--rpm" => rpm = Some(limit(a, &val()?)?),
            "--tps" => tps = Some(limit(a, &val()?)?),
            "--no-jev-aliases" => jev = false,
            "--help" | "-h" => return Err(USAGE.into()),
            other => return Err(format!("unknown option {other:?}\n{USAGE}")),
        }
    }
    if names.is_empty() {
        return Err("--models needs at least one model".into());
    }
    if let Some(t) = threads {
        match dev {
            Device::Auto | Device::Cpu { .. } => dev = Device::Cpu { threads: t },
            _ => return Err("--threads is for the CPU".into()),
        }
    }
    let ip: IpAddr = host.parse().map_err(|e| format!("--host {host:?}: {e}"))?;
    let auth = auth(keys_file.as_deref())?.with_defaults(rpm, tps);
    let mut models = Vec::with_capacity(names.len());
    for name in &names {
        let t = Instant::now();
        let k = Kime::builder()
            .model(name)
            .device(dev)
            .precision(prec)
            .build()
            .map_err(|e| format!("{name}: {e}"))?;
        eprintln!("kime serve: {} on {}, loaded in {:.2?}", k.model_id(), k.device(), t.elapsed());
        models.push(k);
    }
    let mut cfg = kime_serve::Config::new(SocketAddr::new(ip, port), models);
    cfg.jev_aliases = jev;
    cfg.max_batch = max_batch.unwrap_or(cfg.max_batch);
    cfg.max_body = max_body.unwrap_or(cfg.max_body);
    cfg.io_threads = io.unwrap_or(cfg.io_threads);
    if let Some(ms) = max_queue {
        cfg.max_queue = std::time::Duration::from_millis(ms as u64);
    }
    cfg.auth = auth;
    Ok(cfg)
}

fn limit(flag: &str, v: &str) -> Result<u32, String> {
    v.parse::<u32>()
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| format!("{flag} must be a whole number above 0, got {v:?}"))
}

/// The keys from `--api-keys-file` and `KIME_API_KEYS`, which answer with Jev's errors. Alone,
/// laya-serve's `LAYA_API_KEY` answers with laya-serve's 401, so its clients see no change. With
/// other keys it is one more key.
fn auth(file: Option<&str>) -> Result<kime_serve::Auth, String> {
    let laya = std::env::var("LAYA_API_KEY").ok().filter(|k| !k.is_empty());
    let env = std::env::var("KIME_API_KEYS").ok().filter(|k| !k.trim().is_empty());
    if file.is_none() && env.is_none() {
        return Ok(laya.map_or_else(kime_serve::Auth::off, |k| {
            eprintln!("kime serve: requests need LAYA_API_KEY as a bearer token");
            kime_serve::Auth::laya(&k)
        }));
    }
    let mut a = kime_serve::Auth::off();
    if let Some(path) = file {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("--api-keys-file {path}: {e}"))?;
        a.add_keys_file(&text).map_err(|e| format!("--api-keys-file {path}: {e}"))?;
    }
    for k in env.iter().flat_map(|e| e.split(',')).map(str::trim).filter(|k| !k.is_empty()) {
        a.add(k, None, None).map_err(|e| format!("KIME_API_KEYS: {e}"))?;
    }
    if let Some(k) = laya {
        a.add(&k, None, None).map_err(|e| format!("LAYA_API_KEY: {e}"))?;
    }
    if !a.is_on() {
        return Err("the API keys file and KIME_API_KEYS hold no keys".into());
    }
    eprintln!("kime serve: requests need one of {} API keys", a.len());
    Ok(a)
}

pub(crate) fn run(args: &[String]) -> ExitCode {
    let cfg = match config(args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("kime serve: {e}");
            return ExitCode::from(2);
        }
    };
    match kime_serve::run(cfg) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kime serve: {e}");
            ExitCode::FAILURE
        }
    }
}
