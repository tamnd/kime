//! `kime serve`: the HTTP server from spec/03-api.md on the models named.

use std::net::{IpAddr, SocketAddr};
use std::process::ExitCode;
use std::time::Instant;

use kime::{Device, Kime, Precision};

use crate::predict::{device, precision};

const USAGE: &str =
    "usage: kime serve [--host 127.0.0.1] [--port 8000] [--models laya,laya-multilingual]
options: --device auto|cpu|cuda[:N]  --threads N  --precision f16|f32|int8
         --max-batch N  --max-body BYTES  --io-threads N  --no-jev-aliases
The first model answers requests that name no model.";

fn config(args: &[String]) -> Result<kime_serve::Config, String> {
    let (mut host, mut port) = ("127.0.0.1".to_string(), 8000u16);
    let mut names = vec!["laya".to_string()];
    let (mut dev, mut prec, mut threads) = (Device::Auto, Precision::F16, None);
    let (mut max_batch, mut max_body, mut io, mut jev) = (None, None, None, true);
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
            "--io-threads" => io = Some(num(val()?)?),
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
    Ok(cfg)
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
