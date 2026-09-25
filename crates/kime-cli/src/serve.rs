//! `kime serve`: the HTTP server from spec/03-api.md on the models named.
//!
//! Settings come from five places, each one overriding the ones before it: the defaults,
//! laya-serve's `LAYA_*` variables, a kime.toml given with `--config` or `KIME_CONFIG`, the
//! `KIME_*` variables and the flags. Run as `laya-serve` (a symlink or a copy with that name),
//! kime also takes laya-serve's defaults, so a laya-serve unit or container switches binaries
//! with no config change.

use std::net::{IpAddr, SocketAddr};
use std::process::ExitCode;
use std::time::Instant;

use kime::{Device, Kime};
use serde::Deserialize;

use crate::predict::{device, precision};

const USAGE: &str =
    "usage: kime serve [--config kime.toml] [--host 127.0.0.1] [--port 8000] [--models laya,laya-multilingual]
options: --device auto|cpu|cuda[:N]  --threads N  --precision f16|f32|int8
         --max-batch N  --max-batch-tokens N  --max-body BYTES  --max-request-tokens N
         --max-queue-ms MS (0 is off)  --max-pending N (0 is off)  --io-threads N  --no-jev-aliases
         --answer-cache N (answers kept per model, 100000 by default, 0 is off)
         --api-keys-file PATH  --rpm N  --tps N  --log-level info
         --log-requests  --log-format text|json  --otlp-endpoint http://HOST:4318  --otlp-service NAME
Every option is also a kime.toml field (--max-batch is max_batch) and a KIME_ variable
(KIME_MAX_BATCH). Flags win over KIME_ variables, which win over the file, which wins over
laya-serve's LAYA_ variables. OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, OTEL_EXPORTER_OTLP_ENDPOINT and
OTEL_SERVICE_NAME are read when the KIME_ ones are not set. The first model answers requests that name no model. API keys
also come from KIME_API_KEYS (comma separated) and from laya-serve's LAYA_API_KEY. With no
keys anyone can call the API.";

/// Answers the answer cache keeps per model when nothing says otherwise, from spec/11-serving.md.
const ANSWER_CACHE: usize = 100_000;

/// Every setting, as a kime.toml holds it. Each source gives some of them, and a later source's
/// value replaces an earlier one's.
#[derive(Debug, Default, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    host: Option<String>,
    port: Option<u16>,
    models: Option<Vec<String>>,
    device: Option<String>,
    threads: Option<usize>,
    precision: Option<String>,
    max_batch: Option<usize>,
    max_batch_tokens: Option<usize>,
    max_body: Option<usize>,
    max_request_tokens: Option<usize>,
    max_queue_ms: Option<u64>,
    max_pending: Option<usize>,
    answer_cache: Option<usize>,
    io_threads: Option<usize>,
    jev_aliases: Option<bool>,
    api_keys_file: Option<String>,
    rpm: Option<u32>,
    tps: Option<u32>,
    log_level: Option<String>,
    log_requests: Option<bool>,
    log_format: Option<String>,
    otlp_endpoint: Option<String>,
    otlp_service: Option<String>,
}

impl Settings {
    /// `self` with every setting `over` gives replaced.
    fn and(self, over: Settings) -> Settings {
        Settings {
            host: over.host.or(self.host),
            port: over.port.or(self.port),
            models: over.models.or(self.models),
            device: over.device.or(self.device),
            threads: over.threads.or(self.threads),
            precision: over.precision.or(self.precision),
            max_batch: over.max_batch.or(self.max_batch),
            max_batch_tokens: over.max_batch_tokens.or(self.max_batch_tokens),
            max_body: over.max_body.or(self.max_body),
            max_request_tokens: over.max_request_tokens.or(self.max_request_tokens),
            max_queue_ms: over.max_queue_ms.or(self.max_queue_ms),
            max_pending: over.max_pending.or(self.max_pending),
            answer_cache: over.answer_cache.or(self.answer_cache),
            io_threads: over.io_threads.or(self.io_threads),
            jev_aliases: over.jev_aliases.or(self.jev_aliases),
            api_keys_file: over.api_keys_file.or(self.api_keys_file),
            rpm: over.rpm.or(self.rpm),
            tps: over.tps.or(self.tps),
            log_level: over.log_level.or(self.log_level),
            log_requests: over.log_requests.or(self.log_requests),
            log_format: over.log_format.or(self.log_format),
            otlp_endpoint: over.otlp_endpoint.or(self.otlp_endpoint),
            otlp_service: over.otlp_service.or(self.otlp_service),
        }
    }

    /// kime's own defaults, or laya-serve's when kime runs under its name: every address rather
    /// than loopback, and all three Laya checkpoints.
    fn defaults(laya: bool) -> Settings {
        let (host, models) = if laya {
            ("0.0.0.0", LAYA_NAMES.iter().map(|(_, id)| (*id).to_string()).collect())
        } else {
            ("127.0.0.1", vec!["laya".to_string()])
        };
        Settings {
            host: Some(host.into()),
            port: Some(8000),
            models: Some(models),
            ..Settings::default()
        }
    }

    /// A kime.toml. An unknown field is an error that names it.
    fn file(path: &str) -> Result<Settings, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        toml::from_str(&text).map_err(|e| format!("{path}: {}", e.to_string().trim_end()))
    }

    /// The `KIME_*` variables, one per setting, named after the field in capitals.
    fn kime_env(get: &dyn Fn(&str) -> Option<String>) -> Result<Settings, String> {
        let var = |name: &str| get(name).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        fn num<T: std::str::FromStr>(name: &str, v: Option<String>) -> Result<Option<T>, String>
        where
            T::Err: std::fmt::Display,
        {
            v.map(|v| v.parse().map_err(|e| format!("{name}={v:?}: {e}"))).transpose()
        }
        let bool = |name: &str| -> Result<Option<bool>, String> {
            var(name)
                .map(|v| {
                    parse_bool(&v).ok_or_else(|| format!("{name}={v:?}: expected true or false"))
                })
                .transpose()
        };
        Ok(Settings {
            host: var("KIME_HOST"),
            port: num("KIME_PORT", var("KIME_PORT"))?,
            models: var("KIME_MODELS").map(|v| list(&v)),
            device: var("KIME_DEVICE"),
            threads: num("KIME_THREADS", var("KIME_THREADS"))?,
            precision: var("KIME_PRECISION"),
            max_batch: num("KIME_MAX_BATCH", var("KIME_MAX_BATCH"))?,
            max_batch_tokens: num("KIME_MAX_BATCH_TOKENS", var("KIME_MAX_BATCH_TOKENS"))?,
            max_body: num("KIME_MAX_BODY", var("KIME_MAX_BODY"))?,
            max_request_tokens: num("KIME_MAX_REQUEST_TOKENS", var("KIME_MAX_REQUEST_TOKENS"))?,
            max_queue_ms: num("KIME_MAX_QUEUE_MS", var("KIME_MAX_QUEUE_MS"))?,
            max_pending: num("KIME_MAX_PENDING", var("KIME_MAX_PENDING"))?,
            answer_cache: num("KIME_ANSWER_CACHE", var("KIME_ANSWER_CACHE"))?,
            io_threads: num("KIME_IO_THREADS", var("KIME_IO_THREADS"))?,
            jev_aliases: bool("KIME_JEV_ALIASES")?,
            api_keys_file: var("KIME_API_KEYS_FILE"),
            rpm: num("KIME_RPM", var("KIME_RPM"))?,
            tps: num("KIME_TPS", var("KIME_TPS"))?,
            log_level: var("KIME_LOG_LEVEL"),
            log_requests: bool("KIME_LOG_REQUESTS")?,
            log_format: var("KIME_LOG_FORMAT"),
            // The OpenTelemetry names, when kime's own are not set. The general endpoint is a
            // base that traces go under.
            otlp_endpoint: var("KIME_OTLP_ENDPOINT")
                .or_else(|| var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
                .or_else(|| {
                    var("OTEL_EXPORTER_OTLP_ENDPOINT")
                        .map(|b| format!("{}/v1/traces", b.trim_end_matches('/')))
                }),
            otlp_service: var("KIME_OTLP_SERVICE").or_else(|| var("OTEL_SERVICE_NAME")),
        })
    }

    /// laya-serve's variables, read the way laya-serve reads them. `LAYA_THREADS` that is not a
    /// whole number above 0 is ignored, as laya-serve ignores it. `LAYA_API_KEY` is read with the
    /// other keys.
    fn laya_env(get: &dyn Fn(&str) -> Option<String>) -> Result<Settings, String> {
        let var = |name: &str| get(name).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let models = match var("LAYA_MODELS") {
            Some(v) => Some(
                list(&v)
                    .iter()
                    .map(|m| laya_model(m).ok_or_else(|| format!("LAYA_MODELS: unknown model {m:?}, expected english, multilingual or typed-decisions")))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            None => None,
        };
        if var("LAYA_PRELOAD").as_deref().and_then(parse_bool) == Some(false) {
            eprintln!(
                "kime serve: LAYA_PRELOAD is off, but kime always loads its models at startup"
            );
        }
        if var("LAYA_AUTO_TASK").as_deref().and_then(parse_bool) == Some(true) {
            eprintln!(
                "kime serve: LAYA_AUTO_TASK is on, but kime does not route to typed-decisions by itself yet, name the model in the request"
            );
        }
        Ok(Settings {
            host: var("LAYA_HOST"),
            port: var("LAYA_PORT")
                .map(|v| v.parse().map_err(|e| format!("LAYA_PORT={v:?}: {e}")))
                .transpose()?,
            models,
            // torch's "cuda" and "cuda:N" are kime's too. torch has no "auto", laya-serve's
            // default, and kime has it.
            device: var("LAYA_DEVICE"),
            threads: var("LAYA_THREADS").and_then(|v| v.parse().ok()).filter(|&n| n > 0),
            log_level: var("LAYA_LOG_LEVEL"),
            ..Settings::default()
        })
    }

    /// The flags, and the `--config` path if one is given.
    fn flags(args: &[String]) -> Result<(Settings, Option<String>), String> {
        let mut s = Settings::default();
        let mut config = None;
        let mut it = args.iter();
        while let Some(a) = it.next() {
            let mut val = || it.next().cloned().ok_or_else(|| format!("{a} needs a value"));
            fn num<T: std::str::FromStr>(a: &str, v: &str) -> Result<T, String>
            where
                T::Err: std::fmt::Display,
            {
                v.parse().map_err(|e| format!("{a} {v:?}: {e}"))
            }
            match a.as_str() {
                "--config" => config = Some(val()?),
                "--host" => s.host = Some(val()?),
                "--port" => s.port = Some(num(a, &val()?)?),
                "--models" | "--model" => s.models = Some(list(&val()?)),
                "--device" => s.device = Some(val()?),
                "--precision" => s.precision = Some(val()?),
                "--threads" => s.threads = Some(num(a, &val()?)?),
                "--max-batch" => s.max_batch = Some(num(a, &val()?)?),
                "--max-batch-tokens" => s.max_batch_tokens = Some(num(a, &val()?)?),
                "--max-body" => s.max_body = Some(num(a, &val()?)?),
                "--max-request-tokens" => s.max_request_tokens = Some(num(a, &val()?)?),
                "--max-queue-ms" => s.max_queue_ms = Some(num(a, &val()?)?),
                "--max-pending" => s.max_pending = Some(num(a, &val()?)?),
                "--answer-cache" => s.answer_cache = Some(num(a, &val()?)?),
                "--io-threads" => s.io_threads = Some(num(a, &val()?)?),
                "--api-keys-file" => s.api_keys_file = Some(val()?),
                "--rpm" => s.rpm = Some(num(a, &val()?)?),
                "--tps" => s.tps = Some(num(a, &val()?)?),
                "--log-level" => s.log_level = Some(val()?),
                "--log-requests" => s.log_requests = Some(true),
                "--log-format" => s.log_format = Some(val()?),
                "--otlp-endpoint" => s.otlp_endpoint = Some(val()?),
                "--otlp-service" => s.otlp_service = Some(val()?),
                "--no-jev-aliases" => s.jev_aliases = Some(false),
                "--help" | "-h" => return Err(USAGE.into()),
                other => return Err(format!("unknown option {other:?}\n{USAGE}")),
            }
        }
        Ok((s, config))
    }

    /// Every source merged in order.
    fn gather(
        args: &[String],
        laya: bool,
        get: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Settings, String> {
        let (flags, config) = Settings::flags(args)?;
        let file = match config.or_else(|| get("KIME_CONFIG").filter(|p| !p.trim().is_empty())) {
            Some(path) => Settings::file(&path)?,
            None => Settings::default(),
        };
        Ok(Settings::defaults(laya)
            .and(Settings::laya_env(get)?)
            .and(file)
            .and(Settings::kime_env(get)?)
            .and(flags))
    }
}

/// laya-serve's checkpoint names and kime's for the same checkpoint.
const LAYA_NAMES: [(&str, &str); 3] = [
    ("english", "laya"),
    ("multilingual", "laya-multilingual"),
    ("typed-decisions", "laya-typed-decisions"),
];

/// A `LAYA_MODELS` entry as kime names it, with the aliases Laya's router takes.
fn laya_model(name: &str) -> Option<String> {
    let key = match name.to_ascii_lowercase().as_str() {
        "english" | "en" | "laya" | "default" => "english",
        "multilingual" | "multi" | "ml" | "laya-multilingual" => "multilingual",
        "typed-decisions" | "typed" | "typed_decisions" | "laya-typed-decisions" | "decisions" => {
            "typed-decisions"
        }
        _ => return None,
    };
    LAYA_NAMES.iter().find(|(k, _)| *k == key).map(|(_, id)| (*id).to_string())
}

/// laya-serve's reading of a yes or no: 1, true, yes and on are yes.
fn parse_bool(v: &str) -> Option<bool> {
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn list(v: &str) -> Vec<String> {
    v.split(',').map(|m| m.trim().to_string()).filter(|m| !m.is_empty()).collect()
}

/// Uvicorn's log levels, which laya-serve takes. kime logs only at startup for now, and above
/// info it says nothing there.
const LOG_LEVELS: [&str; 6] = ["critical", "error", "warning", "info", "debug", "trace"];

fn config(s: Settings) -> Result<kime_serve::Config, String> {
    let names = s.models.unwrap_or_default();
    if names.is_empty() {
        return Err("models needs at least one model".into());
    }
    let level = s.log_level.unwrap_or_else(|| "info".into()).to_ascii_lowercase();
    if !LOG_LEVELS.contains(&level.as_str()) {
        return Err(format!("log level {level:?} is not one of {}", LOG_LEVELS.join(", ")));
    }
    let quiet = matches!(level.as_str(), "critical" | "error" | "warning");
    let log = match (s.log_requests.unwrap_or(false), s.log_format.as_deref()) {
        (_, Some(f)) if !matches!(f, "text" | "json") => {
            return Err(format!("log format {f:?} is not text or json"));
        }
        (false, _) => kime_serve::Log::Off,
        (true, Some("json")) => kime_serve::Log::Json,
        (true, _) => kime_serve::Log::Text,
    };
    let mut dev = device(s.device.as_deref().unwrap_or("auto"))?;
    let prec = precision(s.precision.as_deref().unwrap_or("f16"))?;
    if let Some(t) = s.threads {
        match dev {
            Device::Auto | Device::Cpu { .. } => dev = Device::Cpu { threads: t },
            _ => return Err("threads is for the CPU".into()),
        }
    }
    let host = s.host.unwrap_or_default();
    let ip: IpAddr = host.parse().map_err(|e| format!("host {host:?}: {e}"))?;
    let rpm = s.rpm.map(|n| limit("rpm", n)).transpose()?;
    let tps = s.tps.map(|n| limit("tps", n)).transpose()?;
    let auth = auth(s.api_keys_file.as_deref(), quiet)?.with_defaults(rpm, tps);
    let mut models = Vec::with_capacity(names.len());
    for name in &names {
        let t = Instant::now();
        let k = Kime::builder()
            .model(name)
            .device(dev)
            .precision(prec)
            .answer_cache(s.answer_cache.unwrap_or(ANSWER_CACHE))
            .build()
            .map_err(|e| format!("{name}: {e}"))?;
        if !quiet {
            eprintln!(
                "kime serve: {} on {}, loaded in {:.2?}",
                k.model_id(),
                k.device(),
                t.elapsed()
            );
        }
        models.push(k);
    }
    let mut cfg = kime_serve::Config::new(SocketAddr::new(ip, s.port.unwrap_or(8000)), models);
    cfg.jev_aliases = s.jev_aliases.unwrap_or(cfg.jev_aliases);
    if cfg.jev_aliases && !quiet {
        // The TypeSafe SDKs and jev-ultrafast all send jev-latest unless told otherwise, so this
        // is on by default and said out loud.
        eprintln!(
            "kime serve: jev, jev-latest and other jev-* names are answered by {} (--no-jev-aliases turns this off)",
            cfg.models[0].model_id()
        );
    }
    cfg.max_batch = s.max_batch.unwrap_or(cfg.max_batch);
    cfg.max_batch_tokens = s.max_batch_tokens.unwrap_or(cfg.max_batch_tokens);
    cfg.max_pending = s.max_pending.unwrap_or(cfg.max_pending);
    cfg.max_body = s.max_body.unwrap_or(cfg.max_body);
    if s.max_request_tokens == Some(0) {
        return Err("max_request_tokens must be a whole number above 0".into());
    }
    cfg.max_request_tokens = s.max_request_tokens.unwrap_or(cfg.max_request_tokens);
    cfg.io_threads = s.io_threads.unwrap_or(cfg.io_threads);
    if let Some(ms) = s.max_queue_ms {
        cfg.max_queue = std::time::Duration::from_millis(ms);
    }
    cfg.auth = auth;
    cfg.log = log;
    if let Some(e) = &s.otlp_endpoint {
        let mut o = kime_serve::Otlp::new(e)?;
        if let Some(name) = s.otlp_service {
            o.service = name;
        }
        if !quiet {
            eprintln!("kime serve: sending a span per request to {e} as {}", o.service);
        }
        cfg.otlp = Some(o);
    }
    Ok(cfg)
}

fn limit(what: &str, n: u32) -> Result<u32, String> {
    if n == 0 { Err(format!("{what} must be a whole number above 0")) } else { Ok(n) }
}

/// The keys from `--api-keys-file` and `KIME_API_KEYS`, which answer with Jev's errors. Alone,
/// laya-serve's `LAYA_API_KEY` answers with laya-serve's 401, so its clients see no change. With
/// other keys it is one more key.
fn auth(file: Option<&str>, quiet: bool) -> Result<kime_serve::Auth, String> {
    let laya = std::env::var("LAYA_API_KEY").ok().filter(|k| !k.is_empty());
    let env = std::env::var("KIME_API_KEYS").ok().filter(|k| !k.trim().is_empty());
    if file.is_none() && env.is_none() {
        return Ok(laya.map_or_else(kime_serve::Auth::off, |k| {
            if !quiet {
                eprintln!("kime serve: requests need LAYA_API_KEY as a bearer token");
            }
            kime_serve::Auth::laya(&k)
        }));
    }
    let mut a = kime_serve::Auth::off();
    if let Some(path) = file {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("api keys file {path}: {e}"))?;
        a.add_keys_file(&text).map_err(|e| format!("api keys file {path}: {e}"))?;
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
    if !quiet {
        eprintln!("kime serve: requests need one of {} API keys", a.len());
    }
    Ok(a)
}

/// Runs the server. `laya` is set when kime was started as `laya-serve`.
pub(crate) fn run(args: &[String], laya: bool) -> ExitCode {
    let get = |name: &str| std::env::var(name).ok();
    let cfg = match Settings::gather(args, laya, &get).and_then(config) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn env(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
        let vars: Vec<(String, String)> =
            vars.iter().map(|(k, v)| ((*k).to_string(), (*v).to_string())).collect();
        move |name| vars.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
    }

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn laya_serve_variables() {
        let get = env(&[
            ("LAYA_HOST", "10.0.0.5"),
            ("LAYA_PORT", "8080"),
            ("LAYA_DEVICE", "cuda:1"),
            ("LAYA_MODELS", "english, ml"),
            ("LAYA_THREADS", "0"),
            ("LAYA_LOG_LEVEL", "warning"),
        ]);
        let s = Settings::gather(&[], false, &get).unwrap();
        assert_eq!(s.host.as_deref(), Some("10.0.0.5"));
        assert_eq!(s.port, Some(8080));
        assert_eq!(s.device.as_deref(), Some("cuda:1"));
        assert_eq!(s.models, Some(args(&["laya", "laya-multilingual"])));
        // laya-serve ignores a thread count of 0.
        assert_eq!(s.threads, None);
        assert_eq!(s.log_level.as_deref(), Some("warning"));
        let bad = env(&[("LAYA_MODELS", "english,klingon")]);
        assert!(Settings::gather(&[], false, &bad).unwrap_err().contains("\"klingon\""));
    }

    #[test]
    fn otlp_settings() {
        let otel =
            env(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "http://c:4318/"), ("OTEL_SERVICE_NAME", "svc")]);
        let s = Settings::gather(&[], false, &otel).unwrap();
        assert_eq!(s.otlp_endpoint.as_deref(), Some("http://c:4318/v1/traces"));
        assert_eq!(s.otlp_service.as_deref(), Some("svc"));
        let get = env(&[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://c:4318"),
            ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://t:4318/traces"),
        ]);
        let s = Settings::gather(&[], false, &get).unwrap();
        assert_eq!(s.otlp_endpoint.as_deref(), Some("http://t:4318/traces"));
        let get = env(&[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://c:4318"),
            ("KIME_OTLP_ENDPOINT", "http://k:4318"),
        ]);
        let s = Settings::gather(&args(&["--otlp-service", "flag"]), false, &get).unwrap();
        assert_eq!(s.otlp_endpoint.as_deref(), Some("http://k:4318"));
        assert_eq!(s.otlp_service.as_deref(), Some("flag"));
        assert!(Settings::gather(&[], false, &env(&[])).unwrap().otlp_endpoint.is_none());
    }

    #[test]
    fn named_laya_serve_takes_its_defaults() {
        let s = Settings::gather(&[], true, &env(&[])).unwrap();
        assert_eq!(s.host.as_deref(), Some("0.0.0.0"));
        assert_eq!(s.models, Some(args(&["laya", "laya-multilingual", "laya-typed-decisions"])));
        let s = Settings::gather(&[], false, &env(&[])).unwrap();
        assert_eq!(s.host.as_deref(), Some("127.0.0.1"));
        assert_eq!(s.models, Some(args(&["laya"])));
    }

    #[test]
    fn later_sources_win() {
        let dir = std::env::temp_dir().join(format!("kime-config-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("kime.toml");
        std::fs::write(&path, "port = 9000\nmax_batch = 64\nrpm = 60\nmodels = [\"laya\"]\n")
            .unwrap();
        let get = env(&[
            ("LAYA_PORT", "8001"),
            ("LAYA_HOST", "0.0.0.0"),
            ("KIME_CONFIG", path.to_str().unwrap()),
            ("KIME_MAX_BATCH", "32"),
            ("KIME_JEV_ALIASES", "off"),
        ]);
        let s = Settings::gather(&args(&["--rpm", "5"]), false, &get).unwrap();
        // LAYA_HOST is the only source for the host, the file beats LAYA_PORT, KIME_MAX_BATCH
        // beats the file and the flag beats both.
        assert_eq!(s.host.as_deref(), Some("0.0.0.0"));
        assert_eq!(s.port, Some(9000));
        assert_eq!(s.max_batch, Some(32));
        assert_eq!(s.rpm, Some(5));
        assert_eq!(s.jev_aliases, Some(false));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn unknown_fields_are_errors() {
        let dir = std::env::temp_dir().join(format!("kime-config-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("kime.toml");
        std::fs::write(&path, "port = 9000\nmax_batches = 64\n").unwrap();
        let e = Settings::gather(&args(&["--config", path.to_str().unwrap()]), false, &env(&[]))
            .unwrap_err();
        assert!(e.contains("max_batches"), "{e}");
        assert!(e.contains("line 2"), "{e}");
        std::fs::write(&path, "port = \"high\"\n").unwrap();
        let e = Settings::gather(&args(&["--config", path.to_str().unwrap()]), false, &env(&[]))
            .unwrap_err();
        assert!(e.contains("port"), "{e}");
        let _ = std::fs::remove_dir_all(dir);
        let e = Settings::gather(&[], false, &env(&[("KIME_PORT", "x")])).unwrap_err();
        assert!(e.starts_with("KIME_PORT="), "{e}");
    }
}
