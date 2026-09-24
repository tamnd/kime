//! The HTTP server. `/v1/systemone` and the rest of the API in spec/03-api.md, keys, rate limits, overload handling and metrics, as a thin layer over `kime-engine`. See spec/11-serving.md.
//!
//! Each loaded model gets a worker thread of its own. Handlers parse and validate on the tokio
//! threads, queue the request on the model's worker and await the answer, so a forward pass never
//! runs on the runtime. Whatever queues up while the device is busy goes into the next forward
//! pass together, which is how concurrent requests share the device without a wait window.

#![forbid(unsafe_code)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use kime_engine::Kime;

mod api;
mod auth;
mod models;

pub use auth::Auth;

/// How the server runs.
#[derive(Debug)]
pub struct Config {
    /// The address to listen on.
    pub addr: SocketAddr,
    /// The loaded models. The first one answers requests that name no model.
    pub models: Vec<Kime>,
    /// Map `jev`, `jev-latest` and every other `jev-*` name to the default model.
    pub jev_aliases: bool,
    /// The largest request body in bytes.
    pub max_body: usize,
    /// The most requests one forward pass takes from a model's queue.
    pub max_batch: usize,
    /// Threads for the async runtime, which only parses, validates and writes JSON.
    pub io_threads: usize,
    /// New requests get 529 when the queue ahead of them would take longer than this. Zero
    /// turns the check off.
    pub max_queue: Duration,
    /// API keys and rate limits. Off unless set.
    pub auth: Auth,
}

impl Config {
    /// The defaults from spec/03-api.md for these models.
    #[must_use]
    pub fn new(addr: SocketAddr, models: Vec<Kime>) -> Self {
        Config {
            addr,
            models,
            jev_aliases: true,
            max_body: 8 << 20,
            max_batch: 256,
            io_threads: 2,
            max_queue: Duration::from_millis(500),
            auth: Auth::off(),
        }
    }
}

/// Serves until SIGINT or SIGTERM, then stops taking connections and lets the open requests
/// finish.
///
/// # Errors
///
/// When the runtime does not start or the address cannot be bound.
pub fn run(cfg: Config) -> std::io::Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(cfg.io_threads.max(1))
        .thread_name("kime-io")
        .enable_all()
        .build()?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(cfg.addr).await?;
        let addr = listener.local_addr()?;
        eprintln!("kime serve: listening on http://{addr}");
        if !cfg.auth.is_on() && !addr.ip().is_loopback() {
            eprintln!(
                "kime serve: warning: no API keys are set and the server is reachable from other machines, so anyone who can reach it can use it (set --api-keys-file, KIME_API_KEYS or LAYA_API_KEY)"
            );
        }
        serve(listener, cfg, shutdown()).await
    })
}

/// Serves on a bound listener until `stop` resolves. Tests use this with an ephemeral port.
///
/// # Errors
///
/// When accepting connections fails.
pub async fn serve(
    listener: tokio::net::TcpListener,
    cfg: Config,
    stop: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    use axum::serve::ListenerExt;
    let state = Arc::new(api::State {
        models: models::Models::new(cfg.models, cfg.jev_aliases, cfg.max_batch, cfg.max_queue),
        buckets: cfg.auth.buckets(),
        auth: cfg.auth,
        max_body: cfg.max_body,
        metrics: api::Metrics::default(),
    });
    let listener = listener.tap_io(|tcp| {
        // Answers are small and latency is the product, so Nagle only gets in the way.
        let _ = tcp.set_nodelay(true);
    });
    axum::serve(listener, api::router(state)).with_graceful_shutdown(stop).await
}

async fn shutdown() {
    let int = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut term =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(s) => s,
                Err(_) => {
                    let _ = int.await;
                    return;
                }
            };
        tokio::select! {
            _ = int => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    let _ = int.await;
    eprintln!("kime serve: shutting down, finishing open requests");
}
