//! The Prometheus metrics from spec/11-serving.md. Everything on the request path is a relaxed
//! atomic add, so keeping them costs a few nanoseconds a request and no lock. The one lock is
//! the table of counts by route and status, which is small and touched once per request.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

/// Bucket bounds for times, from 50 µs to 5 s, in seconds.
const SECONDS: [f64; 16] = [
    50e-6, 100e-6, 250e-6, 500e-6, 1e-3, 2.5e-3, 5e-3, 10e-3, 25e-3, 50e-3, 100e-3, 250e-3, 0.5,
    1.0, 2.5, 5.0,
];

/// Bucket bounds for counts of requests, questions or device batches in one forward pass.
const COUNTS: [f64; 11] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0];

/// Bucket bounds for input tokens in one forward pass.
const TOKENS: [f64; 11] =
    [128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0, 131072.0];

/// A Prometheus histogram. The sum is kept in whole units of `unit`, nanoseconds for times.
#[derive(Debug)]
pub(crate) struct Histogram {
    bounds: &'static [f64],
    unit: f64,
    counts: Vec<AtomicU64>,
    sum: AtomicU64,
}

impl Histogram {
    fn new(bounds: &'static [f64], unit: f64) -> Self {
        let counts = (0..=bounds.len()).map(|_| AtomicU64::new(0)).collect();
        Histogram { bounds, unit, counts, sum: AtomicU64::new(0) }
    }

    fn seconds() -> Self {
        Histogram::new(&SECONDS, 1e-9)
    }

    fn counts() -> Self {
        Histogram::new(&COUNTS, 1.0)
    }

    fn tokens() -> Self {
        Histogram::new(&TOKENS, 1.0)
    }

    /// Adds one observation, in the histogram's own unit for counts and in seconds for times.
    fn observe(&self, v: f64) {
        let at = self.bounds.iter().position(|&b| v <= b).unwrap_or(self.bounds.len());
        self.counts[at].fetch_add(1, Ordering::Relaxed);
        // Rounded to whole units, which for times is a nanosecond.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        self.sum.fetch_add((v / self.unit).round() as u64, Ordering::Relaxed);
    }

    pub(crate) fn time(&self, d: Duration) {
        self.observe(d.as_secs_f64());
    }

    pub(crate) fn count(&self, n: usize) {
        self.observe(n as f64);
    }

    /// The `_bucket`, `_sum` and `_count` lines, with `labels` inside the braces.
    fn render(&self, out: &mut String, name: &str, labels: &str) {
        let sep = if labels.is_empty() { "" } else { "," };
        let mut total = 0;
        for (i, c) in self.counts.iter().enumerate() {
            total += c.load(Ordering::Relaxed);
            let le = self.bounds.get(i).map_or_else(|| "+Inf".to_string(), |b| format!("{b}"));
            let _ = writeln!(out, "{name}_bucket{{{labels}{sep}le=\"{le}\"}} {total}");
        }
        let sum = self.sum.load(Ordering::Relaxed) as f64 * self.unit;
        let braces = if labels.is_empty() { String::new() } else { format!("{{{labels}}}") };
        let _ = writeln!(out, "{name}_sum{braces} {sum}");
        let _ = writeln!(out, "{name}_count{braces} {total}");
    }
}

/// What one model's worker and queue did.
#[derive(Debug)]
pub(crate) struct Stats {
    questions: AtomicU64,
    input_tokens: AtomicU64,
    passes: AtomicU64,
    /// From submission to the start of the forward pass that answered it.
    pub(crate) queue: Histogram,
    tokenize: Histogram,
    device: Histogram,
    pass_requests: Histogram,
    pass_questions: Histogram,
    pass_tokens: Histogram,
    pass_batches: Histogram,
}

impl Default for Stats {
    fn default() -> Self {
        Stats {
            questions: AtomicU64::new(0),
            input_tokens: AtomicU64::new(0),
            passes: AtomicU64::new(0),
            queue: Histogram::seconds(),
            tokenize: Histogram::seconds(),
            device: Histogram::seconds(),
            pass_requests: Histogram::counts(),
            pass_questions: Histogram::counts(),
            pass_tokens: Histogram::tokens(),
            pass_batches: Histogram::counts(),
        }
    }
}

/// One forward pass, as the worker saw it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Pass {
    pub(crate) requests: usize,
    pub(crate) questions: usize,
    pub(crate) tokens: usize,
    pub(crate) batches: usize,
    pub(crate) tokenize: Duration,
    pub(crate) device: Duration,
}

impl Stats {
    pub(crate) fn pass(&self, p: Pass) {
        self.passes.fetch_add(1, Ordering::Relaxed);
        self.questions.fetch_add(p.questions as u64, Ordering::Relaxed);
        self.input_tokens.fetch_add(p.tokens as u64, Ordering::Relaxed);
        self.tokenize.time(p.tokenize);
        self.device.time(p.device);
        self.pass_requests.count(p.requests);
        self.pass_questions.count(p.questions);
        self.pass_tokens.count(p.tokens);
        self.pass_batches.count(p.batches);
    }
}

/// A model's name and live numbers, for [`Metrics::render`].
#[derive(Debug)]
pub(crate) struct ModelView<'a> {
    pub(crate) id: &'a str,
    pub(crate) stats: &'a Stats,
    pub(crate) pending: usize,
    pub(crate) per_request: Duration,
}

/// The routes, a fixed set so clients cannot grow the metrics.
pub(crate) const ROUTES: [&str; 8] = [
    "/v1/systemone",
    "/v1/systemone/batch",
    "/v1/models",
    "/v1/models/{id}",
    "/health",
    "/ready",
    "/metrics",
    "other",
];

/// The route label for a path.
pub(crate) fn route(path: &str) -> usize {
    match path {
        "/v1/systemone" => 0,
        "/v1/systemone/batch" => 1,
        "/v1/models" => 2,
        p if p.starts_with("/v1/models/") => 3,
        "/health" => 4,
        "/ready" => 5,
        "/metrics" => 6,
        _ => 7,
    }
}

/// Server wide numbers: requests by route and status, and time by route.
#[derive(Debug)]
pub(crate) struct Metrics {
    by: Mutex<BTreeMap<(usize, u16), u64>>,
    took: Vec<Histogram>,
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics {
            by: Mutex::default(),
            took: ROUTES.iter().map(|_| Histogram::seconds()).collect(),
        }
    }
}

/// A per model metric: its name, its help text and where it is in [`Stats`].
type Named<T> = (&'static str, &'static str, fn(&Stats) -> &T);

/// Why a status means a request was turned away, for `kime_rejected_total`.
fn rejected(status: u16) -> Option<&'static str> {
    Some(match status {
        401 | 403 => "auth",
        413 => "too_large",
        429 => "rate_limit",
        504 => "deadline",
        529 => "overloaded",
        _ => return None,
    })
}

impl Metrics {
    pub(crate) fn record(&self, route: usize, status: u16, took: Duration) {
        *self
            .by
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry((route, status))
            .or_default() += 1;
        self.took[route].time(took);
    }

    pub(crate) fn render(&self, models: &[ModelView<'_>]) -> String {
        let mut out = String::with_capacity(16 << 10);
        let by = self.by.lock().unwrap_or_else(PoisonError::into_inner).clone();
        let head = |out: &mut String, name: &str, kind: &str, help: &str| {
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
        };
        head(&mut out, "kime_requests_total", "counter", "Requests answered, by route and status.");
        for ((r, status), n) in &by {
            let _ = writeln!(
                out,
                "kime_requests_total{{route=\"{}\",status=\"{status}\"}} {n}",
                ROUTES[*r]
            );
        }
        head(
            &mut out,
            "kime_rejected_total",
            "counter",
            "Requests turned away, by reason: auth, too_large, rate_limit, deadline, overloaded.",
        );
        let mut why: BTreeMap<&str, u64> = BTreeMap::new();
        for ((_, status), n) in &by {
            if let Some(w) = rejected(*status) {
                *why.entry(w).or_default() += n;
            }
        }
        for (w, n) in why {
            let _ = writeln!(out, "kime_rejected_total{{reason=\"{w}\"}} {n}");
        }
        head(
            &mut out,
            "kime_request_duration_seconds",
            "histogram",
            "Time from the request line to the response, by route.",
        );
        for (r, h) in self.took.iter().enumerate() {
            if h.counts.iter().any(|c| c.load(Ordering::Relaxed) > 0) {
                h.render(
                    &mut out,
                    "kime_request_duration_seconds",
                    &format!("route=\"{}\"", ROUTES[r]),
                );
            }
        }
        let counters: [Named<AtomicU64>; 3] = [
            ("kime_questions_total", "Questions answered.", |s| &s.questions),
            ("kime_input_tokens_total", "Input tokens read, the usage.input_tokens unit.", |s| {
                &s.input_tokens
            }),
            ("kime_forward_passes_total", "Forward passes run.", |s| &s.passes),
        ];
        for (name, help, get) in counters {
            head(&mut out, name, "counter", help);
            for m in models {
                let _ = writeln!(
                    out,
                    "{name}{{model=\"{}\"}} {}",
                    m.id,
                    get(m.stats).load(Ordering::Relaxed)
                );
            }
        }
        head(&mut out, "kime_queue_depth", "gauge", "Requests queued or running.");
        for m in models {
            let _ = writeln!(out, "kime_queue_depth{{model=\"{}\"}} {}", m.id, m.pending);
        }
        head(
            &mut out,
            "kime_device_seconds_per_request",
            "gauge",
            "Recent device time per request, which the overload and deadline checks use.",
        );
        for m in models {
            let _ = writeln!(
                out,
                "kime_device_seconds_per_request{{model=\"{}\"}} {}",
                m.id,
                m.per_request.as_secs_f64()
            );
        }
        let hists: [Named<Histogram>; 7] = [
            ("kime_queue_seconds", "Wait from submission to the forward pass.", |s| &s.queue),
            ("kime_tokenize_seconds", "Tokenize time of a forward pass.", |s| &s.tokenize),
            ("kime_device_seconds", "Device time of a forward pass.", |s| &s.device),
            ("kime_pass_requests", "Requests in a forward pass.", |s| &s.pass_requests),
            ("kime_pass_questions", "Questions in a forward pass.", |s| &s.pass_questions),
            ("kime_pass_input_tokens", "Input tokens in a forward pass.", |s| &s.pass_tokens),
            ("kime_pass_device_batches", "Device batches a forward pass took.", |s| {
                &s.pass_batches
            }),
        ];
        for (name, help, get) in hists {
            head(&mut out, name, "histogram", help);
            for m in models {
                get(m.stats).render(&mut out, name, &format!("model=\"{}\"", m.id));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram() {
        let h = Histogram::seconds();
        h.time(Duration::from_micros(40));
        h.time(Duration::from_millis(3));
        h.time(Duration::from_secs(9));
        let mut out = String::new();
        h.render(&mut out, "t", "m=\"x\"");
        assert!(out.contains("t_bucket{m=\"x\",le=\"0.00005\"} 1\n"), "{out}");
        assert!(out.contains("t_bucket{m=\"x\",le=\"0.0025\"} 1\n"), "{out}");
        assert!(out.contains("t_bucket{m=\"x\",le=\"0.005\"} 2\n"), "{out}");
        assert!(out.contains("t_bucket{m=\"x\",le=\"5\"} 2\n"), "{out}");
        assert!(out.contains("t_bucket{m=\"x\",le=\"+Inf\"} 3\n"), "{out}");
        assert!(out.contains("t_sum{m=\"x\"} 9.00304\n"), "{out}");
        assert!(out.contains("t_count{m=\"x\"} 3\n"), "{out}");
    }

    #[test]
    fn rejections_by_reason() {
        let m = Metrics::default();
        m.record(0, 200, Duration::from_millis(2));
        m.record(0, 529, Duration::from_micros(80));
        m.record(1, 529, Duration::from_micros(80));
        m.record(0, 401, Duration::from_micros(80));
        let out = m.render(&[]);
        assert!(out.contains("kime_rejected_total{reason=\"overloaded\"} 2\n"), "{out}");
        assert!(out.contains("kime_rejected_total{reason=\"auth\"} 1\n"), "{out}");
        assert!(
            out.contains("kime_requests_total{route=\"/v1/systemone\",status=\"200\"} 1\n"),
            "{out}"
        );
    }
}
