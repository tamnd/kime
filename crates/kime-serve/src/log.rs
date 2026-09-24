//! The request log from spec/11: one line per request on stdout, off unless asked for. Bodies are
//! never logged, only their size and a blake3 hash, so a line can be matched to a request the
//! client kept without the log holding what was in it.
//!
//! Off, the server does not look at the body at all and the only cost is one branch per request.

use std::io::Write;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::json;

/// Whether requests are logged, and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Log {
    /// No request log.
    #[default]
    Off,
    /// One line of text per request.
    Text,
    /// One JSON object per line.
    Json,
}

/// What the handler knew about a request it answered.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Served {
    /// The index of the model that answered.
    pub(crate) model: usize,
    pub(crate) questions: usize,
    pub(crate) tokens: u64,
}

/// One logged request.
#[derive(Debug)]
pub(crate) struct Line<'a> {
    pub(crate) at: SystemTime,
    pub(crate) id: &'a str,
    pub(crate) method: &'a str,
    pub(crate) path: &'a str,
    pub(crate) status: u16,
    pub(crate) took: Duration,
    pub(crate) bytes: usize,
    pub(crate) hash: blake3::Hash,
    /// The model id, questions and input tokens, for an answered request.
    pub(crate) served: Option<(&'a str, usize, u64)>,
}

impl Line<'_> {
    fn render(&self, log: Log) -> String {
        let ts = timestamp(self.at);
        let ms = self.took.as_secs_f64() * 1e3;
        let hash = &self.hash.to_hex()[..16];
        match log {
            Log::Json => {
                let mut v = json!({
                    "ts": ts,
                    "request_id": self.id,
                    "method": self.method,
                    "path": self.path,
                    "status": self.status,
                    "ms": (ms * 1e3).round() / 1e3,
                    "body_bytes": self.bytes,
                    "body_blake3": hash,
                });
                if let Some((model, questions, tokens)) = self.served {
                    v["model"] = model.into();
                    v["questions"] = questions.into();
                    v["input_tokens"] = tokens.into();
                }
                let mut s = v.to_string();
                s.push('\n');
                s
            }
            _ => {
                let mut s = format!(
                    "{ts} {} {} {} {} {ms:.3}ms body={}:{hash}",
                    self.id, self.method, self.path, self.status, self.bytes
                );
                if let Some((model, questions, tokens)) = self.served {
                    s += &format!(" model={model} questions={questions} tokens={tokens}");
                }
                s.push('\n');
                s
            }
        }
    }

    /// Writes the line to stdout in one call, so lines from different requests never mix.
    pub(crate) fn write(&self, log: Log) {
        let line = self.render(log);
        let _ = std::io::stdout().lock().write_all(line.as_bytes());
    }
}

/// RFC 3339 in UTC with milliseconds.
fn timestamp(at: SystemTime) -> String {
    let d = at.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs();
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60,
        d.subsec_millis()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(served: Option<(&str, usize, u64)>) -> Line<'_> {
        Line {
            at: UNIX_EPOCH + Duration::from_millis(1_790_246_096_789),
            id: "req_1",
            method: "POST",
            path: "/v1/systemone",
            status: 200,
            took: Duration::from_micros(12_345),
            bytes: 3,
            hash: blake3::hash(b"abc"),
            served,
        }
    }

    #[test]
    fn timestamps() {
        assert_eq!(timestamp(UNIX_EPOCH), "1970-01-01T00:00:00.000Z");
        assert_eq!(
            timestamp(UNIX_EPOCH + Duration::from_millis(951_782_400_001)),
            "2000-02-29T00:00:00.001Z"
        );
        assert_eq!(timestamp(line(None).at), "2026-09-24T10:34:56.789Z");
    }

    #[test]
    fn text_and_json() {
        assert_eq!(
            line(Some(("laya", 2, 96))).render(Log::Text),
            "2026-09-24T10:34:56.789Z req_1 POST /v1/systemone 200 12.345ms body=3:6437b3ac38465133 model=laya questions=2 tokens=96\n"
        );
        let v: serde_json::Value = serde_json::from_str(&line(None).render(Log::Json)).unwrap();
        assert_eq!(v["ms"], 12.345);
        assert_eq!(v["body_blake3"], "6437b3ac38465133");
        assert!(v.get("model").is_none());
    }
}
