//! The native part of the kime Python package. It takes and returns JSON text, and the Python side
//! turns that into dicts, so this layer stays small and the wire format is the same one the server
//! speaks. Forward passes run with the GIL released.

use kime_core::request::{Limits, Problem, parse};
use kime_engine::{Device, Error, Kime, Precision};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde_json::{Map, Value, json};

fn json(text: &str, what: &str) -> PyResult<Value> {
    serde_json::from_str(text)
        .map_err(|e| PyValueError::new_err(format!("{what} is not JSON: {e}")))
}

fn error(e: Error) -> PyErr {
    match e {
        Error::Invalid(_) | Error::TooLong { .. } => PyValueError::new_err(e.to_string()),
        e => PyRuntimeError::new_err(e.to_string()),
    }
}

fn device(d: &str, threads: usize) -> PyResult<Device> {
    let d = d.to_ascii_lowercase();
    Ok(match d.as_str() {
        "auto" if threads > 0 => Device::Cpu { threads },
        "auto" => Device::Auto,
        "cpu" => Device::Cpu { threads },
        "cuda" => Device::Cuda(0),
        d => match d.strip_prefix("cuda:").and_then(|n| n.parse().ok()) {
            Some(n) => Device::Cuda(n),
            None => return Err(PyValueError::new_err(format!("unknown device {d:?}"))),
        },
    })
}

fn precision(p: &str) -> PyResult<Precision> {
    Ok(match p.to_ascii_lowercase().as_str() {
        "f16" | "fp16" | "float16" => Precision::F16,
        "f32" | "fp32" | "float32" => Precision::F32,
        "int8" => Precision::Int8,
        p => return Err(PyValueError::new_err(format!("unknown precision {p:?}"))),
    })
}

/// A loaded model.
#[pyclass(frozen, module = "kime._native")]
struct Engine {
    kime: Kime,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (model, device="auto", threads=0, precision=None))]
    fn new(
        py: Python<'_>,
        model: &str,
        device: &str,
        threads: usize,
        precision: Option<&str>,
    ) -> PyResult<Self> {
        let mut b = Kime::builder().model(model).device(self::device(device, threads)?);
        if let Some(p) = precision {
            b = b.precision(self::precision(p)?);
        }
        let kime = py.detach(|| b.preload(true).build()).map_err(error)?;
        Ok(Engine { kime })
    }

    /// The answers to one request body, `{"state": ..., "questions": {...}}`, as JSON text.
    fn system_one(&self, py: Python<'_>, body: &str) -> PyResult<String> {
        let req = parse(&json(body, "the request")?, &Limits::LAYA)
            .map_err(|p| error(Error::Invalid(p)))?;
        let resp = py.detach(|| self.kime.decide(&req)).map_err(error)?;
        Ok(resp.to_json().to_string())
    }

    /// The answers to many request bodies, in order, from as few forward passes as fit.
    fn predict_batch(&self, py: Python<'_>, bodies: Vec<String>) -> PyResult<Vec<String>> {
        let mut reqs = Vec::with_capacity(bodies.len());
        for (i, body) in bodies.iter().enumerate() {
            let v = json(body, &format!("request {i}"))?;
            reqs.push(parse(&v, &Limits::LAYA).map_err(|p| error(Error::Invalid(p)))?);
        }
        let resps = py.detach(|| self.kime.decide_batch(&reqs)).map_err(error)?;
        Ok(resps.iter().map(|r| r.to_json().to_string()).collect())
    }

    /// What `kime serve` answers a Jev client for this body, as the status and the JSON body:
    /// 200 with Jev's shape, 400 when it is not JSON, or 422 with every validation problem.
    fn decide_jev(&self, py: Python<'_>, body: &str) -> PyResult<(u16, String)> {
        let reply = |status: u16, v: Value| Ok((status, v.to_string()));
        let Ok(v) = serde_json::from_str::<Value>(body) else {
            return reply(400, json!({"detail": "request body must be valid JSON"}));
        };
        let (precision, entropy) = match jev_opts(v.get("kime")) {
            Ok(o) => o,
            Err(msg) => return reply(422, json!({"detail": msg})),
        };
        let req = match parse(&v, &Limits::JEV) {
            Ok(r) => r,
            Err(p) => {
                return reply(
                    422,
                    json!({"detail": p.iter().map(Problem::to_json).collect::<Vec<_>>()}),
                );
            }
        };
        match py.detach(|| self.kime.decide(&req)) {
            Ok(r) => {
                let answers: Map<String, Value> = r
                    .answers
                    .iter()
                    .map(|(id, a)| (id.clone(), a.to_jev_json(precision, entropy)))
                    .collect();
                reply(
                    200,
                    json!({"model": self.kime.model_id(), "answers": answers,
                        "usage": {"input_tokens": r.input_tokens, "output_tokens": 0}}),
                )
            }
            Err(e @ (Error::Invalid(_) | Error::TooLong { .. })) => {
                reply(422, json!({"detail": e.to_string()}))
            }
            Err(e) => Err(error(e)),
        }
    }

    /// The tokens a request reads over all its sequences.
    fn count_tokens(&self, body: &str) -> PyResult<usize> {
        let req = parse(&json(body, "the request")?, &Limits::LAYA)
            .map_err(|p| error(Error::Invalid(p)))?;
        Ok(self.kime.count_tokens(&req))
    }

    /// Each text's encoder output mean pooled, one row per text, with each text cut to
    /// `max_length` tokens with the specials.
    fn embed(&self, py: Python<'_>, texts: Vec<String>, max_length: usize) -> PyResult<Vec<Vec<f32>>> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        py.detach(|| self.kime.embed(&refs, max_length)).map_err(error)
    }

    #[getter]
    fn model_id(&self) -> &str {
        self.kime.model_id()
    }

    #[getter]
    fn device(&self) -> String {
        self.kime.device()
    }
}

/// The `kime` options that change a Jev answer: `precision` (0 to 6 or null, default 2) and
/// `confidence` ("jev" or "entropy"). The rest are for the server.
fn jev_opts(kime: Option<&Value>) -> Result<(Option<u32>, bool), String> {
    let Some(k) = kime.filter(|k| !k.is_null()) else { return Ok((Some(2), false)) };
    let k = k.as_object().ok_or("kime must be an object")?;
    let precision = match k.get("precision") {
        None => Some(2),
        Some(Value::Null) => None,
        Some(v) => Some(
            v.as_u64()
                .filter(|&p| p <= 6)
                .and_then(|p| u32::try_from(p).ok())
                .ok_or("precision must be 0 to 6 or null")?,
        ),
    };
    let entropy = match k.get("confidence").map(Value::as_str) {
        None | Some(Some("jev")) => false,
        Some(Some("entropy")) => true,
        Some(_) => return Err("confidence must be \"jev\" or \"entropy\"".into()),
    };
    Ok((precision, entropy))
}

/// Laya's `clean_email_body`.
#[pyfunction]
#[pyo3(signature = (body, max_chars=kime_core::email::MAX_CHARS))]
fn clean_email_body(py: Python<'_>, body: &str, max_chars: usize) -> String {
    py.detach(|| kime_core::email::clean_email_body(body, max_chars))
}

/// A preset's questions as JSON text. `categories` is JSON text too, for the email preset.
#[pyfunction]
#[pyo3(signature = (name, categories=None))]
fn preset(name: &str, categories: Option<&str>) -> PyResult<String> {
    let q = match (name, categories) {
        ("email", Some(c)) => match json(c, "categories")? {
            Value::Object(m) => kime_core::presets::email(Some(m)),
            _ => return Err(PyValueError::new_err("categories must be a dict")),
        },
        (name, _) => kime_core::presets::get(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown preset {name:?}")))?,
    };
    Ok(q.to_string())
}

/// Laya's `analyse` of a state given as JSON text, as JSON text.
#[pyfunction]
fn analyse(py: Python<'_>, state: &str) -> PyResult<String> {
    let state = json(state, "the state")?;
    Ok(py.detach(|| kime_route::lang::analyse(&state).to_json().to_string()))
}

/// Laya's `detect_script`.
#[pyfunction]
fn detect_script(text: &str) -> &'static str {
    kime_route::lang::detect_script(text)
}

/// The checkpoint kime's detection picks for a state given as JSON text: Laya's rules with the
/// language identifier on top, as `kime serve` routes. JSON text with `english` (true, false or
/// null for the default), `reason` and `detection`.
#[pyfunction]
fn detect(py: Python<'_>, state: &str, default_english: bool) -> PyResult<String> {
    let state = json(state, "the state")?;
    let d = py.detach(|| kime_route::router::detect(&state, default_english));
    let mut out = Map::new();
    out.insert("english".into(), d.english.into());
    out.insert("reason".into(), d.reason.into());
    out.insert("detection".into(), d.detection.map_or(Value::Null, |a| a.to_json()));
    Ok(Value::Object(out).to_string())
}

/// One step of a browser agent, from jev-ultrafast.
#[pyclass(frozen, module = "kime._native")]
struct AgentStep {
    step: kime_core::agent::AgentStep,
}

#[pymethods]
impl AgentStep {
    #[new]
    #[pyo3(signature = (snapshot, goal, history="[]"))]
    fn new(snapshot: &str, goal: &str, history: &str) -> PyResult<Self> {
        let snapshot = json(snapshot, "the snapshot")?;
        let history = match json(history, "the history")? {
            Value::Array(h) => h,
            _ => return Err(PyValueError::new_err("history must be a list")),
        };
        Ok(AgentStep { step: kime_core::agent::agent_step(&snapshot, goal, &history) })
    }

    /// The state, as JSON text.
    #[getter]
    fn state(&self) -> String {
        self.step.state.to_string()
    }

    /// The questions, as JSON text.
    #[getter]
    fn questions(&self) -> String {
        serde_json::to_string(&self.step.questions).unwrap_or_default()
    }

    /// The request body for `model`, as JSON text.
    fn body(&self, model: &str) -> String {
        self.step.body_json(model)
    }

    /// The action the answers pick, as JSON text. Raises ValueError for an answer jev-ultrafast
    /// would not act on.
    fn decide(&self, answers: &str) -> PyResult<String> {
        let answers = json(answers, "the answers")?;
        let d = self.step.decide(&answers).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut out = Map::new();
        out.insert("choice".into(), d.choice.into());
        out.insert("operation".into(), d.operation.into());
        out.insert("target".into(), d.target.into());
        out.insert("confidence".into(), d.confidence.into());
        out.insert("probabilities".into(), Value::Object(d.probabilities));
        out.insert("target_confidence".into(), d.target_confidence.into());
        Ok(Value::Object(out).to_string())
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("RELEASE_DATE", kime_core::RELEASE_DATE)?;
    m.add_class::<Engine>()?;
    m.add_class::<AgentStep>()?;
    m.add_function(wrap_pyfunction!(clean_email_body, m)?)?;
    m.add_function(wrap_pyfunction!(preset, m)?)?;
    m.add_function(wrap_pyfunction!(analyse, m)?)?;
    m.add_function(wrap_pyfunction!(detect_script, m)?)?;
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    Ok(())
}
