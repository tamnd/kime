//! `agent_step` against jev-ultrafast's own `choose`: for generated snapshots in the shape
//! `snapshot.js` returns, the same request body key for key, and for generated answers, some of
//! them broken, the same decision or the same refusal. `tools/agent/record.py` writes
//! `tests/agent/jev-ultrafast.json`.

use kime_core::agent::agent_step;
use serde_json::Value;

#[test]
fn same_as_jev_ultrafast() {
    // KIME_AGENT_CASES points at a bigger recording than the committed one.
    let path = std::env::var("KIME_AGENT_CASES").unwrap_or_else(|_| {
        format!("{}/tests/agent/jev-ultrafast.json", env!("CARGO_MANIFEST_DIR"))
    });
    let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let (mut wrong, mut refused) = (Vec::new(), 0);
    for (i, c) in cases.iter().enumerate() {
        let history = c["history"].as_array().unwrap();
        let step = agent_step(&c["snapshot"], c["goal"].as_str().unwrap(), history);
        let mut body = step.body("jev-latest");
        body.as_object_mut().unwrap().shift_remove("model");
        // Compared as text so key order counts too, which Value's == ignores.
        #[allow(clippy::cmp_owned)]
        if body.to_string() != c["body"].to_string() {
            wrong.push(format!("case {i}: body\n  jev  {}\n  kime {body}", c["body"]));
            continue;
        }
        let want = &c["result"];
        match step.decide(&c["answers"]) {
            Err(_) if want.get("error").is_some() => refused += 1,
            Ok(d)
                if want.get("error").is_none()
                    && d.choice == want["choice"]
                    && d.operation == want["operation"]
                    && d.target.as_deref() == want["target"].as_str()
                    && Some(d.confidence) == want["confidence"].as_f64()
                    && Value::Object(d.probabilities.clone()) == want["probabilities"]
                    && d.target_confidence == want["target_confidence"].as_f64() => {}
            got => wrong.push(format!("case {i}: jev {want}\n  kime {got:?}")),
        }
    }
    assert!(wrong.is_empty(), "{} of {} differ:\n{}", wrong.len(), cases.len(), wrong.join("\n"));
    assert!(refused > 0 && refused < cases.len(), "{refused} refused");
}
