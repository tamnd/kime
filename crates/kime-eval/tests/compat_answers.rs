//! Answers against Laya's: the logits and act logits Laya's model produced for each parity case go
//! through `kime_core::answer`, and the response has to equal the one `Agent.system_one` returned,
//! every rounded number included. This checks the answer building alone, apart from the model.

use kime_core::answer::{LAYA_MODEL, Response, Temperatures, laya_answer};
use kime_core::request::{Limits, parse};
use serde_json::Value;

fn lines(name: &str) -> Vec<Value> {
    let path = format!("{}/fixtures/parity/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn check(model: &str, temps: &Temperatures) {
    let (cases, dumps) = (lines("cases.jsonl"), lines(&format!("{model}.jsonl")));
    let (mut same, mut bad) = (0, Vec::new());
    for (case, dump) in cases.iter().zip(&dumps) {
        let Some(want) = dump.get("answer").filter(|a| a.is_object()) else { continue };
        let req = parse(case, &Limits::LAYA).unwrap();
        let qs = dump["questions"].as_array().map_or(&[][..], Vec::as_slice);
        let mut answers = Vec::new();
        let mut input_tokens = 0;
        for (q, d) in req.questions.iter().zip(qs) {
            let f = |v: &Value| {
                v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<_>>()
            };
            let (logits, act) = (f(&d["logits"]), f(&d["act_probs"]));
            input_tokens += d["ids"].as_array().unwrap().len();
            answers.push((q.id.clone(), laya_answer(q, &logits, [act[0], act[1]], temps)));
        }
        let got = Response { model: LAYA_MODEL.into(), answers, input_tokens }.to_json();
        if &got == want {
            same += 1;
        } else {
            bad.push(format!("{}\n  got  {got}\n  want {want}", case["id"]));
        }
    }
    assert!(
        bad.is_empty(),
        "{model}: {} of {} differ\n{}",
        bad.len(),
        same + bad.len(),
        bad[..bad.len().min(3)].join("\n")
    );
    assert!(same > 150, "{model}: only {same} answers checked");
    eprintln!("{model}: {same} responses equal Laya's");
}

/// The calibration in the English checkpoint's `rl_agent_config.json` on the hub.
#[test]
fn laya_english() {
    let by_options = [
        ("choice:3-5", 1.760_151_863_098_144_5),
        ("choice:6-10", 1.000_015_854_835_510_3),
        ("score:3-5", 1.251_430_034_637_451_2),
        ("noul:2", 1.983_399_510_383_606),
        ("choice:11+", 0.100_582_808_256_149_29),
        ("choice:2", 1.906_356_334_686_279_3),
    ]
    .map(|(k, t)| (k.to_string(), t));
    let t = Temperatures::new(
        [1.636_903_047_561_645_5, 1.251_430_034_637_451_2, 1.983_399_510_383_606],
        &by_options,
    );
    check("laya", &t);
}

/// The multilingual checkpoint ships no calibration, so every temperature is 1.
#[test]
fn laya_multilingual() {
    check("laya-multilingual", &Temperatures::default());
}
