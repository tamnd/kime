//! The kime-v1 split forward against `tools/ref/kime_ref.py`, a float64 numpy forward written
//! from spec/05-model.md. Both draw the same seeded random weights for a tiny config, so the
//! fixture holds only token ids and outputs. The bounds are the CPU tolerances of
//! spec/15-testing.md: logits within 1e-4, probabilities within 1e-5 and argmax on every row.

use kime_cpu::split::{Row, SegmentCache, Split};
use kime_model::kime_v1::{V1Graph, V1Spec, random_weights};
use kime_model::{Tensors, safetensors};
use kime_tensor::Blob;
use serde_json::{Map, Value, json};

fn fixture() -> Value {
    let path = format!("{}/tests/fixtures/kime-v1-random.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn tensors(spec: &V1Spec, seed: u64) -> Tensors {
    let (mut header, mut data) = (Map::new(), Vec::new());
    for (name, shape, v) in random_weights(spec, seed) {
        let start = data.len();
        v.iter().for_each(|x| data.extend_from_slice(&x.to_le_bytes()));
        header.insert(
            name,
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let head = Value::Object(header).to_string().into_bytes();
    let mut bytes = (head.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&head);
    bytes.extend_from_slice(&data);
    safetensors::load(Blob::owned(bytes)).unwrap().0
}

fn softmax(l: &[f64]) -> Vec<f64> {
    let mx = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = l.iter().map(|x| (x - mx).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

fn argmax(v: &[f64]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

#[test]
fn kime_v1_parity() {
    let fx = fixture();
    let spec = V1Spec::from_json(&fx["config"]).unwrap();
    let first: Vec<f64> = nums(&fx["first_weights"]);
    let t = tensors(&spec, fx["seed"].as_u64().unwrap());
    let w = t.get("state.embeddings.tok_embeddings.weight").unwrap().to_f32();
    let same = w.iter().zip(&first).all(|(a, b)| f64::from(*a).to_bits() == b.to_bits());
    assert!(same, "the random weights differ from the reference");
    let graph = V1Graph::bind(&spec, &t).unwrap();
    let split = Split::new(&spec, &graph, &t, 2);
    let (mut logit, mut prob, mut pool, mut rows) = (0f64, 0f64, 0f64, 0);
    let mut cache = SegmentCache::default();
    for case in fx["cases"].as_array().unwrap() {
        let mem = match case.get("segments") {
            Some(g) => {
                let segs: Vec<Vec<u32>> = g.as_array().unwrap().iter().map(ids).collect();
                let segs: Vec<&[u32]> = segs.iter().map(Vec::as_slice).collect();
                cache.reset_counts();
                let mem = split.state_segments(&segs, &mut cache);
                let distinct = segs.iter().collect::<std::collections::HashSet<_>>().len();
                // Every case shares [CLS_S] and [SEP] with the ones before it.
                assert!(cache.misses <= distinct && cache.misses + cache.hits == segs.len());
                cache.reset_counts();
                let again = split.state_segments(&segs, &mut cache);
                assert_eq!((cache.hits, cache.misses), (segs.len(), 0));
                assert_eq!(again.states, mem.states, "a cached segment gives other bits");
                mem
            }
            None => split.state(&ids(&case["state"])),
        };
        for (a, b) in mem.pooled.iter().zip(nums(&case["pooled"])) {
            pool = pool.max((f64::from(*a) - b).abs());
        }
        for r in case["rows"].as_array().unwrap() {
            let options: Vec<Vec<u32>> = r["options"].as_array().unwrap().iter().map(ids).collect();
            let row = Row {
                qtype: r["qtype"].as_u64().unwrap() as usize,
                header: &ids(&r["header"]),
                options: &options,
            };
            let got: Vec<f64> = split.question(&mem, row).into_iter().map(f64::from).collect();
            let want = nums(&r["logits"]);
            assert_eq!(got.len(), want.len());
            for (a, b) in got.iter().zip(&want) {
                logit = logit.max((a - b).abs());
            }
            for (a, b) in softmax(&got).iter().zip(softmax(&want)) {
                prob = prob.max((a - b).abs());
            }
            assert_eq!(argmax(&got), argmax(&want), "argmax differs on {r}");
            rows += 1;
        }
    }
    eprintln!(
        "{rows} rows: logits within {logit:.2e}, probabilities {prob:.2e}, pooled {pool:.2e}"
    );
    assert!(logit <= 1e-4 && prob <= 1e-5 && pool <= 1e-5, "{logit:e} {prob:e} {pool:e}");
}
