//! The published Laya checkpoints against what Laya itself builds from them.
//!
//! `fixtures/laya-tensors.json` comes from `tools/ref/laya_tensors.py`: the parameter names and
//! shapes of Laya's own `DecisionModel` for each checkpoint, and each tensor's dtype, sum and first
//! values as read by the safetensors library. The weights are 1.5 GB and not in CI, so the test
//! reads them from `$KIME_MODELS/laya` and says so and passes when they are missing, unless
//! `KIME_REQUIRE_WEIGHTS` is set.

use std::path::PathBuf;

use kime_model::Model;
use serde_json::Value;

fn weights(sub: &str) -> Option<PathBuf> {
    let root = std::env::var_os("KIME_MODELS").map(PathBuf::from)?.join("laya").join(sub);
    root.join("model.safetensors").is_file().then_some(root)
}

fn fixture(name: &str) -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/laya-tensors.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    v[name].clone()
}

fn check(name: &str, sub: &str) {
    let Some(dir) = weights(sub) else {
        assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "no weights for {name}");
        eprintln!("skipping {name}: set KIME_MODELS to a folder holding laya/ with its weights");
        return;
    };
    let m = Model::open(&dir).unwrap();
    assert_eq!(m.spec.id, name);
    let want = fixture(name);

    // Laya's module and kime's graph must agree on every name and shape.
    let module = want["module"].as_object().unwrap();
    let expected = m.spec.expected();
    assert_eq!(expected.len(), module.len());
    for (n, shape) in &expected {
        let s: Vec<usize> =
            module[n].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        assert_eq!(&s, shape, "{n}");
    }

    // And the bytes kime maps must read as the values torch reads.
    let tensors = want["tensors"].as_array().unwrap();
    assert_eq!(tensors.len(), m.tensors.entries().len());
    for t in tensors {
        let n = t["name"].as_str().unwrap();
        let v = m.tensors.get(n).unwrap();
        let dtype = match v.dtype.name() {
            "F16" => "float16",
            "F32" => "float32",
            "BF16" => "bfloat16",
            other => other,
        };
        assert_eq!(dtype, t["dtype"].as_str().unwrap(), "{n}");
        let x = v.to_f32();
        for (i, f) in t["first"].as_array().unwrap().iter().enumerate() {
            assert_eq!(x[i].to_bits(), (f.as_f64().unwrap() as f32).to_bits(), "{n}[{i}]");
        }
        let sum: f64 = x.iter().map(|&f| f64::from(f)).sum();
        let want = t["sum"].as_f64().unwrap();
        assert!(
            (sum - want).abs() <= 1e-9 * want.abs().max(1.0) * x.len() as f64,
            "{n}: {sum} vs {want}"
        );
    }

    // Pack, reopen from the .kime, and check that the tensors and files come back bit exact and
    // that unpacking would give back the original model.safetensors byte for byte.
    let tmp = std::env::temp_dir().join(format!("kime-model-test-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let packed = tmp.join("model.kime");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&packed).unwrap());
    let hash = m.pack(&mut f).unwrap();
    drop(f);
    let k = Model::open(&packed).unwrap();
    k.verify().unwrap();
    assert_eq!(k.index.as_ref().unwrap().hash, hash);
    assert_eq!(k.spec, m.spec);
    assert_eq!(k.graph, m.graph);
    for (i, e) in m.tensors.entries().iter().enumerate() {
        assert_eq!(k.tensors.view(i).bytes, m.tensors.view(i).bytes, "{}", e.name);
        assert_eq!(k.tensors.entries()[i].start % 4096, 0);
    }
    assert_eq!(k.file_names(), m.file_names());
    for n in m.file_names() {
        assert_eq!(k.file(n), m.file(n), "{n}");
    }
    // Unpacking streams into a hash rather than a file, to keep the test's disk use to one copy.
    let mut h = blake3::Hasher::new();
    kime_model::safetensors::write(&k.tensors, k.metadata.as_ref(), &mut h).unwrap();
    let orig = blake3::hash(&std::fs::read(dir.join("model.safetensors")).unwrap());
    assert_eq!(h.finalize(), orig, "unpacked model.safetensors differs from the original");
    for n in m.file_names() {
        assert_eq!(k.file(n).unwrap(), std::fs::read(dir.join(n)).unwrap(), "{n}");
    }
    std::fs::remove_dir_all(&tmp).unwrap();
    eprintln!(
        "{name}: {} tensors match Laya, pack and unpack are bit exact, hash {hash}",
        expected.len()
    );
}

#[test]
fn laya() {
    check("laya", "");
}

#[test]
fn laya_multilingual() {
    check("laya-multilingual", "multilingual");
}
