//! Opening a compat model from a Laya directory or a `.kime` file.

use std::path::{Path, PathBuf};

use kime_tensor::Blob;
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::laya::{LayaGraph, LayaSpec};
use crate::tensors::Tensors;
use crate::{pack, safetensors};

/// The files of a Laya checkpoint directory that kime reads, beside `model.safetensors`.
/// `tokenizer_config.json` is optional.
pub const LAYA_FILES: [&str; 4] = [
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
];

#[derive(Debug)]
enum Files {
    Loose(Vec<(String, Vec<u8>)>),
    Packed(Vec<pack::FileEntry>),
}

/// A loaded compat model: its spec, its tensors and the graph bound to them.
#[derive(Debug)]
pub struct Model {
    /// Both configs.
    pub spec: LayaSpec,
    /// The weights, mapped.
    pub tensors: Tensors,
    /// The graph, with every weight bound and checked.
    pub graph: LayaGraph,
    /// The `.kime` index, when loaded from one.
    pub index: Option<pack::Index>,
    /// `__metadata__` of the source safetensors, kept so unpacking gives the same file back.
    pub metadata: Option<Map<String, Value>>,
    files: Files,
}

impl Model {
    /// Opens a Laya checkpoint directory or a `.kime` file. Tensor data is mapped, not read, and
    /// the `.kime` model hash is not checked here, see [`Model::verify`].
    ///
    /// # Errors
    ///
    /// [`Error::Io`] for a missing file, [`Error::Format`] for a malformed one and
    /// [`Error::Mismatch`] when the tensors do not match the config.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() { Self::open_dir(path) } else { Self::open_kime(path) }
    }

    fn open_dir(dir: &Path) -> Result<Self> {
        let mut files = Vec::new();
        for name in LAYA_FILES {
            let p = dir.join(name);
            match std::fs::read(&p) {
                Ok(b) => files.push((name.to_string(), b)),
                Err(e)
                    if e.kind() == std::io::ErrorKind::NotFound
                        && name.ends_with("_config.json") => {}
                Err(e) => return Err(Error::Io(p, e)),
            }
        }
        let st = dir.join("model.safetensors");
        let blob = Blob::map(&st).map_err(|e| Error::Io(st.clone(), e))?;
        let (tensors, metadata) =
            safetensors::load(blob).map_err(|e| Error::Format(format!("{}: {e}", st.display())))?;
        let get = |n: &str| files.iter().find(|(k, _)| k == n).map(|(_, b)| b.as_slice());
        let spec = spec_from(get("rl_agent_config.json"), get("encoder/config.json"), dir)?;
        let graph = LayaGraph::bind(&spec, &tensors)?;
        Ok(Self { spec, tensors, graph, index: None, metadata, files: Files::Loose(files) })
    }

    fn open_kime(path: &Path) -> Result<Self> {
        let blob = Blob::map(path).map_err(|e| Error::Io(path.to_path_buf(), e))?;
        let (index, tensors) =
            pack::load(blob).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
        if index.family != "laya" {
            return Err(Error::Format(format!(
                "{}: family {:?} is not supported by this build",
                path.display(),
                index.family
            )));
        }
        let get = |n: &str| {
            index.files.iter().find(|f| f.name == n).map(|f| &tensors.blob()[f.start..f.end])
        };
        let spec = spec_from(get("rl_agent_config.json"), get("encoder/config.json"), path)?;
        let graph = LayaGraph::bind(&spec, &tensors)?;
        let metadata = index.json.get("safetensors_metadata").and_then(Value::as_object).cloned();
        let files = Files::Packed(index.files.clone());
        Ok(Self { spec, tensors, graph, index: Some(index), metadata, files })
    }

    /// A carried file, such as `tokenizer/tokenizer.json`.
    #[must_use]
    pub fn file(&self, name: &str) -> Option<&[u8]> {
        match &self.files {
            Files::Loose(v) => v.iter().find(|(k, _)| k == name).map(|(_, b)| b.as_slice()),
            Files::Packed(v) => {
                v.iter().find(|f| f.name == name).map(|f| &self.tensors.blob()[f.start..f.end])
            }
        }
    }

    /// Names of the carried files.
    #[must_use]
    pub fn file_names(&self) -> Vec<&str> {
        match &self.files {
            Files::Loose(v) => v.iter().map(|(k, _)| k.as_str()).collect(),
            Files::Packed(v) => v.iter().map(|f| f.name.as_str()).collect(),
        }
    }

    /// Checks the `.kime` model hash against the data. A directory has no hash and always passes.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] if the data has changed since it was packed.
    pub fn verify(&self) -> Result<()> {
        match &self.index {
            Some(index) => pack::verify(self.tensors.blob(), index),
            None => Ok(()),
        }
    }

    /// Packs the model as `.kime` and returns the model hash.
    ///
    /// # Errors
    ///
    /// Any error from `out`.
    pub fn pack(&self, out: &mut impl std::io::Write) -> Result<String> {
        let mut extra = Map::new();
        if let Some(m) = &self.metadata {
            extra.insert("safetensors_metadata".into(), Value::Object(m.clone()));
        }
        let files =
            self.file_names().into_iter().map(|n| (n, self.file(n).expect("listed"))).collect();
        pack::write(
            &pack::Contents {
                family: "laya",
                id: &self.spec.id,
                tensors: &self.tensors,
                files,
                extra,
            },
            out,
        )
    }

    /// Writes the model back out as a Laya directory: `model.safetensors` plus the carried files.
    /// Refuses to write into a directory that already has a `model.safetensors`.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] on any write failure or an existing checkpoint.
    pub fn unpack(&self, dir: &Path) -> Result<()> {
        let io = |p: PathBuf| move |e| Error::Io(p, e);
        let st = dir.join("model.safetensors");
        if st.exists() {
            return Err(Error::Io(st, std::io::Error::from(std::io::ErrorKind::AlreadyExists)));
        }
        for name in self.file_names() {
            let p = dir.join(name);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).map_err(io(parent.to_path_buf()))?;
            }
            std::fs::write(&p, self.file(name).expect("listed")).map_err(io(p.clone()))?;
        }
        let f = std::fs::File::create(&st).map_err(io(st.clone()))?;
        let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
        safetensors::write(&self.tensors, self.metadata.as_ref(), &mut w)
            .map_err(io(st.clone()))?;
        std::io::Write::flush(&mut w).map_err(io(st))
    }
}

fn spec_from(agent: Option<&[u8]>, encoder: Option<&[u8]>, at: &Path) -> Result<LayaSpec> {
    let parse = |b: Option<&[u8]>, name: &str| -> Result<Value> {
        let b = b.ok_or_else(|| Error::Format(format!("{}: no {name}", at.display())))?;
        serde_json::from_slice(b)
            .map_err(|e| Error::Format(format!("{}: {name}: {e}", at.display())))
    };
    LayaSpec::from_json(
        &parse(agent, "rl_agent_config.json")?,
        &parse(encoder, "encoder/config.json")?,
    )
}
