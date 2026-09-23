use std::fmt;
use std::path::PathBuf;

/// Everything that can go wrong opening a checkpoint.
#[derive(Debug)]
pub enum Error {
    /// A file could not be read or written.
    Io(PathBuf, std::io::Error),
    /// The bytes are not a valid file of the expected format. The message says where.
    Format(String),
    /// The file is valid but does not hold the model its config describes. One line per problem,
    /// naming the tensor, as Laya's `_verify_compatibility` does.
    Mismatch(Vec<String>),
}

impl Error {
    pub(crate) fn format(msg: impl Into<String>) -> Self {
        Self::Format(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(path, e) => write!(f, "{}: {e}", path.display()),
            Self::Format(msg) => f.write_str(msg),
            Self::Mismatch(problems) => {
                write!(f, "checkpoint does not match its config ({} problems)", problems.len())?;
                for p in problems {
                    write!(f, "\n  {p}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for Error {}

pub(crate) type Result<T> = std::result::Result<T, Error>;
