//! Pack a COGX archive directory into `.cogx.tar.gz` for transport.
//!
//! Python's `POST /api/v1/remember` with `content_type=cogx-archive` takes a
//! single tarball, not a directory, so this is the encoding a Rust `push`
//! needs. It is the sending half of Python's `archive.py`; the receiving half
//! (`unpack_archive`) lives there.

use std::fs::File;
use std::path::{Path, PathBuf};

use flate2::Compression;
use flate2::write::GzEncoder;

use crate::error::{MigrationError, MigrationResult};

/// Suffix Python's `pack_archive` uses. Kept identical so an uploaded file
/// name looks the same coming from either SDK.
pub const ARCHIVE_SUFFIX: &str = ".cogx.tar.gz";

/// Tar and gzip an archive directory so its files sit at the tarball root.
///
/// Root placement matters: Python's `find_archive_root` accepts `manifest.json`
/// at the tarball root or inside a *single* subdirectory, and nothing else.
pub fn pack_archive(
    archive_dir: impl AsRef<Path>,
    tar_path: impl Into<PathBuf>,
) -> MigrationResult<PathBuf> {
    let archive_dir = archive_dir.as_ref();
    let tar_path = tar_path.into();

    let file = File::create(&tar_path).map_err(|error| MigrationError::io(&tar_path, error))?;
    let encoder = GzEncoder::new(file, Compression::default());
    let mut builder = tar::Builder::new(encoder);

    // Only the writer's own files, never everything in the directory.
    // `-o` is caller-supplied, so a user pointing it at a populated directory
    // (`export -o ~/notes --pack`) would otherwise ship the rest of that
    // directory inside a tarball they hand to someone else. Python's
    // equivalent rglobs the whole tree, but only ever over a directory
    // `push()` just created for the purpose.
    let mut entries: Vec<PathBuf> = crate::cogx::OWNED_FILES
        .iter()
        .map(|name| archive_dir.join(name))
        .filter(|path| path.symlink_metadata().is_ok())
        .collect();
    entries.sort();

    for path in entries {
        let name = path
            .file_name()
            .ok_or_else(|| {
                MigrationError::io(
                    &path,
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "archive entry has no file name",
                    ),
                )
            })?
            .to_owned();

        // A planted symlink named like an owned file would otherwise pull
        // an arbitrary readable file into the tarball — `is_file` and
        // `append_path_with_name` both follow symlinks.
        let meta =
            std::fs::symlink_metadata(&path).map_err(|error| MigrationError::io(&path, error))?;
        if !meta.file_type().is_file() {
            return Err(MigrationError::io(
                &path,
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "refusing to archive a symlink or non-regular file",
                ),
            ));
        }
        let mut file = File::open(&path).map_err(|error| MigrationError::io(&path, error))?;
        // Close the swap race: the opened handle must be the lstat'd entry.
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let opened = file
                .metadata()
                .map_err(|error| MigrationError::io(&path, error))?;
            if opened.dev() != meta.dev() || opened.ino() != meta.ino() {
                return Err(MigrationError::io(
                    &path,
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "refusing to archive a symlink or non-regular file",
                    ),
                ));
            }
        }
        builder
            .append_file(&name, &mut file)
            .map_err(|error| MigrationError::io(&path, error))?;
    }

    builder
        .into_inner()
        .map_err(|error| MigrationError::io(&tar_path, error))?
        .finish()
        .map_err(|error| MigrationError::io(&tar_path, error))?;

    Ok(tar_path)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
mod tests {
    use super::*;
    use crate::cogx::MANIFEST_FILE;
    use flate2::read::GzDecoder;

    #[test]
    fn manifest_lands_at_the_tarball_root() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("archive");
        std::fs::create_dir_all(&archive).unwrap();
        std::fs::write(archive.join(MANIFEST_FILE), "{}").unwrap();
        std::fs::write(archive.join("entities.jsonl"), "{}\n").unwrap();

        let tar_path = pack_archive(&archive, dir.path().join("out.cogx.tar.gz")).unwrap();

        let names: Vec<String> = tar::Archive::new(GzDecoder::new(File::open(&tar_path).unwrap()))
            .entries()
            .unwrap()
            .map(|entry| entry.unwrap().path().unwrap().display().to_string())
            .collect();

        assert!(
            names.contains(&MANIFEST_FILE.to_string()),
            "manifest is not at the tarball root: {names:?}"
        );
        assert!(names.contains(&"entities.jsonl".to_string()));
    }

    /// A planted symlink named like an owned file must abort the pack
    /// instead of pulling the link target into the tarball.
    #[cfg(unix)]
    #[test]
    fn a_symlinked_owned_file_aborts_the_pack() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("archive");
        std::fs::create_dir_all(&archive).unwrap();
        std::fs::write(archive.join(MANIFEST_FILE), "{}").unwrap();
        let secret = dir.path().join("secret.txt");
        std::fs::write(&secret, "should never ship").unwrap();
        std::os::unix::fs::symlink(&secret, archive.join("entities.jsonl")).unwrap();

        let result = pack_archive(&archive, dir.path().join("out.cogx.tar.gz"));
        assert!(result.is_err(), "packing a symlinked owned file must fail");
        assert_eq!(
            std::fs::read_to_string(&secret).unwrap(),
            "should never ship"
        );
    }

    /// An export whose destination already holds a symlink at an owned
    /// record-file name must error rather than write through it.
    #[cfg(unix)]
    #[test]
    fn a_planted_symlink_in_the_export_directory_errors() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("export");
        std::fs::create_dir_all(&out).unwrap();
        let victim = dir.path().join("victim.txt");
        std::fs::write(&victim, "precious").unwrap();
        std::os::unix::fs::symlink(&victim, out.join("nodes.jsonl")).unwrap();

        let mut writer = crate::cogx::CogxArchiveWriter::new(&out).unwrap();
        let result = writer.write_raw_node(&serde_json::json!({"id": 1}));
        assert!(result.is_err(), "export must refuse the planted symlink");
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "precious");
    }
}
