use super::storage_trait::{StorageError, StorageTrait, StorageWriter};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::{AsyncRead, AsyncWriteExt};
use tracing::{debug, instrument};
use uuid::Uuid;

pub struct LocalStorage {
    base_path: PathBuf,
}

impl LocalStorage {
    pub fn new(base_path: PathBuf) -> Self {
        Self { base_path }
    }

    /// `file_name` must be exactly one ordinary path component — anything
    /// else (`../x`, `a/b`, an absolute path, `.`, `..`, a backslash or NUL)
    /// would escape `base_path` through the unchecked `join`.
    fn validate_file_name(file_name: &str) -> Result<(), StorageError> {
        let mut components = Path::new(file_name).components();
        let single_normal = matches!(
            (components.next(), components.next()),
            (Some(std::path::Component::Normal(c)), None) if c == file_name
        ) && !file_name.contains(['/', '\\', '\0']);
        if !single_normal {
            return Err(StorageError::InvalidPath(format!(
                "file_name must be a single path component: {file_name:?}"
            )));
        }
        Ok(())
    }

    /// Generate a UUID-based subdirectory structure for organizing files.
    /// Returns a relative path like "ab/cd/<32-hex-uuid>/filename.txt" — the
    /// full UUID becomes the file's own directory, so two stores of the same
    /// `file_name` land in different places instead of colliding under the
    /// 65,536 destinations the old 2+2-hex fan-out allowed.
    fn generate_storage_path(&self, file_name: &str) -> String {
        let id = Uuid::new_v4().simple().to_string();
        format!("{}/{}/{}/{file_name}", &id[..2], &id[2..4], id)
    }

    /// Pick a fresh storage path and create it exclusively — a store never
    /// overwrites an existing file (Python's LocalFileStorage.store does not
    /// overwrite unless asked). On an `AlreadyExists` collision the attempt
    /// retries with a new path, up to 3 attempts.
    async fn create_destination(
        &self,
        file_name: &str,
    ) -> Result<(String, fs::File), StorageError> {
        for attempt in 0..3 {
            let relative_path = self.generate_storage_path(file_name);
            let full_path = self.base_path.join(&relative_path);

            if let Some(parent) = full_path.parent() {
                fs::create_dir_all(parent).await.map_err(|e| {
                    StorageError::IoError(format!("Failed to create directory: {e}"))
                })?;
            }

            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&full_path)
                .await
            {
                Ok(file) => return Ok((relative_path, file)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && attempt + 1 < 3 => {
                    continue;
                }
                Err(e) => {
                    return Err(StorageError::IoError(format!("Failed to create file: {e}")));
                }
            }
        }
        unreachable!("the retry loop returns on success or the last error")
    }

    /// Resolve a location string into a filesystem path.
    ///
    /// Mirrors Python's `get_data_file_path()` + `open_data_file()` which
    /// strips the `file://` scheme and uses the resulting absolute path
    /// directly.
    ///
    /// Accepted inputs:
    /// - plain relative path: `ab/cd/file.txt`  → `base_path/ab/cd/file.txt`
    /// - absolute `file://` URI: `file:///data/ab/cd/file.txt` → `/data/ab/cd/file.txt`
    fn resolve_location(&self, location: &str) -> PathBuf {
        let path_str = location.strip_prefix("file://").unwrap_or(location);
        let path = Path::new(path_str);

        if path.is_absolute() {
            // Absolute path (from a file:// URI) — use directly, just like
            // Python's `open_data_file` does after `get_data_file_path()`.
            path.to_path_buf()
        } else {
            // Relative path (plain storage location) — join with base.
            self.base_path.join(path)
        }
    }
}

#[async_trait]
impl StorageTrait for LocalStorage {
    async fn initialize(&self) -> Result<(), StorageError> {
        fs::create_dir_all(&self.base_path)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to create base directory: {e}")))
    }

    #[instrument(name = "storage.store", skip(self, data), fields(file_name, bytes = data.len()))]
    async fn store(&self, data: &[u8], file_name: &str) -> Result<String, StorageError> {
        Self::validate_file_name(file_name)?;
        let (relative_path, mut file) = self.create_destination(file_name).await?;

        file.write_all(data)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to write file: {e}")))?;

        file.flush()
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to flush file: {e}")))?;

        Ok(relative_path)
    }

    #[instrument(name = "storage.store_stream", skip(self, reader), fields(file_name))]
    async fn store_stream_dyn(
        &self,
        reader: &mut (dyn AsyncRead + Unpin + Send),
        file_name: &str,
    ) -> Result<String, StorageError> {
        Self::validate_file_name(file_name)?;
        let (relative_path, mut file) = self.create_destination(file_name).await?;

        // Stream copy from reader to file
        tokio::io::copy(reader, &mut file)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to write file: {e}")))?;

        file.flush()
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to flush file: {e}")))?;

        Ok(relative_path)
    }

    #[instrument(name = "storage.create_writer", skip(self), fields(file_name))]
    async fn create_writer(&self, file_name: &str) -> Result<StorageWriter, StorageError> {
        Self::validate_file_name(file_name)?;
        let (relative_path, file) = self.create_destination(file_name).await?;
        Ok(StorageWriter::new(file, relative_path))
    }

    #[instrument(name = "storage.retrieve", skip(self), fields(location))]
    async fn retrieve(&self, location: &str) -> Result<Vec<u8>, StorageError> {
        let full_path = self.resolve_location(location);

        let bytes = fs::read(&full_path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                StorageError::NotFound(format!("File not found: {location}"))
            } else {
                StorageError::IoError(format!("Failed to read file: {e}"))
            }
        })?;
        debug!(bytes = bytes.len(), "file retrieved");
        Ok(bytes)
    }

    async fn exists(&self, location: &str) -> Result<bool, StorageError> {
        let full_path = self.resolve_location(location);

        fs::try_exists(&full_path)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to check file existence: {e}")))
    }

    #[instrument(name = "storage.delete", skip(self), fields(location))]
    async fn delete(&self, location: &str) -> Result<(), StorageError> {
        let full_path = self.resolve_location(location);

        fs::remove_file(&full_path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                StorageError::NotFound(format!("File not found: {location}"))
            } else {
                StorageError::IoError(format!("Failed to delete file: {e}"))
            }
        })
    }

    fn get_full_path(&self, location: &str) -> PathBuf {
        self.resolve_location(location)
    }

    fn base_path(&self) -> &str {
        self.base_path.to_str().unwrap_or("")
    }

    async fn remove_all(&self) -> Result<(), StorageError> {
        let mut entries = fs::read_dir(&self.base_path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                // Directory doesn't exist — nothing to remove.
                return StorageError::NotFound(format!(
                    "Base directory not found: {}",
                    self.base_path.display()
                ));
            }
            StorageError::IoError(format!("Failed to read directory: {e}"))
        })?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to iterate directory entry: {e}")))?
        {
            let path = entry.path();
            let file_type = entry
                .file_type()
                .await
                .map_err(|e| StorageError::IoError(format!("Failed to get file type: {e}")))?;
            if file_type.is_dir() {
                fs::remove_dir_all(&path).await.map_err(|e| {
                    StorageError::IoError(format!(
                        "Failed to remove directory {}: {}",
                        path.display(),
                        e
                    ))
                })?;
            } else {
                fs::remove_file(&path).await.map_err(|e| {
                    StorageError::IoError(format!(
                        "Failed to remove file {}: {}",
                        path.display(),
                        e
                    ))
                })?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    reason = "test code — panics are acceptable failures"
)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(temp_dir.path().to_path_buf());

        storage.initialize().await.unwrap();

        let data = b"Hello, World!";
        let location = storage.store(data, "test.txt").await.unwrap();

        let retrieved = storage.retrieve(&location).await.unwrap();
        assert_eq!(data.to_vec(), retrieved);
    }

    #[tokio::test]
    async fn test_exists() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(temp_dir.path().to_path_buf());

        storage.initialize().await.unwrap();

        let data = b"Test data";
        let location = storage.store(data, "exists.txt").await.unwrap();

        assert!(storage.exists(&location).await.unwrap());
        assert!(!storage.exists("nonexistent.txt").await.unwrap());
    }

    #[test]
    fn resolve_plain_relative_path() {
        let storage = LocalStorage::new(PathBuf::from("/data"));
        assert_eq!(
            storage.resolve_location("ab/cd/file.txt"),
            PathBuf::from("/data/ab/cd/file.txt")
        );
    }

    #[test]
    fn resolve_absolute_file_uri() {
        // file:// URI with an absolute path — strip scheme, use path as-is
        // (mirrors Python's get_data_file_path for file:///abs/path)
        let storage = LocalStorage::new(PathBuf::from("/data"));
        assert_eq!(
            storage.resolve_location("file:///data/ab/cd/file.txt"),
            PathBuf::from("/data/ab/cd/file.txt")
        );
    }

    #[test]
    fn resolve_absolute_file_uri_different_base() {
        // URI points to a different directory than base_path — still works
        let storage = LocalStorage::new(PathBuf::from("/data"));
        assert_eq!(
            storage.resolve_location("file:///other/ab/cd/file.txt"),
            PathBuf::from("/other/ab/cd/file.txt")
        );
    }

    #[tokio::test]
    async fn test_retrieve_with_file_uri() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(temp_dir.path().to_path_buf());
        storage.initialize().await.unwrap();

        let data = b"URI test data";
        let relative = storage.store(data, "uri_test.txt").await.unwrap();

        // Build a file:// URI the same way the ingestion pipeline does
        let uri = format!("file://{}", temp_dir.path().join(&relative).display());

        let retrieved = storage.retrieve(&uri).await.unwrap();
        assert_eq!(data.to_vec(), retrieved);
    }

    #[tokio::test]
    async fn test_delete() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(temp_dir.path().to_path_buf());

        storage.initialize().await.unwrap();

        let data = b"To be deleted";
        let location = storage.store(data, "delete.txt").await.unwrap();

        assert!(storage.exists(&location).await.unwrap());

        storage.delete(&location).await.unwrap();

        assert!(!storage.exists(&location).await.unwrap());
    }

    #[tokio::test]
    async fn rejects_file_names_that_are_not_a_single_component() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(temp_dir.path().to_path_buf());
        storage.initialize().await.unwrap();

        for name in [
            "../x",
            "a/b",
            "/abs",
            "",
            ".",
            "..",
            "a\\b",
            "x\0y",
            "../../etc/passwd",
        ] {
            assert!(
                matches!(
                    storage.store(b"d", name).await,
                    Err(StorageError::InvalidPath(_))
                ),
                "{name:?} must be rejected"
            );
        }
    }

    #[tokio::test]
    async fn two_stores_of_the_same_name_get_distinct_locations_and_bytes() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(temp_dir.path().to_path_buf());
        storage.initialize().await.unwrap();

        let loc1 = storage.store(b"first", "same.txt").await.unwrap();
        let loc2 = storage.store(b"second", "same.txt").await.unwrap();

        assert_ne!(loc1, loc2, "same-name stores must not collide");
        assert_eq!(storage.retrieve(&loc1).await.unwrap(), b"first");
        assert_eq!(storage.retrieve(&loc2).await.unwrap(), b"second");
    }

    #[tokio::test]
    async fn storage_locations_carry_the_full_uuid_directory() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(temp_dir.path().to_path_buf());
        storage.initialize().await.unwrap();

        let loc = storage.store(b"x", "name.txt").await.unwrap();
        let parts: Vec<&str> = loc.split('/').collect();
        assert_eq!(
            (parts[0].len(), parts[1].len(), parts[2].len(), parts[3]),
            (2, 2, 32, "name.txt"),
            "location {loc:?} does not match the fan-out shape"
        );
        assert!(
            parts[..3]
                .iter()
                .all(|p| p.chars().all(|c| c.is_ascii_hexdigit())),
            "fan-out dirs must be hex: {loc:?}"
        );
    }

    #[tokio::test]
    async fn store_stream_dyn_and_create_writer_also_validate_the_name() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(temp_dir.path().to_path_buf());
        storage.initialize().await.unwrap();

        let mut reader: &[u8] = b"x";
        assert!(storage.store_stream_dyn(&mut reader, "../x").await.is_err());
        assert!(storage.create_writer("../x").await.is_err());
    }
}
