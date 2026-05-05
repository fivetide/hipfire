use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Spills quantized tensor data to temporary files during quantization,
/// keeping peak RAM usage bounded for large models (27B+).
///
/// Each tensor's quantized bytes are written to a numbered file in a
/// per-process temp directory. Data is read back one tensor at a time
/// during the final `.hfq` write pass.
///
/// Implements `Drop` to auto-clean temp files on panic or early return,
/// preventing 10-90 GB NVMe temp file leaks.
pub struct TensorSpill {
    dir: PathBuf,
    entries: Vec<SpillEntry>,
}

struct SpillEntry {
    path: PathBuf,
    len: usize,
}

impl TensorSpill {
    pub fn new() -> io::Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir()
            .join(format!("hipfire-spill-{}-{}", std::process::id(), id));
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            entries: Vec::new(),
        })
    }

    /// Spill tensor data to a temp file. Returns an index for later retrieval.
    pub fn spill(&mut self, data: Vec<u8>) -> io::Result<usize> {
        let idx = self.entries.len();
        let path = self.dir.join(format!("{idx:06}.bin"));
        let mut f = BufWriter::new(fs::File::create(&path)?);
        f.write_all(&data)?;
        f.flush()?;
        self.entries.push(SpillEntry {
            path,
            len: data.len(),
        });
        Ok(idx)
    }

    /// Read back spilled tensor data by index.
    pub fn read_back(&self, idx: usize) -> io::Result<Vec<u8>> {
        let entry = &self.entries[idx];
        fs::read(&entry.path)
    }

    /// Size in bytes of a spilled entry.
    pub fn entry_len(&self, idx: usize) -> usize {
        self.entries[idx].len
    }

    /// Total bytes spilled to disk.
    pub fn total_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.len).sum()
    }

    /// Explicitly clean up all temp files and the temp directory.
    pub fn cleanup(&mut self) {
        for entry in self.entries.drain(..) {
            let _ = fs::remove_file(&entry.path);
        }
        let _ = fs::remove_dir(&self.dir);
    }
}

impl Drop for TensorSpill {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spill_and_read_back() {
        let mut spill = TensorSpill::new().unwrap();
        let data = vec![1u8, 2, 3, 4, 5];
        let idx = spill.spill(data.clone()).unwrap();
        assert_eq!(spill.entry_len(idx), 5);
        assert_eq!(spill.read_back(idx).unwrap(), data);
    }

    #[test]
    fn cleanup_removes_files() {
        let mut spill = TensorSpill::new().unwrap();
        let idx = spill.spill(vec![0u8; 100]).unwrap();
        let path = spill.entries[idx].path.clone();
        let dir = spill.dir.clone();
        assert!(path.exists());
        spill.cleanup();
        assert!(!path.exists());
        assert!(!dir.exists());
    }

    #[test]
    fn drop_cleans_up() {
        let dir;
        let path;
        {
            let mut spill = TensorSpill::new().unwrap();
            let idx = spill.spill(vec![0u8; 100]).unwrap();
            path = spill.entries[idx].path.clone();
            dir = spill.dir.clone();
            assert!(path.exists());
        } // Drop fires here
        assert!(!path.exists());
        assert!(!dir.exists());
    }

    #[test]
    fn multiple_entries() {
        let mut spill = TensorSpill::new().unwrap();
        let d1 = vec![1u8; 1000];
        let d2 = vec![2u8; 2000];
        let i1 = spill.spill(d1.clone()).unwrap();
        let i2 = spill.spill(d2.clone()).unwrap();
        assert_eq!(spill.total_bytes(), 3000);
        assert_eq!(spill.read_back(i1).unwrap(), d1);
        assert_eq!(spill.read_back(i2).unwrap(), d2);
    }
}
