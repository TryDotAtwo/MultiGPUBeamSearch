use beam_core::CandidateMeta;
use beam_runtime::history::{HistoryConfig, HistoryMode, HistoryStore};
use std::{fs, path::PathBuf};

fn config(mode: HistoryMode, name: &str) -> HistoryConfig {
    let directory = PathBuf::from(std::env::temp_dir())
        .join(format!("beam-history-{}-{name}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    HistoryConfig {
        mode,
        capacity: 8,
        slots: 2,
        workers: 2,
        max_depth: 4,
        moves: 2,
        ram_budget_bytes: 4096,
        disk_budget_bytes: 1024,
        directory,
        disk_path: None,
        prune: false,
    }
}
fn batch(parents: &[u64], route: u32) -> Vec<CandidateMeta> {
    parents
        .iter()
        .map(|&parent_idx| CandidateMeta {
            parent_idx,
            route,
            ..Default::default()
        })
        .collect()
}

#[test]
fn formats_bounds_and_out_of_order_batches() {
    for mode in [
        HistoryMode::Ram,
        HistoryMode::Disk,
        HistoryMode::StaticHybrid,
    ] {
        let cfg = config(mode, &format!("formats-{mode:?}"));
        let directory = cfg.directory.clone();
        let mut history = HistoryStore::new(cfg).unwrap();
        history.write(1, &batch(&[1, 0, 1], 7)).unwrap();
        history.write(0, &batch(&[0, 0], 5)).unwrap();
        history.write(2, &[]).unwrap();
        assert_eq!(history.read(1, 1).unwrap().parent_idx, 0);
        assert_eq!(history.read(0, 0).unwrap().route, 5);
        assert_eq!(history.stats().bytes_received, 5 * 32);
        assert_eq!(history.stats().bytes_stored, 5 * 16);
        assert!(history.write(0, &[]).is_err());
        assert!(history.read(2, 0).is_err());
        assert!(history.read(3, 0).is_err());
        assert!(history
            .write(3, &batch(&[0; 9], 1))
            .unwrap_err()
            .to_string()
            .contains("required=9"));
        if mode == HistoryMode::Disk {
            let bytes = fs::read(directory.join("depth_0.candidate_meta.bin")).unwrap();
            assert_eq!(
                &bytes[..16],
                &[0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0]
            );
        }
        drop(history);
        fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn ram_pruning_preserves_reconstructed_routes() {
    let mut cfg = config(HistoryMode::Ram, "prune");
    cfg.prune = true;
    let directory = cfg.directory.clone();
    let mut h = HistoryStore::new(cfg).unwrap();
    h.write(0, &batch(&[0, 0], 11)).unwrap();
    h.write(1, &batch(&[1, 1, 0, 1], 12)).unwrap();
    h.write(2, &batch(&[0, 3], 13)).unwrap();
    assert_eq!(h.count(0).unwrap(), 1);
    assert_eq!(h.count(1).unwrap(), 2);
    let second = h.read(2, 1).unwrap();
    let first = h.read(1, second.parent_idx).unwrap();
    let root = h.read(0, first.parent_idx).unwrap();
    assert_eq!((root.route, first.route, second.route), (11, 12, 13));
    assert_eq!(h.stats().bytes_pruned, 3 * 16);
    assert_eq!(h.stats().bytes_stored_ram, 4 * 16 + 16);
    drop(h);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn invalid_parent_and_memory_budget_fail_explicitly() {
    let mut cfg = config(HistoryMode::StaticHybrid, "budget");
    cfg.ram_budget_bytes = 1;
    assert!(HistoryStore::new(cfg)
        .err()
        .unwrap()
        .to_string()
        .contains("required="));
    let mut cfg = config(HistoryMode::Ram, "bad-parent");
    cfg.prune = true;
    let directory = cfg.directory.clone();
    let mut h = HistoryStore::new(cfg).unwrap();
    h.write(0, &batch(&[0], 1)).unwrap();
    assert!(h.write(1, &batch(&[2], 2)).is_err());
    drop(h);
    fs::remove_dir_all(directory).unwrap();
}
