use beam_core::{
    history::{Entry, Stats},
    CandidateMeta,
};
use beam_native_sys::history::HistorySink;
use beam_native_sys::NativeLibrary;
use std::sync::{Arc, Mutex};
use std::{path::PathBuf, time::Duration};

struct Sink(Arc<Mutex<Vec<u64>>>);
impl HistorySink for Sink {
    fn initialize(&self, json: &[u8]) -> Result<(), String> {
        assert_eq!(json, b"{}");
        Ok(())
    }
    fn write(&self, depth: u32, batch: &[CandidateMeta]) -> Result<(), String> {
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].route, 37);
        self.0.lock().unwrap().push(depth as u64);
        Ok(())
    }
    fn read(&self, _depth: u32, index: u64) -> Result<Entry, String> {
        Ok(Entry {
            parent_idx: index,
            route: 37,
            pad: 0,
        })
    }
    fn stats(&self) -> Result<Stats, String> {
        Ok(Stats::default())
    }
    fn close(&self) -> Result<(), String> {
        self.0.lock().unwrap().push(u64::MAX);
        Ok(())
    }
}

#[test]
#[ignore = "requires BEAM_TEST_LIBRARY built from the CPU ABI fixture"]
fn real_ffi_context_reuses_copied_jobs_and_propagates_errors() {
    let library = NativeLibrary::load(&PathBuf::from(
        std::env::var_os("BEAM_TEST_LIBRARY").unwrap(),
    ))
    .unwrap();
    let layout = library.layout().unwrap();
    assert_eq!(
        (
            layout.state_len,
            layout.storage_len,
            layout.candidate_alignment
        ),
        (96, 112, 32)
    );
    {
        let mut ctx = library.context().unwrap();
        ctx.prepare(&["test".into(), "ok".into()]).unwrap();
        assert!(ctx.wait(Duration::from_secs(1)).unwrap());
        let mut start = vec![253];
        ctx.submit(0, &start, &std::env::temp_dir()).unwrap();
        start[0] = 255;
        assert!(!ctx.wait(Duration::from_millis(1)).unwrap());
        assert!(ctx.wait(Duration::from_secs(1)).unwrap());
        assert_eq!(ctx.status().unwrap().completed_jobs, 1);
        ctx.submit(1, &[1], &std::env::temp_dir()).unwrap();
        assert!(ctx.wait(Duration::from_secs(1)).unwrap());
        assert_eq!(ctx.status().unwrap().completed_jobs, 2);
    }
    let mut ctx = library.context().unwrap();
    ctx.prepare(&["test".into(), "fail-prepare".into()])
        .unwrap();
    let error = ctx.wait(Duration::from_secs(1)).unwrap_err();
    assert!(error.to_string().contains("prepare failure preserved"));
    drop(ctx);
    let observations = Arc::new(Mutex::new(Vec::new()));
    {
        let mut ctx = library.context().unwrap();
        ctx.set_history(Box::new(Sink(observations.clone())))
            .unwrap();
        ctx.prepare(&["test".into(), "ok".into()]).unwrap();
        assert!(ctx.wait(Duration::from_secs(1)).unwrap());
        ctx.submit(0, &[1], &std::env::temp_dir()).unwrap();
        assert!(ctx.wait(Duration::from_secs(1)).unwrap());
        ctx.submit(1, &[2], &std::env::temp_dir()).unwrap();
        assert!(ctx.wait(Duration::from_secs(1)).unwrap());
    }
    assert_eq!(*observations.lock().unwrap(), vec![0, 1, u64::MAX]);
    observations.lock().unwrap().clear();
    assert_eq!(
        library
            .run_with_history(&["test".into()], Box::new(Sink(observations.clone())))
            .unwrap(),
        7
    );
    assert_eq!(*observations.lock().unwrap(), vec![0, u64::MAX]);
}
