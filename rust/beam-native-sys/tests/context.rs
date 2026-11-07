use beam_native_sys::NativeLibrary;
use std::{path::PathBuf, time::Duration};

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
}
