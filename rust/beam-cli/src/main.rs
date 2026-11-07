#![forbid(unsafe_code)]
use std::{env, path::PathBuf};
fn run() -> Result<i32, Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(String::as_str) == Some("--resident-protocol-info") {
        println!("{{\"protocol\":\"resident-search-v1\"}}");
        return Ok(0);
    }
    let library = env::var_os("BEAM_NATIVE_LIBRARY")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            env::current_exe()
                .unwrap_or_else(|_| PathBuf::from(&args[0]))
                .with_file_name("libbeam_native.so")
        });
    let native = beam_native_sys::NativeLibrary::load(&library)?;
    if args.get(1).map(String::as_str) == Some("--native-abi-info") {
        if args.len() != 2 {
            return Err("--native-abi-info takes no arguments".into());
        }
        let layout = native.layout()?;
        println!(
            "{{\"abi_version\":{},\"state_len\":{},\"storage_len\":{},\"state_alignment\":{},\"move_count\":{},\"candidate_bytes\":{},\"candidate_alignment\":{},\"hash_bytes\":{},\"hash_alignment\":{}}}",
            beam_native_sys::ABI_VERSION,
            layout.state_len,
            layout.storage_len,
            layout.state_alignment,
            layout.move_count,
            layout.candidate_bytes,
            layout.candidate_alignment,
            layout.hash_bytes,
            layout.hash_alignment
        );
        return Ok(0);
    }
    if let Some(root) = env::var_os("BEAM_SESSION_DIR") {
        beam_runtime::run_native_worker(&native, &args, &PathBuf::from(root))?;
        Ok(0)
    } else {
        Ok(native.run_with_history(
            &args,
            Box::<beam_runtime::history::HistoryService>::default(),
        )?)
    }
}
fn main() {
    let result = std::panic::catch_unwind(run);
    let code = match result {
        Ok(Ok(code)) => code,
        Ok(Err(error)) => {
            eprintln!("beam_runtime_error={error}");
            1
        }
        Err(_) => {
            eprintln!("beam_runtime_error=Rust worker panic");
            1
        }
    };
    std::process::exit(code);
}
