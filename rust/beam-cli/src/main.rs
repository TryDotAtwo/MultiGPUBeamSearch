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
    if let Some(root) = env::var_os("BEAM_SESSION_DIR") {
        beam_runtime::run_native_worker(&native, &args, &PathBuf::from(root))?;
        Ok(0)
    } else {
        Ok(native.run(&args)?)
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
