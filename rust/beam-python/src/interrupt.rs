//! Keep Python's main-thread signal checks alive during a coarse Rust request.
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    time::Duration,
};

pub fn run<T: Send, F: FnOnce() -> PyResult<T> + Send>(
    py: Python<'_>,
    cancel: &Arc<AtomicBool>,
    work: F,
) -> PyResult<T> {
    std::thread::scope(|scope| {
        let (send, receive) = mpsc::sync_channel(1);
        let receive = Mutex::new(receive);
        let worker = scope.spawn(move || {
            let _ = send.send(work());
        });
        let mut signal = None;
        let result = loop {
            let next = py.allow_threads(|| {
                receive
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_millis(20))
            });
            if signal.is_none() {
                if let Err(error) = py.check_signals() {
                    cancel.store(true, Ordering::Release);
                    signal = Some(error);
                }
            }
            match next {
                Ok(result) => break result,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break Err(PyRuntimeError::new_err("Rust request worker terminated"))
                }
            }
        };
        let joined = py.allow_threads(|| worker.join());
        if let Some(error) = signal {
            return Err(error);
        }
        if joined.is_err() {
            return Err(PyRuntimeError::new_err("Rust request worker panicked"));
        }
        result
    })
}
