//! Rank-local control of the existing CNOT Stream 5 record protocol.
//!
//! CUDA buffers and the communicator remain with the backend. The controller
//! keeps both overflow votes in the same order on every rank and never sees a
//! GPU pointer.

pub struct Counts {
    pub send: Vec<u32>,
    pub receive: Vec<u32>,
    /// Includes negative/out-of-range values in the original host counts.
    pub invalid_send: bool,
}

pub trait Backend {
    type Error;
    type Output;

    fn send_count(&mut self) -> Result<u64, Self::Error>;
    fn any_overflow(&mut self, local: bool) -> Result<bool, Self::Error>;
    fn exchange_counts(&mut self) -> Result<Counts, Self::Error>;
    fn exchange_rows(
        &mut self,
        send: &[u32],
        receive: &[u32],
        size: u32,
    ) -> Result<Self::Output, Self::Error>;
    fn send_overflow(&mut self) -> Self::Error;
    fn receive_overflow(&mut self) -> Self::Error;
    fn invalid_counts(&mut self) -> Self::Error;
}

pub fn exchange<B: Backend>(backend: &mut B) -> Result<B::Output, B::Error> {
    let send_size = backend.send_count()?;
    if backend.any_overflow(send_size > u64::from(u32::MAX))? {
        return Err(backend.send_overflow());
    }
    let counts = backend.exchange_counts()?;
    // A world has at most 65536 ranks; sums therefore fit in u64.
    let receive_size: u64 = counts.receive.iter().map(|&n| u64::from(n)).sum();
    let send_total: u64 = counts.send.iter().map(|&n| u64::from(n)).sum();
    let invalid = counts.invalid_send || send_total != send_size;
    if backend.any_overflow(invalid || receive_size > u64::from(u32::MAX))? {
        return Err(if invalid {
            backend.invalid_counts()
        } else {
            backend.receive_overflow()
        });
    }
    backend.exchange_rows(&counts.send, &counts.receive, receive_size as u32)
}
