//! Exact global CNOT cutoff, preserving the reference radix-pass schedule.
//!
//! Each backend pass handles a complete distributed histogram and its batch
//! of local candidates. No device address or candidate crosses this layer.

pub trait Backend {
    type Error;
    type Output;
    fn begin(&mut self, width: u64) -> Result<u64, Self::Error>;
    fn radix_pass(
        &mut self,
        key: u32,
        shift: u32,
        remaining: u64,
    ) -> Result<(u64, u64), Self::Error>;
    fn accept_boundary(&mut self) -> Result<(), Self::Error>;
    fn finish(&mut self, count: u64, width: u64) -> Result<Self::Output, Self::Error>;
    fn invalid(&self, message: &'static str) -> Self::Error;
}

pub fn select<B: Backend>(backend: &mut B, width: u64) -> Result<B::Output, B::Error> {
    if width == 0 {
        return Err(backend.invalid("CNOT global width must be positive"));
    }
    let count = backend.begin(width)?;
    if count > width {
        let mut remaining = width;
        'keys: for key in 0..2 {
            for digit in (0..8).rev() {
                let shift = digit * 8;
                let (below, boundary) = backend.radix_pass(key, shift, remaining)?;
                remaining = remaining.checked_sub(below).ok_or_else(|| {
                    backend.invalid("CNOT radix histogram exceeds remaining width")
                })?;
                if remaining == boundary {
                    backend.accept_boundary()?;
                    break 'keys;
                }
            }
        }
    }
    backend.finish(count, width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct Mock {
        count: u64,
        passes: VecDeque<(u64, u64)>,
        calls: Vec<String>,
    }
    impl Backend for Mock {
        type Error = String;
        type Output = (u64, u64);
        fn begin(&mut self, width: u64) -> Result<u64, String> {
            self.calls.push(format!("begin:{width}"));
            Ok(self.count)
        }
        fn radix_pass(
            &mut self,
            key: u32,
            shift: u32,
            remaining: u64,
        ) -> Result<(u64, u64), String> {
            self.calls.push(format!("pass:{key}:{shift}:{remaining}"));
            Ok(self.passes.pop_front().unwrap())
        }
        fn accept_boundary(&mut self) -> Result<(), String> {
            self.calls.push("accept".into());
            Ok(())
        }
        fn finish(&mut self, count: u64, width: u64) -> Result<Self::Output, String> {
            self.calls.push("finish".into());
            Ok((count, width))
        }
        fn invalid(&self, message: &'static str) -> String {
            message.into()
        }
    }
    #[test]
    fn no_cutoff_for_small_or_empty_frontier() {
        for count in [0, 3] {
            let mut b = Mock {
                count,
                passes: [].into(),
                calls: vec![],
            };
            assert_eq!(select(&mut b, 3).unwrap(), (count, 3));
            assert_eq!(b.calls, ["begin:3", "finish"]);
        }
    }
    #[test]
    fn score_passes_then_word_tie_break_and_exact_boundary() {
        let mut passes: VecDeque<_> = [(0, 10); 8].into();
        passes.push_back((1, 4));
        passes.push_back((0, 2));
        let mut b = Mock {
            count: 10,
            passes,
            calls: vec![],
        };
        assert_eq!(select(&mut b, 3).unwrap(), (10, 3));
        assert_eq!(b.calls[1], "pass:0:56:3");
        assert_eq!(b.calls[8], "pass:0:0:3");
        assert_eq!(b.calls[9], "pass:1:56:3");
        assert_eq!(b.calls[10], "pass:1:48:2");
        assert_eq!(&b.calls[11..], ["accept", "finish"]);
    }
    #[test]
    fn invalid_count_stops_before_later_collectives() {
        let mut b = Mock {
            count: 5,
            passes: [(4, 1)].into(),
            calls: vec![],
        };
        assert!(select(&mut b, 3).err().unwrap().contains("histogram"));
        assert_eq!(b.calls, ["begin:3", "pass:0:56:3"]);
    }
}
