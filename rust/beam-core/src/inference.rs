//! Bounded inference-microbatch policy. Numerical scoring remains native.
use crate::{
    calibration::{estimate, select, Estimate, Measurement},
    ContractError,
};
use std::collections::{BTreeMap, BTreeSet};
type Result<T> = std::result::Result<T, ContractError>;

const MT_N: usize = 624;
const MT_M: usize = 397;

/// Reproduce Python random.Random(seed).shuffle so persisted calibration policy
/// keeps the exact candidate order used by the public Python implementation.
struct PythonMt19937 {
    state: [u32; MT_N],
    index: usize,
}
impl PythonMt19937 {
    fn seeded(seed: u64) -> Self {
        let high = (seed >> 32) as u32;
        let key = if high == 0 {
            vec![seed as u32]
        } else {
            vec![seed as u32, high]
        };
        let mut state = [0u32; MT_N];
        state[0] = 19_650_218;
        for i in 1..MT_N {
            let previous = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(i as u32);
        }
        let mut i = 1;
        let mut j = 0;
        for _ in 0..MT_N.max(key.len()) {
            let previous = state[i - 1];
            let mixed = (previous ^ (previous >> 30)).wrapping_mul(1_664_525);
            state[i] = (state[i] ^ mixed)
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= MT_N {
                state[0] = state[MT_N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..MT_N - 1 {
            let previous = state[i - 1];
            let mixed = (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941);
            state[i] = (state[i] ^ mixed).wrapping_sub(i as u32);
            i += 1;
            if i >= MT_N {
                state[0] = state[MT_N - 1];
                i = 1;
            }
        }
        state[0] = 0x8000_0000;
        Self { state, index: MT_N }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= MT_N {
            for i in 0..MT_N {
                let joined =
                    (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % MT_N] & 0x7fff_ffff);
                let odd = if joined & 1 == 0 { 0 } else { 0x9908_b0df };
                self.state[i] = self.state[(i + MT_M) % MT_N] ^ (joined >> 1) ^ odd;
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    fn below(&mut self, upper: usize) -> usize {
        let bits = 32 - (upper as u32).leading_zeros();
        loop {
            let value = self.next_u32() >> (32 - bits);
            if (value as usize) < upper {
                return value as usize;
            }
        }
    }
}
fn shuffle(values: &mut [u64], seed: u64) {
    let mut rng = PythonMt19937::seeded(seed);
    for i in (1..values.len()).rev() {
        let j = rng.below(i + 1);
        values.swap(i, j);
    }
}
pub fn coarse_batches(cap: u64) -> Result<Vec<u64>> {
    if !(1..=65536).contains(&cap) {
        return Err(ContractError::Invalid("inference cap"));
    }
    let baseline = cap.min(256);
    let values = [
        32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    ];
    let mut larger: Vec<_> = values
        .into_iter()
        .filter(|v| *v > baseline && *v < cap)
        .collect();
    let mut smaller: Vec<_> = values.into_iter().filter(|v| *v < baseline).collect();
    shuffle(&mut larger, 20261009);
    shuffle(&mut smaller, 20261009);
    let mut result = vec![baseline];
    if cap != baseline {
        result.push(cap);
    }
    result.extend(larger);
    result.extend(smaller);
    Ok(result)
}
pub fn refinement_batches(winner: u64, cap: u64, visited: &BTreeSet<u64>) -> Result<Vec<u64>> {
    if winner == 0 || winner > cap || cap > 65536 {
        return Err(ContractError::Invalid("inference refinement bounds"));
    }
    let grain = (winner / 8).clamp(1, 32);
    let mut values = BTreeSet::new();
    for (numerator, denominator) in [(5, 8), (3, 4), (7, 8), (9, 8), (5, 4), (3, 2)] {
        let batch =
            ((winner * numerator + denominator * grain / 2) / (denominator * grain)) * grain;
        if batch > 0 && batch <= cap && !visited.contains(&batch) {
            values.insert(batch);
        }
    }
    let mut values: Vec<_> = values.into_iter().collect();
    shuffle(&mut values, 20261010);
    Ok(values)
}
pub fn select_inference(
    samples: &[Measurement],
    baseline: &str,
) -> Result<(Estimate, BTreeMap<String, String>)> {
    let mut rejected = BTreeMap::new();
    let baseline = match estimate(baseline, samples) {
        Ok(_) => baseline.to_string(),
        Err(error) => {
            rejected.insert(baseline.into(), error.to_string());
            let mut valid = Vec::new();
            for name in samples
                .iter()
                .map(|r| r.profile.as_str())
                .collect::<BTreeSet<_>>()
            {
                if name == baseline {
                    continue;
                }
                match estimate(name, samples) {
                    Ok(value) => valid.push(value),
                    Err(error) => {
                        rejected.insert(name.into(), error.to_string());
                    }
                }
            }
            valid.sort_by(|a, b| {
                a.median
                    .total_cmp(&b.median)
                    .then_with(|| a.profile.cmp(&b.profile))
            });
            valid
                .first()
                .ok_or(ContractError::Invalid(
                    "no stable complete inference candidate",
                ))?
                .profile
                .clone()
        }
    };
    // Preserve the existing microbatch-selection threshold. Kernel promotion is
    // the separate paired full-step 5% / memory gate, never this component test.
    let (winner, other) = select(samples, &baseline, 0.02)?;
    rejected.extend(other);
    Ok((winner, rejected))
}
