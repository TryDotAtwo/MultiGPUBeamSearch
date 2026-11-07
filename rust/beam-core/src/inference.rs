//! Bounded inference-microbatch policy. Numerical scoring remains native.
use crate::{
    calibration::{estimate, select, Estimate, Measurement},
    ContractError,
};
use std::collections::{BTreeMap, BTreeSet};
type Result<T> = std::result::Result<T, ContractError>;

fn shuffle(values: &mut [u64], mut seed: u64) {
    for i in (1..values.len()).rev() {
        seed = seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        let j = ((z as u128 * (i + 1) as u128) >> 64) as usize;
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
