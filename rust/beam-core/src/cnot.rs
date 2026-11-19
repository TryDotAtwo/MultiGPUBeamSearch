//! Packed CNOT input contracts and independent construction-path replay.
//!
//! A matrix contains `wires` rows of little-endian u64 words. A physical
//! CNOT XORs the complete control row into the target row. Hashes are never
//! used to decide whether two matrices are equal here.
use crate::ContractError;

type Result<T> = std::result::Result<T, ContractError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SearchControlsError {
    #[error("invalid CNOT wires")]
    Wires,
    #[error("invalid global width, shortlist, or depth")]
    WidthOrShortlist,
    #[error("mode must be qhat or proposal")]
    Mode,
}

fn words_per_row(wires: u32) -> Result<usize> {
    if !(2..=256).contains(&wires) {
        return Err(ContractError::Invalid("CNOT wires"));
    }
    Ok((wires as usize).div_ceil(64))
}

/// Validate controls shared by the Python and native CNOT search paths.
///
/// `width * alphabet` is stored in signed 64-bit lexical keys by the current
/// public adapter, so preserve that bound before any rank starts work.
pub fn validate_search_controls(
    wires: u32,
    width: u64,
    shortlist: u64,
    mode: &str,
) -> std::result::Result<(), SearchControlsError> {
    if !(2..=256).contains(&wires) {
        return Err(SearchControlsError::Wires);
    }
    let alphabet = u64::from(wires) * u64::from(wires - 1);
    if width == 0 || width > (i64::MAX as u64) / alphabet || shortlist == 0 {
        return Err(SearchControlsError::WidthOrShortlist);
    }
    if mode != "qhat" && mode != "proposal" {
        return Err(SearchControlsError::Mode);
    }
    Ok(())
}

/// The physical alphabet is control-major with the diagonal omitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CnotGate {
    pub control: u32,
    pub target: u32,
}

impl CnotGate {
    pub fn decode(wires: u32, id: u32) -> Result<Self> {
        words_per_row(wires)?;
        if id >= wires * (wires - 1) {
            return Err(ContractError::Invalid("physical CNOT ID outside alphabet"));
        }
        let control = id / (wires - 1);
        let offset = id % (wires - 1);
        Ok(Self {
            control,
            target: offset + u32::from(offset >= control),
        })
    }

    pub fn encode(self, wires: u32) -> Result<u32> {
        words_per_row(wires)?;
        if self.control >= wires || self.target >= wires || self.control == self.target {
            return Err(ContractError::Invalid("physical CNOT gate"));
        }
        Ok(self.control * (wires - 1) + self.target - u32::from(self.target > self.control))
    }
}

/// An owned, validated input. Its words are never borrowed across the Python
/// or native boundary; every logical bit participates in equality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CnotMatrix {
    wires: u32,
    words: Vec<u64>,
}

impl CnotMatrix {
    pub fn from_le_bytes(wires: u32, bytes: &[u8]) -> Result<Self> {
        let stride = words_per_row(wires)?;
        let n = wires as usize;
        if bytes.len() != n * stride * 8 {
            return Err(ContractError::Invalid("CNOT packed matrix length"));
        }
        let words: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("exactly eight bytes")))
            .collect();
        let tail_bits = wires % 64;
        if tail_bits != 0 {
            let tail_mask = (1u64 << tail_bits) - 1;
            if (0..n).any(|row| words[row * stride + stride - 1] & !tail_mask != 0) {
                return Err(ContractError::Invalid("CNOT row padding"));
            }
        }

        // Elimination uses a bounded temporary copy so the retained input stays
        // immutable for later replay. At 256 wires this is at most 8192 bytes.
        let mut work = words.clone();
        for col in 0..n {
            let word = col / 64;
            let bit = 1u64 << (col % 64);
            let pivot = (col..n)
                .find(|&row| work[row * stride + word] & bit != 0)
                .ok_or(ContractError::Invalid("CNOT matrix is singular"))?;
            if pivot != col {
                for part in 0..stride {
                    work.swap(pivot * stride + part, col * stride + part);
                }
            }
            for row in col + 1..n {
                if work[row * stride + word] & bit != 0 {
                    for part in word..stride {
                        work[row * stride + part] ^= work[col * stride + part];
                    }
                }
            }
        }
        Ok(Self { wires, words })
    }

    /// Whether every packed row equals the corresponding identity row.
    pub fn is_identity(&self) -> bool {
        let stride = self.wires.div_ceil(64) as usize;
        self.words.iter().enumerate().all(|(index, &word)| {
            let row = index / stride;
            let part = index % stride;
            let expected = if part == row / 64 {
                1u64 << (row % 64)
            } else {
                0
            };
            word == expected
        })
    }

    /// Replay construction from identity, in the returned gate order.
    pub fn verifies_construction(&self, moves: &[u32]) -> Result<bool> {
        let n = self.wires as usize;
        let stride = words_per_row(self.wires)?;
        let mut state = vec![0u64; n * stride];
        for row in 0..n {
            state[row * stride + row / 64] |= 1u64 << (row % 64);
        }
        for &id in moves {
            let gate = CnotGate::decode(self.wires, id)?;
            let control = gate.control as usize * stride;
            let target = gate.target as usize * stride;
            for part in 0..stride {
                state[target + part] ^= state[control + part];
            }
        }
        Ok(state == self.words)
    }
}
