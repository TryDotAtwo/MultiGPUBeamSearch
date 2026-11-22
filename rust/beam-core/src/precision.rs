//! Validation of immutable host-side model bytes without converting the weights.
use crate::ContractError;

#[derive(Clone, Copy, Debug)]
pub enum Float16Format {
    Fp16,
    Bf16,
}

/// The input is borrowed for the duration of the call; no tensor is created.
/// Both encodings use an all-ones exponent for infinity and NaN.
pub fn finite_16le(bytes: &[u8], format: Float16Format) -> Result<bool, ContractError> {
    let mut words = bytes.chunks_exact(2);
    if !words.remainder().is_empty() {
        return Err(ContractError::Invalid("16-bit model byte length"));
    }
    let exponent_mask = match format {
        Float16Format::Fp16 => 0x7c00,
        Float16Format::Bf16 => 0x7f80,
    };
    Ok(words.all(|word| u16::from_le_bytes([word[0], word[1]]) & exponent_mask != exponent_mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_encodings_and_truncation() {
        for format in [Float16Format::Fp16, Float16Format::Bf16] {
            assert!(finite_16le(&[], format).unwrap());
            assert!(finite_16le(&[0, 0, 1, 0, 0, 0x80], format).unwrap());
            assert!(finite_16le(&[0], format).is_err());
        }
        assert!(finite_16le(&0x7bff_u16.to_le_bytes(), Float16Format::Fp16).unwrap());
        assert!(!finite_16le(&0x7c00_u16.to_le_bytes(), Float16Format::Fp16).unwrap());
        assert!(!finite_16le(&0xfe01_u16.to_le_bytes(), Float16Format::Fp16).unwrap());
        assert!(finite_16le(&0x7f7f_u16.to_le_bytes(), Float16Format::Bf16).unwrap());
        assert!(!finite_16le(&0xff80_u16.to_le_bytes(), Float16Format::Bf16).unwrap());
        assert!(!finite_16le(&0x7fc1_u16.to_le_bytes(), Float16Format::Bf16).unwrap());
    }
}
