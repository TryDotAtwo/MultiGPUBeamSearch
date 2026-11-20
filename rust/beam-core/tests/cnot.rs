use beam_core::cnot::{CnotGate, CnotMatrix};

fn identity(n: usize) -> Vec<u8> {
    let words = n.div_ceil(64);
    let mut bytes = vec![0; n * words * 8];
    for row in 0..n {
        bytes[row * words * 8 + row / 8] |= 1 << (row % 8);
    }
    bytes
}

#[test]
fn cnot_search_controls_keep_public_width_and_mode_contract() {
    let wires = 144;
    let alphabet = u64::from(wires) * u64::from(wires - 1);
    let maximum_width = (i64::MAX as u64) / alphabet;
    assert!(beam_core::cnot::validate_search_controls(wires, 1, 1, "qhat").is_ok());
    assert!(
        beam_core::cnot::validate_search_controls(wires, maximum_width, 99, "proposal").is_ok()
    );
    assert!(beam_core::cnot::validate_search_controls(wires, 0, 1, "qhat").is_err());
    assert!(
        beam_core::cnot::validate_search_controls(wires, maximum_width + 1, 1, "qhat").is_err()
    );
    assert!(beam_core::cnot::validate_search_controls(wires, 1, 0, "qhat").is_err());
    assert!(beam_core::cnot::validate_search_controls(wires, 1, 1, "greedy").is_err());
    assert!(beam_core::cnot::validate_search_controls(1, 1, 1, "qhat").is_err());
}

#[test]
fn physical_gate_alphabet_is_control_major_and_round_trips() {
    for n in [2, 3, 63, 64, 65, 127, 128, 129, 144, 255, 256] {
        let mut id = 0;
        for control in 0..n {
            for target in 0..n {
                if control == target {
                    continue;
                }
                let gate = CnotGate::decode(n, id).unwrap();
                assert_eq!((gate.control, gate.target), (control, target));
                assert_eq!(gate.encode(n).unwrap(), id);
                id += 1;
            }
        }
        assert_eq!(id, n * (n - 1));
        assert!(CnotGate::decode(n, id).is_err());
    }
    assert_eq!(
        CnotGate::decode(144, 20591).unwrap(),
        CnotGate {
            control: 143,
            target: 142
        }
    );
    assert!(CnotGate::decode(1, 0).is_err());
    assert!(CnotGate::decode(257, 0).is_err());
    assert!(CnotGate {
        control: 1,
        target: 1
    }
    .encode(2)
    .is_err());
}

#[test]
fn matrix_rejects_bad_geometry_length_padding_and_singular_rows() {
    assert!(CnotMatrix::from_le_bytes(1, &[]).is_err());
    assert!(CnotMatrix::from_le_bytes(257, &[]).is_err());
    let raw = identity(65);
    assert!(CnotMatrix::from_le_bytes(65, &raw[..raw.len() - 1]).is_err());
    let mut extra = raw.clone();
    extra.push(0);
    assert!(CnotMatrix::from_le_bytes(65, &extra).is_err());
    let mut padding = raw.clone();
    padding[8] |= 2;
    assert!(CnotMatrix::from_le_bytes(65, &padding).is_err());
    let mut dependent = vec![0; 3 * 8];
    dependent[0] = 1;
    dependent[8] = 2;
    dependent[16] = 3;
    assert!(CnotMatrix::from_le_bytes(3, &dependent).is_err());
    for n in [2, 63, 64, 65, 127, 128, 129, 144, 255, 256] {
        assert!(CnotMatrix::from_le_bytes(n as u32, &identity(n)).is_ok());
    }
}

#[test]
fn construction_replay_checks_order_all_words_and_move_bounds() {
    // At n=3: 0->1, 1->0 yields rows [2, 3, 4]. Reversing differs.
    let mut expected = vec![0; 3 * 8];
    expected[0] = 2;
    expected[8] = 3;
    expected[16] = 4;
    let target = CnotMatrix::from_le_bytes(3, &expected).unwrap();
    assert!(target.verifies_construction(&[0, 2]).unwrap());
    assert!(!target.verifies_construction(&[2, 0]).unwrap());
    assert!(target.verifies_construction(&[6]).is_err());
    assert!(target.verifies_construction(&[u32::MAX]).is_err());

    let mut wide = identity(256);
    let row_254_word_3_byte = 254 * 32 + 3 * 8 + 7;
    wide[row_254_word_3_byte] ^= 0x80; // 255 -> 254
    let target = CnotMatrix::from_le_bytes(256, &wide).unwrap();
    assert!(target.verifies_construction(&[65279]).unwrap());
    assert!(!target.verifies_construction(&[]).unwrap());
}

#[test]
fn apply_gate_words_matches_cnot_replay_and_clears_tail_padding() {
    for wires in [3u32, 65, 129, 256] {
        let stride = (wires as usize).div_ceil(64);
        let mut words = vec![0u64; wires as usize * stride];
        for row in 0..wires as usize {
            words[row * stride + row / 64] |= 1u64 << (row % 64);
        }
        let tail_bits = wires % 64;
        if tail_bits != 0 {
            let invalid_bit = 1u64 << 63;
            for row in 0..wires as usize {
                words[row * stride + stride - 1] |= invalid_bit;
            }
        }
        let id = CnotGate {
            control: wires - 1,
            target: 0,
        }
        .encode(wires)
        .unwrap();
        beam_core::cnot::apply_gate_words(wires, &mut words, id).unwrap();

        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let matrix = CnotMatrix::from_le_bytes(wires, &bytes).unwrap();
        assert!(matrix.verifies_construction(&[id]).unwrap());
        if tail_bits != 0 {
            let valid_mask = (1u64 << tail_bits) - 1;
            for row in 0..wires as usize {
                assert_eq!(words[row * stride + stride - 1] & !valid_mask, 0);
            }
        }
    }

    let mut words = vec![0u64; 3];
    assert!(beam_core::cnot::apply_gate_words(3, &mut words, 6).is_err());
    assert!(beam_core::cnot::apply_gate_words(3, &mut words[..2], 0).is_err());
}

#[test]
fn identity_check_covers_the_complete_packed_matrix() {
    for n in [2, 63, 64, 65, 127, 128, 129, 256] {
        let mut packed = identity(n as usize);
        let target = CnotMatrix::from_le_bytes(n, &packed).unwrap();
        assert!(target.is_identity(), "identity matrix n={n}");

        let stride = (n as usize).div_ceil(64);
        let row = n as usize - 1;
        let control = row - 1;
        let word = control / 64;
        let bit = control % 64;
        let byte = (row * stride + word) * 8 + bit / 8;
        packed[byte] |= 1 << (bit % 8);
        let changed = CnotMatrix::from_le_bytes(n, &packed).unwrap();
        assert!(!changed.is_identity(), "changed matrix n={n}");
    }

    let mut packed = identity(129);
    packed[2 * 8] |= 1; // A non-diagonal bit in the third word of row zero.
    let changed = CnotMatrix::from_le_bytes(129, &packed).unwrap();
    assert!(!changed.is_identity());
}
