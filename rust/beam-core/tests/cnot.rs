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
