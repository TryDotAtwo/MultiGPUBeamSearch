use beam_native_sys::depth_control::{Api, Bridge, DepthControlSink, Input};

struct Panics;
impl DepthControlSink for Panics {
    fn decide(&self, _: beam_native_sys::depth_control::Checkpoint) -> Result<bool, String> {
        panic!("depth decision panic")
    }
}

#[test]
fn callback_contains_panics_and_reports_the_failure() {
    let mut owner = Box::new(Bridge::new(Box::new(Panics)));
    let api: Api = owner.table();
    assert_eq!(std::mem::size_of::<Input>(), 24);
    assert_eq!(std::mem::size_of::<Api>(), 32);
    let input = Input {
        abi_version: 1,
        struct_size: std::mem::size_of::<Input>() as u32,
        checkpoint: 1,
        calibration_limit_reached: 0,
        frontier_live: 0,
        tracked_path_stop: 0,
    };
    let mut output = beam_native_sys::depth_control::Output {
        abi_version: 1,
        struct_size: std::mem::size_of::<beam_native_sys::depth_control::Output>() as u32,
        continue_search: 0,
    };
    // SAFETY: the boxed bridge remains live, and both descriptors are complete
    // caller-owned structures for the duration of the callback.
    let code = unsafe { (api.decide)(api.user_data, &input, &mut output) };
    assert_eq!(code, 3);
    let mut required = 0;
    // SAFETY: this is the documented error-buffer size query.
    assert_eq!(
        unsafe { (api.error)(api.user_data, std::ptr::null_mut(), 0, &mut required) },
        6
    );
    let mut message = vec![0u8; required as usize];
    // SAFETY: message covers the capacity returned by the preceding query.
    assert_eq!(
        unsafe {
            (api.error)(
                api.user_data,
                message.as_mut_ptr().cast(),
                message.len() as u64,
                &mut required,
            )
        },
        0
    );
    assert!(String::from_utf8(message).unwrap().contains("panicked"));
}
