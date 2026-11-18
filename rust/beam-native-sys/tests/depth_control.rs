use beam_native_sys::depth_control::{
    Api, ApiV2, Bridge, Decision, DepthControlSink, Input, InputV2, Output, OutputV2,
    STOP_REASON_CAPACITY,
};

struct Panics;
impl DepthControlSink for Panics {
    fn decide(&self, _: beam_native_sys::depth_control::Checkpoint) -> Result<bool, String> {
        panic!("depth decision panic")
    }
}

struct Bucket;
impl DepthControlSink for Bucket {
    fn decide(&self, _: beam_native_sys::depth_control::Checkpoint) -> Result<bool, String> {
        Ok(false)
    }
    fn decide_v2(
        &self,
        checkpoint: beam_native_sys::depth_control::CheckpointV2,
    ) -> Result<Decision, String> {
        match checkpoint {
            beam_native_sys::depth_control::CheckpointV2::SolveBucketStop {
                max_solutions: 3,
                accepted_solutions: 3,
                completed_depths: 1,
                found_any: true,
                ..
            } => Ok(Decision {
                continue_search: false,
                stop_reason: STOP_REASON_CAPACITY,
            }),
            _ => Err("solve-bucket fields were not preserved through the ABI".into()),
        }
    }
}

#[test]
fn callback_contains_panics_and_reports_the_failure() {
    let mut owner = Box::new(Bridge::new(Box::new(Panics)));
    let api: Api = owner.table();
    assert_eq!(std::mem::size_of::<Input>(), 24);
    assert_eq!(std::mem::align_of::<Input>(), 4);
    assert_eq!(std::mem::size_of::<Output>(), 12);
    assert_eq!(std::mem::align_of::<Output>(), 4);
    assert_eq!(std::mem::size_of::<Api>(), 32);
    let input = Input {
        abi_version: 1,
        struct_size: std::mem::size_of::<Input>() as u32,
        checkpoint: 1,
        calibration_limit_reached: 0,
        frontier_live: 0,
        tracked_path_stop: 0,
        ..Input::default()
    };
    let mut output = Output {
        abi_version: 1,
        struct_size: std::mem::size_of::<Output>() as u32,
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

#[test]
fn solve_bucket_checkpoint_round_trips_scalar_fields_and_reason_code() {
    let mut owner = Box::new(Bridge::new(Box::new(Bucket)));
    let api: ApiV2 = owner.table_v2();
    assert_eq!(std::mem::size_of::<InputV2>(), 64);
    assert_eq!(std::mem::align_of::<InputV2>(), 8);
    assert_eq!(std::mem::size_of::<OutputV2>(), 16);
    assert_eq!(std::mem::align_of::<OutputV2>(), 4);
    assert_eq!(std::mem::size_of::<ApiV2>(), 32);
    let input = InputV2 {
        abi_version: 2,
        struct_size: std::mem::size_of::<InputV2>() as u32,
        checkpoint: 3,
        max_solutions: 3,
        accepted_solutions: 3,
        completed_depths: 1,
        solve_bucket_found_any: 1,
        ..InputV2::default()
    };
    let mut output = OutputV2 {
        abi_version: 2,
        struct_size: std::mem::size_of::<OutputV2>() as u32,
        continue_search: 1,
        stop_reason: 0,
    };
    // SAFETY: the boxed bridge and complete descriptors remain live through
    // this synchronous callback.
    assert_eq!(
        unsafe { (api.decide)(api.user_data, &input, &mut output) },
        0
    );
    assert_eq!(output.continue_search, 0);
    assert_eq!(output.stop_reason, STOP_REASON_CAPACITY);
}
