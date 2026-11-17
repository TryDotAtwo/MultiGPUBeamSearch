use beam_native_sys::cnot_domain::{Config, DeviceMoveType, DeviceView, DomainLibrary};
use libloading::Library;
use std::{
    env,
    ffi::c_void,
    mem::{align_of, size_of},
    path::Path,
    ptr,
};

#[test]
#[ignore = "requires old/new standalone CUDA domain libraries and a GPU"]
fn borrowed_expansion_and_device_only_context() {
    assert_eq!(size_of::<DeviceView>(), 24);
    assert_eq!(align_of::<DeviceView>(), 8);
    let old_path = env::var("BEAM_CNOT_OLD_LIBRARY").expect("old native domain library");
    let new_path = env::var("BEAM_CNOT_OWNED_LIBRARY").expect("new native domain library");
    let old = unsafe { DomainLibrary::load(Path::new(&old_path)) }.unwrap();
    let new = unsafe { DomainLibrary::load(Path::new(&new_path)) }.unwrap();
    assert!(old.supports_device_owners());
    assert!(!old.supports_device_expansion());
    assert!(new.supports_device_owners());
    assert!(new.supports_device_expansion());
    let config = Config {
        device: 0,
        n: 4,
        world: 1,
        parent_capacity: 1,
        child_capacity: 1,
        closed_capacity: 0,
    };
    assert!(old.create_device(config).is_err());
    let mut domain = new.create_device(config).unwrap();
    assert!(domain.set_parents(&[]).is_err());
    drop(old);
    drop(new);
    unsafe {
        let cudart = Library::new("libcudart.so.12").unwrap();
        let malloc = *cudart
            .get::<unsafe extern "C" fn(*mut *mut c_void, usize) -> i32>(b"cudaMalloc\0")
            .unwrap();
        let free = *cudart
            .get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"cudaFree\0")
            .unwrap();
        let copy = *cudart
            .get::<unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32>(
                b"cudaMemcpy\0",
            )
            .unwrap();
        let stream_create = *cudart
            .get::<unsafe extern "C" fn(*mut *mut c_void) -> i32>(b"cudaStreamCreate\0")
            .unwrap();
        let stream_destroy = *cudart
            .get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"cudaStreamDestroy\0")
            .unwrap();
        let (mut parents, mut indices, mut moves, mut children, mut stream) = (
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        );
        assert_eq!(malloc(&mut parents, 4 * size_of::<i64>()), 0);
        assert_eq!(malloc(&mut indices, size_of::<i64>()), 0);
        assert_eq!(malloc(&mut moves, size_of::<u16>()), 0);
        assert_eq!(malloc(&mut children, 4 * size_of::<i64>()), 0);
        assert_eq!(stream_create(&mut stream), 0);
        let state = [1i64, 2, 4, 8];
        let parent_index = 0i64;
        let physical_move = 0u16;
        assert_eq!(
            copy(parents, state.as_ptr().cast(), 4 * size_of::<i64>(), 1),
            0
        );
        assert_eq!(
            copy(
                indices,
                (&parent_index as *const i64).cast(),
                size_of::<i64>(),
                1
            ),
            0
        );
        assert_eq!(
            copy(
                moves,
                (&physical_move as *const u16).cast(),
                size_of::<u16>(),
                1
            ),
            0
        );
        domain
            .expand_device(
                DeviceView::from_raw(parents, 4, 0),
                1,
                DeviceView::from_raw(indices, 1, 0),
                DeviceView::from_raw(moves, 1, 0),
                DeviceMoveType::U16,
                1,
                DeviceView::from_raw(children, 4, 0),
                stream,
            )
            .unwrap();
        let mut actual = [0i64; 4];
        assert_eq!(
            copy(
                actual.as_mut_ptr().cast(),
                children,
                4 * size_of::<i64>(),
                2
            ),
            0
        );
        assert_eq!(actual, [1, 3, 4, 8]);
        assert_eq!(stream_destroy(stream), 0);
        assert_eq!(free(children), 0);
        assert_eq!(free(moves), 0);
        assert_eq!(free(indices), 0);
        assert_eq!(free(parents), 0);
    }
    domain.close().unwrap();
}
