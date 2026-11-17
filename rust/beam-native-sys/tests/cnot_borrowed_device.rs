use beam_native_sys::cnot_domain::{Config, DeviceView, DomainLibrary};
use libloading::Library;
use std::{env, ffi::c_void, mem::size_of, path::Path, ptr};

#[test]
#[ignore = "requires old/new standalone CUDA domain libraries and a GPU"]
fn borrowed_device_stream_and_older_library_compatibility() {
    let old_path = env::var("BEAM_CNOT_OLD_LIBRARY").expect("old native domain library");
    let new_path = env::var("BEAM_CNOT_OWNED_LIBRARY").expect("new native domain library");
    let old = unsafe { DomainLibrary::load(Path::new(&old_path)) }.unwrap();
    let new = unsafe { DomainLibrary::load(Path::new(&new_path)) }.unwrap();
    assert!(!old.supports_device_owners());
    assert!(new.supports_device_owners());
    let config = Config {
        device: 0,
        n: 4,
        world: 1,
        parent_capacity: 0,
        child_capacity: 1,
        closed_capacity: 0,
    };
    let mut old_domain = old.create(config).unwrap();
    let mut domain = new.create(config).unwrap();
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
        let (mut input, mut output, mut stream) =
            (ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
        assert_eq!(malloc(&mut input, 4 * size_of::<i64>()), 0);
        assert_eq!(malloc(&mut output, size_of::<i64>()), 0);
        assert_eq!(stream_create(&mut stream), 0);
        let states = [1i64, 2, 4, 8];
        assert_eq!(
            copy(input, states.as_ptr().cast(), 4 * size_of::<i64>(), 1),
            0
        );
        let src = DeviceView::from_raw(input, 4, 0);
        let dst = DeviceView::from_raw(output, 1, 0);
        assert!(old_domain.owners_device(src, 1, dst, stream).is_err());
        domain.owners_device(src, 1, dst, stream).unwrap();
        let mut owner = -1i64;
        assert_eq!(
            copy((&mut owner as *mut i64).cast(), output, size_of::<i64>(), 2),
            0
        );
        assert_eq!(owner, 0);
        assert_eq!(stream_destroy(stream), 0);
        assert_eq!(free(output), 0);
        assert_eq!(free(input), 0);
    }
    old_domain.close().unwrap();
    domain.close().unwrap();
}
