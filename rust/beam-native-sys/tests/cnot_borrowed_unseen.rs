use beam_native_sys::cnot_domain::{Config, DeviceUnseenWorkspace, DeviceView, DomainLibrary};
use libloading::Library;
use std::{env, ffi::c_void, mem::size_of, path::Path, ptr};

#[test]
#[ignore = "requires old/new standalone CUDA domain libraries and a GPU"]
fn borrowed_unseen_and_older_library_compatibility() {
    assert_eq!(size_of::<DeviceUnseenWorkspace>(), 120);
    let old_path = env::var("BEAM_CNOT_OLD_LIBRARY").expect("old native domain library");
    let new_path = env::var("BEAM_CNOT_OWNED_LIBRARY").expect("new native domain library");
    let old = unsafe { DomainLibrary::load(Path::new(&old_path)) }.unwrap();
    let new = unsafe { DomainLibrary::load(Path::new(&new_path)) }.unwrap();
    assert!(old.supports_device_expansion());
    assert!(!old.supports_device_unseen());
    assert!(new.supports_device_expansion());
    assert!(new.supports_device_unseen());
    let config = Config {
        device: 0,
        n: 4,
        world: 1,
        parent_capacity: 0,
        child_capacity: 3,
        closed_capacity: 1,
    };
    let mut previous = old.create_device(config).unwrap();
    assert!(previous.unseen_scratch_bytes(1, 3).is_err());
    let mut domain = new.create_device(config).unwrap();
    drop(old);
    drop(new);
    let bytes = domain.unseen_scratch_bytes(1, 3).unwrap();
    assert!(bytes > 0);
    assert_eq!(domain.unseen_scratch_bytes(1, 0).unwrap(), 0);
    assert!(domain.unseen_scratch_bytes(2, 3).is_err());
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
        let mut pointers = [ptr::null_mut(); 7];
        for (pointer, len) in pointers
            .iter_mut()
            .zip([32, 96, 16, 3, 24, 4, bytes as usize])
        {
            assert_eq!(malloc(pointer, len), 0);
        }
        let mut stream = ptr::null_mut();
        assert_eq!(stream_create(&mut stream), 0);
        let closed = [1i64, 2, 4, 8];
        let children = [1i64, 2, 4, 8, 1, 3, 4, 8, 1, 3, 4, 8];
        assert_eq!(copy(pointers[0], closed.as_ptr().cast(), 32, 1), 0);
        assert_eq!(copy(pointers[1], children.as_ptr().cast(), 96, 1), 0);
        let workspace = DeviceUnseenWorkspace {
            ordinals: DeviceView::from_raw(pointers[2], 4, 0),
            flags: DeviceView::from_raw(pointers[3], 3, 0),
            selected: DeviceView::from_raw(pointers[4], 3, 0),
            count: DeviceView::from_raw(pointers[5], 1, 0),
            scratch: DeviceView::from_raw(pointers[6], bytes, 0),
        };
        domain
            .first_unseen_device(
                DeviceView::from_raw(pointers[0], 4, 0),
                1,
                DeviceView::from_raw(pointers[1], 12, 0),
                3,
                &workspace,
                stream,
            )
            .unwrap();
        let mut count = -1i32;
        let mut selected = [-1i64; 3];
        assert_eq!(copy((&mut count as *mut i32).cast(), pointers[5], 4, 2), 0);
        assert_eq!(copy(selected.as_mut_ptr().cast(), pointers[4], 8, 2), 0);
        assert_eq!(count, 1);
        assert_eq!(selected[0], 1);
        domain
            .first_unseen_device(
                DeviceView::from_raw(pointers[0], 4, 0),
                1,
                DeviceView::from_raw(ptr::null_mut(), 0, 0),
                0,
                &workspace,
                stream,
            )
            .unwrap();
        assert_eq!(copy((&mut count as *mut i32).cast(), pointers[5], 4, 2), 0);
        assert_eq!(count, 0);
        assert_eq!(stream_destroy(stream), 0);
        for pointer in pointers {
            assert_eq!(free(pointer), 0);
        }
    }
    previous.close().unwrap();
    domain.close().unwrap();
}
