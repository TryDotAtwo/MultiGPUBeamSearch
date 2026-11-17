use beam_native_sys::cnot_domain::{Config, DomainLibrary};
use std::{env, path::Path};

#[test]
#[ignore = "requires an explicitly built standalone CUDA domain and a GPU"]
fn owned_domain_survives_library_handle_and_reuses_resources() {
    let path =
        env::var("BEAM_CNOT_OWNED_LIBRARY").expect("native CNOT domain library must be supplied");
    let lib = unsafe { DomainLibrary::load(Path::new(&path)) }.unwrap();
    let mut domain = lib
        .create(Config {
            device: 0,
            n: 4,
            world: 1,
            parent_capacity: 1,
            child_capacity: 2,
            closed_capacity: 1,
        })
        .unwrap();
    drop(lib);
    domain.set_parents(&[1, 2, 4, 8]).unwrap();
    domain.set_closed(&[]).unwrap();
    domain.set_descriptors(&[0, 0], &[0, 0]).unwrap();
    domain.expand().unwrap();
    assert_eq!(domain.children().unwrap(), vec![1, 3, 4, 8, 1, 3, 4, 8]);
    assert_eq!(domain.owners().unwrap(), vec![0, 0]);
    assert_eq!(domain.first_unseen().unwrap(), vec![0]);
    domain.set_closed(&[1, 3, 4, 8]).unwrap();
    assert_eq!(domain.first_unseen().unwrap(), Vec::<i64>::new());
    assert!(domain.set_descriptors(&[0, 0, 0], &[0, 0, 0]).is_err());
    domain.set_parents(&[1, 2, 4, 8]).unwrap();
    assert!(domain.expand().is_err());
    assert!(domain.set_parents(&[1, 2]).is_err());
    domain.close().unwrap();
}
