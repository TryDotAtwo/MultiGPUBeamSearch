use beam_runtime::search_session::SearchSession;

#[test]
fn preparation_identity_is_owned_and_invalidated_by_close() {
    let mut session = SearchSession::new();

    assert!(!session.preparation_matches("first").unwrap());
    session.remember_preparation("first").unwrap();
    assert!(session.preparation_matches("first").unwrap());
    assert!(!session.preparation_matches("second").unwrap());

    session.close().unwrap();
    assert!(session.preparation_matches("first").is_err());
}
