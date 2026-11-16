use beam_runtime::cnot_exchange::{self, Backend, Counts};

struct Fixture {
    events: Vec<&'static str>,
    sends: u64,
    receive: Vec<u32>,
    peer_overflow: bool,
    invalid_send: bool,
}

impl Backend for Fixture {
    type Error = &'static str;
    type Output = Vec<u32>;

    fn send_count(&mut self) -> Result<u64, Self::Error> {
        self.events.push("send_count");
        Ok(self.sends)
    }

    fn any_overflow(&mut self, local: bool) -> Result<bool, Self::Error> {
        self.events.push("vote");
        Ok(local || self.peer_overflow)
    }

    fn exchange_counts(&mut self) -> Result<Counts, Self::Error> {
        self.events.push("counts");
        Ok(Counts {
            send: vec![self.sends as u32, 0],
            receive: self.receive.clone(),
            invalid_send: self.invalid_send,
        })
    }

    fn exchange_rows(
        &mut self,
        send: &[u32],
        recv: &[u32],
        len: u32,
    ) -> Result<Self::Output, Self::Error> {
        self.events.push("rows");
        assert_eq!(send, &[self.sends as u32, 0]);
        assert_eq!(recv, self.receive);
        Ok(vec![len])
    }

    fn send_overflow(&mut self) -> Self::Error {
        "send overflow"
    }
    fn receive_overflow(&mut self) -> Self::Error {
        "receive overflow"
    }
    fn invalid_counts(&mut self) -> Self::Error {
        "invalid counts"
    }
}

fn fixture() -> Fixture {
    Fixture {
        events: Vec::new(),
        sends: 3,
        receive: vec![2, 1],
        peer_overflow: false,
        invalid_send: false,
    }
}

#[test]
fn exchanges_counts_before_rows_and_votes_before_each() {
    let mut b = fixture();
    assert_eq!(cnot_exchange::exchange(&mut b), Ok(vec![3]));
    assert_eq!(b.events, ["send_count", "vote", "counts", "vote", "rows"]);
}

#[test]
fn rejects_oversized_or_peer_send_before_count_exchange() {
    let mut b = fixture();
    b.sends = u64::from(u32::MAX) + 1;
    assert_eq!(cnot_exchange::exchange(&mut b), Err("send overflow"));
    assert_eq!(b.events, ["send_count", "vote"]);
    let mut b = fixture();
    b.peer_overflow = true;
    assert_eq!(cnot_exchange::exchange(&mut b), Err("send overflow"));
    assert_eq!(b.events, ["send_count", "vote"]);
}

#[test]
fn rejects_receive_overflow_before_payload_exchange() {
    let mut b = fixture();
    b.receive = vec![u32::MAX, 1];
    assert_eq!(cnot_exchange::exchange(&mut b), Err("receive overflow"));
    assert_eq!(b.events, ["send_count", "vote", "counts", "vote"]);
}

#[test]
fn rejects_invalid_counts_before_payload_exchange() {
    let mut b = fixture();
    b.invalid_send = true;
    assert_eq!(cnot_exchange::exchange(&mut b), Err("invalid counts"));
    assert_eq!(b.events, ["send_count", "vote", "counts", "vote"]);
}

#[test]
fn allows_empty_exchange() {
    let mut b = fixture();
    b.sends = 0;
    b.receive = vec![0, 0];
    assert_eq!(cnot_exchange::exchange(&mut b), Ok(vec![0]));
    assert_eq!(b.events, ["send_count", "vote", "counts", "vote", "rows"]);
}
