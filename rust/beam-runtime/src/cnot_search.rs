//! Control flow and history for the distributed packed-matrix CNOT search.
//!
//! Stage implementations own their tensors and communicators. The controller
//! never receives a device pointer; each call represents a complete batch or
//! collective, and the caller keeps the backend on its original rank thread.

/// Public preference for the CNOT search controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestedEngine {
    Auto,
    Python,
    Rust,
}

/// Search mode passed to the Rust CNOT controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CnotMode {
    Qhat,
    Proposal,
}

impl CnotMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Qhat => "qhat",
            Self::Proposal => "proposal",
        }
    }
}

/// Input validation path requested by the compatible Python API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CnotValidation {
    Auto,
    Python,
    Rust,
}

/// Errors while converting public scalar options to one typed CNOT request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CnotRequestError {
    #[error("CNOT engine must be auto, python, or rust")]
    Engine,
    #[error("CNOT validation must be auto, python, or rust")]
    Validation,
    #[error(transparent)]
    Controls(#[from] beam_core::cnot::SearchControlsError),
}

/// A single admitted CNOT search configuration owned by beam-runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CnotRequest {
    pub wires: u32,
    pub width: u64,
    pub shortlist: u32,
    pub max_depth: u64,
    pub mode: CnotMode,
    pub audit: bool,
    pub validation: CnotValidation,
    pub requested_engine: RequestedEngine,
}

impl CnotRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wires: u32,
        width: u64,
        shortlist: u64,
        max_depth: u64,
        mode: &str,
        audit: bool,
        validation: &str,
        engine: &str,
    ) -> Result<Self, CnotRequestError> {
        let mode = match mode {
            "qhat" => CnotMode::Qhat,
            "proposal" => CnotMode::Proposal,
            _ => {
                return Err(CnotRequestError::Controls(
                    beam_core::cnot::SearchControlsError::Mode,
                ))
            }
        };
        let validation = match validation {
            "auto" => CnotValidation::Auto,
            "python" => CnotValidation::Python,
            "rust" => CnotValidation::Rust,
            _ => return Err(CnotRequestError::Validation),
        };
        let requested_engine = match engine {
            "auto" => RequestedEngine::Auto,
            "python" => RequestedEngine::Python,
            "rust" => RequestedEngine::Rust,
            _ => return Err(CnotRequestError::Engine),
        };
        beam_core::cnot::validate_search_controls(wires, width, shortlist, mode.as_str())?;
        let alphabet = u64::from(wires) * u64::from(wires.saturating_sub(1));
        let shortlist = u32::try_from(shortlist.min(alphabet)).map_err(|_| {
            CnotRequestError::Controls(beam_core::cnot::SearchControlsError::WidthOrShortlist)
        })?;
        Ok(Self {
            wires,
            width,
            shortlist,
            max_depth,
            mode,
            audit,
            validation,
            requested_engine,
        })
    }

    pub fn alphabet(self) -> u32 {
        self.wires * (self.wires - 1)
    }

    pub fn selected_engine(self) -> SelectedEngine {
        match self.requested_engine {
            RequestedEngine::Python => SelectedEngine::Python,
            RequestedEngine::Auto | RequestedEngine::Rust => SelectedEngine::Rust,
        }
    }
}

/// Controller selected for one admitted CNOT request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedEngine {
    Python,
    Rust,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineSelectionError {
    RustDepthExceedsU64,
}

/// Resolve controller selection in the runtime so adapters share one policy.
/// A depth that cannot fit the native controller remains valid for the Python
/// reference path when the request is automatic or explicitly Python.
pub fn select_engine(
    requested: RequestedEngine,
    depth_fits_u64: bool,
) -> Result<SelectedEngine, EngineSelectionError> {
    match requested {
        RequestedEngine::Python => Ok(SelectedEngine::Python),
        RequestedEngine::Auto if depth_fits_u64 => Ok(SelectedEngine::Rust),
        RequestedEngine::Auto => Ok(SelectedEngine::Python),
        RequestedEngine::Rust if depth_fits_u64 => Ok(SelectedEngine::Rust),
        RequestedEngine::Rust => Err(EngineSelectionError::RustDepthExceedsU64),
    }
}

/// Capabilities agreed by all ranks before choosing the public controller path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DispatchCapabilities {
    pub native_controller: bool,
    pub checked_control: bool,
    pub native_stages: bool,
    pub stream5_start: bool,
    pub world_exchange: bool,
    pub fixed_gather: bool,
    pub row_gather: bool,
    pub broadcast: bool,
    pub compatibility_controller: bool,
}

/// Public CNOT dispatch route selected from the request and rank capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchPath {
    RustController,
    RustCompatibility,
    PythonReference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchError {
    EngineSelection(EngineSelectionError),
    StrictRequiresRustController,
    StrictBackendUnavailable,
    RustControllerUnavailable,
}

/// Resolve the CNOT control path once from typed options and the rank-wide
/// capability vote. Strict acceptance requires the complete Rust data path;
/// ordinary auto mode retains the Python reference fallback.
pub fn select_dispatch(
    requested: RequestedEngine,
    depth_fits_u64: bool,
    strict_backend: bool,
    capabilities: DispatchCapabilities,
) -> Result<DispatchPath, DispatchError> {
    let selected =
        select_engine(requested, depth_fits_u64).map_err(DispatchError::EngineSelection)?;
    if strict_backend {
        if selected != SelectedEngine::Rust {
            return Err(DispatchError::StrictRequiresRustController);
        }
        return if capabilities.native_controller
            && capabilities.checked_control
            && capabilities.native_stages
            && capabilities.stream5_start
            && capabilities.world_exchange
            && capabilities.fixed_gather
            && capabilities.row_gather
            && capabilities.broadcast
        {
            Ok(DispatchPath::RustController)
        } else {
            Err(DispatchError::StrictBackendUnavailable)
        };
    }

    match selected {
        SelectedEngine::Python => Ok(DispatchPath::PythonReference),
        SelectedEngine::Rust if capabilities.native_controller && capabilities.checked_control => {
            Ok(DispatchPath::RustController)
        }
        SelectedEngine::Rust if capabilities.compatibility_controller => {
            Ok(DispatchPath::RustCompatibility)
        }
        SelectedEngine::Rust if requested == RequestedEngine::Auto => {
            Ok(DispatchPath::PythonReference)
        }
        SelectedEngine::Rust => Err(DispatchError::RustControllerUnavailable),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub rank: u32,
    pub world: u32,
    pub width: u64,
    pub shortlist: u32,
    pub alphabet: u32,
    pub max_depth: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Goal {
    pub slot: u64,
    pub movement: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    pub parent: u64,
    pub route: u32,
    pub owner: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Route {
    pub source: u32,
    pub hash_owner: u32,
    pub target: u32,
    pub movement: u32,
}

pub struct Round<L, H> {
    pub count: u64,
    pub layer: L,
    pub records: Option<H>,
}

pub struct Outcome<L> {
    pub status: &'static str,
    pub gates: Option<Vec<u32>>,
    pub layers: Vec<L>,
    pub routes: Vec<Route>,
}

/// Operations are ordered exactly as in the reference search. In particular,
/// expansion and evaluation each include the existing coordinated error check.
pub trait Backend {
    type Error;
    type Layer;
    type Records;

    fn root_is_goal(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
    fn initialize(&mut self) -> Result<(), Self::Error>;
    fn goal(&mut self, begin: u64) -> Result<Option<Goal>, Self::Error>;
    fn expand(&mut self, begin: u64, k: u32) -> Result<(), Self::Error>;
    fn exchange_candidates(&mut self) -> Result<(), Self::Error>;
    fn deduplicate_and_score(&mut self) -> Result<(), Self::Error>;
    fn select(&mut self, width: u64) -> Result<(), Self::Error>;
    fn redistribute(
        &mut self,
        depth: u64,
    ) -> Result<Round<Self::Layer, Self::Records>, Self::Error>;
    fn history_edge(
        &mut self,
        records: &Self::Records,
        target: u32,
        local_index: u64,
    ) -> Result<Edge, Self::Error>;
    fn verifies_construction(&mut self, gates: &[u32]) -> Result<bool, Self::Error>;
    fn invalid(&self, message: &'static str) -> Self::Error;
}

struct History<H> {
    count: u64,
    previous_count: u64,
    records: H,
}

fn begin(rank: u32, count: u64, world: u32) -> u64 {
    // A valid result is <= count. u128 avoids intermediate overflow.
    ((u128::from(rank) * u128::from(count)).div_ceil(u128::from(world))) as u64
}

fn reconstruct<B: Backend>(
    backend: &mut B,
    options: Options,
    history: &[History<B::Records>],
    goal: Goal,
) -> Result<(Vec<u32>, Vec<Route>), B::Error> {
    let mut slot = goal.slot;
    let mut gates = vec![goal.movement];
    let mut routes = Vec::with_capacity(history.len());
    for entry in history.iter().rev() {
        if slot >= entry.count || entry.previous_count == 0 {
            return Err(backend.invalid("CNOT history has an invalid parent slot"));
        }
        let target =
            (u128::from(slot) * u128::from(options.world) / u128::from(entry.count)) as u32;
        let local = slot - begin(target, entry.count, options.world);
        let edge = backend.history_edge(&entry.records, target, local)?;
        let source = edge.route >> 16;
        let movement = edge.route & 0xffff;
        if edge.parent >= entry.previous_count
            || u128::from(source)
                != u128::from(edge.parent) * u128::from(options.world)
                    / u128::from(entry.previous_count)
        {
            return Err(backend.invalid("CNOT history disagrees with parent world rank"));
        }
        slot = edge.parent;
        gates.push(movement);
        routes.push(Route {
            source,
            hash_owner: edge.owner,
            target,
            movement,
        });
    }
    if slot != 0 || !backend.verifies_construction(&gates)? {
        return Err(backend.invalid("CNOT construction failed bitwise replay from the identity"));
    }
    routes.reverse();
    Ok((gates, routes))
}

pub fn search<B: Backend>(
    backend: &mut B,
    options: Options,
) -> Result<Outcome<B::Layer>, B::Error> {
    if options.world == 0
        || options.rank >= options.world
        || options.world > 65536
        || options.width == 0
        || options.shortlist == 0
        || options.alphabet == 0
    {
        return Err(backend.invalid("invalid CNOT search geometry"));
    }
    if backend.root_is_goal()? {
        return Ok(Outcome {
            status: "solved",
            gates: Some(Vec::new()),
            layers: Vec::new(),
            routes: Vec::new(),
        });
    }
    backend.initialize()?;
    let mut history = Vec::new();
    let mut layers = Vec::new();
    let mut begin_slot = 0;
    let mut previous_count = 1;
    for step in 0..options.max_depth {
        if let Some(goal) = backend.goal(begin_slot)? {
            let (gates, routes) = reconstruct(backend, options, &history, goal)?;
            return Ok(Outcome {
                status: "solved",
                gates: Some(gates),
                layers,
                routes,
            });
        }
        if step + 1 == options.max_depth {
            break;
        }
        backend.expand(begin_slot, options.shortlist.min(options.alphabet))?;
        backend.exchange_candidates()?;
        backend.deduplicate_and_score()?;
        backend.select(options.width)?;
        let round = backend.redistribute(step + 1)?;
        let count = round.count;
        layers.push(round.layer);
        if count == 0 {
            return Ok(Outcome {
                status: "exhausted",
                gates: None,
                layers,
                routes: Vec::new(),
            });
        }
        let records = round
            .records
            .ok_or_else(|| backend.invalid("CNOT history records are missing"))?;
        history.push(History {
            count,
            previous_count,
            records,
        });
        begin_slot = begin(options.rank, count, options.world);
        previous_count = count;
    }
    Ok(Outcome {
        status: "depth_limit",
        gates: None,
        layers,
        routes: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[test]
    fn typed_cnot_request_preserves_controls_and_caps_shortlist_to_alphabet() {
        let request = CnotRequest::new(4, 9, 99, 7, "proposal", true, "auto", "auto").unwrap();

        assert_eq!(request.wires, 4);
        assert_eq!(request.width, 9);
        assert_eq!(request.shortlist, 12);
        assert_eq!(request.max_depth, 7);
        assert_eq!(request.mode, CnotMode::Proposal);
        assert!(request.audit);
        assert_eq!(request.validation, CnotValidation::Auto);
        assert_eq!(request.requested_engine, RequestedEngine::Auto);
        assert_eq!(request.selected_engine(), SelectedEngine::Rust);

        let options = request.search_options(1, 2);
        assert_eq!(options.rank, 1);
        assert_eq!(options.world, 2);
        assert_eq!(options.width, 9);
        assert_eq!(options.shortlist, 12);
        assert_eq!(options.alphabet, 12);
        assert_eq!(options.max_depth, 7);
    }

    #[test]
    fn typed_cnot_request_rejects_invalid_public_controls() {
        assert!(CnotRequest::new(4, 0, 1, 7, "qhat", false, "auto", "auto").is_err());
        assert!(CnotRequest::new(4, 9, 1, 7, "unknown", false, "auto", "auto").is_err());
        assert!(CnotRequest::new(4, 9, 1, 7, "qhat", false, "auto", "unknown").is_err());
        assert!(CnotRequest::new(4, 9, 1, 7, "qhat", false, "unknown", "auto").is_err());
    }

    #[test]
    fn engine_selection_preserves_auto_reference_and_explicit_rust_depth_rules() {
        assert_eq!(
            select_engine(RequestedEngine::Auto, true),
            Ok(SelectedEngine::Rust)
        );
        assert_eq!(
            select_engine(RequestedEngine::Auto, false),
            Ok(SelectedEngine::Python)
        );
        assert_eq!(
            select_engine(RequestedEngine::Python, false),
            Ok(SelectedEngine::Python)
        );
        assert_eq!(
            select_engine(RequestedEngine::Rust, true),
            Ok(SelectedEngine::Rust)
        );
        assert_eq!(
            select_engine(RequestedEngine::Rust, false),
            Err(EngineSelectionError::RustDepthExceedsU64)
        );
    }

    fn complete_dispatch_capabilities() -> DispatchCapabilities {
        DispatchCapabilities {
            native_controller: true,
            checked_control: true,
            native_stages: true,
            stream5_start: true,
            world_exchange: true,
            fixed_gather: true,
            row_gather: true,
            broadcast: true,
            compatibility_controller: true,
        }
    }

    #[test]
    fn runtime_selects_public_cnot_path_from_typed_capabilities() {
        let complete = complete_dispatch_capabilities();
        assert_eq!(
            select_dispatch(RequestedEngine::Auto, true, false, complete),
            Ok(DispatchPath::RustController)
        );

        let mut compatibility = complete;
        compatibility.native_controller = false;
        compatibility.checked_control = false;
        assert_eq!(
            select_dispatch(RequestedEngine::Auto, true, false, compatibility),
            Ok(DispatchPath::RustCompatibility)
        );

        assert_eq!(
            select_dispatch(
                RequestedEngine::Auto,
                true,
                false,
                DispatchCapabilities::default()
            ),
            Ok(DispatchPath::PythonReference)
        );
        assert_eq!(
            select_dispatch(RequestedEngine::Python, true, false, complete),
            Ok(DispatchPath::PythonReference)
        );
        assert_eq!(
            select_dispatch(
                RequestedEngine::Rust,
                true,
                false,
                DispatchCapabilities::default()
            ),
            Err(DispatchError::RustControllerUnavailable)
        );
    }

    #[test]
    fn strict_public_cnot_path_requires_all_native_rank_capabilities() {
        let complete = complete_dispatch_capabilities();
        assert_eq!(
            select_dispatch(RequestedEngine::Auto, true, true, complete),
            Ok(DispatchPath::RustController)
        );
        assert_eq!(
            select_dispatch(RequestedEngine::Python, true, true, complete),
            Err(DispatchError::StrictRequiresRustController)
        );

        let mut missing = complete;
        missing.native_stages = false;
        assert_eq!(
            select_dispatch(RequestedEngine::Auto, true, true, missing),
            Err(DispatchError::StrictBackendUnavailable)
        );
        assert_eq!(
            select_dispatch(RequestedEngine::Auto, false, true, complete),
            Err(DispatchError::StrictRequiresRustController)
        );
    }

    #[derive(Default)]
    struct Mock {
        calls: Vec<String>,
        goals: VecDeque<Option<Goal>>,
        counts: VecDeque<u64>,
        fail_stage: Option<&'static str>,
        valid: bool,
        root_goal: bool,
    }
    impl Mock {
        fn call(
            &mut self,
            stage: &'static str,
            detail: impl std::fmt::Display,
        ) -> Result<(), String> {
            self.calls.push(format!("{stage}:{detail}"));
            if self.fail_stage == Some(stage) {
                Err(format!("failed:{stage}"))
            } else {
                Ok(())
            }
        }
    }
    impl Backend for Mock {
        type Error = String;
        type Layer = u64;
        type Records = u64;
        fn root_is_goal(&mut self) -> Result<bool, String> {
            let goal = self.root_goal;
            self.call("root_goal", goal)?;
            Ok(goal)
        }
        fn initialize(&mut self) -> Result<(), String> {
            self.call("initialize", 0)
        }
        fn goal(&mut self, b: u64) -> Result<Option<Goal>, String> {
            self.call("goal", b)?;
            Ok(self.goals.pop_front().unwrap_or(None))
        }
        fn expand(&mut self, b: u64, k: u32) -> Result<(), String> {
            self.call("expand", format!("{b},{k}"))
        }
        fn exchange_candidates(&mut self) -> Result<(), String> {
            self.call("exchange", 0)
        }
        fn deduplicate_and_score(&mut self) -> Result<(), String> {
            self.call("dedup", 0)
        }
        fn select(&mut self, w: u64) -> Result<(), String> {
            self.call("select", w)
        }
        fn redistribute(&mut self, d: u64) -> Result<Round<u64, u64>, String> {
            self.call("redistribute", d)?;
            let count = self.counts.pop_front().unwrap();
            Ok(Round {
                count,
                layer: d,
                records: (count > 0).then_some(d),
            })
        }
        fn history_edge(&mut self, r: &u64, t: u32, i: u64) -> Result<Edge, String> {
            self.call("edge", format!("{r},{t},{i}"))?;
            Ok(if *r == 2 {
                Edge {
                    parent: 2,
                    route: (1 << 16) | 6,
                    owner: 0,
                }
            } else {
                Edge {
                    parent: 0,
                    route: 5,
                    owner: 1,
                }
            })
        }
        fn verifies_construction(&mut self, g: &[u32]) -> Result<bool, String> {
            self.call("verify", format!("{g:?}"))?;
            Ok(self.valid)
        }
        fn invalid(&self, s: &'static str) -> String {
            s.into()
        }
    }
    fn opts(depth: u64) -> Options {
        Options {
            rank: 1,
            world: 2,
            width: 5,
            shortlist: 30,
            alphabet: 12,
            max_depth: depth,
        }
    }
    #[test]
    fn final_goal_and_distributed_history_replay() {
        let mut b = Mock {
            goals: [
                None,
                None,
                Some(Goal {
                    slot: 1,
                    movement: 9,
                }),
            ]
            .into(),
            counts: [3, 2].into(),
            valid: true,
            ..Mock::default()
        };
        let out = search(&mut b, opts(3)).unwrap();
        assert_eq!(out.status, "solved");
        assert_eq!(out.gates.unwrap(), [9, 6, 5]);
        assert_eq!(out.layers, [1, 2]);
        assert_eq!(
            out.routes,
            [
                Route {
                    source: 0,
                    hash_owner: 1,
                    target: 1,
                    movement: 5
                },
                Route {
                    source: 1,
                    hash_owner: 0,
                    target: 1,
                    movement: 6
                }
            ]
        );
        assert_eq!(
            b.calls,
            [
                "root_goal:false",
                "initialize:0",
                "goal:0",
                "expand:0,12",
                "exchange:0",
                "dedup:0",
                "select:5",
                "redistribute:1",
                "goal:2",
                "expand:2,12",
                "exchange:0",
                "dedup:0",
                "select:5",
                "redistribute:2",
                "goal:1",
                "edge:2,1,0",
                "edge:1,1,0",
                "verify:[9, 6, 5]"
            ]
        );
    }
    #[test]
    fn depth_limit_and_empty_frontier_preserve_stage_order() {
        for depth in [0, 1] {
            let mut b = Mock::default();
            let out = search(&mut b, opts(depth)).unwrap();
            assert_eq!(out.status, "depth_limit");
            assert_eq!(b.calls.len(), 2 + depth as usize);
        }
        let mut b = Mock {
            counts: [0].into(),
            ..Mock::default()
        };
        let out = search(&mut b, opts(4)).unwrap();
        assert_eq!(out.status, "exhausted");
        assert_eq!(out.layers, [1]);
        assert_eq!(b.calls.last().unwrap(), "redistribute:1");
    }
    #[test]
    fn identity_root_returns_empty_solution_before_initialization() {
        let mut b = Mock {
            root_goal: true,
            ..Mock::default()
        };
        let out = search(&mut b, opts(0)).unwrap();
        assert_eq!(out.status, "solved");
        assert_eq!(out.gates, Some(Vec::new()));
        assert!(out.layers.is_empty());
        assert!(out.routes.is_empty());
        assert_eq!(b.calls, ["root_goal:true"]);
    }
    #[test]
    fn errors_stop_before_the_next_collective_and_replay_is_required() {
        let mut b = Mock {
            fail_stage: Some("dedup"),
            ..Mock::default()
        };
        assert_eq!(search(&mut b, opts(3)).err().unwrap(), "failed:dedup");
        assert_eq!(b.calls.last().unwrap(), "dedup:0");
        let mut b = Mock {
            goals: [Some(Goal {
                slot: 0,
                movement: 2,
            })]
            .into(),
            ..Mock::default()
        };
        assert!(search(&mut b, opts(1))
            .err()
            .unwrap()
            .contains("bitwise replay"));
    }
}
