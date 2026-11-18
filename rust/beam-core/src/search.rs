//! Incremental terminal receipts; every returned path is replayed independently.
pub use crate::permutation::{search_budget, PermutationGraph, SearchBudget};
use crate::ContractError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
type Result<T> = std::result::Result<T, ContractError>;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaggedSearchValue {
    pub kind: String,
    pub value: Value,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicSearchControls {
    pub beam_width: TaggedSearchValue,
    pub max_steps: TaggedSearchValue,
    pub beam_mode: TaggedSearchValue,
    pub history_depth: TaggedSearchValue,
    pub destination_state_present: bool,
    pub bfs_result_for_mitm_present: bool,
    pub return_path: TaggedSearchValue,
    pub use_child_scores: TaggedSearchValue,
}

/// CayleyPy values after the Python adapter has checked their original types.
/// Decimal strings keep oversized Python integers lossless at the FFI edge.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicGraphContractRequest {
    pub center: Vec<String>,
    pub start: Vec<String>,
    pub generators: Vec<Vec<String>>,
    pub generator_name_count: usize,
}

/// Validated graph data used by the public adapter and native search session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PublicGraphContract {
    pub state_len: usize,
    pub num_classes: u16,
    pub generators: Vec<Vec<u32>>,
    pub center: Vec<u8>,
    pub start: Vec<u8>,
    pub graph_hash: String,
}

fn hash_decimal_array(hash: &mut Sha256, values: impl IntoIterator<Item = u32>) {
    hash.update(b"[");
    for (index, mut value) in values.into_iter().enumerate() {
        if index != 0 {
            hash.update(b", ");
        }
        let mut digits = [0u8; 10];
        let mut start = digits.len();
        loop {
            start -= 1;
            digits[start] = b'0' + (value % 10) as u8;
            value /= 10;
            if value == 0 {
                break;
            }
        }
        hash.update(&digits[start..]);
    }
    hash.update(b"]");
}

/// Match `cayleypy.models.checkpoint.graph_hash`'s sorted-key JSON encoding.
pub fn permutation_graph_hash(graph: &PermutationGraph) -> String {
    let mut hash = Sha256::new();
    hash.update(b"{\"central_state\": ");
    hash_decimal_array(&mut hash, graph.center.iter().copied().map(u32::from));
    hash.update(b", \"generators\": [");
    for (index, generator) in graph.generators.iter().enumerate() {
        if index != 0 {
            hash.update(b", ");
        }
        hash_decimal_array(&mut hash, generator.iter().copied());
    }
    hash.update(b"], \"generators_type\": \"PERMUTATION\"}");
    format!("{:x}", hash.finalize())
}

/// Validate the normalized public graph and compute its stable CayleyPy identity.
pub fn build_public_graph_contract(
    request: PublicGraphContractRequest,
) -> std::result::Result<PublicGraphContract, String> {
    let size = request.center.len();
    if !(1..=150).contains(&size) {
        return Err("native logical state length must be in [1, 150]".into());
    }
    if request.start.len() != size {
        return Err(format!(
            "start_state must have exactly one state of length {size}"
        ));
    }
    if request.generators.is_empty() || request.generators.len() > 255 {
        return Err("native move count must be in [1, 255]".into());
    }
    if request.generator_name_count != request.generators.len() {
        return Err("generator_names must contain one string per ordered generator".into());
    }

    let value_max = if size > 128 { 255 } else { 127 };
    let parse_state = |values: &[String]| -> std::result::Result<Vec<u8>, String> {
        values
            .iter()
            .map(|value| {
                value
                    .parse::<u16>()
                    .ok()
                    .filter(|value| *value <= value_max)
                    .map(|value| value as u8)
                    .ok_or_else(|| {
                        format!("native state values must be integers in [0, {value_max}]")
                    })
            })
            .collect()
    };
    let center = parse_state(&request.center)?;
    let start = parse_state(&request.start)?;
    let mut expected = center.clone();
    let mut actual = start.clone();
    expected.sort_unstable();
    actual.sort_unstable();
    if actual != expected {
        return Err("start_state multiset must exactly match central_state".into());
    }

    let mut generators = Vec::with_capacity(request.generators.len());
    for row in request.generators {
        if row.len() != size {
            return Err("every native generator must be a complete gather permutation".into());
        }
        let mut seen = vec![false; size];
        let mut generator = Vec::with_capacity(size);
        for value in row {
            let source = value
                .parse::<u32>()
                .ok()
                .filter(|source| (*source as usize) < size)
                .ok_or_else(|| {
                    "every native generator must be a complete gather permutation".to_string()
                })?;
            let slot = &mut seen[source as usize];
            if *slot {
                return Err("every native generator must be a complete gather permutation".into());
            }
            *slot = true;
            generator.push(source);
        }
        generators.push(generator);
    }

    let graph = PermutationGraph {
        center: center.clone(),
        generators: generators.clone(),
    };
    Ok(PublicGraphContract {
        state_len: size,
        num_classes: u16::from(*center.iter().max().expect("non-empty center")) + 1,
        generators,
        center,
        start,
        graph_hash: permutation_graph_hash(&graph),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SearchControlError {
    Invalid(String),
    Unsupported(String),
}

fn tagged_string<'a>(value: &'a TaggedSearchValue, kind: &str) -> Option<&'a str> {
    (value.kind == kind).then(|| value.value.as_str()).flatten()
}

fn integer_in_range(
    value: &TaggedSearchValue,
    name: &str,
    maximum: u64,
    minimum: u64,
) -> std::result::Result<u64, SearchControlError> {
    let message = || {
        format!(
            "{name} must be an integer {}",
            if minimum == 1 { "> 0" } else { ">= 0" }
        )
    };
    let Some(text) = tagged_string(value, "int") else {
        return Err(SearchControlError::Invalid(message()));
    };
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(SearchControlError::Invalid(message()));
    }
    let maximum_text = maximum.to_string();
    if text.len() > maximum_text.len()
        || (text.len() == maximum_text.len() && text > maximum_text.as_str())
    {
        return Err(SearchControlError::Invalid(format!(
            "{name} exceeds native integer range ({maximum})"
        )));
    }
    let parsed = text
        .parse::<u64>()
        .map_err(|_| SearchControlError::Invalid(message()))?;
    if parsed < minimum {
        return Err(SearchControlError::Invalid(message()));
    }
    Ok(parsed)
}

/// Validate the normalized public CayleyPy request without owning its signature adapter.
/// The Python edge tags values so Rust can preserve exact Python type semantics.
pub fn validate_public_search_controls(
    controls: &PublicSearchControls,
) -> std::result::Result<(), SearchControlError> {
    if tagged_string(&controls.beam_width, "str") == Some("max") {
        // The runner still checks whether maximum-width autotuning is enabled.
    } else {
        integer_in_range(&controls.beam_width, "beam_width", u64::MAX, 1)?;
    }
    integer_in_range(&controls.max_steps, "max_steps", u32::MAX.into(), 0)?;

    if tagged_string(&controls.beam_mode, "str") != Some("simple") {
        return Err(SearchControlError::Unsupported(
            "native does not implement the upstream advanced mode; use beam_mode='simple'".into(),
        ));
    }

    let history_is_zero = match controls.history_depth.kind.as_str() {
        "int" => controls.history_depth.value.as_str() == Some("0"),
        "bool" => controls.history_depth.value.as_bool() == Some(false),
        "float" => controls
            .history_depth
            .value
            .as_str()
            .and_then(|value| value.parse::<f64>().ok())
            .is_some_and(|value| value == 0.0),
        _ => false,
    };
    if !history_is_zero {
        return Err(SearchControlError::Unsupported(
            "upstream history_depth is a taboo policy, not native reconstruction history".into(),
        ));
    }
    if controls.destination_state_present {
        return Err(SearchControlError::Unsupported(
            "native adapter currently targets graph.central_state; destination_state is not supported".into(),
        ));
    }
    if controls.bfs_result_for_mitm_present {
        return Err(SearchControlError::Unsupported(
            "upstream BfsResult hashes cannot be reused by native; configure touch_bfs_radius instead".into(),
        ));
    }
    for (name, value) in [
        ("return_path", &controls.return_path),
        ("use_child_scores", &controls.use_child_scores),
    ] {
        if value.kind != "bool" || value.value.as_bool().is_none() {
            return Err(SearchControlError::Invalid(format!("{name} must be bool")));
        }
    }
    Ok(())
}

/// Reserve a suffix of the requested step budget for touch-BFS completion.
///
/// The solved start is handled before launch, so any nontrivial run keeps at
/// least one native forward depth when `max_steps` is nonzero.
pub fn native_depth_budget(max_steps: u64, touch_bfs_radius: u32) -> (u64, u32) {
    if max_steps == 0 {
        return (0, 0);
    }
    let effective_radius = u64::from(touch_bfs_radius).min(max_steps - 1) as u32;
    (max_steps - u64::from(effective_radius), effective_radius)
}

/// Return the duplicate-free geometric touch-BFS bound when it fits the cap.
///
/// Checked multiplication and addition match the shared search-budget contract.
pub fn touch_bfs_worst_case_entries(move_count: u32, radius: u32, max_entries: u64) -> Result<u64> {
    if move_count == 0 || move_count > 255 || radius > 12 || max_entries == 0 {
        return Err(ContractError::Invalid("touch BFS configuration"));
    }
    let moves = u64::from(move_count);
    let mut total = 1_u64;
    let mut frontier = 1_u64;
    for _ in 0..radius {
        frontier = frontier
            .checked_mul(moves)
            .ok_or(ContractError::Overflow("touch BFS frontier"))?;
        total = total
            .checked_add(frontier)
            .ok_or(ContractError::Overflow("touch BFS entries"))?;
        crate::require_capacity("touch BFS entries", total, max_entries)?;
    }
    Ok(total)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TerminalResult {
    pub path: Option<Vec<u32>>,
    pub effective_beam_width: Option<u64>,
    pub metadata: Value,
}
#[derive(Debug, Serialize)]
struct Collected {
    path: Vec<u32>,
    solved_depth: u32,
    touch_depth: u32,
    owner_rank: u32,
}
fn fields(text: &str) -> Result<BTreeMap<&str, &str>> {
    let mut values = BTreeMap::new();
    for token in text.split_whitespace() {
        let (key, value) = token
            .split_once('=')
            .ok_or(ContractError::Invalid("malformed native terminal record"))?;
        if values.insert(key, value).is_some() {
            return Err(ContractError::Invalid("duplicate terminal field"));
        }
    }
    Ok(values)
}
fn value<'a>(fields: &BTreeMap<&str, &'a str>, key: &str) -> Result<&'a str> {
    fields
        .get(key)
        .copied()
        .ok_or(ContractError::Invalid("missing native terminal field"))
}
fn number<T: std::str::FromStr>(fields: &BTreeMap<&str, &str>, key: &str) -> Result<T> {
    value(fields, key)?
        .parse()
        .map_err(|_| ContractError::Invalid("native terminal number"))
}
pub struct TerminalParser<'a> {
    graph: &'a PermutationGraph,
    start: &'a [u8],
    max_steps: Option<u32>,
    path: Option<Option<Vec<u32>>>,
    seconds: f64,
    records: u64,
    effective: BTreeSet<u64>,
    batches: BTreeMap<&'static str, BTreeSet<u64>>,
    collected: Vec<Collected>,
    counts: Vec<u64>,
    status: Option<String>,
}
impl<'a> TerminalParser<'a> {
    pub fn new(
        graph: &'a PermutationGraph,
        start: &'a [u8],
        max_steps: Option<u32>,
    ) -> Result<Self> {
        graph.validate_start(start)?;
        Ok(Self {
            graph,
            start,
            max_steps,
            path: None,
            seconds: 0.,
            records: 0,
            effective: BTreeSet::new(),
            batches: BTreeMap::from([
                ("B_MICRO", BTreeSet::new()),
                ("STREAM1_ROWS_PER_JOB", BTreeSet::new()),
                ("STREAM3_BATCH_CANDIDATES", BTreeSet::new()),
            ]),
            collected: vec![],
            counts: vec![],
            status: None,
        })
    }
    fn solution(&self, row: &BTreeMap<&str, &str>) -> Result<Vec<u32>> {
        let text = value(row, "solution")?;
        let mut moves = Vec::new();
        if !text.is_empty() {
            for token in text.split('.') {
                let id = token
                    .strip_prefix('m')
                    .filter(|s| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()))
                    .ok_or(ContractError::Invalid("synthetic move name"))?;
                moves.push(
                    id.parse::<u32>()
                        .map_err(|_| ContractError::Invalid("solution generator IDs"))?,
                );
            }
        }
        let length = number::<u64>(row, "solution_length")?;
        if length != moves.len() as u64 {
            return Err(ContractError::Invalid("solution length"));
        }
        if self.max_steps.is_some_and(|n| length > n as u64) {
            return Err(ContractError::Invalid(
                "solution exceeds requested max_steps",
            ));
        }
        if !self.graph.replay(self.start, &moves)? {
            return Err(ContractError::Invalid("solution independent replay"));
        }
        Ok(moves)
    }
    pub fn line(&mut self, line: &str) -> Result<()> {
        if line.contains("production_runner_error=") {
            return Err(ContractError::Invalid("native error record"));
        }
        for token in line.split_whitespace() {
            let Some((name, value)) = token.split_once('=') else {
                continue;
            };
            if matches!(
                name,
                "GLOBAL_BEAM_WIDTH_EFFECTIVE" | "global_beam_width_effective"
            ) {
                let width = value
                    .parse::<u64>()
                    .map_err(|_| ContractError::Invalid("effective beam width"))?;
                if width == 0 {
                    return Err(ContractError::Invalid("effective beam width"));
                }
                self.effective.insert(width);
            } else if let Some(values) = self.batches.get_mut(name) {
                values.insert(
                    value
                        .parse()
                        .map_err(|_| ContractError::Invalid("observed native batch"))?,
                );
            }
        }
        if line.starts_with("solve_bucket_solution=1 ") {
            let row = fields(line)?;
            if number::<u32>(&row, "puzzle_id")? != 0 {
                return Err(ContractError::Invalid("collection puzzle"));
            }
            let path = self.solution(&row)?;
            let solved_depth = number::<u32>(&row, "found_depth")?;
            let touch_depth = u32::try_from(path.len())
                .ok()
                .and_then(|n| n.checked_sub(solved_depth))
                .ok_or(ContractError::Invalid("collection found depth"))?;
            self.collected.push(Collected {
                path,
                solved_depth,
                touch_depth,
                owner_rank: number(&row, "owner_rank")?,
            });
        }
        if line.starts_with("solve_bucket_summary=1 ") {
            // The native writer appends an unquoted diagnostic path; it may
            // contain spaces and is not part of the count/identity contract.
            let row = fields(line.split(" result_tsv=").next().unwrap_or(line))?;
            if number::<u32>(&row, "puzzle_id")? != 0 {
                return Err(ContractError::Invalid("collection puzzle"));
            }
            self.counts.push(number(&row, "records")?);
        }
        if let Some(status) = line.strip_prefix("collection_status=") {
            self.status = Some(status.trim().into());
        }
        if let Some(index) = line.find("puzzle_solved=") {
            let row = fields(&line[index..])?;
            let solved = number::<u32>(&row, "puzzle_solved")?;
            let seconds = number::<f64>(&row, "seconds")?;
            if number::<u32>(&row, "puzzle_id")? != 0
                || !seconds.is_finite()
                || seconds < 0.
                || solved > 1
            {
                return Err(ContractError::Invalid("native puzzle/time/status"));
            }
            let path = if solved == 0 {
                if value(&row, "solution_length")? != "-1" || !value(&row, "solution")?.is_empty() {
                    return Err(ContractError::Invalid("inconsistent not-found path"));
                }
                None
            } else {
                Some(self.solution(&row)?)
            };
            if self.path.as_ref().is_some_and(|previous| previous != &path) {
                return Err(ContractError::Invalid("contradictory terminal paths"));
            }
            if self.path.is_none() {
                self.path = Some(path);
            }
            self.seconds = self.seconds.max(seconds);
            self.records = self
                .records
                .checked_add(1)
                .ok_or(ContractError::Overflow("terminal count"))?;
        }
        Ok(())
    }
    pub fn finish(self) -> Result<TerminalResult> {
        let path = self.path.ok_or(ContractError::Invalid(
            "native process returned without a terminal result",
        ))?;
        if self.effective.len() > 1 {
            return Err(ContractError::Invalid("inconsistent effective beam widths"));
        }
        let mut metadata = json!({"native_seconds":self.seconds,"terminal_records":self.records});
        if !self.counts.is_empty() || !self.collected.is_empty() {
            if self.counts != [self.collected.len() as u64] {
                return Err(ContractError::Invalid("collection summary/count mismatch"));
            }
            let mut indices: Vec<_> = (0..self.collected.len()).collect();
            indices.sort_by(|a, b| self.collected[*a].path.cmp(&self.collected[*b].path));
            if indices
                .windows(2)
                .any(|p| self.collected[p[0]].path == self.collected[p[1]].path)
            {
                return Err(ContractError::Invalid("duplicate collected path"));
            }
            if !self.collected.is_empty() != path.is_some()
                || path
                    .as_ref()
                    .is_some_and(|p| !self.collected.iter().any(|row| &row.path == p))
            {
                return Err(ContractError::Invalid("collection and terminal disagree"));
            }
            metadata["collected_solutions"] = serde_json::to_value(self.collected)
                .map_err(|_| ContractError::Invalid("collection encoding"))?;
            metadata["collection_status"] =
                json!(self.status.unwrap_or_else(|| "depth_reached".into()));
        }
        for (key, name) in [
            ("B_MICRO", "observed_parent_batch"),
            ("STREAM1_ROWS_PER_JOB", "observed_inference_rows"),
            (
                "STREAM3_BATCH_CANDIDATES",
                "observed_stream3_batch_candidates",
            ),
        ] {
            let values = &self.batches[key];
            metadata[name] = if values.len() == 1 {
                json!(values.first())
            } else {
                Value::Null
            };
            if values.len() > 1 {
                metadata[format!("{name}_by_rank_values")] = json!(values);
            }
        }
        Ok(TerminalResult {
            path,
            effective_beam_width: self.effective.first().copied(),
            metadata,
        })
    }
}
