//! Incremental terminal receipts; every returned path is replayed independently.
pub use crate::permutation::{search_budget, PermutationGraph, SearchBudget};
use crate::ContractError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
type Result<T> = std::result::Result<T, ContractError>;
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
