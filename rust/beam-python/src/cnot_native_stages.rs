//! Rank-local CNOT stage ownership for the Rust search controller.
//!
//! Tensor storage remains owned by PyTorch/CUDA. Existing ATen operators and
//! rank collectives are invoked on the caller's thread in their former order.
use crate::{cnot_native_batch, cnot_native_checked::checked, cnot_native_exchange};
use beam_runtime::cnot_search::{self, Backend, Edge, Goal, Options, Round};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyTuple},
};
use std::path::PathBuf;

fn required<'py>(value: &Option<Bound<'py, PyAny>>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    value
        .clone()
        .ok_or_else(|| PyRuntimeError::new_err(format!("CNOT stage {name} is unavailable")))
}

fn slice<'py>(
    py: Python<'py>,
    begin: Option<i64>,
    end: Option<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("builtins")?.getattr("slice")?.call1((begin, end))
}

fn column<'py>(tensor: &Bound<'py, PyAny>, index: i64) -> PyResult<Bound<'py, PyAny>> {
    tensor.get_item((slice(tensor.py(), None, None)?, index))
}

struct NativeStages<'py> {
    py: Python<'py>,
    world: Bound<'py, PyAny>,
    root: Bound<'py, PyAny>,
    identity: Bound<'py, PyAny>,
    input: Bound<'py, PyAny>,
    scorer: Bound<'py, PyAny>,
    mode: String,
    audit: bool,
    n: i64,
    words: i64,
    rank: u32,
    size: u32,
    parents: Option<Bound<'py, PyAny>>,
    lex: Option<Bound<'py, PyAny>>,
    closed: Option<Bound<'py, PyAny>>,
    candidates: Option<Bound<'py, PyAny>>,
    owners: Option<Bound<'py, PyAny>>,
    incoming: Option<Bound<'py, PyAny>>,
    children: Option<Bound<'py, PyAny>>,
    meta: Option<Bound<'py, PyAny>>,
    values: Option<Bound<'py, PyAny>>,
    ranks: Option<Bound<'py, PyAny>>,
    local: Option<Bound<'py, PyAny>>,
    new_lex: Option<Bound<'py, PyAny>>,
    device_domain_path: Option<PathBuf>,
    device_domain: Option<beam_native_sys::cnot_device_bridge::DeviceDomain>,
}

impl<'py> NativeStages<'py> {
    fn new(
        py: Python<'py>,
        world: Bound<'py, PyAny>,
        root: Bound<'py, PyAny>,
        identity: Bound<'py, PyAny>,
        input: Bound<'py, PyAny>,
        scorer: Bound<'py, PyAny>,
        mode: String,
        audit: bool,
    ) -> PyResult<Self> {
        let n = root.call_method1("size", (1,))?.extract()?;
        let words = root.call_method1("size", (2,))?.extract()?;
        let rank = world.getattr("rank")?.extract()?;
        let size = world.getattr("size")?.extract()?;
        Ok(Self {
            py,
            world,
            root,
            identity,
            input,
            scorer,
            mode,
            audit,
            n,
            words,
            rank,
            size,
            parents: None,
            lex: None,
            closed: None,
            candidates: None,
            owners: None,
            incoming: None,
            children: None,
            meta: None,
            values: None,
            ranks: None,
            local: None,
            new_lex: None,
            device_domain_path: std::env::var_os("MGBS_CNOT_OWNED_LIBRARY").map(PathBuf::from),
            device_domain: None,
        })
    }

    fn native(&self) -> PyResult<Bound<'py, PyModule>> {
        self.py.import("multigpubeamsearch.cnot_native")
    }

    fn ops(&self, tensor: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if tensor.getattr("is_cuda")?.extract()? {
            Ok(Some(self.native()?.call_method0("_cuda_ops")?))
        } else {
            Ok(None)
        }
    }

    fn ensure_device_domain(&mut self, tensor: &Bound<'py, PyAny>) -> PyResult<bool> {
        if !tensor.getattr("is_cuda")?.extract::<bool>()? {
            return Ok(false);
        }
        let Some(path) = &self.device_domain_path else {
            return Ok(false);
        };
        if self.device_domain.is_none() {
            let device = tensor.getattr("device")?.getattr("index")?.extract()?;
            self.device_domain = Some(beam_native_sys::cnot_device_bridge::DeviceDomain::load(
                path,
                device,
                self.n as i32,
                self.size as i32,
            )?);
        }
        Ok(true)
    }

    fn expand_batch(&mut self, begin: u64, k: u32) -> PyResult<Bound<'py, PyAny>> {
        let parents = required(&self.parents, "parents")?;
        let lex = required(&self.lex, "lex")?;
        if self.ensure_device_domain(&parents)? {
            return cnot_native_batch::expand(
                self.py,
                &self.world,
                &parents,
                &lex,
                begin,
                &self.scorer,
                &self.mode,
                k,
                self.n,
                self.words,
                self.device_domain.as_mut(),
            );
        }
        let ops = self.ops(&parents)?;
        let native = match &ops {
            Some(ops) => ops.hasattr("expand_candidates")?,
            None => false,
        };
        if native {
            let ops =
                ops.ok_or_else(|| PyRuntimeError::new_err("CNOT expansion operators unavailable"))?;
            let count = parents.len()?;
            let alphabet = self.n * (self.n - 1);
            let scores = if count > 0 && i64::from(k) < alphabet {
                let raw = self
                    .scorer
                    .call_method1("parent_scores", (&parents, &self.mode))?;
                cnot_native_batch::scores(
                    self.py,
                    raw,
                    &[count as i64, alphabet],
                    &self.world.getattr("device")?,
                )?
            } else {
                self.py.None().into_bound(self.py)
            };
            ops.call_method1(
                "expand_candidates",
                (
                    parents,
                    lex,
                    scores,
                    begin,
                    self.rank,
                    self.size,
                    k,
                    self.mode == "proposal",
                ),
            )
        } else {
            cnot_native_batch::expand(
                self.py,
                &self.world,
                &parents,
                &lex,
                begin,
                &self.scorer,
                &self.mode,
                k,
                self.n,
                self.words,
                None,
            )
        }
    }

    fn distinct_batch(&mut self) -> PyResult<Bound<'py, PyAny>> {
        let incoming = required(&self.incoming, "incoming")?;
        let closed = required(&self.closed, "closed")?;
        if self.ensure_device_domain(&incoming)? {
            return cnot_native_batch::distinct(
                self.py,
                &self.world,
                &incoming,
                &closed,
                self.n,
                self.words,
                &self.scorer,
                self.device_domain.as_mut(),
            );
        }
        let ops = self.ops(&incoming)?;
        let native = match &ops {
            Some(ops) => ops.hasattr("distinct_candidates")?,
            None => false,
        };
        if native {
            let ops =
                ops.ok_or_else(|| PyRuntimeError::new_err("CNOT distinct operators unavailable"))?;
            let pair = ops.call_method1(
                "distinct_candidates",
                (&incoming, &closed, self.n, self.words),
            )?;
            let children = pair.get_item(0)?;
            let meta = pair.get_item(1)?;
            let values = if children.len()? > 0 {
                let raw = self.scorer.call_method1("child_values", (&children,))?;
                cnot_native_batch::scores(
                    self.py,
                    raw,
                    &[children.len()? as i64],
                    &self.world.getattr("device")?,
                )?
            } else {
                let torch = self.py.import("torch")?;
                let kwargs = PyDict::new(self.py);
                kwargs.set_item("dtype", torch.getattr("float64")?)?;
                kwargs.set_item("device", self.world.getattr("device")?)?;
                torch.call_method("empty", (0,), Some(&kwargs))?
            };
            Ok(PyTuple::new(self.py, [children, meta, values])?.into_any())
        } else {
            cnot_native_batch::distinct(
                self.py,
                &self.world,
                &incoming,
                &closed,
                self.n,
                self.words,
                &self.scorer,
                None,
            )
        }
    }

    fn exchange(
        &self,
        rows: &Bound<'py, PyAny>,
        targets: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        cnot_native_exchange::exchange(self.py, &self.world, rows, targets)
    }

    fn fallback_winner_packet(
        &self,
        ranks: &Bound<'py, PyAny>,
        local: &Bound<'py, PyAny>,
        new_lex: &Bound<'py, PyAny>,
        meta: &Bound<'py, PyAny>,
        children: &Bound<'py, PyAny>,
        closed: &Bound<'py, PyAny>,
        count: usize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let torch = self.py.import("torch")?;
        let slots = ranks
            .call_method1("__eq__", (self.rank,))?
            .call_method0("nonzero")?
            .call_method0("flatten")?;
        let kept = local.get_item(&slots)?;
        let owned = children.get_item(&kept)?;
        let new_closed = torch.call_method1("cat", (PyTuple::new(self.py, [closed, &owned])?,))?;
        let columns = PyTuple::new(
            self.py,
            [
                slots.clone(),
                new_lex.get_item(&slots)?,
                meta.get_item((&kept, 0))?,
                meta.get_item((&kept, 3))?,
                ranks.get_item(&slots)?,
            ],
        )?;
        let records = torch.call_method1("stack", (columns, 1))?;
        let rows = owned.call_method1("reshape", (kept.len()?, self.n * self.words))?;
        let wire = torch.call_method1("cat", (PyTuple::new(self.py, [records, rows])?, 1))?;
        let targets = slots
            .call_method1("__mul__", (self.size,))?
            .call_method1("__floordiv__", (count.max(1),))?;
        Ok((new_closed, wire, targets))
    }
}

impl Backend for NativeStages<'_> {
    type Error = PyErr;
    type Layer = Py<PyAny>;
    type Records = Py<PyAny>;

    fn root_is_goal(&mut self) -> PyResult<bool> {
        self.py
            .import("torch")?
            .call_method1("equal", (&self.root, &self.identity))?
            .extract()
    }

    fn initialize(&mut self) -> PyResult<()> {
        let empty = self.root.call_method1("narrow", (0, 0, 0))?;
        self.parents = Some(if self.rank == 0 {
            self.root.clone()
        } else {
            empty.clone()
        });
        let initial: Vec<i64> = if self.rank == 0 { vec![0] } else { Vec::new() };
        self.lex = Some(self.world.call_method1("longs", (initial,))?);
        let root = self.root.clone();
        let owner = (|| {
            if self.ensure_device_domain(&root)? {
                let domain = self
                    .device_domain
                    .as_mut()
                    .ok_or_else(|| PyRuntimeError::new_err("CNOT device domain is unavailable"))?;
                domain.owners(self.py, &root)?.call_method0("item")
            } else {
                self.native()?
                    .call_method1("cnot_owners", (&root, self.size))?
                    .call_method0("item")
            }
        })();
        let owner: u32 =
            checked(self.py, &self.world, "initialize CNOT domain", owner)?.extract()?;
        self.closed = Some(if owner == self.rank {
            self.root.clone()
        } else {
            empty
        });
        Ok(())
    }

    fn goal(&mut self, begin: u64) -> PyResult<Option<Goal>> {
        let parents = required(&self.parents, "parents")?;
        let lex = required(&self.lex, "lex")?;
        let goal = crate::cnot_native_goal::goal(
            self.py,
            &self.world,
            &parents,
            &lex,
            begin,
            &self.identity,
        )?;
        Ok(goal.map(|(slot, movement)| Goal { slot, movement }))
    }

    fn expand(&mut self, begin: u64, k: u32) -> PyResult<()> {
        let batch = self.expand_batch(begin, k);
        let pair = checked(
            self.py,
            &self.world,
            "rank parent actions and expand",
            batch,
        )?;
        self.candidates = Some(pair.get_item(0)?);
        self.owners = Some(pair.get_item(1)?);
        Ok(())
    }

    fn exchange_candidates(&mut self) -> PyResult<()> {
        self.incoming = Some(self.exchange(
            &required(&self.candidates, "candidates")?,
            &required(&self.owners, "owners")?,
        )?);
        Ok(())
    }

    fn deduplicate_and_score(&mut self) -> PyResult<()> {
        let batch = self.distinct_batch();
        let result = checked(
            self.py,
            &self.world,
            "deduplicate and evaluate distinct children",
            batch,
        )?;
        self.children = Some(result.get_item(0)?);
        self.meta = Some(result.get_item(1)?);
        self.values = Some(result.get_item(2)?);
        Ok(())
    }

    fn select(&mut self, width: u64) -> PyResult<()> {
        let result = crate::cnot_native_selection::select(
            self.py,
            self.world.clone(),
            required(&self.values, "values")?,
            column(&required(&self.meta, "meta")?, 1)?,
            width,
        )?
        .into_bound(self.py);
        self.ranks = Some(result.get_item(0)?);
        self.local = Some(result.get_item(1)?);
        self.new_lex = Some(result.get_item(2)?);
        Ok(())
    }

    fn redistribute(&mut self, depth: u64) -> PyResult<Round<Self::Layer, Self::Records>> {
        let ranks = required(&self.ranks, "ranks")?;
        let local = required(&self.local, "local")?;
        let new_lex = required(&self.new_lex, "new_lex")?;
        let meta = required(&self.meta, "meta")?;
        let children = required(&self.children, "children")?;
        let closed = required(&self.closed, "closed")?;
        let count = local.len()?;
        let ops = self.ops(&children)?;
        let use_native = match &ops {
            Some(ops) => {
                ops.hasattr("winner_packet")?
                    && ops.hasattr("sort_received")?
                    && ops.hasattr("frontier_views")?
            }
            None => false,
        };
        let (new_closed, wire, targets) = if use_native {
            let packet = ops
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("CNOT winner operators unavailable"))?
                .call_method1(
                    "winner_packet",
                    (
                        &ranks, &local, &new_lex, &meta, &children, &closed, self.rank, self.size,
                    ),
                )?;
            (
                packet.get_item(0)?,
                packet.get_item(1)?,
                packet.get_item(2)?,
            )
        } else {
            self.fallback_winner_packet(&ranks, &local, &new_lex, &meta, &children, &closed, count)?
        };
        self.closed = Some(new_closed);
        let received = self.exchange(&wire, &targets)?;
        let received = if use_native {
            ops.as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("CNOT winner operators unavailable"))?
                .call_method1("sort_received", (received,))?
        } else {
            let order = self
                .py
                .import("torch")?
                .call_method1("argsort", (column(&received, 0)?,))?;
            received.get_item(order)?
        };
        let parents = required(&self.parents, "parents")?;
        let candidates = required(&self.candidates, "candidates")?;
        let owners = required(&self.owners, "owners")?;
        let remote: i64 = owners
            .call_method1("__ne__", (self.rank,))?
            .call_method0("sum")?
            .call_method0("item")?
            .extract()?;
        let stat_values = vec![
            parents.len()? as i64,
            candidates.len()? as i64,
            remote,
            children.len()? as i64,
            received.len()? as i64,
        ];
        let stats = self
            .world
            .call_method1(
                "gather_fixed",
                (self.world.call_method1("longs", (stat_values,))?,),
            )?
            .call_method0("tolist")?;
        let rows: Vec<Vec<i64>> = stats.extract()?;
        let distinct = rows.iter().try_fold(0_i64, |sum, row| {
            let count = row
                .get(3)
                .ok_or_else(|| PyRuntimeError::new_err("invalid CNOT rank statistics"))?;
            sum.checked_add(*count)
                .ok_or_else(|| PyRuntimeError::new_err("CNOT rank statistics overflow"))
        })?;
        let layer = PyDict::new(self.py);
        layer.set_item("depth", depth)?;
        layer.set_item("by_rank", stats)?;
        layer.set_item("distinct", distinct)?;
        layer.set_item("retained", count)?;
        if self.audit {
            let torch = self.py.import("torch")?;
            let values = required(&self.values, "values")?
                .call_method1("to", (torch.getattr("float64")?,))?
                .call_method0("contiguous")?
                .call_method1("view", (torch.getattr("int64")?,))?
                .call_method1("unsqueeze", (1,))?;
            let flat = children.call_method1("reshape", (children.len()?, self.n * self.words))?;
            let payload =
                torch.call_method1("cat", (PyTuple::new(self.py, [values, meta, flat])?, 1))?;
            let candidates = self
                .world
                .call_method1("gather_rows", (payload,))?
                .call_method0("cpu")?
                .call_method0("tolist")?;
            let winners = self
                .world
                .call_method1("gather_rows", (&received,))?
                .call_method0("cpu")?
                .call_method0("tolist")?;
            layer.set_item("candidates", candidates)?;
            layer.set_item("winners", winners)?;
        }
        if count == 0 {
            return Ok(Round {
                count: 0,
                layer: layer.into_any().unbind(),
                records: None,
            });
        }
        let (history, next_parents, lex) = if use_native {
            let parts = ops
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("CNOT winner operators unavailable"))?
                .call_method1("frontier_views", (&received, self.n, self.words))?;
            (parts.get_item(0)?, parts.get_item(1)?, parts.get_item(2)?)
        } else {
            let history = received
                .get_item((
                    slice(self.py, None, None)?,
                    slice(self.py, Some(2), Some(5))?,
                ))?
                .call_method0("contiguous")?;
            let next = received
                .get_item((slice(self.py, None, None)?, slice(self.py, Some(5), None)?))?
                .call_method1("reshape", (-1, self.n, self.words))?
                .call_method0("contiguous")?;
            let lex = column(&received, 1)?.call_method0("contiguous")?;
            (history, next, lex)
        };
        self.parents = Some(next_parents);
        self.lex = Some(lex);
        Ok(Round {
            count: count as u64,
            layer: layer.into_any().unbind(),
            records: Some(history.unbind()),
        })
    }

    fn history_edge(
        &mut self,
        records: &Self::Records,
        target: u32,
        local_index: u64,
    ) -> PyResult<Edge> {
        let edge = self.world.call_method1("longs", (vec![0_i64, 0, 0],))?;
        if target == self.rank {
            edge.call_method1("copy_", (records.bind(self.py).get_item(local_index)?,))?;
        }
        if self.size > 1 {
            self.py
                .import("torch.distributed")?
                .call_method1("broadcast", (&edge, target))?;
        }
        let row: Vec<i64> = edge.call_method0("tolist")?.extract()?;
        if row.len() != 3 {
            return Err(PyValueError::new_err("invalid CNOT history edge"));
        }
        Ok(Edge {
            parent: u64::try_from(row[0])
                .map_err(|_| PyValueError::new_err("invalid CNOT history parent"))?,
            route: u32::try_from(row[1])
                .map_err(|_| PyValueError::new_err("invalid CNOT history route"))?,
            owner: u32::try_from(row[2])
                .map_err(|_| PyValueError::new_err("invalid CNOT history owner"))?,
        })
    }

    fn verifies_construction(&mut self, gates: &[u32]) -> PyResult<bool> {
        self.input
            .call_method1("verifies_construction", (gates.to_vec(),))?
            .extract()
    }

    fn invalid(&self, message: &'static str) -> PyErr {
        PyRuntimeError::new_err(message)
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_cnot_native<'py>(
    py: Python<'py>,
    world: Bound<'py, PyAny>,
    root: Bound<'py, PyAny>,
    identity: Bound<'py, PyAny>,
    input: Bound<'py, PyAny>,
    scorer: Bound<'py, PyAny>,
    mode: String,
    audit: bool,
    width: u64,
    shortlist: u32,
    alphabet: u32,
    max_depth: u64,
) -> PyResult<Py<PyAny>> {
    let mut backend = NativeStages::new(py, world, root, identity, input, scorer, mode, audit)?;
    let options = Options {
        rank: backend.rank,
        world: backend.size,
        width,
        shortlist,
        alphabet,
        max_depth,
    };
    crate::cnot::outcome_to_python(py, cnot_search::search(&mut backend, options)?)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_cnot_native, module)?)
}

