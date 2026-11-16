//! Rust ownership for the rank-local batches of the existing CNOT cutoff.
//!
//! Tensor storage and the process group remain owned by PyTorch. The same
//! ATen operators, radix schedule and collective order are used as before.
use beam_runtime::cnot_selection::{self, Backend};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyTuple},
};

fn required<'py>(value: &Option<Bound<'py, PyAny>>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    value
        .clone()
        .ok_or_else(|| PyRuntimeError::new_err(format!("CNOT selection {name} is unavailable")))
}

fn all_symbols(ops: &Bound<'_, PyAny>, names: &[&str]) -> PyResult<bool> {
    for name in names {
        if !ops.hasattr(*name)? {
            return Ok(false);
        }
    }
    Ok(true)
}

struct NativeSelection<'py> {
    py: Python<'py>,
    world: Bound<'py, PyAny>,
    values: Bound<'py, PyAny>,
    word_keys: Bound<'py, PyAny>,
    selection_ops: Option<Bound<'py, PyAny>>,
    bits: Option<Bound<'py, PyAny>>,
    selected: Option<Bound<'py, PyAny>>,
    boundary: Option<Bound<'py, PyAny>>,
    score_keys: Option<Bound<'py, PyAny>>,
}

impl<'py> NativeSelection<'py> {
    fn new(
        py: Python<'py>,
        world: Bound<'py, PyAny>,
        values: Bound<'py, PyAny>,
        word_keys: Bound<'py, PyAny>,
    ) -> Self {
        Self {
            py,
            world,
            values,
            word_keys,
            selection_ops: None,
            bits: None,
            selected: None,
            boundary: None,
            score_keys: None,
        }
    }

    fn ops(&self) -> PyResult<Bound<'py, PyAny>> {
        self.py
            .import("multigpubeamsearch.cnot_native")?
            .call_method0("_cuda_ops")
    }

    fn argsort(&self, tensor: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(self.py);
        kwargs.set_item("stable", true)?;
        self.py
            .import("torch")?
            .call_method("argsort", (tensor,), Some(&kwargs))
    }
}

impl Backend for NativeSelection<'_> {
    type Error = PyErr;
    type Output = Py<PyAny>;

    fn begin(&mut self, width: u64) -> PyResult<u64> {
        let local = i64::try_from(self.values.len()?)
            .map_err(|_| PyValueError::new_err("CNOT selection local count exceeds int64"))?;
        let count: u64 = self
            .world
            .call_method1("sum", (self.world.call_method1("longs", (vec![local],))?,))?
            .call_method0("item")?
            .extract()?;
        if self.values.getattr("is_cuda")?.extract()? {
            let ops = self.ops()?;
            if all_symbols(
                &ops,
                &["selection_prepare", "selection_records", "selection_order"],
            )? {
                let parts = ops.call_method1(
                    "selection_prepare",
                    (&self.values, &self.word_keys, count > width),
                )?;
                self.bits = Some(parts.get_item(0)?);
                self.selected = Some(parts.get_item(1)?);
                self.boundary = Some(parts.get_item(2)?);
                self.score_keys = Some(parts.get_item(3)?);
                self.selection_ops = Some(ops);
                return Ok(count);
            }
        }

        let torch = self.py.import("torch")?;
        let scores = self
            .values
            .call_method1("to", (torch.getattr("float64")?,))?
            .call_method1("__add__", (0.0,))?;
        let bits = scores
            .call_method0("contiguous")?
            .call_method1("view", (torch.getattr("int64")?,))?;
        let kwargs = PyDict::new(self.py);
        kwargs.set_item("dtype", torch.getattr("bool")?)?;
        let selected = torch.call_method("ones_like", (&self.word_keys,), Some(&kwargs))?;
        if count > width {
            selected.call_method0("zero_")?;
            let boundary = torch.call_method1("ones_like", (&selected,))?;
            let negative = bits.call_method1("__lt__", (0,))?;
            let complement = bits.call_method0("__invert__")?;
            let flipped = bits.call_method1("__xor__", (i64::MIN,))?;
            self.score_keys = Some(torch.call_method1("where", (negative, complement, flipped))?);
            self.boundary = Some(boundary);
        }
        self.bits = Some(bits);
        self.selected = Some(selected);
        Ok(count)
    }

    fn radix_pass(&mut self, key: u32, shift: u32, remaining: u64) -> PyResult<(u64, u64)> {
        let keys = if key == 0 {
            required(&self.score_keys, "score keys")?
        } else {
            self.word_keys.clone()
        };
        let boundary = required(&self.boundary, "boundary")?;
        let selected = required(&self.selected, "selected mask")?;
        if keys.getattr("is_cuda")?.extract()? {
            let ops = self.ops()?;
            if all_symbols(&ops, &["radix_histogram", "radix_update"])? {
                let pair = ops.call_method1("radix_histogram", (&keys, &boundary, shift))?;
                let digit = pair.get_item(0)?;
                let hist = pair.get_item(1)?;
                self.world.call_method1("sum", (&hist,))?;
                return ops
                    .call_method1("radix_update", (digit, hist, selected, boundary, remaining))?
                    .extract();
            }
        }

        let torch = self.py.import("torch")?;
        let digit = keys
            .call_method1("__rshift__", (shift,))?
            .call_method1("__and__", (255,))?;
        let kwargs = PyDict::new(self.py);
        kwargs.set_item("minlength", 256)?;
        let hist = torch.call_method("bincount", (digit.get_item(&boundary)?,), Some(&kwargs))?;
        let hist = self.world.call_method1("sum", (hist,))?;
        let prefix = torch.call_method1("cumsum", (&hist, 0))?;
        let cutoff: usize = torch
            .call_method1("searchsorted", (&prefix, remaining))?
            .call_method0("item")?
            .extract()?;
        let below = if cutoff == 0 {
            0
        } else {
            prefix
                .get_item(cutoff - 1)?
                .call_method0("item")?
                .extract()?
        };
        let below_digit = digit.call_method1("__lt__", (cutoff,))?;
        let to_add = boundary.call_method1("__and__", (below_digit,))?;
        selected.call_method1("__ior__", (to_add,))?;
        boundary.call_method1("__iand__", (digit.call_method1("__eq__", (cutoff,))?,))?;
        Ok((
            below,
            hist.get_item(cutoff)?.call_method0("item")?.extract()?,
        ))
    }

    fn accept_boundary(&mut self) -> PyResult<()> {
        required(&self.selected, "selected mask")?
            .call_method1("__ior__", (required(&self.boundary, "boundary")?,))?;
        Ok(())
    }

    fn finish(&mut self, count: u64, width: u64) -> PyResult<Self::Output> {
        let bits = required(&self.bits, "score bits")?;
        let selected = required(&self.selected, "selected mask")?;
        let torch = self.py.import("torch")?;
        let records = if let Some(ops) = &self.selection_ops {
            ops.call_method1(
                "selection_records",
                (
                    &bits,
                    &self.word_keys,
                    &selected,
                    self.world.getattr("rank")?,
                ),
            )?
        } else {
            let local = selected.call_method0("nonzero")?.call_method0("flatten")?;
            let columns = PyTuple::new(
                self.py,
                [
                    bits.get_item(&local)?,
                    self.word_keys.get_item(&local)?,
                    torch.call_method1("full_like", (&local, self.world.getattr("rank")?))?,
                    local,
                ],
            )?;
            torch.call_method1("stack", (columns, 1))?
        };
        let records = self.world.call_method1("gather_rows", (records,))?;
        if u64::try_from(records.len()?).ok() != Some(count.min(width)) {
            return Err(PyRuntimeError::new_err(
                "global radix cutoff returned an inconsistent frontier size",
            ));
        }
        if let Some(ops) = &self.selection_ops {
            return Ok(ops.call_method1("selection_order", (records,))?.unbind());
        }
        let by_word = self.argsort(
            &records.get_item((
                self.py
                    .import("builtins")?
                    .getattr("slice")?
                    .call1((self.py.None(),))?,
                1,
            ))?,
        )?;
        let ordered_values = records
            .get_item((&by_word, 0))?
            .call_method0("contiguous")?
            .call_method1("view", (torch.getattr("float64")?,))?;
        let in_order = by_word.get_item(self.argsort(&ordered_values)?)?;
        let lexical_rank = torch.call_method1("empty_like", (&by_word,))?;
        let kwargs = PyDict::new(self.py);
        kwargs.set_item("device", self.world.getattr("device")?)?;
        let indices = torch.call_method("arange", (by_word.len()?,), Some(&kwargs))?;
        lexical_rank.set_item(&by_word, indices)?;
        Ok(PyTuple::new(
            self.py,
            [
                records.get_item((&in_order, 2))?,
                records.get_item((&in_order, 3))?,
                lexical_rank.get_item(&in_order)?,
            ],
        )?
        .into_any()
        .unbind())
    }

    fn invalid(&self, message: &'static str) -> PyErr {
        PyRuntimeError::new_err(message)
    }
}

pub(crate) fn select<'py>(
    py: Python<'py>,
    world: Bound<'py, PyAny>,
    values: Bound<'py, PyAny>,
    word_keys: Bound<'py, PyAny>,
    width: u64,
) -> PyResult<Py<PyAny>> {
    cnot_selection::select(
        &mut NativeSelection::new(py, world, values, word_keys),
        width,
    )
}

#[pyfunction]
fn run_cnot_native_selection<'py>(
    py: Python<'py>,
    world: Bound<'py, PyAny>,
    values: Bound<'py, PyAny>,
    word_keys: Bound<'py, PyAny>,
    width: u64,
) -> PyResult<Py<PyAny>> {
    select(py, world, values, word_keys, width)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_cnot_native_selection, module)?)
}
