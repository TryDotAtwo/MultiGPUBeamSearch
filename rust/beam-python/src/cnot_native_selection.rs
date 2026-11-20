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

    fn checked(
        &self,
        stage: &str,
        result: PyResult<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::cnot_native_checked::checked_compat(self.py, &self.world, stage, result)
    }

    fn checked_local<T>(&self, stage: &str, result: PyResult<T>) -> PyResult<T> {
        let marker = match &result {
            Ok(_) => Ok(self.py.None().into_bound(self.py)),
            Err(error) => Err(error.clone_ref(self.py)),
        };
        self.checked(stage, marker)?;
        result
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

impl<'py> Backend for NativeSelection<'py> {
    type Error = PyErr;
    type Output = Py<PyAny>;

    fn begin(&mut self, width: u64) -> PyResult<u64> {
        let local = self.checked(
            "prepare CNOT selection count",
            (|| -> PyResult<Bound<'py, PyAny>> {
                let local = i64::try_from(self.values.len()?).map_err(|_| {
                    PyValueError::new_err("CNOT selection local count exceeds int64")
                })?;
                self.world.call_method1("longs", (vec![local],))
            })(),
        )?;
        let global = self.world.call_method1("sum", (local,));
        let global = self.checked("sum CNOT selection counts", global)?;
        let count_value = self.checked("read CNOT selection count", global.call_method0("item"))?;
        let count =
            self.checked_local("decode CNOT selection count", count_value.extract::<u64>())?;

        let prepared = (|| -> PyResult<()> {
            self.selection_ops = None;
            self.bits = None;
            self.selected = None;
            self.boundary = None;
            self.score_keys = None;

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
                    return Ok(());
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
                self.score_keys =
                    Some(torch.call_method1("where", (negative, complement, flipped))?);
                self.boundary = Some(boundary);
            }
            self.bits = Some(bits);
            self.selected = Some(selected);
            Ok(())
        })();
        self.checked_local("prepare CNOT selection buffers", prepared)?;
        Ok(count)
    }
    fn radix_pass(&mut self, key: u32, shift: u32, remaining: u64) -> PyResult<(u64, u64)> {
        let prepared = (|| -> PyResult<(
            Bound<'py, PyAny>,
            Bound<'py, PyAny>,
            Bound<'py, PyAny>,
            Bound<'py, PyAny>,
            Option<Bound<'py, PyAny>>,
            bool,
        )> {
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
                    return Ok((digit, hist, selected, boundary, Some(ops), true));
                }
            }

            let torch = self.py.import("torch")?;
            let digit = keys
                .call_method1("__rshift__", (shift,))?
                .call_method1("__and__", (255,))?;
            let kwargs = PyDict::new(self.py);
            kwargs.set_item("minlength", 256)?;
            let hist = torch.call_method("bincount", (digit.get_item(&boundary)?,), Some(&kwargs))?;
            Ok((digit, hist, selected, boundary, None, false))
        })();
        let (digit, hist, selected, boundary, ops, native) =
            self.checked_local("prepare CNOT selection histogram", prepared)?;

        let summed = self.world.call_method1("sum", (&hist,));
        let summed = self.checked("reduce CNOT selection histogram", summed)?;
        // The native CUDA operator historically consumes the in-place reduced
        // histogram. Keep that data flow; the tensor returned by World::sum is
        // used only by the ATen fallback path.
        let global_hist = if native { hist } else { summed };

        let updated = (|| -> PyResult<(u64, u64)> {
            if native {
                let ops = ops.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err("CNOT selection operators unavailable")
                })?;
                return ops
                    .call_method1(
                        "radix_update",
                        (&digit, &global_hist, &selected, &boundary, remaining),
                    )?
                    .extract();
            }

            let torch = self.py.import("torch")?;
            let prefix = torch.call_method1("cumsum", (&global_hist, 0))?;
            let cutoff: usize = torch
                .call_method1("searchsorted", (&prefix, remaining))?
                .call_method0("item")?
                .extract()?;
            let below: u64 = if cutoff == 0 {
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
            let boundary_count = global_hist
                .get_item(cutoff)?
                .call_method0("item")?
                .extract()?;
            Ok((below, boundary_count))
        })();
        self.checked_local("update CNOT selection boundary", updated)
    }

    fn accept_boundary(&mut self) -> PyResult<()> {
        let accepted = (|| -> PyResult<()> {
            required(&self.selected, "selected mask")?
                .call_method1("__ior__", (required(&self.boundary, "boundary")?,))?;
            Ok(())
        })();
        self.checked_local("accept CNOT selection boundary", accepted)
    }

    fn finish(&mut self, count: u64, width: u64) -> PyResult<Self::Output> {
        let records = (|| -> PyResult<Bound<'py, PyAny>> {
            let bits = required(&self.bits, "score bits")?;
            let selected = required(&self.selected, "selected mask")?;
            let torch = self.py.import("torch")?;
            if let Some(ops) = &self.selection_ops {
                ops.call_method1(
                    "selection_records",
                    (
                        &bits,
                        &self.word_keys,
                        &selected,
                        self.world.getattr("rank")?,
                    ),
                )
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
                torch.call_method1("stack", (columns, 1))
            }
        })();
        let records = self.checked("prepare CNOT selected records", records)?;
        let gathered = self.world.call_method1("gather_rows", (records,));
        let records = self.checked("gather CNOT selected records", gathered)?;

        let output = (|| -> PyResult<Bound<'py, PyAny>> {
            if u64::try_from(records.len()?).ok() != Some(count.min(width)) {
                return Err(PyRuntimeError::new_err(
                    "global radix cutoff returned an inconsistent frontier size",
                ));
            }
            if let Some(ops) = &self.selection_ops {
                return ops.call_method1("selection_order", (records,));
            }

            let torch = self.py.import("torch")?;
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
            .into_any())
        })();
        let output = self.checked("select CNOT frontier", output)?;
        Ok(output.unbind())
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
