//! Rank-local C ABI for a pinned TheQuantumProject checkpoint. Built inside
//! run_config::execution by build.py. Tensor storage belongs to the Python caller.
use super::evaluation::load_dev_checkpoint_on;
use crate::device_feature_plan::DeviceFeaturePlan;
use crate::device_search::{ActionHead, BeamConfig};
use crate::device_state::DeviceStateBatch;
use crate::model::FusionDistanceField;
use crate::run_config::manifest::PinnedArtifact;
use crate::CnotAlphabet;
use serde::Deserialize;
use serde_json::json;
use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::{fs, path::Path, ptr};
use tch::{Device, Kind, Tensor};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    training_manifest: PinnedArtifact,
    training_result: PinnedArtifact,
    dev_indices: Vec<usize>,
    #[serde(rename = "search")]
    _search: BeamConfig,
    #[serde(rename = "head")]
    _head: ActionHead,
}

struct Scorer {
    plan: DeviceFeaturePlan,
    model: FusionDistanceField,
    alphabet: CnotAlphabet,
    positions: Tensor,
    metadata: CString,
    device: usize,
    capacity: i64,
}

impl Scorer {
    fn open(path: &Path, n: u32, device: usize) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        let spec: Manifest = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if spec.schema != "cnot103_dev_search_v1" {
            return Err("expected a pinned cnot103_dev_search_v1 manifest".into());
        }
        let mut restored = load_dev_checkpoint_on(
            path,
            &spec.training_manifest,
            &spec.training_result,
            &spec.dev_indices,
            Some(device),
        )?;
        let plan = restored
            .field
            .plans
            .remove(&n)
            .ok_or("no bound feature plan for this width")?;
        let capacity = plan.search_packet_capacity(n)?;
        if restored.field.selected_count != 102 || capacity < 1 || device != restored.device {
            return Err("the complete trained field must admit a live row on this device".into());
        }
        let alphabet = CnotAlphabet::all_to_all_unit(n).map_err(|e| e.to_string())?;
        let indices: Vec<_> = alphabet
            .gates()
            .iter()
            .map(|g| i64::from(g.control) * i64::from(n) + i64::from(g.target))
            .collect();
        let positions = Tensor::from_slice(&indices).to_device(Device::Cuda(device));
        let feature_ids: Vec<_> = crate::schema::all_model_input_specs()
            .iter()
            .map(|spec| spec.id.to_string())
            .collect();
        let metadata = json!({
            "scorer_contract": "dev_v1",
            "upstream_commit": "c5b3e3f93b45aa7e533fd2c06209c28ce516ffeb",
            "checkpoint_sha256": restored.receipt.sha256,
            "feature_registry_sha256": restored.field.registry.contract_sha256(),
            "training_manifest_sha256": spec.training_manifest.sha256,
            "training_result_sha256": spec.training_result.sha256,
            "scorer_output_dtype": "float32",
            "dimension": n, "cuda_device": device, "feature_model_inputs": feature_ids.len(),
            "feature_ids": feature_ids, "max_packet_rows": capacity,
        });
        Ok(Scorer {
            plan,
            model: restored.field.model,
            alphabet,
            positions,
            device,
            capacity,
            metadata: CString::new(metadata.to_string()).map_err(|e| e.to_string())?,
        })
    }

    // All pointers address live, contiguous storage on this CUDA device.
    unsafe fn score(
        &self,
        head: u32,
        batch: i64,
        rows: *const u8,
        output: *mut u8,
        votes: *mut u8,
    ) -> Result<(), String> {
        if batch < 1
            || batch > self.capacity
            || rows.is_null()
            || output.is_null()
            || votes.is_null()
        {
            return Err("invalid scorer storage or packet size".into());
        }
        let device = Device::Cuda(self.device);
        let n = i64::from(self.alphabet.dimension());
        let shape = [batch, n, (n + 63) / 64];
        let borrowed = unsafe { Tensor::f_from_blob(rows, &shape, &[], Kind::Int64, device) }
            .map_err(|e| e.to_string())?;
        let states = DeviceStateBatch::from_packed_tensor_for_alphabet(
            borrowed,
            &self.alphabet,
            self.device,
        )?;
        let packet = self.plan.build(&states)?;
        let coverage = packet.first_coverage_failure().map_err(|e| e.to_string())?;
        let heads = self
            .model
            .forward_device(&packet)
            .map_err(|e| e.to_string())?;
        let scores = match head {
            0 => heads.qhat.flatten(1, -1).index_select(1, &self.positions),
            1 => heads
                .proposal_logits
                .flatten(1, -1)
                .index_select(1, &self.positions),
            2 => heads.value.view([batch]),
            _ => return Err("unknown scorer head".into()),
        };
        let mut target =
            unsafe { Tensor::f_from_blob(output, &scores.size(), &[], Kind::Float, device) }
                .map_err(|e| e.to_string())?;
        let mut status = unsafe { Tensor::f_from_blob(votes, &[batch], &[], Kind::Int64, device) }
            .map_err(|e| e.to_string())?;
        target.f_copy_(&scores).map_err(|e| e.to_string())?;
        status.f_copy_(&coverage).map_err(|e| e.to_string())?;
        Ok(())
    }
}

thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::default()); }

fn boundary<T>(failed: T, f: impl FnOnce() -> Result<T, String>) -> T {
    let err = match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => return value,
        Ok(Err(error)) => error,
        Err(_) => "native CNOT field panicked".into(),
    };
    ERROR.with(|slot| *slot.borrow_mut() = CString::new(err.replace('\0', "?")).unwrap());
    failed
}

#[no_mangle]
pub extern "C" fn mgbs_tqp_abi_version() -> u32 {
    1
}

#[no_mangle]
pub extern "C" fn mgbs_tqp_error() -> *const c_char {
    ERROR.with(|slot| slot.borrow().as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn mgbs_tqp_open(path: *const c_char, n: u32, device: u32) -> *mut c_void {
    boundary(ptr::null_mut(), || {
        if path.is_null() {
            return Err("missing scorer manifest".into());
        }
        let path = unsafe { CStr::from_ptr(path) }
            .to_str()
            .map_err(|e| e.to_string())?;
        let scorer = tch::no_grad(|| Scorer::open(Path::new(path), n, device as usize))?;
        Ok(Box::into_raw(Box::new(scorer)).cast())
    })
}

#[no_mangle]
pub unsafe extern "C" fn mgbs_tqp_metadata(handle: *const c_void) -> *const c_char {
    boundary(ptr::null(), || {
        let scorer = unsafe { handle.cast::<Scorer>().as_ref() }.ok_or("null scorer")?;
        Ok(scorer.metadata.as_ptr())
    })
}

#[no_mangle]
pub unsafe extern "C" fn mgbs_tqp_score(
    handle: *const c_void,
    head: u32,
    batch: i64,
    rows: *const u8,
    output: *mut u8,
    votes: *mut u8,
) -> i32 {
    boundary(-1, || {
        let scorer = unsafe { handle.cast::<Scorer>().as_ref() }.ok_or("null scorer")?;
        tch::no_grad(|| unsafe { scorer.score(head, batch, rows, output, votes) })?;
        Ok(0)
    })
}

/// Score a complete borrowed packet using the existing fixed-size scorer.
/// All buffers belong to the caller and must address the scorer's CUDA device.
/// Byte extents are checked before the first sub-packet is submitted.
#[no_mangle]
pub unsafe extern "C" fn mgbs_tqp_score_many(
    handle: *const c_void,
    head: u32,
    batch: i64,
    batch_limit: i64,
    rows: *const u8,
    rows_bytes: usize,
    output: *mut u8,
    output_bytes: usize,
    votes: *mut u8,
    votes_bytes: usize,
) -> i32 {
    boundary(-1, || {
        let scorer = unsafe { handle.cast::<Scorer>().as_ref() }.ok_or("null scorer")?;
        if head > 2 || batch < 0 || batch_limit < 1 || batch_limit > scorer.capacity {
            return Err("invalid scorer head, packet size or batch limit".into());
        }
        if batch == 0 {
            return Ok(0);
        }
        if rows.is_null() || output.is_null() || votes.is_null() {
            return Err("invalid scorer storage".into());
        }
        let count = usize::try_from(batch).map_err(|_| "scorer packet size overflow")?;
        let n = usize::try_from(scorer.alphabet.dimension()).map_err(|_| "CNOT dimension overflow")?;
        if n < 2 {
            return Err("invalid CNOT dimension".into());
        }
        let words = n.checked_add(63).ok_or("CNOT row width overflow")? / 64;
        let row_stride = n.checked_mul(words).and_then(|v| v.checked_mul(8))
            .ok_or("CNOT row byte stride overflow")?;
        let columns = if head == 2 { 1 } else {
            n.checked_mul(n.checked_sub(1).ok_or("invalid CNOT dimension")?)
                .ok_or("CNOT score width overflow")?
        };
        let output_stride = columns.checked_mul(4).ok_or("CNOT output stride overflow")?;
        for (name, stride, available, address) in [
            ("rows", row_stride, rows_bytes, rows as usize),
            ("output", output_stride, output_bytes, output as usize),
            ("votes", 8usize, votes_bytes, votes as usize),
        ] {
            let required = count.checked_mul(stride).ok_or("CNOT packet byte size overflow")?;
            if available < required {
                return Err(format!("CNOT {name} capacity exceeded: required={required} available={available}"));
            }
            address.checked_add(required - 1).ok_or("CNOT device pointer offset overflow")?;
        }
        let limit = usize::try_from(batch_limit).map_err(|_| "scorer batch limit overflow")?;
        tch::no_grad(|| {
            let mut start = 0usize;
            while start < count {
                let size = std::cmp::min(limit, count - start);
                // These are borrowed CUDA addresses, never Rust references. The
                // full checked extent above bounds every sub-packet offset.
                unsafe {
                    scorer.score(head, size as i64,
                        rows.wrapping_add(start * row_stride),
                        output.wrapping_add(start * output_stride),
                        votes.wrapping_add(start * 8))?;
                }
                start += size;
            }
            Ok::<(), String>(())
        })?;
        Ok(0)
    })
}

#[no_mangle]
pub unsafe extern "C" fn mgbs_tqp_close(handle: *mut c_void) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle.cast::<Scorer>()) });
    }
}
