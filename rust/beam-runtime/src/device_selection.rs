//! Selection of process-local CUDA device ordinals for the public permutation runner.

use crate::{Error, Result};
use std::collections::BTreeSet;

/// Normalized observation from the Python/Torch adapter. Rust owns the policy;
/// Python only translates framework objects into a stable kind/index pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceObservation {
    pub kind: String,
    pub index: Option<i64>,
}

fn invalid_devices() -> Error {
    Error::Message(
        "native devices must be distinct available CUDA indices (at most 128)".into(),
    )
}

/// Resolve the public device request without importing framework objects.
///
/// The adapter supplies only the visible CUDA count and normalized graph
/// observations. Explicit indices and num_gpus take precedence over graph
/// placement; CPU-only or absent graph placement means all visible GPUs.
pub fn resolve(
    visible_device_count: u32,
    explicit_devices: Option<Vec<i64>>,
    num_gpus: Option<i64>,
    graph_devices: Vec<DeviceObservation>,
) -> Result<Vec<u32>> {
    if explicit_devices.is_some() && num_gpus.is_some() {
        return Err(Error::Message(
            "specify num_gpus or devices, not both".into(),
        ));
    }

    let devices = if let Some(requested) = explicit_devices {
        requested
            .into_iter()
            .map(|device| u32::try_from(device).map_err(|_| invalid_devices()))
            .collect::<Result<Vec<_>>>()?
    } else if let Some(requested) = num_gpus {
        if requested <= 0 {
            return Err(Error::Message("num_gpus must be a positive integer".into()));
        }
        if requested > i64::from(visible_device_count) {
            return Err(Error::Message(
                "num_gpus exceeds the number of visible CUDA devices".into(),
            ));
        }
        (0..requested as u32).collect()
    } else if graph_devices.is_empty()
        || graph_devices.iter().all(|device| device.kind == "cpu")
    {
        (0..visible_device_count).collect()
    } else if graph_devices.iter().any(|device| device.kind != "cuda") {
        return Err(Error::Message(
            "graph mixes CPU and CUDA devices; set explicit native devices".into(),
        ));
    } else {
        graph_devices
            .into_iter()
            .map(|device| {
                let index = device.index.unwrap_or(0);
                u32::try_from(index).map_err(|_| invalid_devices())
            })
            .collect::<Result<Vec<_>>>()?
    };

    let unique = devices.iter().copied().collect::<BTreeSet<_>>();
    if devices.is_empty()
        || devices.len() > 128
        || unique.len() != devices.len()
        || devices.iter().any(|device| *device >= visible_device_count)
    {
        return Err(invalid_devices());
    }
    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::{resolve, DeviceObservation};

    fn device(kind: &str, index: Option<i64>) -> DeviceObservation {
        DeviceObservation {
            kind: kind.into(),
            index,
        }
    }

    #[test]
    fn preserves_explicit_count_and_graph_device_selection() {
        assert_eq!(resolve(2, None, None, vec![]).unwrap(), vec![0, 1]);
        assert_eq!(
            resolve(2, None, None, vec![device("cpu", None)]).unwrap(),
            vec![0, 1]
        );
        assert_eq!(
            resolve(2, None, None, vec![device("cuda", Some(1))]).unwrap(),
            vec![1]
        );
        assert_eq!(
            resolve(2, None, None, vec![device("cuda", None)]).unwrap(),
            vec![0]
        );
        assert_eq!(resolve(2, None, Some(1), vec![]).unwrap(), vec![0]);
        assert_eq!(resolve(2, Some(vec![1]), None, vec![]).unwrap(), vec![1]);
    }

    #[test]
    fn rejects_shrinking_duplicates_mixed_devices_and_invalid_indices() {
        assert!(resolve(2, None, Some(3), vec![])
            .unwrap_err()
            .to_string()
            .contains("num_gpus exceeds"));
        assert!(resolve(2, None, None, vec![device("cpu", None), device("cuda", Some(1))])
            .unwrap_err()
            .to_string()
            .contains("graph mixes CPU and CUDA"));
        assert!(resolve(2, Some(vec![0, 0]), None, vec![]).is_err());
        assert!(resolve(2, None, None, vec![device("cuda", Some(2))]).is_err());
        assert!(resolve(2, None, None, vec![device("cuda", Some(-1))]).is_err());
        assert!(resolve(2, Some(vec![]), None, vec![]).is_err());
    }

    #[test]
    fn enforces_the_existing_device_limit() {
        let selected = resolve(129, None, Some(129), vec![]).unwrap_err();
        assert!(selected.to_string().contains("at most 128"));
    }
}
