use beam_core::search::PublicGraphContract;
use std::path::Path;

use crate::Result;

/// Write the runner's legacy task files from the validated Rust graph contract.
pub fn write_legacy_search_inputs(contract: &PublicGraphContract, run_dir: &Path) -> Result<()> {
    let center = contract
        .center
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let generators = contract
        .generators
        .iter()
        .enumerate()
        .map(|(index, generator)| {
            let values = generator
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("\"m{index}\": [{values}]")
        })
        .collect::<Vec<_>>()
        .join(", ");

    let mut puzzle_info = String::new();
    puzzle_info.push_str("{\"central_state\": [");
    puzzle_info.push_str(&center);
    puzzle_info.push_str("], \"generators\": {");
    puzzle_info.push_str(&generators);
    puzzle_info.push_str("}}\n");

    let start = contract
        .start
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let test_csv = format!("initial_state_id,initial_state\n0,\"{start}\"\n");

    std::fs::write(run_dir.join("puzzle_info.json"), puzzle_info)?;
    std::fs::write(run_dir.join("test.csv"), test_csv)?;
    Ok(())
}
