"""Remote-only diagnostic validation of the two owned CUPTI captures."""
import json
from pathlib import Path
from cupti_replay_trace import validate_replay_trace

root = Path('/workspace/cube4_54295544')
reports = []
for device in (0, 1):
    capture = Path('/workspace/cupti_replay_54295544') / f'pilot_gpu{device}'
    trace = (root / 'evidence' / f'cupti_replay_pilot_gpu{device}.log').read_text()
    inventory = json.loads((capture / 'graph_kernel_inventory.json').read_text())
    transfers = json.loads((capture / 'graph_transfers.json').read_text())
    report = validate_replay_trace(trace, inventory, transfers, device_ordinal=device)
    rejected = 0
    for broken in (trace + '\nDropped 1 records', trace.replace('cudaGraphLaunch_v10000', 'unobservedLaunch'), trace + '\nCUPTI_ERROR_UNKNOWN'):
        try:
            validate_replay_trace(broken, inventory, transfers, device_ordinal=device)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('corrupted activity trace accepted')
    reports.append(dict(device=device, rejected_corruptions=rejected, **report))
result = dict(reports=reports, numeric_comparison_checked=False, parameter_bytes_observed=False, production_admitted=False)
(root / 'evidence' / 'cupti_pilots_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
