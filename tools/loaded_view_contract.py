"""Narrow loaded-view check; not kernel/tensor attestation or production admission."""
import math
import re


def validate_visible_device_map(probe):
    """Validate the explicit CUDA-visible ordinal/UUID map, not quality."""
    if (not isinstance(probe, dict) or set(probe) != {'schema_version', 'scope', 'devices', 'production_quality_accepted'} or
            type(probe['schema_version']) is not int or probe['schema_version'] != 1 or
            probe['scope'] != 'observed_cuda_visible_devices_not_quality' or
            probe['production_quality_accepted'] is not False):
        raise ValueError('invalid visible-device probe schema')
    devices = probe['devices']
    if not isinstance(devices, list) or not 1 <= len(devices) <= 1024:
        raise ValueError('invalid visible-device count')
    uuids = set()
    for ordinal, row in enumerate(devices):
        if (not isinstance(row, dict) or set(row) != {'device', 'sm', 'uuid_hex'} or
                type(row['device']) is not int or row['device'] != ordinal or
                type(row['sm']) is not int or row['sm'] < 75 or
                not isinstance(row['uuid_hex'], str) or
                not re.fullmatch('[0-9a-f]{32}', row['uuid_hex']) or
                row['uuid_hex'] == '0' * 32 or row['uuid_hex'] in uuids):
            raise ValueError('invalid visible-device identity')
        uuids.add(row['uuid_hex'])
    return devices


def validate_reference_device_binding(probe, identity, shape):
    """Compare independent visible-device query with current producer identity."""
    devices = validate_visible_device_map(probe)
    if (not isinstance(identity, dict) or set(identity) != {'schema_version', 'device', 'sm', 'uuid_hex', 'production_quality_accepted'} or
            type(identity['schema_version']) is not int or identity['schema_version'] != 1 or
            type(identity['device']) is not int or not 0 <= identity['device'] < len(devices) or
            identity['production_quality_accepted'] is not False):
        raise ValueError('invalid reference device identity')
    actual = devices[identity['device']]
    if type(identity['sm']) is not int or identity['sm'] != actual['sm'] or identity['uuid_hex'] != actual['uuid_hex']:
        raise ValueError('reference device does not match independent probe')
    validate_reference_execution_shape(shape, device=actual['device'], sm=actual['sm'])
    return {'reference_device_matches': True, 'production_quality_accepted': False}


def validate_reference_execution_shape(observed, *, device, sm):
    """Bind the eight-row reference forward only, never the later lane sweep."""
    expected = dict(schema_version=1, scope='execution_shape_not_kernel_admission',
        outer_microbatch=8, transformer_microbatch=8, lanes=1,
        device=device, sm=sm, production_quality_accepted=False)
    if type(device) is not int or device < 0 or type(sm) is not int or sm < 75:
        raise ValueError('invalid expected reference device identity')
    integers = ('schema_version', 'outer_microbatch', 'transformer_microbatch', 'lanes', 'device', 'sm')
    if (not isinstance(observed, dict) or observed != expected or
            any(type(observed.get(key)) is not int for key in integers) or
            type(observed.get('production_quality_accepted')) is not bool):
        raise ValueError('reference execution shape mismatch')
    return {'reference_shape_matches': True, 'production_quality_accepted': False}


def validate_cube4_diagnostic_view(observed, *, compact57=False):
    if type(compact57) is not bool:
        raise ValueError('compact57 policy must be Boolean')
    # Independent fixed native enum mapping: FP16=0, ReLU=1.
    expected = dict(state_len=96, num_classes=6, num_pieces=56, max_piece_size=3,
        seq_len=57, padded_seq_len=57 if compact57 else 64,
        sequence_alignment=1 if compact57 else 16, d_model=256,
        nhead=8, head_dim=32, transformer_layers=4, ff_dim=1024, output_dim=24,
        dtype=0, activation=1)
    if not isinstance(observed, dict) or set(observed) != set(expected) | {'fp16_layouts', 'block_scales'}:
        raise ValueError('diagnostic loaded view schema mismatch')
    for name, value in expected.items():
        if type(observed[name]) is not int or observed[name] != value:
            raise ValueError('diagnostic loaded view differs: ' + name)
    layout_names = {'qkv_packed_fp16', 'ff1_packed_fp16', 'ff2_packed_fp16'}
    scale_names = {'qkv_e4m3_scale', 'ff1_e4m3_scale', 'ff2_e4m3_scale'}
    for group, names in (('fp16_layouts', layout_names), ('block_scales', scale_names)):
        rows = observed[group]
        if not isinstance(rows, list) or len(rows) != 4:
            raise ValueError('diagnostic loaded block count mismatch')
        for index, row in enumerate(rows):
            if (not isinstance(row, dict) or set(row) != names | {'block'} or
                    type(row['block']) is not int or row['block'] != index):
                raise ValueError('diagnostic loaded block identity mismatch')
            for name in names:
                value = row[name]
                if group == 'fp16_layouts':
                    valid = type(value) is bool and value is False
                else:
                    # Zero only: this diagnostic forbids all E4M3 selectors.
                    valid = type(value) in (int, float) and value == 0 and math.isfinite(value)
                if not valid:
                    raise ValueError('unbound diagnostic loaded field: ' + name)
    return {'diagnostic_view_matches': True, 'production_quality_accepted': False}
