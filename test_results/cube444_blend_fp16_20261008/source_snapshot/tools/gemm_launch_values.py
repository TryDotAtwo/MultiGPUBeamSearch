"""Independent canonical classic GEMM launch fields; not full Params admission.

Pinned CUTLASS 0b55a2f: identity swizzle<1>, one K partition, FP16
row-major reductions 256/1024 (no alignment rounding changes those sizes).
The caller must independently validate problem shape and kernel descriptor.
"""

def validate_epilogue_override_endpoints(value,*,broadcast):
    if type(broadcast) is not bool:raise ValueError('invalid epilogue family')
    keys={'alpha','beta'} if broadcast else {'alpha','beta','alpha_array','beta_array'}
    if not isinstance(value,dict) or set(value)!=keys:
        raise ValueError('missing GEMM scalar override endpoints')
    for endpoint in value.values():
        if (not isinstance(endpoint,dict) or set(endpoint)!={'role','offset','available_bytes'} or
            endpoint['role']!='null' or type(endpoint['offset']) is not int or endpoint['offset']!=0 or
            type(endpoint['available_bytes']) is not int or endpoint['available_bytes']!=0):
            raise ValueError('noncanonical GEMM scalar override endpoint')



def validate_gemm_launch_values(node, *, rows, columns, reduction, tile, broadcast):
    if (type(rows) is not int or not 0 < rows <= 65536 * 64 or
            type(columns) is not int or columns not in (24, 256, 512, 768, 1024) or
            type(reduction) is not int or reduction not in (256, 1024) or
            type(broadcast) is not bool or not isinstance(tile, (tuple, list)) or
            len(tile) != 3 or any(type(v) is not int for v in tile) or
            tuple(tile) not in ((128, 64, 32), (128, 128, 32))):
        raise ValueError('unsupported canonical GEMM launch contract')
    expected = dict(grid_tiled_shape=dict(m=(rows + tile[0] - 1) // tile[0],
                                         n=(columns + tile[1] - 1) // tile[1], k=1),
                    swizzle_log_tile=0, gemm_k_size=reduction, semaphore_null=True)
    if broadcast:
        expected.update(mode=0, batch_count=1, batch_stride_D=rows * columns)
    actual = node.get('launch_scalars') if isinstance(node, dict) else None
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError('missing or unknown GEMM launch values')
    for field, wanted in expected.items():
        value = actual[field]
        if isinstance(wanted, dict):
            if (not isinstance(value, dict) or set(value) != set(wanted) or
                    any(type(value[k]) is not int or value[k] != v for k, v in wanted.items())):
                raise ValueError('wrong GEMM tiled grid')
        elif type(value) is not type(wanted) or value != wanted:
            raise ValueError('wrong GEMM launch value: ' + field)
    return True
