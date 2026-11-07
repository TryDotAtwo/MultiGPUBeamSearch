"""Independent FP16 public C/D arithmetic, not complete GEMM admission.

Descriptor must come from an independently compiled, source-bound thread-map
probe, never from captured iterator values. Private A/B Params remain unproved.
"""

def validate_epilogue_iterator(captured, *, row_elements, descriptor):
    if type(row_elements) is not int or not 0 < row_elements <= 2**31-1:
        raise ValueError('unsupported FP16 row geometry')
    groups={'shape':('row','group','cluster','tile'),
        'iterations':('row','group'), 'delta':('row','group','cluster'),
        'count':('row','group')}
    if not isinstance(descriptor,dict) or set(descriptor)!=set(groups):
        raise ValueError('missing independent thread-map descriptor')
    for group,fields in groups.items():
        values=descriptor[group]
        if not isinstance(values,dict) or set(values)!=set(fields):
            raise ValueError('incomplete independent thread-map descriptor')
        minimum=0 if group=='delta' else 1
        if any(type(v) is not int or not minimum<=v<=2**31-1 for v in values.values()):
            raise ValueError('invalid independent thread-map descriptor')
    shape=descriptor['shape'];iterations=descriptor['iterations']
    delta=descriptor['delta'];count=descriptor['count'];stride=row_elements*2
    wanted=dict(stride=stride,
        increment_row=stride*delta['row'],
        increment_group=stride*(delta['group']-delta['row']*(iterations['row']-1)),
        increment_cluster=stride*(delta['cluster']-delta['group']*(iterations['group']-1)
            -delta['row']*(iterations['row']-1)),
        advance_row=stride*shape['row'],
        advance_group=stride*(shape['group']-1)*shape['row']*count['row'],
        advance_cluster=stride*count['group']*shape['group']*count['row']*shape['row'],
        advance_tile=stride*shape['group']*shape['row']*shape['cluster']*shape['tile'])
    if any(not -(2**63)<=v<2**63 for v in wanted.values()):
        raise ValueError('iterator arithmetic exceeds signed64')
    if (not isinstance(captured,dict) or set(captured)!=set(wanted) or
        any(type(captured[k]) is not int or captured[k]!=v for k,v in wanted.items())):
        raise ValueError('wrong captured epilogue iterator value')
    return True


def validate_mainloop_descriptor(descriptor):
    keys={'element_bits','advance_rank','shape_contiguous','shape_strided',
        'iterations_strided','delta_strided'}
    if (not isinstance(descriptor,dict) or set(descriptor)!=keys or
        any(type(v) is not int for v in descriptor.values()) or
        descriptor['element_bits']!=16 or descriptor['advance_rank'] not in (0,1) or
        any(not 1<=descriptor[k]<=2**31-1 for k in keys-{'element_bits','advance_rank'})):
        raise ValueError('unsupported independent FP16 mainloop descriptor')
    return True


def validate_mainloop_iterator(captured, *, row_elements, descriptor):
    if type(row_elements) is not int or not 0<row_elements<=2**31-1:
        raise ValueError('unsupported mainloop row geometry')
    validate_mainloop_descriptor(descriptor)
    step=row_elements*descriptor['delta_strided']*2
    advance=(descriptor['shape_strided']*row_elements if descriptor['advance_rank']
        else descriptor['shape_contiguous'])*2
    wanted=dict(stride=row_elements,increment_strided=step,
        increment_next=advance-(descriptor['iterations_strided']-1)*step,advance=advance)
    if any(not -(2**63)<=v<2**63 for v in wanted.values()):
        raise ValueError('mainloop arithmetic exceeds signed64')
    if (not isinstance(captured,dict) or set(captured)!=set(wanted) or
        any(type(captured[k]) is not int or captured[k]!=v for k,v in wanted.items())):
        raise ValueError('wrong captured mainloop iterator value')
    return True
