"""Bounded CUDA symbol decoding; no expected-profile or replay attestation."""
import os
import re
import subprocess
import tempfile


def template_arguments(text, token):
    start=text.find(token+'<')
    if start<0: raise ValueError('missing kernel template: '+token)
    start+=len(token)+1
    level=0
    arguments=[]
    for position in range(start,len(text)):
        char=text[position]
        if char=='<': level+=1
        elif char=='>':
            if level==0:
                arguments.append(text[start:position].strip())
                return arguments
            level-=1
        elif char==',' and level==0:
            arguments.append(text[start:position].strip())
            start=position+1
    raise ValueError('unterminated kernel template')


def describe_demangled_kernel(name):
    if not isinstance(name,str) or len(name)>262144 or name.startswith('_Z'):
        raise ValueError('undecoded or oversized kernel name')
    non_gemm={
        'stream1_transformer_build_input_layernorm256_generic_kernel':'fused_input',
        'stream1_transformer_layernorm256_copy_kernel':'ln',
        'stream1_transformer_bias_layernorm256_copy_kernel':'bias_ln',
        'stream1_transformer_gather_cls256_kernel':'gather_cls',
        'stream1_transformer_build_input_kernel':'input',
        'stream1_transformer_build_input_kernel_graph_job':'input_graph',
        'stream1_transformer_cls_bias_layernorm_kernel':'cls_ln',
        'stream1_transformer_zero_padded_rows_kernel':'padding',
        'stream1_transformer_zero_padded_rows_legacy_kernel':'padding_legacy',
        'stream1_transformer_score_quantize_graph_job_kernel':'quantize_graph',
        'stream1_transformer_score_quantize_kernel':'quantize_eager'}
    for symbol,family in non_gemm.items():
        if re.match(r'^(?:void )?beam::'+re.escape(symbol)+r'(?:\(|<)',name):
            if family == 'fused_input':
                parameters = template_arguments(name, 'beam::'+symbol)
                if len(parameters) != 1 or parameters[0] not in ('true', 'false'):
                    raise ValueError('unsupported fused-input Dual specialization')
                return dict(family=family, dual=parameters[0] == 'true')
            return dict(family=family)
    if name.startswith('void attention_kernel_batched_impl<AttentionKernel<'):
        parameters=template_arguments(name,'AttentionKernel')
        if len(parameters)!=9 or parameters[0]!='cutlass::half_t':
            raise ValueError('unsupported attention kernel encoding')
        architecture=re.fullmatch(r'cutlass::arch::Sm([0-9]+)',parameters[1])
        if not architecture or any(not re.fullmatch('[0-9]+',v) for v in parameters[3:6]):
            raise ValueError('unsupported attention dimensions')
        if (int(architecture[1]) not in (75, 80) or
                parameters[2] not in ('true', 'false') or
                int(parameters[3]) not in (32, 64) or
                int(parameters[4]) != 64 or int(parameters[5]) not in (32, 64) or
                parameters[6:8] != ['false', 'false'] or
                parameters[8] != 'DefaultToBatchHook'):
            raise ValueError('unsupported attention specialization')
        return dict(family='attention',architecture=int(architecture[1]),
                    tile=list(map(int,parameters[3:6])),parameters=parameters)
    broadcast=name.startswith('void cutlass::Kernel2<cutlass::gemm::kernel::GemmWithFusedEpilogue<')
    regular=name.startswith('void cutlass::Kernel<cutlass::gemm::kernel::Gemm<')
    if not (broadcast or regular): raise ValueError('unknown captured kernel symbol')
    # GNU expands the same kernel again in the trailing ::Params argument.
    # Decode the actual outer template argument, not repeated text globally.
    outer=template_arguments(name,'cutlass::Kernel2' if broadcast else 'cutlass::Kernel')
    if len(outer)!=1: raise ValueError('unsupported outer kernel template')
    name=outer[0]
    shapes=re.findall(r'cutlass::gemm::GemmShape<([0-9]+), ([0-9]+), ([0-9]+)>',name)
    if len(shapes)<3: raise ValueError('missing GEMM tile/warp/instruction')
    tile,warp,instruction=[list(map(int,shape)) for shape in shapes[:3]]
    if 'MmaMultistage<' in name:
        params=template_arguments(name,'MmaMultistage')
        if len(params)<=10 or params[10] not in ('2','3'):
            raise ValueError('unsupported GEMM stage encoding')
        stages=int(params[10])
        family='gemm_residual'
    elif 'MmaPipelined<' in name:
        stages=2
        # SM75 residuals and the output head share this mainloop. Role must be
        # established by independent profile expectations, not this symbol.
        family='gemm_pipelined'
    else: raise ValueError('unknown GEMM mainloop')
    swizzles=re.findall(r'GemmIdentityThreadblockSwizzle<([0-9]+)>',name)
    if len(swizzles)!=1 or swizzles[0] not in ('1','2','4','8'):
        raise ValueError('unsupported GEMM swizzle encoding')
    if 'cutlass::half_t' not in name: raise ValueError('unsupported GEMM dtype')
    if broadcast:
        if 'cutlass::epilogue::thread::ReLu<float>' in name: family='gemm_relu'
        elif 'cutlass::epilogue::thread::Identity<float>' in name: family='gemm_bias'
        else: raise ValueError('unknown fused GEMM epilogue')
    return dict(family=family,tile=tile,warp=warp,instruction=instruction,
                stages=stages,swizzle=int(swizzles[0]))


def decode_kernel_symbols(symbols):
    if os.name!='posix': raise ValueError('bounded demangler requires POSIX')
    import resource
    if (not isinstance(symbols,list) or not 1<=len(symbols)<=65536 or
            any(not isinstance(s,str) or not s.startswith('_Z') or len(s)>65536 or
                any(ord(c)<33 or ord(c)>126 for c in s) for s in symbols)):
        raise ValueError('invalid mangled kernel list')
    encoded=('\n'.join(symbols)+'\n').encode('ascii')
    if len(encoded)>8*1024*1024: raise ValueError('mangled input bound exceeded')
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU,(5,5))
        resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024))
        resource.setrlimit(resource.RLIMIT_STACK,(16*1024*1024,16*1024*1024))
        resource.setrlimit(resource.RLIMIT_FSIZE,(32*1024*1024,32*1024*1024))
    # Default GNU recursion bound leaves real deep CUTLASS names unchanged.
    # Disable that bound only inside this CPU/memory/stack/output-limited child.
    with tempfile.TemporaryFile() as incoming,tempfile.TemporaryFile() as output:
        incoming.write(encoded);incoming.seek(0)
        subprocess.run(['c++filt','--no-recurse-limit'],stdin=incoming,stdout=output,
            stderr=subprocess.DEVNULL,check=True,timeout=10,preexec_fn=limits)
        if output.tell()>32*1024*1024: raise ValueError('demangled output bound exceeded')
        output.seek(0)
        names=output.read().decode('ascii').splitlines()
    if len(names)!=len(symbols): raise ValueError('demangler row count mismatch')
    return [describe_demangled_kernel(name) for name in names]
