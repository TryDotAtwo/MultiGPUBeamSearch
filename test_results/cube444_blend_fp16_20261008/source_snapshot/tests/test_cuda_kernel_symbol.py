"""Decoder grammar tests, not compiled-kernel acceptance."""
import pytest
from tools.cuda_kernel_symbol import template_arguments,describe_demangled_kernel,decode_kernel_symbols


def test_nested_template_parameters_are_not_split_at_inner_commas():
    assert template_arguments('X<A<1, 2>, B<C<3>>, 4>','X')==['A<1, 2>','B<C<3>>','4']


@pytest.mark.parametrize('text',['X<A<1,2>','none','X<'])
def test_malformed_template_rejected(text):
    with pytest.raises(ValueError): template_arguments(text,'X')


@pytest.mark.parametrize('name',['_ZstillEncoded','evil::stream1_transformer_layernorm256_copy_kernel()',
    'void arbitrary_kernel()', 'void cutlass::Kernel<unknown>'])
def test_unknown_symbols_do_not_become_supported_kernels(name):
    with pytest.raises(ValueError): describe_demangled_kernel(name)


def test_known_non_gemm_family_decodes_without_coverage_claim():
    assert describe_demangled_kernel('beam::stream1_transformer_layernorm256_copy_kernel(__half const*)')=={'family':'ln'}


@pytest.mark.parametrize('epilogue,family',[('Identity','gemm_bias'),('ReLu','gemm_relu')])
def test_pipelined_broadcast_mainloop_is_decoded(epilogue,family):
    name=('void cutlass::Kernel2<cutlass::gemm::kernel::GemmWithFusedEpilogue<'
          'MmaPipelined<cutlass::gemm::GemmShape<128, 64, 32>, '
          'cutlass::gemm::GemmShape<64, 32, 32>, '
          'cutlass::gemm::GemmShape<16, 8, 8>>, cutlass::half_t, '
          'cutlass::epilogue::thread::'+epilogue+'<float>, '
          'GemmIdentityThreadblockSwizzle<1>>>(Params)')
    result=describe_demangled_kernel(name)
    assert result['family']==family and result['stages']==2


@pytest.mark.parametrize('dual', [False, True])
def test_fused_input_decoder_preserves_dual_template_argument(dual):
    literal = 'true' if dual else 'false'
    name = ('void beam::stream1_transformer_build_input_layernorm256_generic_kernel<'
            + literal + '>(__half const*)')
    assert describe_demangled_kernel(name) == {'family': 'fused_input', 'dual': dual}


@pytest.mark.parametrize('argument', ['', '1', 'true, false', 'unsupported'])
def test_fused_input_unknown_template_is_not_accepted(argument):
    name = ('void beam::stream1_transformer_build_input_layernorm256_generic_kernel<'
            + argument + '>(__half const*)')
    with pytest.raises(ValueError):
        describe_demangled_kernel(name)


def attention_name(parameters):
    return ('void attention_kernel_batched_impl<AttentionKernel<'
            + ', '.join(parameters) + '>>(AttentionKernelParams)')


@pytest.mark.parametrize('position,replacement', [
    (1, 'cutlass::arch::Sm90'), (2, 'not_a_bool'),
    (3, '0'), (4, '128'), (5, '128'),
    (6, 'true'), (7, 'true'), (8, 'UnknownBatchHook')])
def test_attention_decoder_rejects_unimplemented_specializations(position, replacement):
    # Expected supported scope is precisely the FP16 Sm75/Sm80 no-bias,
    # no-dropout specializations instantiated by stream1_transformer_fmha.cu.
    parameters = ['cutlass::half_t', 'cutlass::arch::Sm80', 'true',
                  '64', '64', '64', 'false', 'false', 'DefaultToBatchHook']
    parameters[position] = replacement
    with pytest.raises(ValueError):
        describe_demangled_kernel(attention_name(parameters))


@pytest.mark.parametrize('symbols',[[],['_Zbad\n_Zother'],['_Z'+'x'*65537],['not-mangled']])
def test_unbounded_or_injected_demangler_input_rejected(symbols):
    with pytest.raises(ValueError): decode_kernel_symbols(symbols)
