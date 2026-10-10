"""Serialized GNN projection boundary, implemented by LibTorch/CUTLASS in C++."""
import torch
from torch import nn
from torch.nn import functional as F

# Keep registrations alive for TorchScript export. Native execution registers
# the same schema in C++; exporting never labels a Python fallback as CUTLASS.
def _reference(input, weight, bias, cutlass):
    if cutlass:
        raise RuntimeError('CUTLASS GNN projections require the native C++ runtime')
    return F.linear(input, weight, bias)

_library = None
if not hasattr(torch.ops.multigpubeamsearch_gnn, 'linear'):
    _library = torch.library.Library('multigpubeamsearch_gnn', 'DEF')
    _library.define('linear(Tensor input, Tensor weight, Tensor? bias, bool cutlass) -> Tensor')
    _library.impl('linear', _reference, 'CompositeExplicitAutograd')

class NativeLinear(nn.Linear):
    use_cutlass: bool

    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        super().__init__(in_features, out_features, bias)
        self.use_cutlass = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.ops.multigpubeamsearch_gnn.linear(input, self.weight, self.bias, self.use_cutlass)
