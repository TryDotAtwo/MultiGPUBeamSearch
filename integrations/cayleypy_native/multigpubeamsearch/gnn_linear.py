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

def _gat_reference(left, right, edge, attention, bias, indices, types):
    if left.is_cuda:
        raise RuntimeError('CUTLASS GNN aggregation requires the native C++ runtime')
    source,target=indices[0],indices[1]
    logits=(F.leaky_relu(left[source]+right[target]+edge[types],.2)*attention).sum(-1).float()
    destinations=target[:,None].expand(-1,4)
    maximum=torch.full((left.size(0),4),-float('inf'),device=left.device)
    maximum.scatter_reduce_(0,destinations,logits,reduce='amax',include_self=True)
    numerator=torch.exp(logits-maximum[target])
    denominator=torch.zeros_like(maximum).index_add(0,target,numerator)
    weights=(numerator/denominator[target]).to(left.dtype)
    return torch.zeros_like(left).index_add(0,target,left[source]*weights[:,:,None]).mean(1)+bias

_library = None
if not hasattr(torch.ops.multigpubeamsearch_gnn, 'linear'):
    _library = torch.library.Library('multigpubeamsearch_gnn', 'DEF')
    _library.define('linear(Tensor input, Tensor weight, Tensor? bias, bool cutlass) -> Tensor')
    _library.impl('linear', _reference, 'CompositeExplicitAutograd')
    _library.define('gat(Tensor left, Tensor right, Tensor edge, Tensor attention, Tensor bias, Tensor indices, Tensor types) -> Tensor')
    _library.impl('gat',_gat_reference,'CompositeExplicitAutograd')

class NativeLinear(nn.Linear):
    use_cutlass: bool

    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        super().__init__(in_features, out_features, bias)
        self.use_cutlass = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.ops.multigpubeamsearch_gnn.linear(input, self.weight, self.bias, self.use_cutlass)
