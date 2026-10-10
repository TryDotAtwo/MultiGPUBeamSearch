"""Our tensor-native two-level pancake GNN; no PyG inference dependency.

Weight names match the issue #5 architecture. The two element graphs, GATv2
attention, readout, neighbor expansion and hop attention retain its semantics.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .gnn_linear import NativeLinear


class GATv2(nn.Module):
    __constants__ = ['heads', 'channels']

    def __init__(self, dim: int, edge_dim: int):
        super().__init__()
        self.heads = 4
        self.channels = dim
        self.lin_l = NativeLinear(dim, 4 * dim)
        self.lin_r = NativeLinear(dim, 4 * dim)
        self.lin_edge = NativeLinear(edge_dim, 4 * dim, bias=False)
        self.att = nn.Parameter(torch.empty(1, 4, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.att)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor,
                attributes: torch.Tensor) -> torch.Tensor:
        left = self.lin_l(nodes).reshape(-1, self.heads, self.channels)
        right = self.lin_r(nodes).reshape(-1, self.heads, self.channels)
        source, target = edges[0], edges[1]
        edge = self.lin_edge(attributes).reshape(-1, self.heads, self.channels)
        logits = (F.leaky_relu(left[source] + right[target] + edge, .2) * self.att).sum(-1).float()
        indices = target[:, None].expand(-1, self.heads)
        maxima = torch.full((nodes.size(0), self.heads), -float('inf'), device=nodes.device)
        maxima.scatter_reduce_(0, indices, logits, reduce='amax', include_self=True)
        numerators = torch.exp(logits - maxima[target])
        denominators = torch.zeros_like(maxima).index_add(0, target, numerators)
        weights = (numerators / denominators[target]).to(left.dtype)
        messages = left[source] * weights[:, :, None]
        result = torch.zeros_like(left).index_add(0, target, messages)
        return result.mean(1) + self.bias


class DualStreamEncoder(nn.Module):
    __constants__ = ['n', 'd_model', 'stream_dim']

    def __init__(self, n: int, dim: int, layers: int, dropout: float):
        super().__init__()
        self.n, self.d_model, self.stream_dim = n, dim, dim // 2
        d = dim // 2
        self.value_node_emb = nn.Sequential(NativeLinear(2,d),nn.GELU(),NativeLinear(d,d))
        self.position_node_emb = nn.Sequential(NativeLinear(2,d),nn.GELU(),NativeLinear(d,d))
        self.value_edge_emb, self.pos_edge_emb = nn.Embedding(3,dim),nn.Embedding(3,dim)
        self.value_convs = nn.ModuleList([GATv2(d,dim) for _ in range(layers)])
        self.pos_convs = nn.ModuleList([GATv2(d,dim) for _ in range(layers)])
        self.value_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.pos_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        a = torch.arange(n-1)
        source = torch.cat((torch.stack((a,a+1),1).flatten(),torch.arange(n)))
        target = torch.cat((torch.stack((a+1,a),1).flatten(),torch.arange(n)))
        self.register_buffer('value_edge_index_template',torch.stack((source,target)),persistent=False)
        types = torch.cat((torch.zeros(n-1,dtype=torch.long),torch.ones(n-1,dtype=torch.long),
                           torch.full((n,),2,dtype=torch.long)))
        value_types = torch.cat((torch.tensor([0,1]).repeat(n-1),torch.full((n,),2,dtype=torch.long)))
        self.register_buffer('value_edge_type_template',value_types,persistent=False)
        self.register_buffer('pos_edge_type_template',types.clone(),persistent=False)
        self.readout_proj = nn.Sequential(NativeLinear(dim*6,dim),nn.GELU(),
                                          nn.Dropout(dropout),NativeLinear(dim,dim))
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        batch = states.size(0)
        values = torch.arange(self.n,device=states.device)
        positions = torch.empty((batch,self.n),device=states.device)
        positions.scatter_(1,states,values.float()[None,:].expand(batch,-1))
        ids = values.float()[None,:].expand(batch,-1)
        vf = torch.stack((ids,positions),-1) / max(1,self.n-1)
        pf = torch.stack((positions,ids),-1) / max(1,self.n-1)
        hv = self.value_node_emb(vf.to(self.value_node_emb[0].weight.dtype)).reshape(-1,self.stream_dim)
        hp = self.position_node_emb(pf.to(self.position_node_emb[0].weight.dtype)).reshape(-1,self.stream_dim)
        offsets = torch.arange(batch,device=states.device).repeat_interleave(3*self.n-2)*self.n
        ve = self.value_edge_index_template.repeat(1,batch)+offsets[None,:]
        pe = torch.cat((torch.stack((states[:,:-1],states[:,1:]),-1),
                        torch.stack((states[:,1:],states[:,:-1]),-1),
                        torch.stack((values,values),-1)[None,:,:].expand(batch,-1,-1)),1)
        pe = pe.reshape(-1,2).t().contiguous()+offsets[None,:]
        va = self.value_edge_emb(self.value_edge_type_template.repeat(batch))
        pa = self.pos_edge_emb(self.pos_edge_type_template.repeat(batch))
        for vc,vn,pc,pn in zip(self.value_convs,self.value_norms,self.pos_convs,self.pos_norms):
            hv = F.gelu(vn(vc(hv,ve,va)+hv))
            hp = F.gelu(pn(pc(hp,pe,pa)+hp))
        h = torch.cat((hv,hp),-1).reshape(batch,self.n,self.d_model)
        rows = torch.arange(batch,device=states.device)
        special = torch.cat((h[rows,states[:,0]],h[rows,states[:,-1]],h[:,0],h[:,-1]),-1)
        return self.out_norm(self.readout_proj(torch.cat((h.mean(1),h.max(1).values,special),-1)))


@dataclass(frozen=True)
class NeighborConfig:
    max_neighbors_per_hop: int = 49
    num_hops: int = 3
    use_stratified_sampling: bool = True
    adaptive_to_n: bool = True
    use_policy_sampling: bool = False
    policy_topk_fraction: float = .5
    max_frontier_states: int = 100000


class NeighborParameters(nn.Module):
    def __init__(self, dim: int, actions: int, hops: int, dropout: float):
        super().__init__()
        self.action_emb = nn.Embedding(actions,dim)
        self.hop_emb = nn.Embedding(hops+1,dim)
        self.msg_mlps = nn.ModuleList([nn.Sequential(NativeLinear(dim*3,dim),nn.GELU(),
            nn.Dropout(dropout),NativeLinear(dim,dim)) for _ in range(hops)])
        self.update_mlps = nn.ModuleList([nn.Sequential(NativeLinear(dim*2,dim),nn.GELU(),
            nn.Dropout(dropout),NativeLinear(dim,dim)) for _ in range(hops)])
        self.hop_attention = nn.MultiheadAttention(dim,4,dropout=dropout,batch_first=True)
        self.hop_query = nn.Parameter(torch.randn(1,1,dim)*.02)
        self.out_proj = NativeLinear(dim,dim)


class PancakeGNN(nn.Module):
    """Own GNN→GNN scorer with issue-compatible weight names.

The native-export path requires deterministic stratified actions and bounds
the inference batch so the reference's random frontier cap never activates.
"""
    __constants__ = ['n','d_model','neighbor_counts','max_frontier_states']

    def __init__(self,n: int,d_model: int=128,num_layers: int=2,
                 dropout: float=.1,neighbors: NeighborConfig=NeighborConfig()):
        super().__init__()
        if n<2 or d_model<=0 or d_model%8 or num_layers<1 or neighbors.num_hops<1:
            raise ValueError('requires n>=2, positive layers/hops and d_model divisible by8')
        if not neighbors.use_stratified_sampling or neighbors.use_policy_sampling:
            raise ValueError('native GNN currently requires deterministic stratified neighbor sampling')
        self.n,self.d_model,self.max_frontier_states=n,d_model,neighbors.max_frontier_states
        self.neighbor_counts=[]
        for hop in range(neighbors.num_hops):
            k=min(neighbors.max_neighbors_per_hop,n-1)
            if neighbors.adaptive_to_n:k=max(1,math.ceil(k/(hop+1)))
            self.neighbor_counts.append(min(max(1,k),n-1))
        self.state_encoder=DualStreamEncoder(n,d_model,num_layers,dropout)
        self.neighbor_agg=NeighborParameters(d_model,n-1,neighbors.num_hops,dropout)
        self.value_head=nn.Sequential(NativeLinear(d_model*2,d_model),nn.GELU(),
                                     nn.Dropout(dropout),NativeLinear(d_model,1))
        self.policy_head=NativeLinear(d_model,n-1)

    def _actions(self,k: int,device: torch.device) -> torch.Tensor:
        if k>=self.n-1:return torch.arange(2,self.n+1,device=device)
        if k==1:return torch.tensor([self.n],device=device)
        if k==2:return torch.tensor([2,self.n],device=device)
        interior=torch.arange(3,self.n,device=device)
        stride=max(1,math.ceil(interior.numel()/float(k-2)))
        return torch.cat((torch.tensor([2],device=device),interior[::stride][:k-2],
                          torch.tensor([self.n],device=device)))

    @torch.jit.export
    def features(self,states: torch.Tensor) -> torch.Tensor:
        states=states.to(torch.long)
        batch=states.size(0)
        current=self.state_encoder(states)
        outputs=[current]
        frontier=states
        roots=torch.arange(batch,device=states.device)
        indices=torch.arange(self.n,device=states.device)
        for hop,(message,update) in enumerate(zip(self.neighbor_agg.msg_mlps,self.neighbor_agg.update_mlps)):
            actions=self._actions(self.neighbor_counts[hop],states.device)
            permutation=torch.where(indices[None,:]<actions[:,None],
                                    actions[:,None]-1-indices[None,:],indices[None,:])
            children=frontier[:,permutation].reshape(-1,self.n)
            child_roots=roots.repeat_interleave(actions.numel())
            child_actions=actions.repeat(frontier.size(0))
            if self.max_frontier_states>0 and children.size(0)>self.max_frontier_states:
                raise RuntimeError('GNN batch would activate stochastic frontier cap; lower inference batch')
            aggregated=torch.zeros((batch,self.d_model),device=states.device)
            count=torch.zeros((batch,1),device=states.device)
            for start in range(0,children.size(0),12000):
                child=self.state_encoder(children[start:start+12000])
                af=self.neighbor_agg.action_emb(child_actions[start:start+12000]-2)
                hf=self.neighbor_agg.hop_emb(torch.full((child.size(0),),hop+1,
                                                      device=states.device,dtype=torch.long))
                messages=message(torch.cat((child,af,hf),-1))
                ids=child_roots[start:start+12000]
                aggregated.index_add_(0,ids,messages.float())
                count.index_add_(0,ids,torch.ones((child.size(0),1),device=states.device))
            mean=(aggregated/count.clamp(min=1)).to(current.dtype)
            outputs.append(update(torch.cat((current,mean),-1)))
            frontier,roots=children,child_roots
        hops=torch.stack(outputs,1)
        query=self.neighbor_agg.hop_query.expand(batch,-1,-1)
        attended,_=self.neighbor_agg.hop_attention(query,hops,hops)
        context=self.neighbor_agg.out_proj(attended.squeeze(1))
        return self.value_head[2](F.gelu(self.value_head[0](torch.cat((current,context),-1))))

    def forward(self,states: torch.Tensor) -> torch.Tensor:
        return self.value_head[3](self.features(states)).squeeze(-1)

