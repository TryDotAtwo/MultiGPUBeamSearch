#!/usr/bin/env python3
"""Experimental full-four-block TE adapter, not a production backend.

Times GPU uint8 states -> logits, including token preparation and activation
quantization. No score conversion, upload, load or graph capture in timing.
FP8 quality is diagnostic, not an acceptance gate. ReLU Cube4 only.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

QKV_MAPPING = dict(layer_norm_weight='ln1_gamma', layer_norm_bias='ln1_beta',
                   weight='qkv_weight_linear', bias='qkv_bias')
MLP_MAPPING = dict(layer_norm_weight='ln2_gamma', layer_norm_bias='ln2_beta',
                   fc1_weight='ff1_weight_linear', fc1_bias='ff1_bias',
                   fc2_weight='ff2_weight_linear', fc2_bias='ff2_bias')


def checked_activation(value):
    if value != 'relu':
        raise ValueError('This experimental adapter is validated only for ReLU')
    return value


def main():
    import torch
    import torch.nn.functional as F
    import transformer_engine as engine
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, Format
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from h200_compare_torch import StaticTokenModel

    ap = argparse.ArgumentParser()
    ap.add_argument('--weight-dir', type=Path, default=Path('/workspace'))
    ap.add_argument('--batches', default='384,768,1536')
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cls-only', action='store_true')
    ap.add_argument('--scaling', choices=('current','delayed'), default='current')
    args = ap.parse_args()
    torch.manual_seed(20260915)
    torch.set_grad_enabled(False)
    reference = StaticTokenModel(args.weight_dir, torch.device('cuda'))
    activation = checked_activation(reference.activation)
    qkvs, mlps, projections = [], [], []

    def copy(module, mapping, block):
        for target, source in mapping.items():
            getattr(module, target).copy_(block[source])
        module.eval()
        return module

    for block in reference.blocks:
        qkvs.append(copy(te.LayerNormLinear(reference.d_model, 3*reference.d_model,
                            eps=1e-5, params_dtype=torch.float16), QKV_MAPPING, block))
        mlps.append(copy(te.LayerNormMLP(reference.d_model, reference.ff_dim,
                            eps=1e-5, activation=activation, params_dtype=torch.float16),
                         MLP_MAPPING, block))
        projections.append(copy(te.Linear(reference.d_model, reference.d_model,
                            params_dtype=torch.float16),
                         dict(weight='attn_out_weight_linear', bias='attn_out_bias'), block))

    recipe = (Float8CurrentScaling(fp8_format=Format.E4M3) if args.scaling=='current' else
              DelayedScaling(fp8_format=Format.E4M3, amax_history_len=16, amax_compute_algo='max'))

    def forward(states, fp8, first=False):
        x = reference.build_tokens(states)
        x = reference.layer_norm(x, reference.input_ln_gamma, reference.input_ln_beta)
        batch = x.shape[0]
        with te.fp8_autocast(enabled=fp8, fp8_recipe=recipe):
            for index,(qkv_layer, mlp, proj) in enumerate(zip(qkvs, mlps, projections)):
                qkv = qkv_layer(x, is_first_microbatch=first).reshape(
                    batch, reference.seq_len, 3, reference.nhead, reference.head_dim)
                q,k,v = [qkv[:,:,i].permute(0,2,1,3) for i in range(3)]
                if args.cls_only and index == reference.num_layers-1:
                    q=q[:,:,:1];x=x[:,:1]
                context = F.scaled_dot_product_attention(q,k,v,dropout_p=0.0)
                context = context.permute(0,2,1,3).reshape(batch,x.shape[1],reference.d_model)
                x = x + proj(context, is_first_microbatch=first)
                x = x + mlp(x, is_first_microbatch=first)
        cls = reference.layer_norm(x[:,0],reference.output_ln_gamma,reference.output_ln_beta)
        return F.linear(cls,reference.output_weight_linear,reference.output_bias)

    with args.output.open('w') as f:
        def emit(**data):
            line=json.dumps(data);print(line,flush=True);f.write(line+'\n');f.flush()
        emit(torch=torch.__version__,te=engine.__version__,gpu=torch.cuda.get_device_name(),
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             manifest_sha256=hashlib.sha256((args.weight_dir/'manifest.json').read_bytes()).hexdigest(),
             topology='final_CLS_attention_MLP_full_QKV' if args.cls_only else 'full_four_blocks',
             torch_control_topology='full_four_blocks',
             recipe=('E4M3 current scaling, activation amax/quantization in graph' if args.scaling=='current'
                     else 'E4M3 delayed scaling, history16, raw graph calibrated scales'),
             weight_cache='20 calibration refreshes then fixed weights',
             flops_per_parent=reference.estimated_flops_per_parent())
        for batch in map(int,args.batches.split(',')):
            states=torch.randint(0,reference.num_classes,(batch,reference.state_len),
                                 device='cuda',dtype=torch.uint8)
            # Include real competition puzzles; remaining states exercise varied tokens.
            csv_path=args.weight_dir/'test.csv'
            if csv_path.exists():
                with csv_path.open() as rows:
                    real=[list(map(int,r['initial_state'].split(','))) for r in csv.DictReader(rows)
                          if int(r['initial_state_id']) in [*range(1,11),1000]]
                states[:len(real)]=torch.tensor(real,device='cuda',dtype=torch.uint8)
            expected=reference.forward(states)
            graphs,outputs={},{}
            for mode in ('torch_fp16','te_fp16','te_fp8'):
                fp8=mode=='te_fp8'
                fn=reference.forward if mode=='torch_fp16' else lambda s: forward(s,fp8)
                if mode!='torch_fp16':
                    # Delayed scaling must settle before caching quantized weights.
                    # Caching the initial scale=1 quantization understates quality.
                    for _ in range(20):forward(states,fp8,True)
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(20):actual=fn(states)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):actual=fn(states)
                graph.replay();torch.cuda.synchronize()
                delta=(actual.float()-expected.float()).abs()
                emit(validation=mode,batch=batch,finite=bool(torch.isfinite(actual).all()),
                     max_abs=float(delta.max()),rmse=float(delta.square().mean().sqrt()),
                     min_score_agreement=float((actual.argmin(-1)==expected.argmin(-1)).float().mean()),
                     real_csv_max_abs=float(delta[:11].max()))
                if not torch.isfinite(actual).all():raise RuntimeError('Nonfinite TE output')
                # Large FP16 deviations indicate an adapter error, not quantization.
                if mode=='te_fp16':torch.testing.assert_close(actual,expected,rtol=.02,atol=.08)
                graphs[mode],outputs[mode]=graph,actual
            # Replay on a distinct input batch. This catches accidental capture
            # of constant inputs and reports quality beyond calibration inputs.
            states.copy_(torch.randint(0,reference.num_classes,states.shape,device='cuda',dtype=torch.uint8))
            expected=reference.forward(states)
            for mode,graph in graphs.items():
                graph.replay();torch.cuda.synchronize()
                actual=outputs[mode]
                delta=(actual.float()-expected.float()).abs()
                emit(heldout=mode,batch=batch,finite=bool(torch.isfinite(actual).all()),
                     max_abs=float(delta.max()),rmse=float(delta.square().mean().sqrt()),
                     min_score_agreement=float((actual.argmin(-1)==expected.argmin(-1)).float().mean()))
                if not torch.isfinite(actual).all():raise RuntimeError('Nonfinite heldout output')
                if mode!='te_fp8':torch.testing.assert_close(actual,expected,rtol=.02,atol=.08)
            modes=list(graphs)
            for rep in range(args.repeats):
                for mode in modes if rep%2==0 else modes[::-1]:
                    torch.cuda.synchronize();begin=time.perf_counter()
                    for _ in range(args.iters):graphs[mode].replay()
                    torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                    emit(mode=mode,batch=batch,repeat=rep,parents_per_sec=batch*args.iters/elapsed,
                         elapsed_ms=elapsed*1000,iters=args.iters)
            del graphs,outputs


if __name__=='__main__':
    main()
