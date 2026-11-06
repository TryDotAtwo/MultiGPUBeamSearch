# CNOT checkpoint on one RTX 3070

Measured on 2026-11-06 with Linux x86-64, one GeForce RTX 3070 Laptop GPU, driver 572.70, PyTorch 2.9.0+cu128, libtorch C++11 ABI, and Python 3.10.12. The private field binding was built with CUDA 12.8.93, Rust 1.89.0, CMake 3.31.6, and the donor's locked tch 0.22.0.

The native loader restored checkpoint `92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4` from the original GL(4,2) one-update fixture. The private git object was `c5b3e3f93b45aa7e533fd2c06209c28ce516ffeb`. The source archive and weights are operator-supplied. Both searches ran from an isolated directory with precisely the eight files listed in the binding README.

The three GPU tests passed in 30.570 s. They exercised raw Qhat, raw proposal logits and scalar V on two physical matrices, reversed batch order, an empty batch, input validation, and rejection of a live state by the upstream F02 feature. On both matrices the V values agreed with the canonical predictions from the upstream checkpoint CLI within `rtol=2e-5, atol=2e-5`. The independent reference report had SHA-256 `d0cef7bfea2f42a4b0e77cd400155c53d5fc76dde9886020261852840804ddfe`.

For the supplied rows `[1,2,7,8]`, both modes generated 11 distinct candidates and retained the global top 8 at depth one. They selected different sets. Both returned the length-two construction `1→2, 0→2` from the identity, independently replayed bit for bit. The JSON files record each survivor digest, route and hardware identity.

A separate demonstration selected `[1,7,4,10]` from the exact GL(4,2) distance-three layer under the frozen feature policy. With Qhat, width 2 and shortlist 3, the public validator retained 2 of 3 distinct children at depth one and 2 of 4 at depth two. The second expansion processed two parents and six proposals. The returned construction was `1→3, 2→1, 0→1`; direct enumeration excluded every word of lengths 0, 1 and 2. Applying the CUDA owner hash for two buckets to those single-GPU audit rows gave counts `[1,2]` and `[1,3]` at the two layers.

| Local binary | Bytes | SHA-256 | CUDA images |
| --- | ---: | --- | --- |
| `libcnot103_native.so` | 8,881,376 | `efd751477ade2bcd73c79876b732341244ef26df0fa2485b76054ddf20ac8332` | 22 each for sm_75 and sm_86 |
| `libmgbs_cnot_domain.so` | 1,553,224 | `0fa14ee2caba39dd3b43274b32e685013f09dea9bac3488e77554c5a974e4978` | 2 each for sm_75 and sm_86 |
| `libmgbs_cnot_stream5.so` | 779,304 | `86cc992905d7a46e47d7e3212d65206f3ba0fcc5a17d03de945187b882092df0` | 3 each for sm_75 and sm_86 |

The Stream 5 library resolved NCCL 2.27.5 and returned ABI version 1. A single rank measures local model, operator and search behavior. Actual inter-rank transport requires a multi-rank run.

```sh
export PYTHONPATH=integrations/cayleypy_native
export MGBS_TQP_LIBRARY=/private/libcnot103_native.so
export MGBS_TQP_MANIFEST=/private/gl4/search-qhat.json
export MGBS_TQP_H07_REFERENCE=/private/view-spread-result.json
export MGBS_CNOT_DOMAIN_LIBRARY=/private/libmgbs_cnot_domain.so
export MGBS_CNOT_STREAM5_LIBRARY=/private/libmgbs_cnot_stream5.so
python -m unittest discover -s integrations/cayleypy_native/tests -p test_cnot_scorer.py -v
for mode in qhat proposal; do
  python tools/verify_cnot.py --case trained --mode "$mode" --width 8 \
    --library "$MGBS_TQP_LIBRARY" --manifest "$MGBS_TQP_MANIFEST" \
    --output "trained_single_${mode}_rtx3070.json"
done
python tools/verify_cnot.py --case trained-deep --mode qhat --width 2 \
  --library "$MGBS_TQP_LIBRARY" --manifest "$MGBS_TQP_MANIFEST" \
  --output trained_deep_single_qhat_rtx3070.json
```
