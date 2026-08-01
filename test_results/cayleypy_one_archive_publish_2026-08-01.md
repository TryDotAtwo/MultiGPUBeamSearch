# CayleyPy one-archive publication gate - 2026-08-01

- Contract: one notebook run becomes one deterministic `application/gzip` request when the compressed payload is at most 32 MiB.
- Overflow policy: split recursively into bounded archives and upload sequentially in original solution order.
- Payload: canonical CayleyPy results-v1 JSON compressed with deterministic gzip (`mtime=0`).
- Local solve and submission remain successful when publication fails.

Client verification:

- RED: archive builder missing; focused test failed with `AttributeError`.
- RED: oversized archive was not split; focused test failed with `ValueError`.
- RED: archive HTTP publisher missing; focused test failed with `AttributeError`.
- GREEN: archive builder, splitter, and sequential uploader focused tests passed.
- Public CayleyPy suite: 275 passed.
