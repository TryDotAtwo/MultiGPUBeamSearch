# Stream Pipeline Smoke

- commit: `62c14bb`
- b_micro: `512`
- concurrency: `2`
- failures: `0`

| mode | window | status | cand/s | depth_like_ms | ring_jobs | stream3_jobs | log |
|---|---:|---|---:|---:|---:|---:|---|
| stream12 | 16 | OK | 563956 | 5577.97 | 256 | 0 | `/kaggle/working/stream_pipeline_smoke_logs/pipeline_stream12_w16.log` |
| stream12 | 32 | OK | 547178 | 5749 | 256 | 0 | `/kaggle/working/stream_pipeline_smoke_logs/pipeline_stream12_w32.log` |
| stream12 | 64 | OK | 533020 | 5901.71 | 256 | 0 | `/kaggle/working/stream_pipeline_smoke_logs/pipeline_stream12_w64.log` |
| stream123 | 16 | OK | 523753 | 6006.12 | 256 | 32 | `/kaggle/working/stream_pipeline_smoke_logs/pipeline_stream123_w16.log` |
| stream123 | 32 | OK | 504416 | 6236.37 | 256 | 32 | `/kaggle/working/stream_pipeline_smoke_logs/pipeline_stream123_w32.log` |
| stream123 | 64 | OK | 493849 | 6369.81 | 256 | 32 | `/kaggle/working/stream_pipeline_smoke_logs/pipeline_stream123_w64.log` |
