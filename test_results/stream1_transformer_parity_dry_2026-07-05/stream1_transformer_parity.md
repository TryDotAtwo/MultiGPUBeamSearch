# Stream1 Transformer Backend Parity

- status=dry_run
- tolerance=3072

| backend | mode | status | return_code | checksum | first_score_keys |
|---|---|---|---:|---:|---|
| pytorch | eager | dry_run | 0 | None | `` |
| libtorch | eager | dry_run | 0 | None | `` |
| native_cutlass | eager | dry_run | 0 | None | `` |

## Comparisons

```json
{
  "backends": [
    {
      "backend": "pytorch",
      "mode": "eager"
    },
    {
      "backend": "libtorch",
      "mode": "eager"
    },
    {
      "backend": "native_cutlass",
      "mode": "eager"
    }
  ],
  "reason": "backend invocations generated without execution",
  "status": "dry_run",
  "tolerance": 3072
}
```
