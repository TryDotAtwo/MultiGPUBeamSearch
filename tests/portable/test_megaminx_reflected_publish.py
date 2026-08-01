from hashlib import sha256

from portable.megaminx_cluster.scripts.finalize_publication import enrich_reflected_results


def test_after_mode_attaches_exact_original_source_to_reflected_record():
    results = [{"search": "original", "path": ["a", "b"], "valid": True}, {"search": "reflected", "path": ["c"], "searched_path": ["-c"], "valid": True}]
    enriched = enrich_reflected_results(results, None)
    reflected = enriched[1]
    assert reflected["reflected_source_path"] == ["a", "b"]
    assert reflected["reflected_source_sha256"] == sha256(b"a.b").hexdigest()


def test_only_mode_uses_supplied_original_source():
    results = [{"search": "reflected", "path": ["c"], "searched_path": ["-c"], "valid": True}]
    assert enrich_reflected_results(results, ["a"])[0]["reflected_source_path"] == ["a"]
