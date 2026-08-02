"""Ensure an exact measured profile exists before a production solve."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import argparse, json, os, shutil, sys
from portable.megaminx_cluster.profile import HardwareKey, select_profile
from portable.megaminx_cluster.profile_cache import install_fragment, load_registry

@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX); yield
    finally: handle.close()

def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--archive-root",type=Path,required=True); parser.add_argument("--run-dir",type=Path,required=True); parser.add_argument("--beam",type=int,required=True)
    args=parser.parse_args(argv)
    try:
        root,run_dir=args.archive_root.resolve(),args.run_dir.resolve()
        manifest=json.loads((root/"MANIFEST.json").read_text(encoding="utf-8-sig")); preflight=json.loads((run_dir/"preflight.json").read_text(encoding="utf-8")); first=preflight["gpus"][0]
        hardware=HardwareKey(str(first["family"]),int(first["vram_mib"]),int(first["sm"]),int(preflight["world_size"])); bundled=root/"profiles"/"registry.json"; cache=Path(os.environ.get("MEGAMINX_PROFILE_CACHE",root/"profile-cache"/"registry.json"))
        expected_provenance = {"driver": str(first["driver_major"]), "solver_commit": str(manifest["solver_commit"]), "model_digest": str(manifest["model_sha256"])}
        def available():
            try:
                registry = load_registry(bundled, cache)
                if cache.exists():
                    cached = json.loads(cache.read_text(encoding="utf-8-sig"))
                    registry["profiles"] = [item for item in registry["profiles"] if not (isinstance(item, dict) and item.get("hardware") == {"gpu_family": hardware.gpu_family, "vram_mib": hardware.vram_mib, "sm": hardware.sm, "world_size": hardware.world_size} and item.get("provenance") not in (None, expected_provenance))]
                select_profile(registry,hardware,args.beam,str(manifest["backend"]),str(manifest["model_class"])); return True
            except ValueError: return False
        if available(): print("profile_source=existing"); return 0
        lock_name=f"sm{hardware.sm}-{hardware.vram_mib}mib-x{hardware.world_size}.lock"
        with _exclusive_lock(cache.parent/"locks"/lock_name):
            if available(): print("profile_source=cache_after_wait"); return 0
            tune_dir=run_dir/"autotune"; tune_dir.mkdir(parents=True,exist_ok=True); shutil.copy2(run_dir/"preflight.json",tune_dir/"preflight.json")
            os.environ.update({"MEGAMINX_AUTOTUNE_RUN_DIR":str(tune_dir),"MEGAMINX_AUTOTUNE_PUZZLES":os.environ.get("MEGAMINX_AUTOTUNE_PUZZLES","900:950:1000"),"MEGAMINX_AUTOTUNE_MIN_BEAM":os.environ.get("MEGAMINX_AUTOTUNE_MIN_BEAM","30000000"),"MEGAMINX_AUTOTUNE_TIME_BUDGET_SECONDS":os.environ.get("MEGAMINX_AUTOTUNE_TIME_BUDGET_SECONDS","21600"),"MEGAMINX_AUTOTUNE_BFS_HASH_BUDGET_MIB":os.environ.get("MEGAMINX_AUTOTUNE_BFS_HASH_BUDGET_MIB","256")})
            from portable.megaminx_cluster.autotune.controller import main as tune
            if tune()!=0: raise ValueError("autotune did not complete; solve was not started")
            fragment=json.loads((tune_dir/"registry.fragment.json").read_text(encoding="utf-8"))
            fragment["profiles"][0]["provenance"] = expected_provenance
            install_fragment(cache,fragment)
            if not available(): raise ValueError("installed profile does not cover requested hardware and beam")
            print(f"profile_source=autotuned cache={cache}")
        return 0
    except (OSError,KeyError,TypeError,ValueError,json.JSONDecodeError) as exc:
        print(f"auto_profile_failed={exc}",file=sys.stderr); return 2
if __name__ == "__main__": raise SystemExit(main())
