"""Collect an allowlisted redistributable ELF runtime closure."""
from __future__ import annotations
import argparse, re, shutil, subprocess
from pathlib import Path

HOST_PREFIXES=("libc.so","libm.so","libpthread.so","librt.so","libdl.so","libstdc++.so","libgcc_s.so","ld-linux","libcuda.so","libnvidia-")
BUNDLE_PREFIXES=("libtorch","libc10","libnccl","libcudart","libnvToolsExt","libgomp")

def dependency_policy(name:str)->str:
    if name.startswith(HOST_PREFIXES): return "host"
    if name.startswith(BUNDLE_PREFIXES): return "bundle"
    raise ValueError(f"unclassified runtime dependency: {name}")

def parse_ldd(text:str)->tuple[Path,...]:
    paths=[]
    for raw in text.splitlines():
        line=raw.strip()
        if not line or "linux-vdso" in line: continue
        if "not found" in line: raise ValueError(f"runtime library not found: {line.split()[0]}")
        match=re.search(r"=>\s+(/\S+)",line) or re.match(r"(/\S+)",line)
        if match: paths.append(Path(match.group(1)))
    return tuple(dict.fromkeys(paths))

def collect_closure(executable:Path,destination:Path)->tuple[Path,...]:
    destination.mkdir(parents=True,exist_ok=True); pending=[executable.resolve()]; inspected=set(); copied={}
    while pending:
        binary=pending.pop()
        if binary in inspected: continue
        inspected.add(binary)
        result=subprocess.run(["ldd",str(binary)],text=True,capture_output=True,check=False)
        if result.returncode!=0: raise ValueError(f"ldd failed for {binary.name}")
        for dependency in parse_ldd(result.stdout):
            resolved=dependency.resolve(); policy=dependency_policy(resolved.name)
            if policy=="host": continue
            if not resolved.is_file(): raise ValueError(f"resolved runtime library is missing: {resolved}")
            prior=copied.get(resolved.name)
            if prior is not None and prior!=resolved: raise ValueError(f"runtime basename collision: {resolved.name}")
            if prior is None:
                shutil.copy2(resolved,destination/resolved.name); copied[resolved.name]=resolved; pending.append(resolved)
    return tuple(copied[name] for name in sorted(copied))

def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("executable",type=Path); parser.add_argument("destination",type=Path); args=parser.parse_args()
    for path in collect_closure(args.executable,args.destination): print(path)
    return 0
if __name__=="__main__": raise SystemExit(main())
