"""Build the optional CNOT field binding from an operator's TheQuantumProject checkout."""

import argparse
import io
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile


SOURCE_COMMIT = "c5b3e3f93b45aa7e533fd2c06209c28ce516ffeb"


def replace_once(path, before, after):
    content = path.read_text(encoding="utf-8")
    if content.count(before) != 1:
        raise ValueError(f"unexpected pinned source layout in {path.name}")
    path.write_text(content.replace(before, after), encoding="utf-8", newline="\n")


def prepare(source, build_dir):
    source = source.resolve(strict=True)
    build_dir = build_dir.resolve()
    if build_dir == source or source in build_dir.parents:
        raise ValueError("choose a private build directory outside the upstream checkout")
    archive = subprocess.check_output([
        "git", "-c", "core.autocrlf=false", "-c", "core.eol=lf",
        "archive", "--format=tar", SOURCE_COMMIT, "native",
    ], cwd=source)
    native = build_dir / "native"
    native.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for item in tar:
            name = PurePosixPath(item.name)
            if not name.parts or name.parts[0] != "native" or ".." in name.parts:
                raise ValueError(f"unexpected archive path {item.name}")
            dest = build_dir.joinpath(*name.parts)
            if item.isdir():
                dest.mkdir(exist_ok=True)
            elif item.isfile():
                with tar.extractfile(item) as src, dest.open("xb") as out:
                    shutil.copyfileobj(src, out)
            else:
                raise ValueError(f"unexpected archive entry {item.name}")

    # The checkpoint loader is private in this upstream revision. Compile the
    # binding in its module so the existing pin, coverage and restore checks run.
    # Only the execution device changes between ranks. Hashes still cover the
    # original run manifest, report, registry, objective and checkpoint.
    execution = native / "src" / "run_config" / "execution"
    loader = execution / "evaluation.rs"
    signature = loader.read_text(encoding="utf-8").split("pub(super) fn load_dev_checkpoint(", 1)[1]
    signature = signature.split(") -> Result<DevCheckpoint, String> {", 1)[0]
    start = f"pub(super) fn load_dev_checkpoint({signature}) -> Result<DevCheckpoint, String> {{"
    replace_once(loader, start, start + """
    load_dev_checkpoint_on(path, training_manifest, training_result, dev_indices, None)
}

pub(super) fn load_dev_checkpoint_on(""" + signature + """    device: Option<usize>,
) -> Result<DevCheckpoint, String> {""")
    replace_once(loader, "let manifest = parse_run_manifest(&run_raw)?;", """
    let mut manifest = parse_run_manifest(&run_raw)?;
    if let Some(index) = device {
        manifest.training.cuda_device = index;
    }""")
    replace_once(execution.with_suffix(".rs"), "mod evaluation;", "mod evaluation;\nmod mgbs_scorer;")
    replace_once(native / "Cargo.toml", 'crate-type = ["rlib"]', 'crate-type = ["rlib", "cdylib"]')
    with (native / "cuda" / "CMakeLists.txt").open("a", encoding="utf-8", newline="\n") as cmake:
        cmake.write("\nset_property(TARGET cnot103_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)\n")
    shutil.copyfile(Path(__file__).with_name("bridge.rs"), execution / "mgbs_scorer.rs")
    return native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="local licensed upstream Git checkout")
    parser.add_argument("--build-dir", type=Path, required=True, help="private directory for native sources and build outputs")
    parser.add_argument("--prepare-only", action="store_true", help="prepare sources for a separate build environment")
    args = parser.parse_args()
    native = prepare(args.source, args.build_dir)
    if args.prepare_only:
        print(native)
        return
    target = args.build_dir.resolve() / "target"
    subprocess.run(
        ["cargo", "build", "--locked", "--release", "--lib", "--features", "cuda", "--target-dir", str(target)],
        cwd=native, check=True, env={**os.environ, "LIBTORCH_USE_PYTORCH": "1"},
    )
    print(target / "release" / ("cnot103_native.dll" if os.name == "nt" else "libcnot103_native.so"))


if __name__ == "__main__":
    main()
