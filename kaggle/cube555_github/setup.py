"""Install the pinned example into a private environment; usable on Kaggle or Linux."""
from pathlib import Path
import os
CODE_COMMIT = "fc75cad08cb90cdf49bd56f92f95f1a45b50808d"
WORK_ROOT = Path(os.environ.get("MULTIGPUBEAMSEARCH_HOME", Path.home() / ".cache/multigpubeamsearch/cube555")).expanduser().resolve()
PREPARED_SOURCE_DIR = WORK_ROOT / "sources" / CODE_COMMIT
import subprocess, sys

if not (PREPARED_SOURCE_DIR / ".git").is_dir():
    PREPARED_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(PREPARED_SOURCE_DIR)], check=True)
    subprocess.run(["git", "-C", str(PREPARED_SOURCE_DIR), "remote", "add", "origin",
                    "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git"], check=True)
    subprocess.run(["git", "-C", str(PREPARED_SOURCE_DIR), "fetch", "--depth", "1",
                    "origin", CODE_COMMIT], check=True)
    subprocess.run(["git", "-C", str(PREPARED_SOURCE_DIR), "checkout", "--detach", "FETCH_HEAD"], check=True)
actual_commit = subprocess.check_output(["git", "-C", str(PREPARED_SOURCE_DIR),
    "rev-parse", "HEAD"], text=True).strip()
if actual_commit != CODE_COMMIT:
    raise RuntimeError(f"Source commit mismatch: {actual_commit}")
# Install into a private environment, including on PEP 668-managed hosts.
import site
ENV_ROOT = WORK_ROOT / ("python-env-" + str(sys.version_info.major) + str(sys.version_info.minor))
ENV_PYTHON = ENV_ROOT / "bin/python"
if not ENV_PYTHON.is_file():
    bootstrap = WORK_ROOT / "bootstrap"
    if not (bootstrap / "virtualenv/__init__.py").is_file():
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "--target", str(bootstrap), "virtualenv"], check=True)
    bootstrap_env = dict(os.environ, PYTHONPATH=str(bootstrap))
    subprocess.run([sys.executable, "-m", "virtualenv", "--system-site-packages",
                    "--no-download", str(ENV_ROOT)], env=bootstrap_env, check=True)
subprocess.run([str(ENV_PYTHON), "-m", "pip", "install", "--quiet",
    "cayleypy @ https://github.com/cayleypy/cayleypy/archive/3cd11ecefc98e54d6fc0ccdec967d0bcc8ca4a93.tar.gz",
    str(PREPARED_SOURCE_DIR / "integrations/cayleypy_native")], check=True)
ENV_SITE = subprocess.check_output([str(ENV_PYTHON), "-c",
    "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip()
site.addsitedir(ENV_SITE)
sys.path.insert(0, ENV_SITE)
os.environ["PYTHONPATH"] = ENV_SITE + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["PATH"] = str(ENV_ROOT / "bin") + os.pathsep + os.environ["PATH"]
import torch
NUM_GPUS = torch.cuda.device_count()
if not torch.cuda.is_available() or NUM_GPUS < 1 or torch.cuda.device_count() < NUM_GPUS:
    raise RuntimeError(f"Requested {NUM_GPUS} GPUs, available {torch.cuda.device_count()}; request GPU resources before starting the notebook")
print("Source:", actual_commit)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(NUM_GPUS)])
