# quick_fix_test.py - Test direct script execution

import modal
import os

# Same configuration as before
REMOTE_REPO_URL = "https://github.com/kael4n/slippi_ai_modal.git"
REMOTE_REPO_PATH = "/root/slippi-ai"

replays_volume = modal.Volume.from_name("slippi-ai-replays")
processed_data_volume = modal.Volume.from_name("slippi-ai-processed-data")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "tzdata", "python3-pip", "python3-dev", "build-essential", "pkg-config", "cmake", 
        "ninja-build", "libssl-dev", "libffi-dev", "zlib1g-dev", "libbz2-dev", 
        "libreadline-dev", "libsqlite3-dev", "libncurses5-dev", "libncursesw5-dev", 
        "xz-utils", "tk-dev", "libxml2-dev", "libxmlsec1-dev", "liblzma-dev", "git", 
        "curl", "wget", "unzip", "software-properties-common", "libgl1-mesa-glx", 
        "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1"
    )
    .run_commands(
        "ln -sf /usr/bin/python3 /usr/bin/python",
        "python3 -m pip install --upgrade pip setuptools wheel",
        "python3 -m pip install maturin",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .run_commands(
        "pip install numpy==1.24.3 --no-deps",
        "pip install --no-build-isolation peppi-py==0.6.0",
    )
    .pip_install([
        "scipy==1.10.1", "jax==0.4.13", "jaxlib==0.4.13", "pandas==2.0.3",
        "tensorflow==2.13.0", "flax==0.7.2", "optax==0.1.7", "dm-haiku==0.0.10",
        "dm-tree==0.1.8", "sacred==0.8.4", "pymongo==4.5.0", "matplotlib==3.7.2",
        "seaborn==0.12.2", "tqdm==4.65.0", "cloudpickle==2.2.1", "absl-py==1.4.0",
        "tensorboard==2.13.0", "gymnasium==0.28.1", "pyarrow", "pyenet", "py-ubjson",
        "multivolumefile", "pybcj", "inflate64", "brotli", "Cython", "decorator", 
        "platformdirs", "portpicker", "psutil", "pycryptodomex", "pydantic", 
        "pyppmd", "pyzstd", "sentry-sdk", "setproctitle", "texttable"
    ])
    .run_commands(
        f"git clone --recurse-submodules {REMOTE_REPO_URL} {REMOTE_REPO_PATH}",
        f"cd {REMOTE_REPO_PATH} && git pull origin main",
    )
    .workdir(REMOTE_REPO_PATH)
    .run_commands(
        "pip install --no-deps -r requirements.txt",
        "pip install --no-deps -e .",
    )
)

app = modal.App(
    name="slippi-ai-quick-fix",
    image=image,
)

@app.function(
    volumes={
        "/replays": replays_volume,
        "/processed": processed_data_volume,
    },
    timeout=1800
)
def quick_fix_test():
    import subprocess
    import os
    
    print("--- 🚀 Quick Fix Test ---")
    
    # Test the most promising approaches based on your debug output
    
    # 1. Try run_parsing.py directly
    print("\n1. Testing run_parsing.py:")
    try:
        result = subprocess.run(
            ["python", "run_parsing.py", "--help"],
            capture_output=True,
            text=True,
            timeout=60
        )
        print(f"Return code: {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Error: {e}")
    
    # 2. Try parse_peppi directly  
    print("\n2. Testing parse_peppi.py directly:")
    try:
        result = subprocess.run(
            ["python", "slippi_db/parse_peppi.py", "--help"],
            capture_output=True,
            text=True,
            timeout=60
        )
        print(f"Return code: {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Error: {e}")
    
    # 3. Try with explicit path and basic args
    print("\n3. Testing with likely arguments:")
    test_args = [
        ["python", "run_parsing.py", "--input_dir", "/replays", "--output_dir", "/processed"],
        ["python", "slippi_db/parse_peppi.py", "--input_dir", "/replays", "--output_dir", "/processed"],
        ["python", "slippi_db/parse_local.py", "--input_dir", "/replays", "--output_dir", "/processed"],
    ]
    
    for args in test_args:
        try:
            print(f"\nTrying: {' '.join(args)}")
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=30
            )
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print("STDOUT:", result.stdout[:300])
            if result.stderr:
                print("STDERR:", result.stderr[:300])
        except Exception as e:
            print(f"Error: {e}")

@app.local_entrypoint()
def main():
    quick_fix_test.remote()