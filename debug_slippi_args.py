# enhanced_debug_slippi_args.py - Get complete help output and find correct arguments

import modal
import os

# Use the same configuration as your main script
REMOTE_REPO_URL = "https://github.com/kael4n/slippi_ai_modal.git"
REMOTE_REPO_PATH = "/root/slippi-ai"

# --- Modal Configuration ---
replays_volume = modal.Volume.from_name("slippi-ai-replays")
processed_data_volume = modal.Volume.from_name("slippi-ai-processed-data")

# Same image as your main script
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
    name="slippi-ai-enhanced-debug",
    image=image,
)

@app.function(
    volumes={
        "/replays": replays_volume,
        "/processed": processed_data_volume,
    },
    timeout=1800
)
def enhanced_debug():
    import subprocess
    import os
    import sys
    
    print("--- 🔍 Enhanced Slippi-AI Debug ---")
    
    # 1. Get COMPLETE help output from parse_peppi
    print("\n1. Getting complete help from slippi_db.parse_peppi:")
    try:
        result = subprocess.run(
            ["python", "-m", "slippi_db.parse_peppi", "--help"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"Return code: {result.returncode}")
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        
    except Exception as e:
        print(f"Error: {e}")
    
    # 2. Check run_parsing.py 
    print("\n2. Testing run_parsing.py:")
    run_parsing_path = "/root/slippi-ai/run_parsing.py"
    if os.path.exists(run_parsing_path):
        try:
            result = subprocess.run(
                ["python", run_parsing_path, "--help"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            print(f"Return code: {result.returncode}")
            print("=== STDOUT ===")
            print(result.stdout)
            print("=== STDERR ===") 
            print(result.stderr)
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("run_parsing.py not found")
    
    # 3. Check example scripts for usage patterns
    print("\n3. Examining example scripts:")
    
    example_files = [
        "/root/slippi-ai/scripts/imitation_example.sh",
        "/root/slippi-ai/scripts/rl_example.sh"
    ]
    
    for example_file in example_files:
        if os.path.exists(example_file):
            print(f"\n--- {os.path.basename(example_file)} ---")
            try:
                with open(example_file, 'r') as f:
                    content = f.read()
                    print(content)
            except Exception as e:
                print(f"Error reading {example_file}: {e}")
    
    # 4. Check README.md for usage instructions
    print("\n4. Checking README.md for usage:")
    readme_path = "/root/slippi-ai/README.md"
    if os.path.exists(readme_path):
        try:
            with open(readme_path, 'r') as f:
                content = f.read()
                # Look for sections about parsing or usage
                lines = content.split('\n')
                in_relevant_section = False
                for i, line in enumerate(lines):
                    if any(keyword in line.lower() for keyword in ['usage', 'parsing', 'getting started', 'example']):
                        in_relevant_section = True
                        print(f"\nFound relevant section at line {i+1}:")
                        # Print this line and next 10 lines
                        for j in range(max(0, i-2), min(len(lines), i+15)):
                            print(f"{j+1:3d}: {lines[j]}")
                        print("...")
                        break
        except Exception as e:
            print(f"Error reading README: {e}")
    
    # 5. Try to examine parse_local.py source to understand the error
    print("\n5. Examining parse_local.py source:")
    parse_local_path = "/root/slippi-ai/slippi_db/parse_local.py"
    if os.path.exists(parse_local_path):
        try:
            with open(parse_local_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                
                # Look for main function and argument parsing
                print("Looking for main function and arguments...")
                for i, line in enumerate(lines):
                    if 'def main' in line or 'if __name__' in line or 'argparse' in line or 'flags' in line:
                        print(f"\nFound at line {i+1}:")
                        # Print surrounding context
                        for j in range(max(0, i-3), min(len(lines), i+10)):
                            print(f"{j+1:3d}: {lines[j]}")
                        print("...")
                        
        except Exception as e:
            print(f"Error reading parse_local.py: {e}")
    
    # 6. Try different ways to call parse_local
    print("\n6. Testing different ways to call parse_local:")
    
    test_commands = [
        ["python", "/root/slippi-ai/slippi_db/parse_local.py", "--help"],
        ["python", "-m", "slippi_db.parse_local", "--help"],
        ["python", "/root/slippi-ai/slippi_db/parse_local.py"],
    ]
    
    for cmd in test_commands:
        try:
            print(f"\nTrying: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print("STDOUT:", result.stdout[:500])
            if result.stderr:
                print("STDERR:", result.stderr[:500])
                
        except Exception as e:
            print(f"Error: {e}")

@app.local_entrypoint()
def main():
    enhanced_debug.remote()