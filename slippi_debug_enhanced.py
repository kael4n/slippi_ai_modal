# enhanced_slippi_debug.py - Debug version with more detailed logging

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
    name="slippi-ai-debug-enhanced",
    image=image,
)

@app.function(
    volumes={
        "/replays": replays_volume,
        "/processed": processed_data_volume,
    },
    timeout=300,
    cpu=2,
    memory=4096,
)
def test_single_file():
    """Test parsing a single .slp file to debug issues"""
    import subprocess
    import os
    import json
    
    print("--- 🔍 Testing Single File Parsing ---")
    
    # Find the first .slp file
    replays_path = "/replays"
    test_file = None
    
    for root, dirs, files in os.walk(replays_path):
        for file in files:
            if file.endswith('.slp'):
                test_file = os.path.join(root, file)
                break
        if test_file:
            break
    
    if not test_file:
        print("❌ No .slp files found")
        return
    
    print(f"🎯 Testing with file: {os.path.basename(test_file)}")
    print(f"File size: {os.path.getsize(test_file)} bytes")
    
    # Test if we can import peppi and parse directly
    print("\n--- 🧪 Testing peppi-py directly ---")
    try:
        import peppi
        print("✅ peppi imported successfully")
        
        # Try to parse the file directly
        print(f"Attempting to parse {test_file}...")
        game = peppi.game(test_file)
        print(f"✅ File parsed successfully!")
        print(f"Game duration: {game.metadata.duration} frames")
        print(f"Players: {len(game.metadata.players)}")
        
        # Print player info
        for i, player in enumerate(game.metadata.players):
            if player:
                print(f"  Player {i}: {player.netplay.name if player.netplay else 'Unknown'}")
        
    except Exception as e:
        print(f"❌ Error parsing with peppi: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test the actual parsing script with minimal setup
    print("\n--- 🧪 Testing parse_local.py ---")
    
    # Create minimal directory structure
    test_root = "/tmp/slippi_test"
    raw_dir = os.path.join(test_root, "Raw")
    parsed_dir = os.path.join(test_root, "Parsed")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(parsed_dir, exist_ok=True)
    
    # Copy single test file
    import shutil
    test_file_copy = os.path.join(raw_dir, os.path.basename(test_file))
    shutil.copy2(test_file, test_file_copy)
    
    # Create minimal raw.json
    raw_json_path = os.path.join(test_root, "raw.json")
    with open(raw_json_path, 'w') as f:
        json.dump({}, f)
    
    # Run parsing on single file
    cmd = [
        "python", "/root/slippi-ai/slippi_db/parse_local.py",
        f"--root={test_root}",
        "--threads=1",
        "--chunk_size=0.1",
        "--in_memory=true",
        "--compression=ZLIB",
        "--reprocess=true",  # Force reprocessing
        "--dry_run=false"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        # Check results
        parsed_pkl = os.path.join(test_root, "parsed.pkl")
        meta_json = os.path.join(test_root, "meta.json")
        
        if os.path.exists(parsed_pkl):
            size = os.path.getsize(parsed_pkl)
            print(f"✅ parsed.pkl created ({size} bytes)")
        
        if os.path.exists(meta_json):
            print(f"✅ meta.json created")
            with open(meta_json, 'r') as f:
                meta = json.load(f)
                print(f"Meta keys: {list(meta.keys())}")
        
        # Check parsed directory
        parsed_files = os.listdir(parsed_dir)
        print(f"Parsed directory has {len(parsed_files)} files")
        
    except subprocess.TimeoutExpired:
        print("❌ Parsing timed out")
    except Exception as e:
        print(f"❌ Error running parsing: {e}")

@app.function(
    volumes={
        "/replays": replays_volume,
        "/processed": processed_data_volume,
    },
    timeout=300,
)
def check_parsing_script():
    """Check if the parsing script exists and what it expects"""
    import os
    
    print("--- 🔍 Checking Parsing Script ---")
    
    script_path = "/root/slippi-ai/slippi_db/parse_local.py"
    
    if os.path.exists(script_path):
        print(f"✅ Found parsing script at {script_path}")
        
        # Check if it's executable
        if os.access(script_path, os.X_OK):
            print("✅ Script is executable")
        else:
            print("⚠️ Script is not executable")
        
        # Read the first few lines to understand usage
        with open(script_path, 'r') as f:
            lines = f.readlines()[:50]  # First 50 lines
            
        print("\n--- Script Header ---")
        for i, line in enumerate(lines):
            if line.strip().startswith('"""') or line.strip().startswith("'''"):
                break
            if 'argparse' in line or 'ArgumentParser' in line:
                print(f"Line {i+1}: {line.strip()}")
            if 'add_argument' in line:
                print(f"Line {i+1}: {line.strip()}")
        
        # Try to run with --help
        print("\n--- Script Help ---")
        import subprocess
        try:
            result = subprocess.run(
                ["python", script_path, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        except Exception as e:
            print(f"Error getting help: {e}")
            
    else:
        print(f"❌ Parsing script not found at {script_path}")
        
        # List what's in the directory
        base_dir = "/root/slippi-ai/slippi_db"
        if os.path.exists(base_dir):
            print(f"Contents of {base_dir}:")
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isfile(item_path):
                    print(f"  📄 {item}")
                else:
                    print(f"  📁 {item}/")
        else:
            print(f"❌ Directory {base_dir} not found")
    
    # Also check what's in the root slippi-ai directory
    print(f"\n--- Contents of /root/slippi-ai ---")
    slippi_root = "/root/slippi-ai"
    if os.path.exists(slippi_root):
        for item in os.listdir(slippi_root):
            item_path = os.path.join(slippi_root, item)
            if os.path.isfile(item_path):
                print(f"  📄 {item}")
            else:
                print(f"  📁 {item}/")
    
    # Check for alternative parsing scripts
    print(f"\n--- Looking for other parsing scripts ---")
    for root, dirs, files in os.walk("/root/slippi-ai"):
        for file in files:
            if 'parse' in file.lower() and file.endswith('.py'):
                print(f"  Found: {os.path.join(root, file)}")

@app.local_entrypoint()
def main():
    print("=== 🔍 Enhanced Slippi AI Debug Session ===\n")
    
    # Check the parsing script first
    print("1. Checking parsing script...")
    check_parsing_script.remote()
    
    print("\n" + "="*50 + "\n")
    
    # Test single file parsing
    print("2. Testing single file parsing...")
    test_single_file.remote()
    
    print("\n=== Debug session completed ===")
