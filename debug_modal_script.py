# run_slippi_ai_on_modal.py (v28 - Fixed to use correct parse_local.py)

import modal
import os

# FIX: Use the user's new repository as the single source of truth.
REMOTE_REPO_URL = "https://github.com/kael4n/slippi_ai_modal.git"
REMOTE_REPO_PATH = "/root/slippi-ai"

# --- Modal Configuration ---
replays_volume = modal.Volume.from_name("slippi-ai-replays")
processed_data_volume = modal.Volume.from_name("slippi-ai-processed-data")

# The image definition, now simplified to clone from the correct source.
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
        # Force fresh clone and pull latest changes
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
    name="slippi-ai-parser-working",
    image=image,
)

@app.function(
    volumes={
        "/replays": replays_volume,
        "/processed": processed_data_volume,
    },
    gpu="any",
    timeout=7200
)
def parse_replays():
    import subprocess
    import os
    import shutil
    import zipfile
    from pathlib import Path
    
    print("\n--- 🏁 Starting Replay Parsing ---")
    
    # Set up the directory structure that parse_local.py expects
    root_dir = "/working"
    raw_dir = f"{root_dir}/Raw"
    parsed_dir = f"{root_dir}/Parsed"
    
    print("Setting up directory structure...")
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(parsed_dir, exist_ok=True)
    
    # Check what's in the replays directory
    print("Checking replays directory...")
    if not os.path.exists("/replays"):
        print("❌ /replays directory not found!")
        return
    
    replay_files = []
    for root, dirs, files in os.walk("/replays"):
        for file in files:
            if file.endswith('.slp'):
                replay_files.append(os.path.join(root, file))
    
    print(f"Found {len(replay_files)} .slp files in /replays")
    if replay_files:
        print(f"First few files: {[os.path.basename(f) for f in replay_files[:5]]}")
    
    if not replay_files:
        print("❌ No .slp files found!")
        return
    
    # Create zip archives from the .slp files
    # parse_local.py expects .zip or .7z archives, not individual .slp files
    print("Creating zip archives from .slp files...")
    
    # Group files into batches to create manageable zip files
    batch_size = 100  # Files per zip
    zip_count = 0
    
    for i in range(0, len(replay_files), batch_size):
        batch = replay_files[i:i + batch_size]
        zip_path = os.path.join(raw_dir, f"replays_batch_{zip_count:03d}.zip")
        
        print(f"Creating {zip_path} with {len(batch)} files...")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for slp_file in batch:
                # Get the relative path for the archive
                arcname = os.path.basename(slp_file)
                zipf.write(slp_file, arcname)
        
        zip_count += 1
        
        # Limit to first few batches for testing
        if zip_count >= 3:  # Only process first 300 files for testing
            print(f"Limited to first {zip_count * batch_size} files for testing")
            break
    
    print(f"Created {zip_count} zip archives in {raw_dir}")
    
    # Create the raw.json metadata file (can be empty initially)
    raw_json_path = os.path.join(root_dir, "raw.json")
    with open(raw_json_path, 'w') as f:
        f.write('{}')  # Empty JSON object
    
    print("Directory structure created:")
    print(f"  {root_dir}/")
    print(f"    Raw/ ({zip_count} zip files)")
    print(f"    Parsed/ (empty)")
    print(f"    raw.json (created)")
    
    # Now run parse_local.py with the correct structure
    parse_local_path = "slippi_db/parse_local.py"
    
    print(f"\n--- Running parse_local.py ---")
    command = [
        "python", parse_local_path,
        f"--root={root_dir}",
        "--threads=4",  # Use multiple threads for faster processing
        "--dry_run"     # Start with dry run to see what it would do
    ]
    
    print(f"Command: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    
    output_lines = []
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        output_lines.append(line)
        
    process.wait()
    
    if process.returncode == 0:
        print(f"\n✅ Dry run successful! Now running actual parsing...")
        
        # Run the actual parsing (remove --dry_run)
        command = [
            "python", parse_local_path,
            f"--root={root_dir}",
            "--threads=4"
        ]
        
        print(f"Actual parsing command: {' '.join(command)}")
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✅ Parsing completed successfully!")
            
            # Copy results to the processed volume
            print("Copying results to processed volume...")
            if os.path.exists(parsed_dir):
                for item in os.listdir(parsed_dir):
                    src = os.path.join(parsed_dir, item)
                    dst = os.path.join("/processed", item)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                        print(f"Copied {item}")
            
            # Also copy metadata files
            for metadata_file in ["parsed.pkl", "meta.json"]:
                src = os.path.join(root_dir, metadata_file)
                if os.path.exists(src):
                    dst = os.path.join("/processed", metadata_file)
                    shutil.copy2(src, dst)
                    print(f"Copied {metadata_file}")
            
            processed_data_volume.commit()
            print("✅ Results saved to processed volume")
        else:
            print(f"\n❌ Actual parsing failed with return code {process.returncode}")
    else:
        print(f"\n❌ Dry run failed with return code {process.returncode}")
        error_output = ''.join(output_lines)
        print("Error details:")
        print(error_output[-1000:])  # Show last 1000 chars of output

@app.local_entrypoint()
def main():
    parse_replays.remote()