# slippi_volume_processor.py - Process individual .slp files efficiently

import modal
import os
import zipfile
import tempfile
import shutil

# Volume configuration
replays_volume = modal.Volume.from_name("slippi-ai-replays")
processed_data_volume = modal.Volume.from_name("slippi-ai-processed-data")

# Same image configuration as before
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "tzdata", "python3-pip", "python3-dev", "build-essential", "pkg-config", "cmake", 
        "ninja-build", "libssl-dev", "libffi-dev", "zlib1g-dev", "libbz2-dev", 
        "libreadline-dev", "libsqlite3-dev", "libncurses5-dev", "libncursesw5-dev", 
        "xz-utils", "tk-dev", "libxml2-dev", "libxmlsec1-dev", "liblzma-dev", "git", 
        "curl", "wget", "unzip", "software-properties-software", "libgl1-mesa-glx", 
        "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1", "p7zip-full"
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
        "git clone --recurse-submodules https://github.com/kael4n/slippi_ai_modal.git /root/slippi-ai",
        "cd /root/slippi-ai && git pull origin main",
    )
    .workdir("/root/slippi-ai")
    .run_commands(
        "pip install --no-deps -r requirements.txt",
        "pip install --no-deps -e .",
    )
)

app = modal.App(
    name="slippi-ai-volume-processor",
    image=image,
)

@app.function(
    volumes={
        "/replays": replays_volume,
        "/processed": processed_data_volume,
    },
    timeout=3600,
    cpu=4,
    memory=8192,
)
def process_individual_slp_files():
    import subprocess
    import json
    
    print("--- 🚀 Processing Individual .slp Files ---")
    
    # Set up processing directory
    work_dir = "/tmp/slippi_processing"
    raw_dir = os.path.join(work_dir, "Raw")
    parsed_dir = os.path.join(work_dir, "Parsed")
    
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(parsed_dir, exist_ok=True)
    
    # Find all .slp files
    replays_path = "/replays"
    slp_files = []
    
    for root, dirs, files in os.walk(replays_path):
        for file in files:
            if file.endswith('.slp'):
                slp_files.append(os.path.join(root, file))
    
    print(f"Found {len(slp_files)} .slp files")
    
    if len(slp_files) == 0:
        print("❌ No .slp files found!")
        return
    
    # Create ONE efficient zip archive with all files
    # This is much better than creating multiple small archives
    archive_path = os.path.join(raw_dir, "all_replays.zip")
    
    print("Creating single optimized zip archive...")
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for i, slp_file in enumerate(slp_files):
            # Use just the filename, not the full path
            arcname = os.path.basename(slp_file)
            zipf.write(slp_file, arcname)
            
            if (i + 1) % 100 == 0:
                print(f"  Added {i + 1}/{len(slp_files)} files...")
    
    archive_size_mb = os.path.getsize(archive_path) / (1024 * 1024)
    print(f"✅ Created optimized archive: {archive_size_mb:.2f} MB")
    
    # Create raw.json metadata
    raw_json_path = os.path.join(work_dir, "raw.json")
    with open(raw_json_path, 'w') as f:
        json.dump({}, f)
    
    # Test peppi-py
    print("\n--- 🧪 Testing peppi-py ---")
    try:
        import peppi_py as peppi
        print("✅ peppi_py imported successfully")
        
        # Test with first file
        test_file = slp_files[0]
        print(f"Testing parse on: {os.path.basename(test_file)}")
        game = peppi.game(test_file)
        print(f"✅ Test parse successful! Game duration: {game.metadata.duration} frames")
    except Exception as e:
        print(f"❌ Error with peppi-py: {e}")
    
    # Run the parsing
    print("\n--- 🔥 Running Slippi AI Parsing ---")
    
    cmd = [
        "python", "/root/slippi-ai/slippi_db/parse_local.py",
        f"--root={work_dir}",
        "--threads=4",
        "--chunk_size=0.5",
        "--in_memory=true",
        "--compression=zlib",
        "--reprocess=false",
        "--dry_run=false"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3000  # 50 minutes timeout
        )
        
        print("STDOUT:")
        print(process.stdout)
        
        if process.stderr:
            print("STDERR:")
            print(process.stderr)
        
        print(f"\nParsing completed with return code: {process.returncode}")
        
        # Copy results to processed volume
        print("\n--- 📊 Copying Results ---")
        
        processed_volume_path = "/processed"
        
        files_to_copy = [
            ("parsed.pkl", os.path.join(work_dir, "parsed.pkl")),
            ("meta.json", os.path.join(work_dir, "meta.json")),
            ("raw.json", raw_json_path)
        ]
        
        for filename, src_path in files_to_copy:
            if os.path.exists(src_path):
                dst_path = os.path.join(processed_volume_path, filename)
                shutil.copy2(src_path, dst_path)
                size_mb = os.path.getsize(src_path) / (1024 * 1024)
                print(f"✅ Copied {filename} ({size_mb:.2f} MB)")
            else:
                print(f"❌ {filename} not found at {src_path}")
        
        # Copy Parsed directory
        if os.path.exists(parsed_dir):
            parsed_dst = os.path.join(processed_volume_path, "Parsed")
            if os.path.exists(parsed_dst):
                shutil.rmtree(parsed_dst)
            shutil.copytree(parsed_dir, parsed_dst)
            print("✅ Copied Parsed directory")
        
        if process.returncode == 0:
            print("\n🎉 SUCCESS! Individual .slp files processed efficiently")
        else:
            print(f"\n❌ Processing failed with return code {process.returncode}")
            
    except subprocess.TimeoutExpired:
        print("❌ Processing timed out")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

@app.local_entrypoint()
def main():
    print("=== 🎮 Process Individual .slp Files ===\n")
    process_individual_slp_files.remote()
