# slippi_fixed_final.py - Properly working Slippi AI parsing

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
    name="slippi-ai-fixed-final",
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
def parse_slippi_replays():
    import subprocess
    import os
    import json
    import zipfile
    import shutil
    
    print("--- 🚀 Setting up Slippi AI Parsing (Fixed Version) ---")
    
    # 1. Create the required directory structure
    root_dir = "/slippi_root"
    raw_dir = os.path.join(root_dir, "Raw")
    parsed_dir = os.path.join(root_dir, "Parsed")
    
    print(f"Creating directory structure at {root_dir}")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(parsed_dir, exist_ok=True)
    
    # 2. Find all .slp files and create zip archives
    print("Finding and archiving replay files...")
    
    replays_path = "/replays"
    slp_files = []
    
    # Collect all .slp files
    for root, dirs, files in os.walk(replays_path):
        for file in files:
            if file.endswith('.slp'):
                slp_files.append(os.path.join(root, file))
    
    print(f"Found {len(slp_files)} .slp files")
    
    if len(slp_files) == 0:
        print("No .slp files found! Please check your replays volume.")
        return
    
    # 3. Create zip archives from .slp files
    # The parsing script expects zip/7z archives, so we need to create them
    print("Creating zip archives (required by parsing script)...")
    
    # Group files into reasonably sized archives (e.g., 10 files per archive)
    files_per_archive = 10
    archive_count = 0
    
    for i in range(0, len(slp_files), files_per_archive):
        archive_count += 1
        batch = slp_files[i:i + files_per_archive]
        
        archive_name = f"replays_batch_{archive_count:03d}.zip"
        archive_path = os.path.join(raw_dir, archive_name)
        
        print(f"Creating {archive_name} with {len(batch)} files...")
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for slp_file in batch:
                # Add file to zip with just its basename (no path)
                zipf.write(slp_file, os.path.basename(slp_file))
        
        archive_size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        print(f"  Created {archive_name} ({archive_size_mb:.2f} MB)")
    
    print(f"Created {archive_count} zip archives in Raw directory")
    
    # 4. Create initial raw.json metadata file
    raw_json_path = os.path.join(root_dir, "raw.json")
    print(f"Creating {raw_json_path}")
    
    # The script will update this file to track which archives have been processed
    initial_metadata = {}
    with open(raw_json_path, 'w') as f:
        json.dump(initial_metadata, f)
    
    # 5. Test peppi-py to make sure it works
    print("\n--- 🧪 Testing peppi-py ---")
    try:
        import peppi_py as peppi
        print("✅ peppi_py imported successfully")
        
        # Test with first file
        if slp_files:
            test_file = slp_files[0]
            print(f"Testing parse on: {os.path.basename(test_file)}")
            game = peppi.game(test_file)
            print(f"✅ Test parse successful! Game duration: {game.metadata.duration} frames")
    except Exception as e:
        print(f"❌ Error with peppi-py: {e}")
        print("This might cause parsing issues...")
    
    # 6. Run the parsing with correct arguments
    print("\n--- 🔥 Running Slippi AI Parsing ---")
    
    cmd = [
        "python", "/root/slippi-ai/slippi_db/parse_local.py",
        f"--root={root_dir}",
        "--threads=4",
        "--chunk_size=0.5",
        "--in_memory=true",
        "--compression=zlib",
        "--reprocess=false",
        "--dry_run=false"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
        
        process.wait()
        
        print(f"\nParsing completed with return code: {process.returncode}")
        
        # 7. Check the results in detail
        print("\n--- 📊 Checking Results ---")
        
        # Check main output files
        parsed_pkl_path = os.path.join(root_dir, "parsed.pkl")
        meta_json_path = os.path.join(root_dir, "meta.json")
        
        if os.path.exists(parsed_pkl_path):
            size_mb = os.path.getsize(parsed_pkl_path) / (1024 * 1024)
            print(f"✅ parsed.pkl created ({size_mb:.2f} MB)")
            
            # Try to load and examine the pickle file
            try:
                import pickle
                with open(parsed_pkl_path, 'rb') as f:
                    parsed_data = pickle.load(f)
                    if isinstance(parsed_data, dict):
                        print(f"   Contains {len(parsed_data)} parsed games")
                        # Show first few game IDs
                        game_ids = list(parsed_data.keys())[:5]
                        print(f"   Sample game IDs: {game_ids}")
                    else:
                        print(f"   Data type: {type(parsed_data)}")
            except Exception as e:
                print(f"   Could not examine pickle file: {e}")
        else:
            print("❌ parsed.pkl not found")
        
        if os.path.exists(meta_json_path):
            size_kb = os.path.getsize(meta_json_path) / 1024
            print(f"✅ meta.json created ({size_kb:.2f} KB)")
            
            try:
                with open(meta_json_path, 'r') as f:
                    meta_data = json.load(f)
                    print(f"   Meta data keys: {list(meta_data.keys())}")
                    if 'games' in meta_data:
                        print(f"   Number of games in meta: {len(meta_data['games'])}")
            except Exception as e:
                print(f"   Error reading meta.json: {e}")
        else:
            print("❌ meta.json not found")
        
        # Check raw.json to see processing status
        if os.path.exists(raw_json_path):
            try:
                with open(raw_json_path, 'r') as f:
                    raw_data = json.load(f)
                    print(f"✅ raw.json updated with {len(raw_data)} entries")
                    
                    # Check processing status
                    processed_count = sum(1 for entry in raw_data.values() if entry.get('processed', False))
                    print(f"   Processed archives: {processed_count}/{len(raw_data)}")
            except Exception as e:
                print(f"   Error reading raw.json: {e}")
        
        # Check Parsed directory
        if os.path.exists(parsed_dir):
            parsed_contents = os.listdir(parsed_dir)
            print(f"✅ Parsed directory contains {len(parsed_contents)} files")
            
            # Show some examples
            parquet_files = [f for f in parsed_contents if f.endswith('.parquet')]
            print(f"   Parquet files: {len(parquet_files)}")
            
            if parquet_files:
                # Show file sizes of first few parquet files
                for i, pf in enumerate(parquet_files[:5]):
                    pf_path = os.path.join(parsed_dir, pf)
                    size_kb = os.path.getsize(pf_path) / 1024
                    print(f"     {pf}: {size_kb:.1f} KB")
        
        # 8. Copy results to processed volume
        print("\n--- 💾 Copying Results to Processed Volume ---")
        
        processed_volume_path = "/processed"
        
        # Copy the important output files
        files_to_copy = [
            ("parsed.pkl", parsed_pkl_path),
            ("meta.json", meta_json_path),
            ("raw.json", raw_json_path)
        ]
        
        for filename, src_path in files_to_copy:
            if os.path.exists(src_path):
                dst_path = os.path.join(processed_volume_path, filename)
                try:
                    shutil.copy2(src_path, dst_path)
                    print(f"✅ Copied {filename} to processed volume")
                except Exception as e:
                    print(f"❌ Error copying {filename}: {e}")
        
        # Copy Parsed directory (with size check)
        parsed_dst = os.path.join(processed_volume_path, "Parsed")
        if os.path.exists(parsed_dir):
            try:
                # Calculate total size first
                total_size = 0
                for root, dirs, files in os.walk(parsed_dir):
                    for file in files:
                        total_size += os.path.getsize(os.path.join(root, file))
                
                size_mb = total_size / (1024 * 1024)
                print(f"Parsed directory size: {size_mb:.2f} MB")
                
                if size_mb < 500:  # Only copy if less than 500MB
                    if os.path.exists(parsed_dst):
                        shutil.rmtree(parsed_dst)
                    shutil.copytree(parsed_dir, parsed_dst)
                    print("✅ Copied Parsed directory to processed volume")
                else:
                    print("⚠️ Parsed directory too large, skipping copy")
                    print("   (You can access it in future runs)")
            except Exception as e:
                print(f"❌ Error copying Parsed directory: {e}")
        
        print("\n🎉 Parsing process completed!")
        
        if process.returncode == 0:
            print("✅ SUCCESS! Your Slippi replays have been processed.")
            print("📊 Results summary:")
            print("   - Individual game data: Parsed/*.parquet files")
            print("   - Metadata summary: parsed.pkl")
            print("   - Processing log: raw.json")
            print("   - All data available in processed-data volume")
        else:
            print(f"❌ Parsing failed with return code {process.returncode}")
            print("Check the output above for error details.")
        
    except Exception as e:
        print(f"Error running parsing: {e}")
        import traceback
        traceback.print_exc()

@app.function(
    volumes={
        "/replays": replays_volume,
        "/processed": processed_data_volume,
    },
    timeout=300
)
def quick_test():
    """Quick test to make sure everything is set up correctly"""
    import os
    
    print("--- 🧪 Quick Setup Test ---")
    
    # Check replay files
    replays_path = "/replays"
    slp_count = 0
    
    if os.path.exists(replays_path):
        for root, dirs, files in os.walk(replays_path):
            for file in files:
                if file.endswith('.slp'):
                    slp_count += 1
        print(f"✅ Found {slp_count} .slp files in replays volume")
    else:
        print("❌ Replays volume not accessible")
        return False
    
    # Test peppi-py import
    try:
        import peppi_py as peppi
        print("✅ peppi_py can be imported")
    except Exception as e:
        print(f"❌ peppi_py import failed: {e}")
        return False
    
    # Check parsing script
    script_path = "/root/slippi-ai/slippi_db/parse_local.py"
    if os.path.exists(script_path):
        print("✅ Parsing script found")
    else:
        print("❌ Parsing script not found")
        return False
    
    print("✅ All checks passed! Ready for full parsing.")
    return True

@app.local_entrypoint()
def main():
    print("=== 🎮 Fixed Slippi AI Parsing ===\n")
    
    # Quick test first
    if quick_test.remote():
        print("\n" + "="*50)
        print("🚀 Starting full parsing process...")
        parse_slippi_replays.remote()
    else:
        print("\n❌ Setup test failed. Please check the errors above.")
