# fixed_slippi_parsing.py - Correct way to run Slippi AI parsing

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
    name="slippi-ai-fixed-parsing",
    image=image,
)

@app.function(
    volumes={
        "/replays": replays_volume,
        "/processed": processed_data_volume,
    },
    timeout=3600,  # Increased timeout for parsing
    cpu=4,  # More CPU for parallel processing
    memory=8192,  # More memory for large datasets
)
def setup_and_parse():
    import subprocess
    import os
    import json
    
    print("--- 🚀 Setting up Slippi AI Parsing ---")
    
    # 1. Create the required directory structure
    root_dir = "/slippi_root"
    raw_dir = os.path.join(root_dir, "Raw")
    parsed_dir = os.path.join(root_dir, "Parsed")
    
    print(f"Creating directory structure at {root_dir}")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(parsed_dir, exist_ok=True)
    
    # 2. Copy .slp files from /replays to the Raw directory
    print("Copying replay files to Raw directory...")
    
    # Check what's in the replays volume
    replays_path = "/replays"
    if os.path.exists(replays_path):
        print(f"Contents of {replays_path}:")
        for item in os.listdir(replays_path):
            item_path = os.path.join(replays_path, item)
            if os.path.isfile(item_path):
                print(f"  File: {item} ({os.path.getsize(item_path)} bytes)")
            else:
                print(f"  Dir: {item}/")
    else:
        print(f"Warning: {replays_path} does not exist!")
        return
    
    # Copy all .slp files to Raw directory
    slp_count = 0
    for root, dirs, files in os.walk(replays_path):
        for file in files:
            if file.endswith('.slp'):
                src = os.path.join(root, file)
                dst = os.path.join(raw_dir, file)
                
                # Avoid filename conflicts
                counter = 1
                original_dst = dst
                while os.path.exists(dst):
                    name, ext = os.path.splitext(original_dst)
                    dst = f"{name}_{counter}{ext}"
                    counter += 1
                
                try:
                    import shutil
                    shutil.copy2(src, dst)
                    slp_count += 1
                    if slp_count <= 5:  # Show first few files
                        print(f"  Copied: {file}")
                except Exception as e:
                    print(f"  Error copying {file}: {e}")
    
    print(f"Copied {slp_count} .slp files to Raw directory")
    
    if slp_count == 0:
        print("No .slp files found! Please check your replays volume.")
        return
    
    # 3. Create initial raw.json metadata file
    raw_json_path = os.path.join(root_dir, "raw.json")
    print(f"Creating {raw_json_path}")
    
    # Simple metadata structure - the script will update this
    initial_metadata = {}
    with open(raw_json_path, 'w') as f:
        json.dump(initial_metadata, f)
    
    # 4. Run the parsing with correct arguments
    print("\n--- 🔥 Running Slippi AI Parsing ---")
    
    cmd = [
        "python", "/root/slippi-ai/slippi_db/parse_local.py",
        f"--root={root_dir}",
        "--threads=4",  # Use multiple threads
        "--chunk_size=0.5",  # 0.5GB chunks
        "--in_memory=true",  # Process in memory for speed
        "--compression=ZLIB",  # Use ZLIB compression
        "--reprocess=false",  # Don't reprocess already processed files
        "--dry_run=false"  # Actually run the processing
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
        
        # 5. Check the results
        print("\n--- 📊 Checking Results ---")
        
        # Check if parsed files were created
        parsed_pkl_path = os.path.join(root_dir, "parsed.pkl")
        meta_json_path = os.path.join(root_dir, "meta.json")
        
        if os.path.exists(parsed_pkl_path):
            size_mb = os.path.getsize(parsed_pkl_path) / (1024 * 1024)
            print(f"✅ parsed.pkl created ({size_mb:.2f} MB)")
        else:
            print("❌ parsed.pkl not found")
        
        if os.path.exists(meta_json_path):
            size_kb = os.path.getsize(meta_json_path) / 1024
            print(f"✅ meta.json created ({size_kb:.2f} KB)")
            
            # Show a preview of meta.json
            try:
                with open(meta_json_path, 'r') as f:
                    meta_data = json.load(f)
                    print(f"Meta data keys: {list(meta_data.keys())}")
                    if 'games' in meta_data:
                        print(f"Number of games processed: {len(meta_data['games'])}")
            except Exception as e:
                print(f"Error reading meta.json: {e}")
        else:
            print("❌ meta.json not found")
        
        # Check Parsed directory
        if os.path.exists(parsed_dir):
            parsed_contents = os.listdir(parsed_dir)
            print(f"Parsed directory contains {len(parsed_contents)} items:")
            for item in parsed_contents[:10]:  # Show first 10 items
                item_path = os.path.join(parsed_dir, item)
                if os.path.isfile(item_path):
                    size_mb = os.path.getsize(item_path) / (1024 * 1024)
                    print(f"  {item} ({size_mb:.2f} MB)")
                else:
                    print(f"  {item}/ (directory)")
        
        # 6. Copy results to processed volume
        print("\n--- 💾 Copying Results to Processed Volume ---")
        
        processed_volume_path = "/processed"
        
        # Copy the important output files
        files_to_copy = [
            ("parsed.pkl", parsed_pkl_path),
            ("meta.json", meta_json_path)
        ]
        
        for filename, src_path in files_to_copy:
            if os.path.exists(src_path):
                dst_path = os.path.join(processed_volume_path, filename)
                try:
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    print(f"✅ Copied {filename} to processed volume")
                except Exception as e:
                    print(f"❌ Error copying {filename}: {e}")
        
        # Also copy the Parsed directory if it's not too large
        parsed_dst = os.path.join(processed_volume_path, "Parsed")
        if os.path.exists(parsed_dir):
            try:
                import shutil
                if os.path.exists(parsed_dst):
                    shutil.rmtree(parsed_dst)
                shutil.copytree(parsed_dir, parsed_dst)
                print("✅ Copied Parsed directory to processed volume")
            except Exception as e:
                print(f"❌ Error copying Parsed directory: {e}")
        
        print("\n🎉 Parsing process completed!")
        
        if process.returncode == 0:
            print("✅ Success! Your Slippi replays have been processed.")
            print("The processed data is now available in your processed-data volume.")
        else:
            print(f"❌ Parsing failed with return code {process.returncode}")
            print("Check the output above for error details.")
        
    except Exception as e:
        print(f"Error running parsing: {e}")

# Test function to check setup before running full parsing
@app.function(
    volumes={
        "/replays": replays_volume,
        "/processed": processed_data_volume,
    },
    timeout=300
)
def test_setup():
    import os
    
    print("--- 🧪 Testing Setup ---")
    
    # Check replay files
    replays_path = "/replays"
    if os.path.exists(replays_path):
        slp_files = []
        for root, dirs, files in os.walk(replays_path):
            for file in files:
                if file.endswith('.slp'):
                    slp_files.append(os.path.join(root, file))
        
        print(f"Found {len(slp_files)} .slp files in replays volume")
        for i, file in enumerate(slp_files[:5]):  # Show first 5
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"  {i+1}: {os.path.basename(file)} ({size_mb:.2f} MB)")
        
        if len(slp_files) > 5:
            print(f"  ... and {len(slp_files) - 5} more files")
            
        return len(slp_files)
    else:
        print("❌ Replays volume not found or empty")
        return 0

@app.local_entrypoint()
def main():
    # First test the setup
    slp_count = test_setup.remote()
    
    if slp_count > 0:
        print(f"\n✅ Found {slp_count} replay files. Starting parsing...")
        setup_and_parse.remote()
    else:
        print("\n❌ No replay files found. Please upload .slp files to your replays volume first.")
        print("\nTo upload files to your Modal volume:")
        print("1. modal volume put slippi-ai-replays local_file_or_directory remote_path")
        print("2. Or use the Modal dashboard to upload files")
