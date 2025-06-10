# slippi_nfs_processor.py - Process replays directly from NFS

import modal
import os
import zipfile
import tempfile
import shutil

# NFS and Volume configuration
nfs = modal.NetworkFileSystem.from_name("slippi-ai-dataset-doesokay")
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
    name="slippi-ai-nfs-processor",
    image=image,
)

@app.function(
    network_file_systems={"/nfs": nfs},
    volumes={"/processed": processed_data_volume},
    timeout=3600,
    cpu=4,
    memory=8192,
)
def process_replays_from_nfs():
    import subprocess
    import json
    
    print("--- 🚀 Processing Slippi Replays from NFS ---")
    
    # Check what's available on NFS
    nfs_path = "/nfs"
    print(f"NFS contents: {os.listdir(nfs_path)}")
    
    # Set up working directory
    work_dir = "/tmp/slippi_processing"
    raw_dir = os.path.join(work_dir, "Raw")
    parsed_dir = os.path.join(work_dir, "Parsed")
    
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(parsed_dir, exist_ok=True)
    
    # Option 1: Use the zip file directly if it exists
    zip_file_path = os.path.join(nfs_path, "replays.zip")
    extracted_replays_path = os.path.join(nfs_path, "extracted_replays")
    
    if os.path.exists(zip_file_path):
        print(f"Found replays.zip ({os.path.getsize(zip_file_path) / (1024*1024):.1f} MB)")
        
        # Copy the zip directly to Raw directory
        dest_zip = os.path.join(raw_dir, "replays.zip")
        print("Copying zip file to processing directory...")
        shutil.copy2(zip_file_path, dest_zip)
        
        # Verify the zip file
        try:
            with zipfile.ZipFile(dest_zip, 'r') as zf:
                file_list = zf.namelist()
                slp_files = [f for f in file_list if f.endswith('.slp')]
                print(f"✅ Zip contains {len(slp_files)} .slp files")
                
                # Show some examples
                for i, fname in enumerate(slp_files[:5]):
                    print(f"   {fname}")
                if len(slp_files) > 5:
                    print(f"   ... and {len(slp_files) - 5} more")
        except Exception as e:
            print(f"❌ Error reading zip file: {e}")
            return
            
    elif os.path.exists(extracted_replays_path):
        print("Using extracted_replays directory")
        # Count .slp files in extracted directory
        slp_count = 0
        for root, dirs, files in os.walk(extracted_replays_path):
            for file in files:
                if file.endswith('.slp'):
                    slp_count += 1
        
        print(f"Found {slp_count} .slp files in extracted_replays")
        
        # Create a zip from the extracted files for processing
        # (The parsing script expects zip/7z archives)
        dest_zip = os.path.join(raw_dir, "replays.zip")
        print("Creating zip archive from extracted files...")
        
        with zipfile.ZipFile(dest_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(extracted_replays_path):
                for file in files:
                    if file.endswith('.slp'):
                        file_path = os.path.join(root, file)
                        # Add with relative path
                        arcname = os.path.relpath(file_path, extracted_replays_path)
                        zipf.write(file_path, arcname)
        
        print(f"Created zip archive ({os.path.getsize(dest_zip) / (1024*1024):.1f} MB)")
        
    else:
        print("❌ Neither replays.zip nor extracted_replays found on NFS")
        return
    
    # Create raw.json metadata file
    raw_json_path = os.path.join(work_dir, "raw.json")
    initial_metadata = {}
    with open(raw_json_path, 'w') as f:
        json.dump(initial_metadata, f)
    
    # Test peppi-py
    print("\n--- 🧪 Testing peppi-py ---")
    try:
        import peppi_py as peppi
        print("✅ peppi_py imported successfully")
        
        # Test with a file from the zip
        with zipfile.ZipFile(dest_zip, 'r') as zf:
            slp_files = [f for f in zf.namelist() if f.endswith('.slp')]
            if slp_files:
                # Extract one file for testing
                test_file = slp_files[0]
                zf.extract(test_file, "/tmp")
                test_path = os.path.join("/tmp", test_file)
                
                print(f"Testing parse on: {test_file}")
                game = peppi.game(test_path)
                print(f"✅ Test parse successful! Game duration: {game.metadata.duration} frames")
                
                # Clean up test file
                os.remove(test_path)
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
        
        # Check and copy results
        print("\n--- 📊 Copying Results ---")
        
        processed_volume_path = "/processed"
        
        # Copy main output files
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
        
        # Copy Parsed directory if it exists and isn't too large
        if os.path.exists(parsed_dir):
            # Calculate size
            total_size = 0
            for root, dirs, files in os.walk(parsed_dir):
                for file in files:
                    total_size += os.path.getsize(os.path.join(root, file))
            
            size_mb = total_size / (1024 * 1024)
            print(f"Parsed directory size: {size_mb:.2f} MB")
            
            parsed_dst = os.path.join(processed_volume_path, "Parsed")
            if size_mb < 1000:  # Copy if less than 1GB
                if os.path.exists(parsed_dst):
                    shutil.rmtree(parsed_dst)
                shutil.copytree(parsed_dir, parsed_dst)
                print("✅ Copied Parsed directory")
            else:
                print("⚠️ Parsed directory too large, skipping copy")
        
        if process.returncode == 0:
            print("\n🎉 SUCCESS! Replays processed from NFS")
        else:
            print(f"\n❌ Processing failed with return code {process.returncode}")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

@app.function(
    network_file_systems={"/nfs": nfs},
    volumes={"/processed": processed_data_volume},
    timeout=3600,
    cpu=4,
    memory=8192,
)
def process_replays_from_volume():
    """Alternative: Process .slp files directly from Modal volume"""
    import subprocess
    import json
    
    print("--- 🚀 Processing Individual .slp Files from Volume ---")
    
    # This would be if you wanted to mount the volume with individual .slp files
    # You'd need to modify the function signature to include the volume mount
    
    replays_volume = modal.Volume.from_name("slippi-ai-replays")
    
    # Set up processing directory
    work_dir = "/tmp/slippi_processing"
    raw_dir = os.path.join(work_dir, "Raw") 
    parsed_dir = os.path.join(work_dir, "Parsed")
    
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(parsed_dir, exist_ok=True)
    
    # Note: You'd need to mount the replays volume here
    # This is just a template showing how you could process individual files
    
    print("This function would process individual .slp files from a volume")
    print("But using the NFS zip is more efficient!")

@app.local_entrypoint()
def main():
    print("=== 🎮 Efficient Slippi Processing ===\n")
    
    print("Processing replays from NFS...")
    process_replays_from_nfs.remote()
