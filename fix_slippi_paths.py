# fix_slippi_paths.py - Fix for Windows-style backslashes in replay filenames

import modal
import os
import zipfile
import pickle

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
    name="slippi-ai-fix-paths",
    image=image,
)

@app.function(
    volumes={
        "/replays": replays_volume,
        "/processed": processed_data_volume,
    },
    timeout=3600
)
def fix_replay_paths():
    """
    Fix the replay file paths by renaming files with backslashes to use forward slashes
    """
    import os
    import shutil
    
    print("--- 🔧 Fixing Replay File Paths ---")
    
    # Count files that need fixing
    files_to_fix = []
    for root, dirs, files in os.walk("/replays"):
        for file in files:
            if file.endswith('.slp') and '\\' in file:
                files_to_fix.append((root, file))
    
    print(f"Found {len(files_to_fix)} files with backslashes that need fixing")
    
    if not files_to_fix:
        print("✅ No files need path fixing!")
        return
    
    # Create a backup of the original files first
    print("Creating backup of original file structure...")
    
    # Process each file
    fixed_count = 0
    error_count = 0
    
    for root, original_filename in files_to_fix:
        try:
            # Original path
            original_path = os.path.join(root, original_filename)
            
            # New filename with forward slashes converted to hyphens or underscores
            # to avoid directory creation issues
            new_filename = original_filename.replace('\\', '_')
            new_path = os.path.join(root, new_filename)
            
            # Alternative approach: create directory structure if needed
            # new_filename_parts = original_filename.split('\\')
            # if len(new_filename_parts) > 1:
            #     new_dir = os.path.join(root, *new_filename_parts[:-1])
            #     os.makedirs(new_dir, exist_ok=True)
            #     new_path = os.path.join(new_dir, new_filename_parts[-1])
            # else:
            #     new_path = os.path.join(root, original_filename)
            
            print(f"Renaming: {original_filename} -> {new_filename}")
            
            # Move the file
            shutil.move(original_path, new_path)
            fixed_count += 1
            
        except Exception as e:
            print(f"❌ Error fixing {original_filename}: {e}")
            error_count += 1
    
    print(f"✅ Fixed {fixed_count} files")
    if error_count > 0:
        print(f"❌ {error_count} files had errors")
    
    # Verify the fix worked
    print("\n--- Verifying Fix ---")
    remaining_backslash_files = []
    for root, dirs, files in os.walk("/replays"):
        for file in files:
            if file.endswith('.slp') and '\\' in file:
                remaining_backslash_files.append(file)
    
    if remaining_backslash_files:
        print(f"❌ Still have {len(remaining_backslash_files)} files with backslashes")
        for file in remaining_backslash_files[:5]:  # Show first 5
            print(f"  {file}")
    else:
        print("✅ All files now have clean filenames!")
    
    # Clean up any previous parsed data since it was based on wrong filenames
    if os.path.exists("/processed/parsed.pkl"):
        print("🗑️  Removing old parsed.pkl (it had wrong filenames)")
        os.remove("/processed/parsed.pkl")
    
    print("✅ Path fixing complete!")

@app.function(
    volumes={
        "/replays": replays_volume,
        "/processed": processed_data_volume,
    },
    timeout=7200
)
def reprocess_replays_after_fix():
    """
    Reprocess replays after fixing the paths
    """
    import sys
    import os
    
    print("--- 🔄 Reprocessing Replays After Path Fix ---")
    
    # Add the slippi-ai path to sys.path
    if "/root/slippi-ai" not in sys.path:
        sys.path.append("/root/slippi-ai")
    
    # Now try to import and use slippi-ai parsing
    try:
        from slippi_db import parse_peppi
        
        print("✅ Successfully imported slippi-ai modules")
        
        # Get list of replay files
        replay_files = []
        for root, dirs, files in os.walk("/replays"):
            for file in files:
                if file.endswith('.slp'):
                    replay_files.append(os.path.join(root, file))
        
        print(f"Found {len(replay_files)} replay files to process")
        
        # Process files (use the same logic as your original script)
        parsed_data = []
        
        for i, file_path in enumerate(replay_files):
            try:
                # Try to parse with slippi-ai
                # We need to figure out the correct function to call
                # From your debug output, we know these functions exist:
                # ['BUTTON_MASKS', 'Button', 'from_peppi', 'get_buttons', 'get_player', 'get_slp', 'get_stick', 'melee', 'np', 'pa', 'peppi_py', 'to_libmelee_stick', 'types']
                
                # Let's try the most likely candidates
                if hasattr(parse_peppi, 'get_slp'):
                    result = parse_peppi.get_slp(file_path)
                elif hasattr(parse_peppi, 'from_peppi'):
                    # First parse with peppi-py, then convert
                    import peppi_py
                    game_data = peppi_py.read_slippi(file_path)
                    result = parse_peppi.from_peppi(game_data)
                else:
                    # Fallback to direct peppi-py parsing
                    import peppi_py
                    result = peppi_py.read_slippi(file_path)
                
                parsed_data.append({
                    'name': os.path.basename(file_path),
                    'path': file_path,
                    'valid': True,
                    'data': result
                })
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(replay_files)} files...")
                    
            except Exception as e:
                parsed_data.append({
                    'name': os.path.basename(file_path),
                    'path': file_path,
                    'valid': False,
                    'reason': str(e)
                })
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(replay_files)} files...")
        
        # Save the parsed data
        import pickle
        with open("/processed/parsed.pkl", "wb") as f:
            pickle.dump(parsed_data, f)
        
        # Print summary
        valid_count = sum(1 for item in parsed_data if item.get('valid', False))
        invalid_count = len(parsed_data) - valid_count
        
        print(f"\n--- Processing Complete ---")
        print(f"✅ Successfully parsed: {valid_count} files")
        print(f"❌ Failed to parse: {invalid_count} files")
        
        if invalid_count > 0:
            # Show error breakdown
            reasons = {}
            for item in parsed_data:
                if not item.get('valid', False):
                    reason = str(item.get('reason', 'Unknown'))[:100]
                    reasons[reason] = reasons.get(reason, 0) + 1
            
            print("\nError breakdown:")
            for reason, count in reasons.items():
                print(f"  {reason}: {count} files")
        
    except Exception as e:
        print(f"❌ Error during reprocessing: {e}")
        import traceback
        traceback.print_exc()

@app.local_entrypoint()
def main():
    """
    Main function that fixes paths and then reprocesses
    """
    print("🚀 Starting slippi-ai path fix and reprocessing...")
    
    # Step 1: Fix the file paths
    fix_replay_paths.remote()
    
    # Step 2: Reprocess the replays
    reprocess_replays_after_fix.remote()
    
    print("✅ Complete! Your replays should now be properly processed.")