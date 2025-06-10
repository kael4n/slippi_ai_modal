# debug_replay_files_fixed.py - Fixed debug script for replay processing

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
    name="slippi-ai-debug-replays-fixed",
    image=image,
)

@app.function(
    volumes={
        "/replays": replays_volume,
        "/processed": processed_data_volume,
    },
    timeout=3600
)
def debug_replay_files():
    import zipfile
    import pickle
    import os
    import sys
    
    print("\n--- 🔍 Debug Replay Files (Fixed) ---")
    
    # Check what was processed
    if os.path.exists("/processed/parsed.pkl"):
        print("Loading parsed.pkl to see what was processed...")
        with open("/processed/parsed.pkl", "rb") as f:
            parsed_data = pickle.load(f)
        
        print(f"Parsed data type: {type(parsed_data)}")
        print(f"Parsed data length: {len(parsed_data) if hasattr(parsed_data, '__len__') else 'N/A'}")
        
        # Count valid vs invalid
        if isinstance(parsed_data, list):
            valid_count = sum(1 for item in parsed_data if isinstance(item, dict) and item.get('valid', False))
            invalid_count = len(parsed_data) - valid_count
            print(f"Valid files: {valid_count}")
            print(f"Invalid files: {invalid_count}")
            
            # Show reasons for invalid files
            reasons = {}
            for item in parsed_data:
                if isinstance(item, dict) and not item.get('valid', False):
                    reason = item.get('reason', 'Unknown')
                    reasons[reason] = reasons.get(reason, 0) + 1
            
            print("\nReasons for invalid files:")
            for reason, count in reasons.items():
                print(f"  {reason}: {count} files")
        
        print("\nFirst few entries in parsed data:")
        if isinstance(parsed_data, list):
            for i, item in enumerate(parsed_data[:5]):
                if isinstance(item, dict):
                    print(f"  [{i}]: {item.get('name', 'No name')} - Valid: {item.get('valid', False)}")
                    if not item.get('valid', False):
                        reason = item.get('reason', 'No reason')
                        print(f"       Reason: {reason[:100]}...")
    else:
        print("❌ No parsed.pkl found in /processed")
    
    # Check file structure
    print("\n--- File Structure Analysis ---")
    
    replay_files = []
    for root, dirs, files in os.walk("/replays"):
        print(f"Directory: {root}")
        print(f"  Subdirectories: {dirs}")
        print(f"  Files: {len(files)} total")
        
        slp_files = [f for f in files if f.endswith('.slp')]
        print(f"  .slp files: {len(slp_files)}")
        
        for file in slp_files[:5]:  # Show first 5 files
            full_path = os.path.join(root, file)
            replay_files.append(full_path)
            print(f"    {file} ({os.path.getsize(full_path)} bytes)")
    
    if not replay_files:
        print("❌ No replay files found!")
        return
    
    # Test manual parsing with corrected API
    print("\n--- Manual Replay Testing (Fixed) ---")
    
    test_file = replay_files[0]
    print(f"Testing replay file: {test_file}")
    print(f"File size: {os.path.getsize(test_file)} bytes")
    
    # Check if file path contains backslashes (Windows-style paths)
    if '\\' in test_file:
        print(f"⚠️  Warning: File path contains backslashes: {test_file}")
        # Try to fix the path
        fixed_path = test_file.replace('\\', '/')
        print(f"Fixed path: {fixed_path}")
        if os.path.exists(fixed_path):
            test_file = fixed_path
            print("✅ Fixed path exists")
        else:
            print("❌ Fixed path doesn't exist")
    
    # Try to parse it directly with peppi-py (FIXED API USAGE)
    try:
        import peppi_py
        
        print("Attempting to parse with peppi-py (using file path)...")
        # FIXED: Pass file path directly, not file handle
        game_data = peppi_py.read_slippi(test_file)
        
        print("✅ Successfully parsed with peppi-py!")
        print(f"Game data type: {type(game_data)}")
        
        # Check game metadata
        if hasattr(game_data, 'start'):
            start = game_data.start
            print(f"Game start info: {start}")
        
        if hasattr(game_data, 'metadata'):
            metadata = game_data.metadata
            print(f"Game metadata: {metadata}")
            
        if hasattr(game_data, 'frames'):
            frames = game_data.frames
            print(f"Frames data available: {frames is not None}")
            if frames:
                # Check if frames have the expected structure
                print(f"Frames type: {type(frames)}")
                # Try to access frame data
                try:
                    if hasattr(frames, 'ports'):
                        print(f"Ports available: {hasattr(frames, 'ports')}")
                        if hasattr(frames.ports, '__len__'):
                            print(f"Number of ports: {len(frames.ports)}")
                except Exception as e:
                    print(f"Error accessing frame data: {e}")
                    
        print("✅ Game has valid structure - should work with slippi-ai")
        
    except Exception as e:
        print(f"❌ Failed to parse with peppi-py: {e}")
        import traceback
        traceback.print_exc()
    
    # Let's check what modules are available in slippi-ai
    print("\n--- slippi-ai Module Structure ---")
    
    try:
        # Add the slippi-ai path to sys.path if not already there
        if "/root/slippi-ai" not in sys.path:
            sys.path.append("/root/slippi-ai")
        
        # Try to import slippi_db
        import slippi_db
        print(f"✅ Successfully imported slippi_db")
        print(f"slippi_db location: {slippi_db.__file__}")
        
        # Check what's in slippi_db
        print(f"slippi_db contents: {dir(slippi_db)}")
        
        # Try to import parse_peppi
        try:
            from slippi_db import parse_peppi
            print(f"✅ Imported parse_peppi")
            print(f"parse_peppi contents: {dir(parse_peppi)}")
        except ImportError as e:
            print(f"❌ Could not import parse_peppi: {e}")
        
        # Look for other parsing modules
        import slippi_db
        for attr in dir(slippi_db):
            if 'parse' in attr.lower():
                print(f"Found parsing-related attribute: {attr}")
                
    except Exception as e:
        print(f"❌ Error exploring slippi-ai modules: {e}")
        import traceback
        traceback.print_exc()
    
    # Try to find the actual parsing function
    print("\n--- Finding Parsing Functions ---")
    
    try:
        # Look for the actual parsing logic
        import slippi_db.parse_peppi as parse_peppi
        
        # Check for different possible function names
        possible_functions = [
            'parse_file', 'parse_replay', 'parse_slippi', 'parse_game',
            'process_file', 'read_file', 'load_replay'
        ]
        
        for func_name in possible_functions:
            if hasattr(parse_peppi, func_name):
                print(f"✅ Found function: {func_name}")
                func = getattr(parse_peppi, func_name)
                print(f"Function type: {type(func)}")
                
                # Try to call it
                try:
                    result = func(test_file)
                    print(f"✅ Function {func_name} returned: {type(result)}")
                    if result is not None:
                        print("✅ File processed successfully!")
                    else:
                        print("❌ Function returned None")
                    break
                except Exception as e:
                    print(f"❌ Error calling {func_name}: {e}")
            else:
                print(f"❌ No function named {func_name}")
                
    except Exception as e:
        print(f"❌ Error testing parsing functions: {e}")
        import traceback
        traceback.print_exc()

@app.local_entrypoint()
def main():
    debug_replay_files.remote()