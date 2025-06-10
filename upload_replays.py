import modal
import os
from pathlib import Path
from tqdm import tqdm

MAX_CONCURRENCY = 50

app = modal.App("replay-uploader-fast")
volume = modal.Volume.from_name("slippi-ai-replays")
image = modal.Image.debian_slim().pip_install("tqdm")

@app.function(
    image=image,
    volumes={"/replays": volume},
    max_containers=MAX_CONCURRENCY,
    timeout=600
)
def upload_file(upload_request: dict):
    content = upload_request["content"]
    remote_path = upload_request["remote_path"]
    remote_full_path = Path("/replays") / remote_path
    remote_full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(remote_full_path, "wb") as f_remote:
        f_remote.write(content)
    volume.commit()
    return f"Uploaded {remote_path}"

@app.local_entrypoint()
def main(local_dir: str):
    local_dir_path = Path(local_dir)
    if not local_dir_path.is_dir():
        print(f"❌ Error: {local_dir} is not a valid directory.")
        return

    slp_files = list(local_dir_path.glob("**/*.slp"))
    if not slp_files:
        print(f"🤷 No .slp files found in {local_dir}.")
        return
        
    print(f"Found {len(slp_files)} files. Preparing for parallel upload...")

    upload_requests = []
    for local_path in slp_files:
        with open(local_path, "rb") as f_local:
            content = f_local.read()
        remote_path = local_path.relative_to(local_dir_path)
        upload_requests.append({"content": content, "remote_path": str(remote_path)})

    print(f"🚀 Uploading {len(upload_requests)} files with a max concurrency of {MAX_CONCURRENCY}...")
    with tqdm(total=len(upload_requests), desc="Uploading files") as pbar:
        for result in upload_file.map(upload_requests, return_exceptions=True):
            if isinstance(result, Exception):
                print(f"Error uploading a file: {result}")
            pbar.update(1)

    print("\n✅ All files have been uploaded.")