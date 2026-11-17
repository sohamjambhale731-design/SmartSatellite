
from huggingface_hub import hf_hub_download
from pathlib import Path
import tarfile

REPO_ID = "orion-ai-lab/Kuro-Siwo-Webdataset"
FILENAME = "test_GRD/shard-00011.tar"

download_dir = Path("datasets/kuro_siwo")
download_dir.mkdir(parents=True, exist_ok=True)

print("Downloading independent test shard...")

tar_path = hf_hub_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    filename=FILENAME,
    local_dir=download_dir
)

print("\nDownloaded:", tar_path)

print("\nInspecting first sample...")

with tarfile.open(tar_path, "r") as tar:
    members = tar.getmembers()

    print("Total archive entries:", len(members))

    for member in members[:20]:
        print(member.name, "|", member.size, "bytes")
