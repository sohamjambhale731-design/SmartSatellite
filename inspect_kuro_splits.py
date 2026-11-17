
from huggingface_hub import HfApi

REPO_ID = "orion-ai-lab/Kuro-Siwo-Webdataset"

api = HfApi()

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

print("Available dataset splits:\n")

splits = set()

for name in files:
    if "/" in name:
        splits.add(name.split("/")[0])

for split in sorted(splits):
    print(split)

print("\nShard counts:")

for split in sorted(splits):
    count = sum(
        1 for name in files
        if name.startswith(split + "/") and name.endswith(".tar")
    )
    print(f"{split}: {count} shards")
    