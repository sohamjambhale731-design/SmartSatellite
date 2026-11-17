
from huggingface_hub import HfApi

REPO_ID = "orion-ai-lab/Kuro-Siwo-Webdataset"

api = HfApi()

files = api.list_repo_tree(
    repo_id=REPO_ID,
    repo_type="dataset",
    path_in_repo="test_GRD",
    recursive=True
)

total_bytes = 0

print("TEST_GRD SHARD DETAILS\n")

for item in files:
    if item.path.endswith(".tar"):
        size = item.size
        total_bytes += size

        print(
            f"{item.path} | "
            f"{size / (1024**3):.2f} GB"
        )

print("\nTotal test data size:")
print(f"{total_bytes / (1024**3):.2f} GB")

