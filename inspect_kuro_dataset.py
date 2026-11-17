
from huggingface_hub import HfApi

REPO_ID = "orion-ai-lab/Kuro-Siwo-Webdataset"

api = HfApi()

print("Checking Kuro Siwo dataset files...\n")

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

grd_files = [
    name for name in files
    if name.startswith("train_GRD/")
]

print(f"Found {len(grd_files)} files in train_GRD:\n")

for name in grd_files[:30]:
    print(name)

