
from pathlib import Path
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import json

REPO_ID = "adityabavadekar/dhara-flood-unet"
MODEL_DIR = Path("models/flood_unet")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

files = [
    "model.onnx",
    "model_card.json",
    "norm.json",
]

print("Downloading flood segmentation model...")

for filename in files:
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=str(MODEL_DIR)
    )
    print(f"Downloaded: {path}")

print("\nReading model configuration...")

with open(MODEL_DIR / "model_card.json", "r") as f:
    model_card = json.load(f)

with open(MODEL_DIR / "norm.json", "r") as f:
    normalization = json.load(f)

print("\nModel card:")
print(json.dumps(model_card, indent=2))

print("\nNormalization settings:")
print(json.dumps(normalization, indent=2))

print("\nInspecting ONNX model...")

ort.preload_dlls(directory="")

session = ort.InferenceSession(
    str(MODEL_DIR / "model.onnx"),
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

print("\nInput details:")
for item in session.get_inputs():
    print(item.name, item.shape, item.type)

print("\nOutput details:")
for item in session.get_outputs():
    print(item.name, item.shape, item.type)

print("\nActive providers:")
print(session.get_providers())

print("\nModel download and inspection completed.")

