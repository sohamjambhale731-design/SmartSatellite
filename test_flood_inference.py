
from pathlib import Path
import os
import json
import numpy as np

BASE = Path(".venv/Lib/site-packages/nvidia").resolve()

# Keep DLL directory handles alive throughout inference
dll_handles = []

dll_dirs = [
    BASE / "cudnn/bin",
    BASE / "cublas/bin",
    BASE / "cuda_runtime/bin",
    BASE / "cuda_nvrtc/bin",
    BASE / "cufft/bin",
    BASE / "curand/bin",
    BASE / "nvjitlink/bin",
]

for folder in dll_dirs:
    folder = folder.resolve()

    if folder.is_dir():
        dll_handles.append(os.add_dll_directory(str(folder)))
        print("Added DLL directory:", folder)
    else:
        print("Directory not found:", folder)
        
# Add NVIDIA DLL folders to this Python process's PATH
existing_path = os.environ.get("PATH", "")

valid_dirs = [
    str(folder.resolve())
    for folder in dll_dirs
    if folder.is_dir()
]

os.environ["PATH"] = os.pathsep.join(valid_dirs) + os.pathsep + existing_path

print("NVIDIA DLL folders added to process PATH.")

import onnxruntime as ort

MODEL_DIR = Path("models/flood_unet")

with open(MODEL_DIR / "model_card.json", "r") as f:
    config = json.load(f)

ort.preload_dlls(directory="")

options = ort.SessionOptions()
options.enable_profiling = True

session = ort.InferenceSession(
    str(MODEL_DIR / "model.onnx"),
    sess_options=options,
    providers=[
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider"
    ]
)

print("\nActive providers:", session.get_providers())

sample = np.zeros((1, 6, 224, 224), dtype=np.float32)

print("\nRunning inference...")
logits = session.run(["logits"], {"x": sample})[0]

print("Output shape:", logits.shape)

profile_path = session.end_profiling()

with open(profile_path, "r") as f:
    events = json.load(f)

provider_counts = {}

for event in events:
    if event.get("cat") == "Node":
        provider = event.get("args", {}).get("provider", "Unknown")
        provider_counts[provider] = provider_counts.get(provider, 0) + 1

print("\nExecution provider counts:", provider_counts)
print("\nProfiling file:", profile_path)
