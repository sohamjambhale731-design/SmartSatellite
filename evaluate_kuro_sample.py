
from pathlib import Path
from io import BytesIO
import tarfile
import json
import os

import numpy as np

# --------------------------------------------------
# 1. Configure NVIDIA DLL directories
# --------------------------------------------------

BASE = Path(".venv/Lib/site-packages/nvidia").resolve()

dll_dirs = [
    BASE / "cudnn/bin",
    BASE / "cublas/bin",
    BASE / "cuda_runtime/bin",
    BASE / "cuda_nvrtc/bin",
    BASE / "cufft/bin",
    BASE / "curand/bin",
    BASE / "nvjitlink/bin",
]

dll_handles = []

for folder in dll_dirs:
    if folder.is_dir():
        dll_handles.append(os.add_dll_directory(str(folder)))

valid_dirs = [str(p.resolve()) for p in dll_dirs if p.is_dir()]
os.environ["PATH"] = os.pathsep.join(valid_dirs) + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort

# --------------------------------------------------
# 2. Load model configuration
# --------------------------------------------------

MODEL_DIR = Path("models/flood_unet")
TAR_PATH = Path("datasets/kuro_siwo/train_GRD/shard-00000.tar")

with open(MODEL_DIR / "model_card.json", "r") as f:
    config = json.load(f)

mean = np.array(config["norm"]["mean"], dtype=np.float32).reshape(1, 6, 1, 1)
std = np.array(config["norm"]["std"], dtype=np.float32).reshape(1, 6, 1, 1)

temperature = float(config.get("temperature", 1.0))

# --------------------------------------------------
# 3. Read one sample from TAR
# --------------------------------------------------

sample_id = "000000"

channel_names = [
    "flood_vv",
    "flood_vh",
    "sec1_vv",
    "sec1_vh",
    "sec2_vv",
    "sec2_vh",
]

arrays = {}

with tarfile.open(TAR_PATH, "r") as tar:
    for name in channel_names + ["mask", "valid_mask"]:
        filename = f"{sample_id}.{name}.npy"
        file_obj = tar.extractfile(filename)

        if file_obj is None:
            raise FileNotFoundError(filename)

        arrays[name] = np.load(BytesIO(file_obj.read()))


# --------------------------------------------------
# 4. Convert linear sigma-zero to dB
# --------------------------------------------------

channels = []

for name in channel_names:
    arr = arrays[name].astype(np.float32)

    arr = np.maximum(arr, 1e-10)
    arr_db = 10.0 * np.log10(arr)

    # Remove the leading singleton dimension: (1,224,224)
    arr_db = np.squeeze(arr_db, axis=0)

    channels.append(arr_db)

# Stack into (6, 224, 224)
x = np.stack(channels, axis=0)

# Add batch dimension: (1, 6, 224, 224)
x = np.expand_dims(x, axis=0)

x = (x - mean) / std
x = x.astype(np.float32)

print("Corrected input shape:", x.shape)
print("Input min/max:", float(x.min()), float(x.max()))


# --------------------------------------------------
# 5. Load ONNX model on GPU
# --------------------------------------------------

ort.preload_dlls(directory="")

session = ort.InferenceSession(
    str(MODEL_DIR / "model.onnx"),
    providers=[
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider"
    ]
)

print("\nExecution providers:", session.get_providers())

# --------------------------------------------------
# 6. Run inference
# --------------------------------------------------

print("\nRunning inference...")

logits = session.run(
    ["logits"],
    {"x": x}
)[0]

print("Logits shape:", logits.shape)

# Temperature-scaled probabilities
scaled = logits / temperature
scaled = scaled - np.max(scaled, axis=1, keepdims=True)

exp_values = np.exp(scaled)
probabilities = exp_values / np.sum(
    exp_values, axis=1, keepdims=True
)

prediction = np.argmax(probabilities, axis=1)[0]

# --------------------------------------------------
# 7. Evaluate against ground truth
# --------------------------------------------------

ground_truth = arrays["mask"][0].astype(np.int32)
valid = arrays["valid_mask"][0] > 0

y_true = ground_truth[valid]
y_pred = prediction[valid]

print("\nGround truth classes:", np.unique(y_true))
print("Predicted classes:", np.unique(y_pred))

accuracy = np.mean(y_true == y_pred)
print(f"\nOverall accuracy: {accuracy:.4f}")

classes = [0, 1, 2]
class_names = ["No water", "Permanent water", "Flood"]

for class_id, class_name in zip(classes, class_names):
    tp = np.sum((y_true == class_id) & (y_pred == class_id))
    fp = np.sum((y_true != class_id) & (y_pred == class_id))
    fn = np.sum((y_true == class_id) & (y_pred != class_id))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0

    print(f"\n{class_name} (class {class_id})")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  IoU:       {iou:.4f}")

# --------------------------------------------------
# 8. Save prediction outputs
# --------------------------------------------------

output_dir = Path("outputs/kuro_siwo")
output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir / "sample_000000_prediction.npy", prediction)
np.save(output_dir / "sample_000000_ground_truth.npy", ground_truth)
np.save(output_dir / "sample_000000_probabilities.npy", probabilities[0])

print("\nSaved prediction files in:", output_dir.resolve())
print("\nInference and evaluation completed.")

