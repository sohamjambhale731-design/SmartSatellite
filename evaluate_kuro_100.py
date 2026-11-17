
from pathlib import Path
from io import BytesIO
import tarfile
import json
import os
import csv

import numpy as np

# ==================================================
# 1. Configure NVIDIA DLL directories
# ==================================================

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
os.environ["PATH"] = (
    os.pathsep.join(valid_dirs)
    + os.pathsep
    + os.environ.get("PATH", "")
)

import onnxruntime as ort

# ==================================================
# 2. Configuration
# ==================================================

MODEL_DIR = Path("models/flood_unet")
TAR_PATH = Path("datasets/kuro_siwo/train_GRD/shard-00000.tar")
OUTPUT_DIR = Path("outputs/kuro_siwo/evaluation_100")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SAMPLES = 100

CHANNEL_NAMES = [
    "flood_vv",
    "flood_vh",
    "sec1_vv",
    "sec1_vh",
    "sec2_vv",
    "sec2_vh",
]

REQUIRED_FILES = set(CHANNEL_NAMES + ["mask", "valid_mask"])

with open(MODEL_DIR / "model_card.json", "r") as f:
    config = json.load(f)

mean = np.array(
    config["norm"]["mean"], dtype=np.float32
).reshape(1, 6, 1, 1)

std = np.array(
    config["norm"]["std"], dtype=np.float32
).reshape(1, 6, 1, 1)

temperature = float(config.get("temperature", 1.0))

# ==================================================
# 3. Load ONNX model
# ==================================================

ort.preload_dlls(directory="")

session = ort.InferenceSession(
    str(MODEL_DIR / "model.onnx"),
    providers=[
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ],
)

print("Execution providers:", session.get_providers())
print("Target samples:", TARGET_SAMPLES)

# ==================================================
# 4. Inference function
# ==================================================

def predict_sample(arrays):

    channels = []

    for name in CHANNEL_NAMES:
        arr = arrays[name].astype(np.float32)

        # Linear sigma-zero to dB
        arr = np.maximum(arr, 1e-10)
        arr_db = 10.0 * np.log10(arr)

        # Remove singleton channel dimension
        arr_db = np.squeeze(arr_db, axis=0)

        channels.append(arr_db)

    # (1, 6, 224, 224)
    x = np.stack(channels, axis=0)[None, ...]
    x = ((x - mean) / std).astype(np.float32)

    logits = session.run(["logits"], {"x": x})[0]

    # Temperature-scaled softmax
    scaled = logits / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)

    exp_values = np.exp(scaled)
    probabilities = exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )

    prediction = np.argmax(
        probabilities, axis=1
    )[0]

    return prediction


# ==================================================
# 5. Metric calculation
# ==================================================

confusion = np.zeros((3, 3), dtype=np.int64)

sample_results = []

processed = 0
samples_with_flood = 0
excluded_label_pixels = 0

# ==================================================
# 6. Read samples sequentially from TAR
# ==================================================

print("\nStarting evaluation...\n")

with tarfile.open(TAR_PATH, "r") as tar:

    current_id = None
    arrays = {}

    for member in tar:

        filename = member.name.rsplit("/", 1)[-1]

        if "." not in filename:
            continue

        sample_id, suffix = filename.split(".", 1)

        if not sample_id.isdigit():
            continue

        if int(sample_id) >= TARGET_SAMPLES:
            break

        if suffix not in {
            name + ".npy" for name in REQUIRED_FILES
        }:
            continue

        if current_id is None:
            current_id = sample_id

        if sample_id != current_id:
            continue

        file_obj = tar.extractfile(member)

        if file_obj is None:
            continue

        array = np.load(BytesIO(file_obj.read()))
        key = suffix.removesuffix(".npy")
        arrays[key] = array

        # Process when all required files are available
        if REQUIRED_FILES.issubset(arrays.keys()):

            prediction = predict_sample(arrays)

            ground_truth = arrays["mask"][0].astype(np.int32)
            valid = arrays["valid_mask"][0] > 0

            # Exclude invalid pixels and unexpected labels
            valid &= np.isin(ground_truth, [0, 1, 2])

            excluded_label_pixels += int(
                np.sum((arrays["valid_mask"][0] > 0) & ~np.isin(
                    ground_truth, [0, 1, 2]
                ))
            )

            y_true = ground_truth[valid]
            y_pred = prediction[valid]

            # Update global confusion matrix
            encoded = 3 * y_true + y_pred

            confusion += np.bincount(
                encoded, minlength=9
            ).reshape(3, 3)

            # Per-sample flood IoU
            gt_flood = y_true == 2
            pred_flood = y_pred == 2

            tp = int(np.sum(gt_flood & pred_flood))
            fp = int(np.sum(~gt_flood & pred_flood))
            fn = int(np.sum(gt_flood & ~pred_flood))

            union = tp + fp + fn

            flood_iou = tp / union if union else float("nan")

            if np.sum(gt_flood) > 0:
                samples_with_flood += 1
                sample_results.append({
                    "sample_id": current_id,
                    "flood_iou": flood_iou,
                    "flood_pixels": int(np.sum(gt_flood)),
                    "predicted_flood_pixels": int(np.sum(pred_flood)),
                })

            processed += 1

            print(
                f"Processed {processed}/{TARGET_SAMPLES} "
                f"| Sample {current_id} "
                f"| Flood IoU: "
                f"{flood_iou:.4f}" if union else
                f"Processed {processed}/{TARGET_SAMPLES} "
                f"| Sample {current_id} | No flood pixels"
            )

            if processed >= TARGET_SAMPLES:
                break

            current_id = None
            arrays = {}

# ==================================================
# 7. Calculate aggregate metrics
# ==================================================

if processed == 0:
    raise RuntimeError("No samples were processed.")

tp_flood = int(confusion[2, 2])
fp_flood = int(confusion[:, 2].sum() - tp_flood)
fn_flood = int(confusion[2, :].sum() - tp_flood)

total_pixels = int(confusion.sum())
correct_pixels = int(np.trace(confusion))

accuracy = (
    correct_pixels / total_pixels
    if total_pixels else 0.0
)

precision = (
    tp_flood / (tp_flood + fp_flood)
    if tp_flood + fp_flood else 0.0
)

recall = (
    tp_flood / (tp_flood + fn_flood)
    if tp_flood + fn_flood else 0.0
)

flood_iou = (
    tp_flood / (tp_flood + fp_flood + fn_flood)
    if tp_flood + fp_flood + fn_flood else 0.0
)

dice = (
    2 * tp_flood / (2 * tp_flood + fp_flood + fn_flood)
    if 2 * tp_flood + fp_flood + fn_flood else 0.0
)

sample_ious = [
    row["flood_iou"]
    for row in sample_results
    if np.isfinite(row["flood_iou"])
]

mean_sample_iou = (
    float(np.mean(sample_ious))
    if sample_ious else 0.0
)

# ==================================================
# 8. Print results
# ==================================================

print("\n" + "=" * 50)
print("KURO SIWO FLOOD MODEL - EVALUATION RESULTS")
print("=" * 50)

print("Samples processed:", processed)
print("Samples containing flood:", samples_with_flood)
print("Valid pixels evaluated:", total_pixels)
print("Unexpected labels excluded:", excluded_label_pixels)

print(f"\nOverall accuracy: {accuracy:.4f}")
print(f"Flood precision:  {precision:.4f}")
print(f"Flood recall:     {recall:.4f}")
print(f"Flood IoU:        {flood_iou:.4f}")
print(f"Flood Dice:       {dice:.4f}")
print(f"Mean sample IoU:  {mean_sample_iou:.4f}")

print("\nFlood confusion counts:")
print("True positives:", tp_flood)
print("False positives:", fp_flood)
print("False negatives:", fn_flood)

print("\nFull confusion matrix:")
print(confusion)

# ==================================================
# 9. Save reports
# ==================================================

report = {
    "samples_processed": processed,
    "samples_containing_flood": samples_with_flood,
    "valid_pixels": total_pixels,
    "unexpected_labels_excluded": excluded_label_pixels,
    "overall_accuracy": accuracy,
    "flood_precision": precision,
    "flood_recall": recall,
    "flood_iou": flood_iou,
    "flood_dice": dice,
    "mean_sample_flood_iou": mean_sample_iou,
    "flood_true_positives": tp_flood,
    "flood_false_positives": fp_flood,
    "flood_false_negatives": fn_flood,
    "confusion_matrix": confusion.tolist(),
}

with open(OUTPUT_DIR / "evaluation_report.json", "w") as f:
    json.dump(report, f, indent=4)

with open(OUTPUT_DIR / "sample_metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sample_id",
            "flood_iou",
            "flood_pixels",
            "predicted_flood_pixels",
        ],
    )
    writer.writeheader()
    writer.writerows(sample_results)

print("\nReports saved in:", OUTPUT_DIR.resolve())
print("Evaluation completed.")
