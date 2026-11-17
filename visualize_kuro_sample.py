
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE = Path("outputs/kuro_siwo")

prediction = np.load(BASE / "sample_000000_prediction.npy")
ground_truth = np.load(BASE / "sample_000000_ground_truth.npy")
probabilities = np.load(BASE / "sample_000000_probabilities.npy")

# Load the post-event VV image from the original archive
import tarfile
from io import BytesIO

with tarfile.open(
    "datasets/kuro_siwo/train_GRD/shard-00000.tar", "r"
) as tar:
    data = tar.extractfile("000000.flood_vv.npy")
    post_vv = np.load(BytesIO(data.read()))[0]

# Create figure
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(post_vv, cmap="gray")
axes[0].set_title("Post-event SAR (VV)")

axes[1].imshow(ground_truth, vmin=0, vmax=2, cmap="viridis")
axes[1].set_title("Ground-truth mask")

axes[2].imshow(prediction, vmin=0, vmax=2, cmap="viridis")
axes[2].set_title("Predicted mask")

im = axes[3].imshow(probabilities[2], cmap="inferno", vmin=0, vmax=1)
axes[3].set_title("Flood probability")
fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

for ax in axes:
    ax.axis("off")

fig.suptitle("Kuro Siwo Flood Detection — Sample 000000")
plt.tight_layout()

output = BASE / "sample_000000_comparison.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
print("Saved visualization:", output.resolve())

plt.show()
