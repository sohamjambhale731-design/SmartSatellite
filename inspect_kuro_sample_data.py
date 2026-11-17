
import tarfile
import numpy as np
import json
from io import BytesIO
from pathlib import Path

TAR_PATH = Path(
    "datasets/kuro_siwo/train_GRD/shard-00000.tar"
)

sample_id = "000000"

channels = [
    "flood_vv",
    "flood_vh",
    "sec1_vv",
    "sec1_vh",
    "sec2_vv",
    "sec2_vh",
]

with tarfile.open(TAR_PATH, "r") as tar:
    print("INSPECTING SAMPLE:", sample_id)

    for channel in channels + ["dem", "mask", "valid_mask"]:
        filename = f"{sample_id}.{channel}.npy"

        file_obj = tar.extractfile(filename)
        if file_obj is None:
            print("Missing:", filename)
            continue

        array = np.load(BytesIO(file_obj.read()))

        print(f"\n{channel}")
        print("Shape:", array.shape)
        print("Data type:", array.dtype)
        print("Min:", np.nanmin(array))
        print("Max:", np.nanmax(array))
        print("Mean:", np.nanmean(array))

        if channel in ["mask", "valid_mask"]:
            print("Unique values:", np.unique(array))

    info_file = tar.extractfile(f"{sample_id}.info.json")

    if info_file:
        info = json.loads(info_file.read().decode("utf-8"))

        print("\nSAMPLE METADATA:")
        print(json.dumps(info, indent=2))
