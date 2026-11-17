
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

# Explicitly preload CUDA and cuDNN from NVIDIA Python packages
ort.preload_dlls(directory="")

# Create a tiny ONNX model
a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 2])
b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 2])
c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 2])

node = helper.make_node("Add", ["A", "B"], ["C"])

graph = helper.make_graph([node], "GPU_Test", [a, b], [c])
model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 13)]
)
model.ir_version = 10
onnx.save(model, "gpu_test.onnx")

# Create session with CUDA
session = ort.InferenceSession(
    "gpu_test.onnx",
    providers=[
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider"
    ]
)

print("Active providers:", session.get_providers())

x = np.array([[2.0, 3.0]], dtype=np.float32)
y = np.array([[4.0, 5.0]], dtype=np.float32)

result = session.run(["C"], {"A": x, "B": y})

print("Result:", result[0])
