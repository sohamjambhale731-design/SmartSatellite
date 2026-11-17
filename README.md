# Smart Satellite Change Detection

AI-based multi-temporal satellite image analysis for detecting and monitoring environmental changes.

## Project Overview

Smart Satellite Change Detection is an ongoing research and development project focused on using satellite imagery and deep learning to identify changes on Earth's surface.

The broader project aims to support:

- Flood-affected region detection
- Landslide-affected region detection
- Urbanization monitoring

The current implemented research component focuses on **flood segmentation using Sentinel-1 SAR imagery**.

## Current Implementation

The current flood-detection pipeline uses a pretrained U-Net model with a ResNet34 encoder.

The model accepts six SAR input channels:

1. Post-event VV
2. Post-event VH
3. First pre-event VV
4. First pre-event VH
5. Second pre-event VV
6. Second pre-event VH

Input data is converted from linear sigma-zero values to decibels (dB), normalized using the supplied model configuration, and passed to the ONNX model.

The model predicts three classes:

| Class ID | Class |
|---|---|
| 0 | No water |
| 1 | Permanent water |
| 2 | Flood |

Inference is performed using ONNX Runtime with CUDA GPU acceleration when available.

## Technology Stack

- Python 3.13
- NumPy
- ONNX Runtime GPU
- CUDA and cuDNN
- Hugging Face Hub
- Matplotlib
- ONNX

## Model and Dataset

### Flood Segmentation Model

Model repository: [Dhara Flood U-Net](https://huggingface.co/adityabavadekar/dhara-flood-unet)

Architecture: U-Net with ResNet34 encoder.

Model format: ONNX.

### Dataset

Dataset: [Kuro Siwo Webdataset](https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset)

The project uses the GRD data split for flood-model evaluation.

The dataset contains pre-event and post-event SAR imagery, flood masks, and validity masks.

## Evaluation Results

The following results were obtained by evaluating 100 samples from one training shard and 100 samples from one held-out test shard.

| Metric | Training shard | Held-out test shard |
|---|---:|---:|
| Overall accuracy | 96.64% | 97.62% |
| Flood precision | 93.34% | 83.21% |
| Flood recall | 91.81% | 78.12% |
| Flood IoU | 86.17% | 67.48% |
| Flood Dice | 92.57% | 80.59% |

**Evaluation scope:** These results represent 100 samples from a training shard and 100 samples from a single held-out test shard. They are not full-dataset benchmark results.

The difference between training and held-out test performance indicates that further evaluation and model improvement are needed.

## Project Structure

```text
SmartSatellite/
â”œâ”€â”€ datasets/
â”œâ”€â”€ models/
â”‚   â””â”€â”€ flood_unet/
â”‚       â”œâ”€â”€ model.onnx
â”‚       â”œâ”€â”€ model_card.json
â”‚       â””â”€â”€ norm.json
â”œâ”€â”€ outputs/
â”œâ”€â”€ services/
â”‚   â”œâ”€â”€ evaluation.py
â”‚   â”œâ”€â”€ flood_detection.py
â”‚   â””â”€â”€ preprocessing.py
â”œâ”€â”€ static/
â”œâ”€â”€ templates/
â”œâ”€â”€ download_flood_model.py
â”œâ”€â”€ download_kuro_sample.py
â”œâ”€â”€ download_kuro_test.py
â”œâ”€â”€ evaluate_kuro_100.py
â”œâ”€â”€ evaluate_kuro_sample.py
â”œâ”€â”€ evaluate_kuro_test_100.py
â”œâ”€â”€ inspect_kuro_dataset.py
â”œâ”€â”€ inspect_kuro_splits.py
â”œâ”€â”€ test_flood_inference.py
â”œâ”€â”€ test_gpu.py
â”œâ”€â”€ visualize_kuro_sample.py
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
