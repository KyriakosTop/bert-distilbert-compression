## Efficient Compression Techniques for Large Language Models with Limited Compute Resources

This repository presents a systematic study on compressing DistilBERT using 8-bit and 4-bit quantization, as well as pruning, to enable efficient inference on low-resource hardware. The goal is to evaluate the trade-offs between model size, accuracy, latency, and memory consumption when deploying compressed models on CPUs and GPUs with limited compute capacity.

Experiments are conducted using the SST-2 sentiment classification task from the GLUE benchmark. We assess full-precision (FP32), 8-bit, and 4-bit inference modes—alongside structured pruning—across multiple hardware configurations, including:

- CPU-only inference
- NVIDIA T4 GPU (Google Colab)
- Consumer GPU: GTX 1050 Ti
- CPU inference with pruning and quantization combined
  
## Table of Contents

- [Motivation](#motivation)
- [Methodology](#methodology)
- [Experiments](#experiments)
  - [1. CPU Inference (FP32 vs 8-bit)](#1-cpu-inference-fp32-vs-8-bit)
  - [2. GPU Quantization on T4](#2-gpu-quantization-on-t4)
  - [3. CPU Pruning + Quantization](#3-cpu-pruning--quantization)
  - [4. GPU Quantization on GTX 1050 Ti](#4-gpu-quantization-on-gtx-1050-ti)
- [Key Findings](#key-findings)
- [Reproducing the Experiments](#how-to-run)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Motivation

Large language models (LLMs) have achieved state-of-the-art results in a wide range of natural language processing tasks. However, their computational and memory requirements make them difficult to deploy on everyday hardware such as laptops, older GPUs, or free-tier cloud environments.

This project explores whether aggressive compression techniques—such as 8-bit and 4-bit quantization, as well as structured pruning—can make models like DistilBERT practical to run on limited compute resources. By quantifying trade-offs between efficiency and accuracy, the project aims to provide actionable insights for developers working in constrained environments.

## Methodology

The study focuses on the DistilBERT model fine-tuned for the SST-2 sentiment classification task. All experiments use a consistent data pipeline and tokenization process to ensure comparability across models and hardware.

Compression techniques applied:

- **8-bit quantization**: Using PyTorch (CPU) and bitsandbytes (GPU)
- **4-bit quantization**: Using bitsandbytes (where supported)
- **Structured pruning**: L1 unstructured pruning applied to linear layers, followed by quantization (CPU)

Hardware tested:

- CPU-only inference (Google Colab, Intel Xeon @ 2.20GHz)
- NVIDIA T4 GPU (Google Colab)
- NVIDIA GTX 1050 Ti (local machine; CPU fallback to Intel Core i5-7500 @ 3.40GHz)

Each configuration was evaluated using:

- Accuracy (on the SST-2 validation set)
- Latency per sample (in milliseconds)
- Peak RAM usage (for CPU)
- VRAM usage and memory deltas (for GPU)

All models used were loaded from Hugging Face's Transformers library. Quantization and pruning were applied post-training, and all evaluations were done without additional fine-tuning.

## Experiments

The project consists of four experimental notebooks, each focused on a different hardware and compression configuration. All tests were performed on 100 validation samples from the SST-2 sentiment classification task.

### 1. CPU Inference (FP32 vs 8-bit)

- Environment: Google Colab (Intel Xeon @ 2.20GHz)
- Tools: PyTorch dynamic quantization
- 8-bit dynamic quantization reduced:
  - Latency by ~57%
  - Memory usage by ~98%
- Accuracy dropped slightly (from 91.06% to 89.33%)

### 2. GPU Quantization on T4

- Environment: Google Colab (NVIDIA T4 GPU)
- Tools: bitsandbytes 8-bit and 4-bit quantization
- Accuracy remained stable (94% FP32 and INT8, 93% for 4-bit)
- Latency increased for 8-bit due to kernel overhead
- 4-bit offered the best latency and lowest memory delta, but highest VRAM load

### 3. CPU Pruning + Quantization

- Environment: Google Colab (CPU only)
- Applied 30%, 40%, and 50% L1 unstructured pruning followed by 8-bit quantization
- Observed:
  - Gradual accuracy drop (from 90.5% to 87.2%)
  - Consistent latency improvement
  - Increased RAM usage due to PyTorch overhead
- Demonstrated pruning is effective for latency but not for memory reduction without sparse-aware backends

### 4. GPU Quantization on GTX 1050 Ti

- Environment: Local machine (GTX 1050 Ti + Intel Core i5-7500)
- bitsandbytes models loaded, but likely fell back to CPU
- Evidence:
  - Higher latency for 8-bit and 4-bit compared to FP32
  - Negative VRAM deltas during inference
- Confirms that older GPUs without compute capability 7.5 are not suitable for quantized LLM inference

## Key Findings

## Performance Summary Table

| Configuration               | Accuracy (%) | Latency (ms) | RAM ↑ (MB) | VRAM ↑ (MB) | Notes                             |
|----------------------------|--------------|--------------|------------|-------------|-----------------------------------|
| CPU FP32 (Colab)           | 91.06        | 352.0        | 273.00     | N/A         | Baseline                          |
| CPU INT8 (Colab)           | 89.33        | 151.3        | 5.81       | N/A         | 98% RAM drop, ~57% faster         |
| CPU Pruned+INT8 (30%)      | 90.48        | ~60,000      | 2330       | N/A         | Latency in full sweep, batch=16   |
| CPU Pruned+INT8 (50%)      | 87.16        | ~47,500      | 2697       | N/A         | Lower latency, higher RAM         |
| T4 FP32 (Colab)            | 94.00        | 12.4         | 213.52     | 30.00       | Fast and balanced                 |
| T4 INT8                    | 94.00        | 77.2         | 130.85     | 14.00       | Higher latency due to overhead    |
| T4 4-bit                   | 93.00        | 16.5         | 5.73       | 2.00        | Best latency + lowest RAM         |
| GTX 1050 Ti FP32           | 94.00        | 9.15         | 97.40      | 20.13       | True GPU execution                |
| GTX 1050 Ti INT8           | 94.00        | 98.44        | 48.71      | -12.38      | Likely CPU fallback               |
| GTX 1050 Ti 4-bit          | 93.00        | 11.27        | -4.93      | -3.00       | Likely CPU fallback               |


- **8-bit quantization** is highly effective on CPU: it drastically reduces memory usage and inference time with only a small drop in accuracy.
- On **T4 GPUs**, 8-bit and 4-bit quantization retain accuracy but may introduce latency overheads due to kernel fusion and runtime reordering.
- **4-bit quantization** significantly lowers memory deltas but increases total VRAM footprint—suitable for large models but less beneficial for small models like DistilBERT.
- **Structured pruning** improves latency but does not reduce memory unless combined with sparse-aware runtimes (e.g., ONNX Runtime or DeepSparse).
- **Older GPUs (like GTX 1050 Ti)** may load quantized models but fall back to CPU for inference, negating performance gains.
- Consistent data preprocessing across all notebooks ensured fair comparisons and reproducibility.

## Reproducing the Experiments

Each experiment is provided as a standalone Jupyter notebook:

- `baseline_cpu.ipynb` – CPU FP32 vs 8-bit quantization
- `quantization_gpu.ipynb` – T4 GPU quantization (FP32, 8-bit, 4-bit)
- `pruning_cpu.ipynb` – CPU pruning + 8-bit quantization
- `quantization_gtx1050ti.ipynb` – GTX 1050 Ti results

Run each notebook sequentially. They include:

- Environment setup and dependencies
- Model and dataset loading
- Inference and benchmarking
- Result summaries with metrics and tables

> Note: bitsandbytes 4-bit inference requires CUDA compute capability ≥7.5. On older GPUs (e.g., GTX 1050 Ti), fallback to CPU is likely.

## Dependencies

The notebooks rely on the following Python packages:

- `transformers`
- `datasets`
- `torch`
- `evaluate`
- `psutil`
- `bitsandbytes` (for 8-bit and 4-bit GPU quantization)
- `accelerate`
- `pynvml` (for GPU memory tracking)

These can be installed via `pip`:

```bash
pip install transformers datasets torch evaluate psutil bitsandbytes accelerate pynvml
```

> Some packages (like `bitsandbytes`) require a compatible CUDA environment and are only supported on Linux-based systems with NVIDIA GPUs.

## Acknowledgements

This project was conducted as part of the MSc Artificial Intelligence programme at the University of Hull (online).

Thanks to Hugging Face, PyTorch, and the open-source community for providing the tools used throughout this study.

The SST-2 dataset is part of the GLUE benchmark and was accessed via the Hugging Face Datasets library.
