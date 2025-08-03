# Efficient Compression Techniques for Large Language Models on Limited Compute Resources

**Why run large language models on limited hardware at all?**  
Because most developers don't have access to high-end GPUs. Because deploying models to laptops, embedded devices, or free-tier cloud services is increasingly common. And because not every task needs a trillion-parameter model — but everyone still wants fast, accurate results.

This project tackles a practical and relevant challenge:  
**Can transformer-based models like DistilBERT or BERT-base be made efficient enough to run on CPUs or older GPUs — without retraining and without sacrificing too much accuracy?**

To answer this, we apply **8-bit and 4-bit quantization** and **structured pruning** to fine-tuned transformer models and benchmark their performance on the SST-2 sentiment classification task across a range of hardware environments. The goal is simple:  
**Make these models smaller, faster, and deployable — without compromising usefulness.**

---

## 📌 Table of Contents

1. [Motivation](#1-motivation)  
2. [Methodology](#2-methodology)  
3. [Experiments and Results](#3-experiments-and-results)  
   - [3.1 DistilBERT on CPU (n1)](#31-distilbert-on-cpu-n1)  
   - [3.2 DistilBERT on CPU — Pruned + Quantized (n2)](#32-distilbert-on-cpu--pruned--quantized-n2)  
   - [3.3 DistilBERT on T4 GPU (n3)](#33-distilbert-on-t4-gpu-n3)  
   - [3.4 DistilBERT on GTX 1050 Ti (n4)](#34-distilbert-on-gtx-1050-ti-n4)  
   - [3.5 BERT-base on T4 GPU (n5)](#35-bert-base-on-t4-gpu-n5)  
   - [3.6 DistilBERT via ONNX Runtime (n6)](#36-distilbert-via-onnx-runtime-n6)  
4. [Key Insights](#4-key-insights)  
5. [How to Reproduce](#5-how-to-reproduce)  
6. [Dependencies](#6-dependencies)  
7. [References](#7-references)  
8. [Acknowledgements](#8-acknowledgements)

---

## 1. Motivation

Transformer models are powerful — but deploying them on limited hardware remains a challenge. This project explores whether post-training compression techniques can make models like DistilBERT and BERT practical to run on:

- Free-tier CPUs (e.g., Google Colab)
- Consumer GPUs (e.g., GTX 1050 Ti)
- Older edge or embedded systems

We measure how much accuracy, latency, and memory can be traded off using 8-bit/4-bit quantization and structured pruning — without retraining the model.

---

## 2. Methodology

- **Task:** SST-2 sentiment classification  
- **Models:** DistilBERT (66M) and BERT-base (110M), fine-tuned  
- **Compression Techniques:**
  - 8-bit dynamic quantization (PyTorch, ONNX, bitsandbytes)
  - 4-bit quantization (bitsandbytes QLoRA-style)
  - L1 structured pruning (PyTorch)
- **Evaluation Metrics:** Accuracy, per-sample latency (ms), RAM/VRAM usage (MB)
- **Hardware Platforms:**
  - Intel Xeon (Colab CPU)
  - NVIDIA T4 (Colab GPU)
  - NVIDIA GTX 1050 Ti + Intel Core i5-7500 (local)

Each experiment reuses a consistent data loading and tokenization pipeline for fair comparison.

---

## 3. Experiments and Results

### 3.1 DistilBERT on CPU (n1)

| Metric        | FP32      | INT8      | Δ (%)    |
|---------------|-----------|-----------|----------|
| Accuracy      | 91.06%    | 89.33%    | -1.73%   |
| Latency (ms)  | 352.0     | 151.3     | -57%     |
| RAM (MB)      | 273.00    | 5.81      | -98%     |

✅ 8-bit quantization drastically improves latency and RAM usage with minimal accuracy drop.

---

### 3.2 DistilBERT on CPU – Pruned + Quantized (n2)

| Pruning | Accuracy | Latency (s) | RAM (MB) |
|---------|----------|-------------|----------|
| 30%     | 90.48%   | 60          | 2330     |
| 40%     | 88.87%   | 54          | 2500     |
| 50%     | 87.16%   | 47.5        | 2697     |

✅ Pruning reduces latency further.  
⚠️ RAM usage increases due to PyTorch masking. No memory benefits without sparse-aware runtimes.

---

### 3.3 DistilBERT on T4 GPU (n3)

| Precision | Accuracy | Latency (ms) | VRAM Total (MB) | VRAM Δ (MB) | RAM Δ (MB) |
|-----------|----------|--------------|------------------|-------------|------------|
| FP32      | 94.00%   | 12.40        | 659.88           | 30.00       | 213.52     |
| 8-bit     | 94.00%   | 77.20        | 843.88           | 14.00       | 130.85     |
| 4-bit     | 93.00%   | 16.50        | 953.88           | 2.00        | 5.73       |

⚠️ 8-bit slower due to kernel overhead.  
✅ 4-bit has lowest memory deltas but increases total VRAM.

---

### 3.4 DistilBERT on GTX 1050 Ti (n4)

| Precision | Accuracy | Latency (ms) | VRAM Δ (MB) | RAM Δ (MB) |
|-----------|----------|--------------|-------------|------------|
| FP32      | 94.00%   | 9.15         | +20.13      | 97.40      |
| 8-bit     | 94.00%   | 98.44        | -12.38      | 48.71      |
| 4-bit     | 93.00%   | 11.27        | -3.00       | -4.93      |

⚠️ Quantized models fell back to CPU → higher latency, negative VRAM delta.

---

### 3.5 BERT-base on T4 GPU (n5)

| Precision | Accuracy | Latency (ms) | VRAM Total (MB) | VRAM Δ (MB) | RAM Δ (MB) |
|-----------|----------|--------------|------------------|-------------|------------|
| FP32      | 92.00%   | 12.83        | 1289.88          | 32.00       | 63.26      |
| 8-bit     | 92.00%   | 94.04        | 1081.88          | 12.00       | 20.96      |
| 4-bit     | 92.00%   | 20.75        | 1195.88          | 6.00        | 0.51       |

✅ All variants maintain accuracy.  
✅ 4-bit offers best memory efficiency.  
⚠️ 8-bit slower due to fused kernel overhead.

---

### 3.6 DistilBERT via ONNX Runtime (n6)

| Precision | Accuracy | Inference Time (s) | RAM Δ (MB) |
|-----------|----------|--------------------|------------|
| FP32      | 91.06%   | 191.68             | ~0.00      |
| INT8      | 90.48%   | 121.08             | ~0.00      |

✅ ONNX quantization yields ~36.8% speedup with minimal accuracy loss.  
⚠️ RAM unchanged due to measurement granularity.

---

## 4. Key Insights

- **8-bit quantization is highly effective on CPUs** (speed + memory, minimal accuracy loss).
- **Structured pruning reduces latency** but doesn't save RAM without sparse-aware inference engines.
- **bitsandbytes 4-bit quantization** works best on larger models and modern GPUs.
- **Quantized models on old GPUs (e.g., GTX 1050 Ti)** fall back to CPU — misleading gains.
- **BERT benefits more from quantization than DistilBERT**, justifying compression for bigger models.
- **ONNX Runtime** is a lightweight, portable option for CPU quantized inference.

---

## 5. How to Reproduce

Run the following notebooks in order:

| Notebook Filename                  | Description                              |
|-----------------------------------|------------------------------------------|
| `n1_dbert_quant_cpu.ipynb`        | DistilBERT CPU FP32 vs 8-bit             |
| `n2_dbert_quant_prun_cpu.ipynb`   | DistilBERT CPU pruning + quantization    |
| `n3_dbert_quant_gpu_t4.ipynb`     | DistilBERT on T4 (FP32, 8-bit, 4-bit)    |
| `n4_dbert_quant_gpu_gtx.ipynb`    | DistilBERT on GTX 1050 Ti                |
| `n5_bert_quant_gpu_t4.ipynb`      | BERT-base on T4 (FP32, 8-bit, 4-bit)     |
| `n6_dbert_onnx_cpu.ipynb`         | DistilBERT with ONNX Runtime             |

Each notebook is self-contained with metrics and visualizations.

---

## 6. References

- Sanh, V. et al. (2019). [DistilBERT: A distilled version of BERT](https://arxiv.org/abs/1910.01108)  
- Devlin, J. et al. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)  
- Dettmers, T. et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)  
- Han, S. et al. (2016). [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)  
- ONNX Runtime. [Quantization Overview](https://onnxruntime.ai/docs/performance/quantization.html)


---

## 7. Acknowledgements

This project was conducted as part of the MSc Artificial Intelligence programme at the University of Hull.

Thanks to Hugging Face, PyTorch, bitsandbytes, and ONNX Runtime for making this work possible. SST-2 is part of the GLUE benchmark.


