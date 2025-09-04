# Efficient Compression Techniques for Large Language Models on Limited Compute Resources

**Why run large language models on limited hardware at all?**  
Because most developers don't have access to high-end GPUs. Because deploying models to laptops, embedded devices, or free-tier cloud services is increasingly common. And because not every task needs a trillion-parameter model — but everyone still wants fast, accurate results.

This project tackles a practical and relevant challenge:  
**Can transformer-based models like DistilBERT or BERT-base be made efficient enough to run on CPUs or older GPUs — with post-training compression and/or light fine-tuning — without sacrificing too much accuracy?**

We apply **8-bit and 4-bit quantization**, **L1 unstructured pruning**, and **parameter-efficient fine-tuning** (LoRA/QLoRA) to transformer models and benchmark their performance on **SST-2** across diverse hardware. The goal is simple:  
**Make these models smaller, faster, and deployable — while keeping them useful.**

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
   - [3.7 DistilBERT + LoRA on T4 (n7)](#37-distilbert--lora-on-t4-n7)  
   - [3.8 BERT-base FP32 vs QLoRA (4-bit) on T4 (n8)](#38-bert-base-fp32-vs-qlora-4-bit-on-t4-n8)  
4. [Key Insights](#4-key-insights)  
5. [How to Reproduce](#5-how-to-reproduce)  
   - [5.1 Run order](#51-run-order)  
   - [5.2 Environment notes](#52-environment-notes)  
   - [5.3 Repro tips](#53-repro-tips)  
6. [References](#6-references)  
7. [Acknowledgements](#7-acknowledgements)

---

## 1. Motivation

Transformer models are powerful — but deploying them on limited hardware remains a challenge. This project explores whether compression and PEFT techniques can make models like **DistilBERT** and **BERT-base** practical on:

- Free-tier CPUs (e.g., Google Colab)
- Consumer GPUs (e.g., GTX 1050 Ti)
- Older edge or embedded systems

We measure accuracy, latency, and memory under quantization, pruning, and PEFT (LoRA/QLoRA).

---

## 2. Methodology

- **Task:** SST-2 sentiment classification  
- **Models:** DistilBERT (≈66M) and BERT-base (≈110M)  
- **Techniques:**
  - **8-bit quantization:** PyTorch dynamic INT8 on CPU; bitsandbytes INT8 on GPU; ONNX Runtime INT8
  - **4-bit quantization:** bitsandbytes (inference) and **QLoRA** (4-bit base + LoRA adapters for training)
  - **L1 unstructured pruning** (PyTorch on `nn.Linear`)
  - **PEFT:** **LoRA** and **QLoRA**
- **Metrics:** Accuracy; latency (CPU: seconds or total seconds, GPU: ms/sample); memory:
  - **RAM Δ (MB)** and **VRAM Δ (MB)** = end of evaluation − start of evaluation (retained, not peak)
  - **VRAM Total (MB)** = GPU memory right after model load (pre-inference footprint)
  - **n2** reports **RAM (MB, total RSS at end)** and **Total Latency (s)** for the full validation set  
  *Note:* Δ metrics can look counter-intuitive (e.g., very small RAM deltas or negative VRAM deltas on unsupported GPUs due to CPU fallback). Each results table labels columns explicitly as **Δ** or **Total**.

- **Hardware:**
  - Intel Xeon (Colab CPU)
  - NVIDIA T4 (Colab GPU)
  - NVIDIA GTX 1050 Ti + Intel Core i5-7500 (local)

All experiments share a consistent GLUE/SST-2 data pipeline for comparability.

---

## 3. Experiments and Results

### 3.1 DistilBERT on CPU (n1)

| Metric        | FP32   | INT8   | Δ (%)  |
|---------------|--------|--------|--------|
| Accuracy      | 91.06% | 89.33% | -1.73% |
| Latency (s)   | 0.3520 | 0.1513 | -57%   |
| RAM (MB)      | 273.00 | 5.81   | -98%   |

✅ 8-bit quantization drastically improves latency and RAM usage with minimal accuracy drop.  
⚠️ 4-bit quantization is not supported on CPU (PyTorch only provides 8-bit dynamic quantization).

---

### 3.2 DistilBERT on CPU – Pruned + Quantized (n2)

| Pruning | Accuracy | Latency (s) | RAM (MB) |
|---------|----------|-------------|----------|
| 30%     | 90.48%   | 60.0        | 2330     |
| 40%     | 88.87%   | 54.0        | 2500     |
| 50%     | 87.16%   | 47.5        | 2697     |

✅ Pruning reduces latency.  
⚠️ RAM increases in PyTorch due to masks; sparse-aware runtimes are needed to see memory wins.

---

### 3.3 DistilBERT on T4 GPU (n3)

| Precision | Accuracy | Latency (ms) | VRAM Total (MB) | VRAM Δ (MB) | RAM Δ (MB) |
|-----------|----------|--------------|------------------|-------------|------------|
| FP32      | 94.00%   | 12.40        | 659.88           | 30.00       | 213.52     |
| 8-bit     | 94.00%   | 77.20        | 843.88           | 14.00       | 130.85     |
| 4-bit     | 93.00%   | 16.50        | 953.88           | 2.00        | 5.73       |

⚠️ 8-bit slower due to kernel overhead.  
✅ 4-bit cuts memory deltas most, with modest latency impact.

---

### 3.4 DistilBERT on GTX 1050 Ti (n4)

| Precision | Accuracy | Latency (ms) | VRAM Δ (MB) | RAM Δ (MB) |
|-----------|----------|--------------|-------------|------------|
| FP32      | 94.00%   | 9.15         | +20.13      | 97.40      |
| 8-bit     | 94.00%   | 98.44        | -12.38      | 48.71      |
| 4-bit     | 93.00%   | 11.27        | -3.00       | -4.93      |

⚠️ Quantized models fell back to CPU on older GPU → higher latency, negative VRAM deltas.

---

### 3.5 BERT-base on T4 GPU (n5)

| Precision | Accuracy | Latency (ms) | VRAM Total (MB) | VRAM Δ (MB) | RAM Δ (MB) |
|-----------|----------|--------------|------------------|-------------|------------|
| FP32      | 92.00%   | 12.83        | 1289.88          | 32.00       | 63.26      |
| 8-bit     | 92.00%   | 94.04        | 1081.88          | 12.00       | 20.96      |
| 4-bit     | 92.00%   | 20.75        | 1195.88          | 6.00        | 0.51       |

✅ All variants maintain accuracy; 4-bit provides the best memory efficiency on BERT.

---

### 3.6 DistilBERT via ONNX Runtime (n6)

| Precision | Accuracy | Inference Time (s) | RAM Δ (MB) |
|-----------|----------|--------------------|------------|
| FP32      | 91.06%   | 191.68             | ~0.00      |
| INT8      | 90.48%   | 121.08             | ~0.00      |

✅ ONNX INT8 yields ~36.8% speedup with minimal accuracy loss.

---

### 3.7 DistilBERT + LoRA on T4 (n7)

**Goal.** Compare **FP16 LoRA** (adapters on full-precision DistilBERT) vs **QLoRA** (4-bit base + LoRA) on SST-2 using a T4 GPU.

| Precision / Method | Accuracy | Latency (ms) | VRAM Total (MB) | VRAM Δ (MB) | RAM Δ (MB) |
|--------------------|----------|--------------|------------------|-------------|------------|
| **LoRA (FP16)**    | **90.48%** | **10.52**   | **1021.88**      | ~0.00       | ~0.00      |
| QLoRA (4-bit)      | —        | —            | —                | —           | —          |

✅ FP16 LoRA trains and evaluates reliably on T4 with DistilBERT.  
⚠️ QLoRA on DistilBERT is **not reliable** with current bitsandbytes 4-bit kernels (assertion during quantization state recovery); see **BERT-base** results below.

---

### 3.8 BERT-base FP32 vs QLoRA (4-bit) on T4 (n8)

**Setup:** SST-2 `train[:20,000]`, 2 epochs, max length 128, T4 GPU.  
(Training details and per-model reload evaluation are in the notebook.)

| Precision | Accuracy | Latency (ms) | VRAM Total (MB) | VRAM Δ (MB) | RAM Δ (MB) |
|-----------|----------|--------------|------------------|-------------|------------|
| **FP32**  | **91.63%** | **7.60**    | **420.99**       | ~0.00       | ~0.00      |
| **4-bit QLoRA** | **90.60%** | **3.25** | **99.00**       | ~0.00       | ~0.00      |

✅ **QLoRA** nearly matches FP32 accuracy while being faster and far lighter in VRAM on T4.  
(Notebook also reports training wall time and peak VRAM; inference table here matches the style of 3.1–3.6.)

---

## 4. Key Insights

- **8-bit quantization is highly effective on CPUs** (large latency & RAM cuts, small accuracy hit).
- **Unstructured L1 pruning** reduces latency but **doesn’t save RAM** in PyTorch without sparse-aware runtimes.
- **bitsandbytes 4-bit** helps most on **larger models and modern GPUs** (e.g., BERT-base on T4).
- On **older GPUs**, quantized models may **fall back to CPU** — always check device placement.
- **LoRA/QLoRA**: with tuned adapters (higher LR, sensible target modules), you get **near-FP32 accuracy** with **~0.5% trainable params** and much lower VRAM.
- For **fair VRAM comparisons**, measure with **per-model reload** so only one checkpoint sits on the GPU at a time.
- **DistilBERT + QLoRA** is currently **unreliable/fails** with some bitsandbytes 4-bit kernels; use **BERT-base** for QLoRA (n8).

---

## 5. How to Reproduce

> Tip: run notebooks in order. Each is self-contained and will download SST-2 automatically from 🤗 Datasets.

### 5.1 Run order

| Notebook Filename                | What it does                                             |
|----------------------------------|----------------------------------------------------------|
| `n1_dbert_quant_cpu.ipynb`       | DistilBERT on CPU: FP32 vs dynamic INT8                  |
| `n2_dbert_quant_prun_cpu.ipynb`  | DistilBERT on CPU: L1 unstructured pruning + quantization|
| `n3_dbert_quant_gpu_t4.ipynb`    | DistilBERT on T4: FP32 / 8-bit / 4-bit                   |
| `n4_dbert_quant_gpu_gtx.ipynb`   | DistilBERT on GTX 1050 Ti (older GPU quirks)             |
| `n5_bert_quant_gpu_t4.ipynb`     | BERT-base on T4: FP32 / 8-bit / 4-bit                    |
| `n6_dbert_onnx_cpu.ipynb`        | DistilBERT via ONNX Runtime: FP32 vs INT8                |
| `n7_dbert_lora_gpu_t4.ipynb`     | DistilBERT on T4: **LoRA (FP16)** and **QLoRA attempt**  |
| `n8_bert_qlora_gpu_t4.ipynb`     | BERT-base on T4: **FP32 vs QLoRA (4-bit)**               |

### 5.2 Environment notes

- **Python:** 3.10+  
- **PyTorch:** 2.1+ (CUDA build if using GPU)  
- **GPU:** NVIDIA T4 (Colab) for n3, n5, n7, n8; GTX 1050 Ti for n4  
- **Dataset:** `glue/sst2` is pulled on first run

### 5.3 Repro tips

- All training notebooks set `SEED = 42`. For exact reproducibility, keep batch sizes/epochs the same and avoid interrupting the runtime.  
- For **T4 runs**, ensure the Colab runtime is set to **GPU**.  
- On **older GPUs** (e.g., 1050 Ti), quantized models may **fallback to CPU**; verify with `next(model.parameters()).device`.  
- QLoRA on **DistilBERT** (n7) can fail due to bitsandbytes 4-bit kernel shape checks; use **BERT-base** (n8) for QLoRA results.

---

## 6. References

- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT: A distilled version of BERT.* arXiv:1910.01108. https://arxiv.org/abs/1910.01108  
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv:1810.04805. https://arxiv.org/abs/1810.04805  
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, L., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685. https://arxiv.org/abs/2106.09685  
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314. https://arxiv.org/abs/2305.14314  
- Han, S., Mao, H., & Dally, W. J. (2016). *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.* ICLR. https://arxiv.org/abs/1510.00149  
- Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., & Bowman, S. R. (2018). *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding.* arXiv:1804.07461. https://arxiv.org/abs/1804.07461  
- Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank (SST).* EMNLP. https://aclanthology.org/D13-1170/  
- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing.* EMNLP (System Demos). https://arxiv.org/abs/1910.03771  
- Lhoest, Q., Delangue, C., von Platen, P., et al. (2021). *Datasets: A Community Library for Natural Language Processing.* NeurIPS (Datasets and Benchmarks). https://arxiv.org/abs/2109.02846  
- Hugging Face PEFT. *Parameter-Efficient Fine-Tuning library.* https://github.com/huggingface/peft  
- bitsandbytes. *8-bit/4-bit quantization and optimizers.* https://github.com/TimDettmers/bitsandbytes  
- ONNX Runtime Docs. *Quantization Overview.* https://onnxruntime.ai/docs/performance/quantization.html

---

## 7. Acknowledgements

This project was conducted as part of the MSc Artificial Intelligence programme at the University of Hull.

Thanks to Hugging Face, PyTorch, bitsandbytes, and ONNX Runtime for making this work possible. SST-2 is part of the GLUE benchmark.
