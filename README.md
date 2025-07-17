# DistilBERT Compression for Efficient Inference

This repository contains the experiments and findings of my MSc project on compressing transformer models for low-resource hardware.

The focus is on applying quantization and pruning techniques to a fine-tuned DistilBERT model on the SST-2 sentiment classification task. The project evaluates trade-offs between accuracy, inference time, and memory usage across different devices (CPU, GTX 1050 Ti, and T4 GPU).

## Methods & Tools Used

- **Model**: DistilBERT (`distilbert-base-uncased`) fine-tuned on the SST-2 sentiment classification dataset (GLUE benchmark).
- **Compression Techniques**:
  - *8-bit Quantization*: Using PyTorch dynamic quantization and Hugging Face transformers.
  - *Structured Pruning*: Applied to linear layers with pruning ratios of 30%, 40%, and 50%.
- **Evaluation Metrics**:
  - Accuracy
  - Inference latency (seconds)
  - Memory usage (MB)
- **Hardware Platforms**:
  - CPU (Intel Xeon, Google Colab)
  - NVIDIA GTX 1050 Ti (local)
  - NVIDIA T4 GPU (Colab)
- **Frameworks**: PyTorch, Hugging Face Transformers, Datasets, ONNX (optional), bitsandbytes (attempted)
