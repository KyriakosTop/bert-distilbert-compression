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

## Key Findings

| Device        | Precision | Pruned | Accuracy | Latency (s) | Memory (MB) | Notes                          |
|---------------|-----------|--------|----------|-------------|-------------|--------------------------------|
| CPU           | FP32      | 0%     | 91.06%   | 0.072       | 1666.8      | Baseline                       |
| CPU           | 8-bit     | 0%     | 89.68%   | 0.158       | 60.1        | Best CPU config                |
| CPU           | 8-bit     | 30%    | 90.48%   | 70.55       | 2216.4      | Higher latency/memory          |
| CPU           | 8-bit     | 40%    | 88.53%   | 63.06       | 2388.0      | Some accuracy loss             |
| CPU           | 8-bit     | 50%    | 87.16%   | 53.99       | 2563.8      | Costly trade-off               |
| GTX 1050 Ti   | FP32      | 0%     | 91.06%   | 0.012       | 282.9       | Baseline                       |
| GTX 1050 Ti   | 8-bit     | 0%     | 90.71%   | 0.205       | 377.3       | Fallback to CPU                |
| T4 GPU        | FP32      | 0%     | 91.06%   | 0.007       | 280.0       | Baseline                       |
| T4 GPU        | 8-bit     | 0%     | 90.71%   | 0.047       | 213.6       | Native support                 |
| T4 GPU        | 4-bit     | 0%     | 91.17%   | 0.027       | 89.5        | Best overall result            |

## Reproducing the Results

You can run the notebooks directly in Google Colab or in a local environment with the required dependencies.

### Installation

To install required packages locally:

```bash
pip install -r requirements.txt
```

Alternatively, you can open the notebooks in Google Colab for GPU support (recommended).
