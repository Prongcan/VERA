<div align="center">

# 👀 VERA: Visual Evidence Retrieval Augmentation

### VLM Attention Capturing, Masking, Analysis, Visualization and Utilization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)

*Codebase of paper: VERA: Identifying and Leveraging Visual Evidence Retrieval Heads in Long-Context Understanding*

</div>

---

## 📋 Table of Contents

- [✨ Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
- [📖 Documentation](#-documentation)

---

## ✨ Overview

VERA provides a comprehensive toolkit for VLM attention capture, masking, analysis, and utilization for retrieval tasks.

| Feature | Description |
|---------|-------------|
| 🎯 **Attention-Based Retrieval** | Extract relevant information using attention mechanisms from large vision-language models |
| 🎨 **Document Rendering** | Convert text to document images with customizable fonts and layouts |
| 🔍 **Evidence Highlighting** | Automatically highlight important information in rendered documents |
| 📊 **Attention Visualization** | Generate heatmaps and visualizations of model attention |
| 🤖 **Head Masking** | Mask specific attention heads to analyze their contributions |
| 🔄 **Multi-Stage Retrieval** | Combine attention-based and embedding-based retrieval methods |
| 📈 **Batch Analysis** | Process multiple documents with full pipeline support |
| 🎯 **RAG Integration** | End-to-end Retrieval-Augmented Generation for visual QA |

---

```
┌─────────────────────────────────────────────────────────────────┐
│                         VERA Framework                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │   Models     │      │  Rendering   │      │  Retrieval   │ │
│  │              │      │              │      │              │ │
│  │ • Qwen3-VL   │──────│ • Text→Img   │──────│ • Attention  │ │
│  │ • Masking    │      │ • Evidence   │      │ • Embedding  │ │
│  │ • Inference  │      │   Highlight  │      │ • Top-K Patches│ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Analysis Module                        │  │
│  │                                                           │  │
│  │  • Heatmap Generation  • Top-K Extraction  • Statistics  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

In collaboration with Claude Code, I have refactored the original paper's codebase into a universal engine tool for VLM attention analysis and visualization. To ensure reusability and decoupling, some KV Cache acceleration optimizations have been omitted. The Thinking model code and complete datasets will be uploaded soon. The codebase is under active maintenance - contributions to add new model support are welcome!

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 or higher
- **CUDA**: 12.1 (for GPU acceleration)
- **GPU**: NVIDIA GPU with 16GB+ VRAM recommended
- **OS**: Linux (tested on Ubuntu)

### Installation

<details>
<summary><b>1. Create Conda Environment</b></summary>

```bash
# Create new environment
conda create -n vera python=3.10 -y
conda activate vera
```

</details>

<details>
<summary><b>2. Install PyTorch & Flash Attention</b></summary>

```bash
# Install CUDA compiler
conda install -c nvidia cuda-nvcc=12.1 -y

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

</details>

<details>
<summary><b>3. Install Dependencies</b></summary>

```bash
# Install all required packages
pip install -r requirements.txt
```

</details>

<details>
<summary><b>4. Configure Your Models</b></summary>

Download models and add the local model paths to `config/model_config.json`.

```python
# Download models using Python (currently supports Qwen3-VL-8B and GLM)
import os

# Switch to domestic mirror for HuggingFace to avoid connection failures
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

# Specify repository ID
repo_id = "Qwen/Qwen3-VL-8B"  # or "zai-org/GLM-4.1V-9B-Thinking"
# Local directory (customizable)
local_dir = "./Qwen3-VL-8B"
snapshot_download(repo_id=repo_id, local_dir=local_dir)
```

</details>

<details>
<summary><b>5. Let's Begin!</b></summary>

Run the experiment scripts in the recommended order:

```bash
# Basic inference: renders images, captures first-token attention, word mapping, and answers
# Outputs saved to tem/ directory with corresponding folder names
python experiments/qasper_qwen_img.py

# Masked inference: runs inference with specified attention heads masked
# Results saved to corresponding folders
python3 experiments/qasper_qwen_img_masked.py

# VERA pipeline: runs the complete VERA retrieval pipeline
python3 experiments/qasper_qwen_RAG_VER.py

# Attention visualization and VERA evidence extraction
python3 anylasis/attention_data_anylasis.py

# Qwen Embedding baseline retrieval (optional, can be skipped)
python3 experiments/calculate_qwen_embedding_retrieval.py

# Colpali baseline retrieval (optional, can be skipped)
python3 experiments/calculte_colpali_embedding_retrieval.py

# Retrieval evaluation (optional, can be skipped)
python3 anylasis/evaluate_retrieval.py
```

</details>

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Cookbook README](cookbook/README.md) | Detailed cookbook guide |
| [Quick Start](cookbook/QUICKSTART.md) | Fast-track to running examples |
| [Summary](cookbook/SUMMARY.md) | Complete feature summary |
| [Usage Guide](vera/USAGE.md) | API documentation |

---

<div align="center">

**Built with ❤️ for visual document understanding**

</div>
