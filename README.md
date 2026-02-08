# VERA - Visual Extraction and Reasoning Assistant

## Project Overview

- Python 3.10+
- CUDA 12.1 
- Conda/Miniconda

## Quick Start

### 1. Create conda environment

```bash
conda create -n vera python=3.10
conda activate vera
```

### 2. Install PyTorch and flash attention

```bash
conda install -c nvidia cuda-nvcc=12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 3. Install other dependencies

```bash
pip install -r requirements.txt
```
