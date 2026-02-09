# VERA Cookbook - Visual Evidence Retrieval and Analysis

This is a comprehensive collection of VERA engine usage examples, demonstrating the functionality of each module.

## Directory Structure

```
cookbook/
├── README.md                          # This file
├── 01_models_basic_inference.py      # Basic model inference
├── 02_models_masked_inference.py     # Inference with masked attention heads
├── 03_rendering_basic.py             # Text to image rendering (basic)
├── 04_rendering_with_evidence.py     # Text to image rendering (with evidence highlighting)
├── 05_retrieval_attention.py         # Attention-based retrieval
├── 06_retrieval_qwen_embedding.py    # Qwen embedding retrieval
├── 07_analysis_heatmap.py            # Heatmap generation
├── 08_analysis_full_pipeline.py      # Complete analysis pipeline
├── 09_end_to_end_rag.py              # End-to-end RAG example
├── data/                             # Sample data
│   ├── sample_text.txt
│   ├── sample_evidence.json
│   └── sample_image.png
└── output/                           # Output directory
```

## Quick Start

### 1. Basic Inference - Model Initialization and Usage
```bash
python 01_models_basic_inference.py
```
Demonstrates how to initialize the Qwen3-VL model and perform basic inference.

### 2. Inference with Masked Attention Heads
```bash
python 02_models_masked_inference.py
```
Demonstrates how to use QwenEngineMasked to mask specific attention heads.

### 3. Text to Image Rendering
```bash
python 03_rendering_basic.py
python 04_rendering_with_evidence.py
```
Demonstrates how to render text as images and how to highlight evidence.

### 4. Retrieval Functions
```bash
python 05_retrieval_attention.py
python 06_retrieval_qwen_embedding.py
```
Demonstrates different retrieval methods: attention-based retrieval and embedding-based retrieval.

### 5. Analysis Functions
```bash
python 07_analysis_heatmap.py
python 08_analysis_full_pipeline.py
```
Demonstrates how to generate heatmaps and perform complete analysis pipelines.

### 6. End-to-End RAG Example
```bash
python 09_end_to_end_rag.py
```
Complete RAG pipeline: rendering → inference → attention capture → evidence extraction → re-inference.

## Module Overview

### vera.models - Model Inference Module

**Features**:
- `models.initialize()` - Initialize model engine
- `engine.run()` - Run inference
- Supports standard version and masked version

**Use Cases**:
- Single-turn/multi-turn image Q&A
- Attention analysis
- Attention head masking experiments

### vera.rendering - Text Rendering Module

**Features**:
- `rendering.text_to_image()` - Render text as image
- Supports evidence highlighting

**Use Cases**:
- Document visualization
- Evidence highlighting
- Preparing input for vision models

### vera.retrieval - Retrieval Module

**Features**:
- `retrieval.extract_evidence_from_patches()` - Extract evidence from patches
- `retrieval.qwen_embedding()` - Qwen embedding retrieval
- `retrieval.colpali()` - ColPali retrieval
- `retrieval.retrieve_by_attention()` - Complete attention retrieval pipeline

**Use Cases**:
- Retrieving relevant evidence from documents
- Comparing multiple retrieval methods
- Retrieval component of RAG systems

### vera.analysis - Analysis Module

**Features**:
- `analysis.create_heatmap()` - Create attention heatmaps
- `analysis.get_top_k_patches()` - Get Top-K patches
- `analysis.run_full_analysis()` - Complete three-stage analysis

**Use Cases**:
- Visualize attention distribution
- Batch analyze multiple samples
- Generate global statistics

## Dependencies

Ensure the following dependencies are installed:

```bash
pip install torch transformers
pip install opencv-python numpy pillow tqdm
pip install matplotlib seaborn
```

Optional dependencies (for ColPali retrieval):
```bash
pip install colpali-engine
```

## Notes

1. **Model Paths**: Before running examples, ensure correct model paths are set in the code
2. **CUDA Memory**: Some examples require GPU, ensure sufficient VRAM
3. **Configuration Files**: Rendering module requires config files, defaults to `config/config_en.json`
4. **Output Directory**: All outputs are saved in the `cookbook/output/` directory

## Advanced Usage

Check detailed comments in each example file to understand function parameters and usage.

## Feedback

For questions, please refer to VERA documentation or submit an issue.