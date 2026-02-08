# VERA Project Documentation

## Overview

**VERA (Visual Extraction and Reasoning Assistant)** is a comprehensive system for visual question answering that combines vision-language models with advanced attention mechanisms and retrieval-augmented generation.

### Key Capabilities
- Multi-modal document understanding and QA
- Attention monitoring and masking for interpretable reasoning
- Support for multiple QA datasets (Qasper, Hotpot, DocMath, Musique, LongBench)
- Integration with external OCR systems (Glyph)
- GPU-optimized inference with distributed training support

---

## Project Structure

```
VERA/
├── data/                          # Dataset loaders and data
│   ├── dataset_loader.py          # Unified dataset loader for multiple QA datasets
│   ├── qasper/                    # Qasper dataset data
│   ├── hotpot/                    # HotpotQA dataset data
│   ├── musique/                   # Musique dataset data
│   ├── docmath/                   # DocMath dataset data
│   └── longbench_pro/             # LongBench and Pro dataset data
│
├── experiments/                   # Experiment scripts organized by dataset
│   ├── docmath_*.py              # DocMath dataset experiments
│   ├── hotpot_*.py               # HotpotQA dataset experiments
│   ├── musique_*.py              # Musique dataset experiments
│   ├── qasper_*.py               # Qasper dataset experiments
│   ├── longbench_*.py            # LongBench dataset experiments
│   └── pro_*.py                  # Pro dataset experiments
│
├── models/                        # Core model implementations
│   ├── wrapper.py                # Main Qwen model wrapper with attention monitoring
│   ├── modeling_qwen3_vl.py      # Custom Qwen3-VL with masked attention
│   ├── wrapper_glm.py            # GLM model wrapper
│   ├── wrapper_glyph.py          # Glyph OCR wrapper
│   ├── wrapper_qwen3_thinking.py # Qwen3 with thinking capabilities
│   └── intervener.py             # Attention monitoring and masking utilities
│
├── models_thinking/              # Extended models with reasoning capabilities
│   ├── wrapper_qwen3_thinking.py # Qwen3 with thinking and few-shot attention
│   └── model_*.py                # Enhanced model implementations
│
├── config/                       # Text-to-image generation configurations
│   ├── config_en.json            # English text rendering config
│   ├── config_en_*.json          # Quality variants (high, low, ultra)
│   └── config_zh.json            # Chinese text rendering config
│
├── scripts/                      # Utility scripts
│   └── *.py                      # Text processing and conversion utilities
│
├── anylasis/                     # Analysis and evaluation scripts
│   ├── data_anylasis_*.py        # Performance and attention analysis
│   └── evaluate_retrieval.py     # Retrieval evaluation
│
├── VLM_and_LLM/                  # Vision-Language Model integration
│   └── [integration components]
│
├── requirements.txt              # Python dependencies
├── README.md                     # Setup instructions
└── .gitignore                    # Git ignore rules

```

---

## Supported Datasets

| Dataset | Description | Experiment Prefix |
|---------|-------------|-------------------|
| **Qasper** | Academic QA with research papers | `qasper_*.py` |
| **Hotpot** | Multi-hop QA with context documents | `hotpot_*.py` |
| **DocMath** | Document-based mathematical reasoning | `docmath_*.py` |
| **Musique** | Multi-hop QA with complex reasoning | `musique_*.py` |
| **LongBench** | Long-form question answering | `longbench_*.py` |
| **Pro** | Professional domain QA | `pro_*.py` |

---

## Key Components

### 1. Model Wrappers (`models/`)

#### `wrapper.py`
Main Qwen model wrapper with:
- Attention monitoring for all layers and heads
- GPU memory optimization with automatic device detection
- Support for multi-GPU inference
- Entropy calculation for attention analysis

**Key Usage:**
```python
from models.wrapper import QwenEngine_img

# Initialize with config
config = {
    "model_path": "path/to/qwen-model",
    "device": "cuda:0",
    "max_new_tokens": 512
}
model = QwenEngine_img(config)

# Generate answer (note: method name is generate_answer, not generate)
input_ids = [...]  # tokenized input
attention_mask = [...]  # attention mask
seq_len = len(input_ids)
output = model.generate_answer(input_ids, attention_mask, seq_len)
```

**Available Engine Classes:**
```python
# Text mode
from models.wrapper import QwenEngine_txt
model = QwenEngine_txt(config)

# Image mode (most common)
from models.wrapper import QwenEngine_img
model = QwenEngine_img(config)

# VLLM accelerated version
from models.wrapper import VllmEngine_img
model = VllmEngine_img(config)

# GLM model
from models.wrapper_glm import GlmEngine_img
model = GlmEngine_img(config)

# With attention masking
from models.wrapper import QwenEngine_img_no_eager_masked
model = QwenEngine_img_no_eager_masked(config)
```

#### `modeling_qwen3_vl.py`
Custom Qwen3-VL implementation featuring:
- Masked attention mechanisms
- Configurable attention head selection
- Integration with attention intervenor

#### `intervenor.py`
Utilities for:
- Attention head masking
- Attention pattern visualization
- Layer-wise attention analysis

### 2. Dataset Loader (`data/dataset_loader.py`)

**`DatasetUtils` class** provides unified interface for:
- Loading different QA datasets
- Preprocessing documents and questions
- Converting documents to images
- Managing evidence retrieval strategies

**Example:**
```python
from data.dataset_loader import DatasetUtils

dataset_utils = DatasetUtils(dataset_name="qasper")
data = dataset_utils.load_data(split="dev")
documents = dataset_utils.preprocess_documents(data)
```

### 3. Experiment Scripts

All experiments follow a similar pattern:
```python
# Typical experiment structure
1. Load dataset using DatasetUtils
2. Initialize model wrapper
3. Run inference with specific methodology
4. Save results for analysis
```

**Methodology Variants:**
- `VER`: Visual Extraction and Reasoning pipeline
- `GLM`: Experiments with GLM models
- `Qwen`: Experiments with Qwen models
- `SFT`: Supervised Fine-Tuning approaches
- `Masked`: Attention masking experiments
- `RAG`: Retrieval-Augmented Generation variants
- `Thinking`: Enhanced reasoning with thinking capabilities

---

## Configuration

### Text-to-Image Config (`config/`)

Controls how text documents are rendered as images:

```json
{
  "font": "Arial",
  "font_size": 12,
  "margin": 20,
  "dpi": 150,
  "page_width": 800,
  "page_height": 1000
}
```

**Available configs:**
- `config_en.json` - Standard English rendering
- `config_en_high.json` - High quality
- `config_en_low.json` - Fast rendering
- `config_en_ultra.json` - Maximum quality
- `config_zh.json` - Chinese text support

---

## Dependencies

### Core Requirements
- Python 3.10+
- PyTorch 2.0+ with CUDA 12.1
- Transformers 4.45+

### Vision & Processing
- opencv-python
- pdf2image, pdfplumber, pypdfium2
- Pillow

### NLP
- sentencepiece, tiktoken
- nltk
- scikit-learn

### Performance
- flash-attn (2.8.3+)
- deepspeed
- peft
- xformers

**Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## Common Workflows

### Running an Experiment

1. **Select appropriate experiment script:**
   ```bash
   cd experiments
   ```

2. **Run with specific dataset:**
   ```python
   python qasper_qwen_img.py
   ```

3. **Results are saved** in the experiment's output directory

### Analyzing Results

1. **Use analysis scripts in `anylasis/`:**
   ```python
   python anylasis/data_anylasis_dev_20_best_5.py
   ```

2. **Attention visualization** is handled by the model wrapper

### Adding a New Dataset

1. **Extend `DatasetUtils`** in `data/dataset_loader.py`
2. **Create experiment script** following existing patterns
3. **Add configuration** if text-to-image conversion is needed

---

## Key Features

### Attention Monitoring
- All attention heads are tracked during inference
- Layer-wise attention patterns can be analyzed
- Support for selective head masking (few-shot approaches)

### GPU Optimization
- Automatic device detection and allocation
- Multi-GPU support with DataParallel
- Memory-efficient attention with Flash Attention 2
- Automatic batch size adjustment

### RAG Strategies
- **Golden:** Ground truth evidence
- **Random:** Random document segments
- **Top-N:** Top-N relevant passages
- **BM25:** BM25 retrieval
- **Dense:** Dense passage retrieval

### Integration with External Systems
- **Glyph OCR:** Advanced text extraction
- **VLLM:** Very Large Language Model inference
- **Thinking Models:** Enhanced reasoning chains

---

## Model-Specific Notes

### Qwen Models
- Primary model family for VERA
- Supports both SFT and base variants
- Attention masking available in all layers

### GLM Models
- Alternative VLM option
- Used in comparative experiments
- Different attention architecture

### Thinking Models (`models_thinking/`)
- Extended with reasoning capabilities
- Few-shot attention head selection
- Enhanced interpretability

---

## File Naming Conventions

Experiments use descriptive naming:
```
{dataset}_{model}_{method}_{variant}.py
```

Examples:
- `qasper_qwen_img.py` - Qasper with Qwen and image input
- `hotpot_glm_rag_top5.py` - Hotpot with GLM using top-5 RAG
- `musique_qwen_masked.py` - Musique with attention masking

---

## Development Guidelines

### Adding New Experiments
1. Copy similar existing experiment script
2. Modify dataset and model configuration
3. Update methodology-specific parameters
4. Document changes in comments

### Modifying Model Wrappers
- Maintain backward compatibility
- Add attention hooks for new layers if needed
- Update GPU memory management for large models
- Test with multiple datasets before committing

### Analysis Scripts
- Save results in structured format (JSON)
- Include attention visualizations when applicable
- Provide clear metrics and comparisons
- Support batch processing of multiple results

---

## Performance Tips

1. **Use Flash Attention 2** for faster inference
2. **Enable mixed precision** (FP16) for memory efficiency
3. **Batch similar examples** when possible
4. **Cache preprocessed documents** to save time
5. **Use appropriate image resolution** for your GPU memory

---

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in experiment script
- Use lower resolution images
- Enable gradient checkpointing
- Use FP16 mixed precision

### Attention Masking Not Working
- Verify model wrapper initialization
- Check attention hook registration
- Ensure compatible model version
- Review intervenor configuration

### Dataset Loading Errors
- Check dataset path configuration
- Verify data format matches expected schema
- Review preprocessing steps in `dataset_loader.py`
- Ensure required dependencies are installed

---

## Citation and References

If you use VERA in your research, please cite the relevant papers describing the methodology.

---

## Contact and Support

For questions or issues:
1. Check experiment script comments for methodology details
2. Review analysis scripts for evaluation procedures
3. Examine model wrapper code for implementation details
4. Refer to config files for parameter settings