# VERA Cookbook - Visual Evidence Retrieval and Analysis

这是一个完整的VERA引擎使用示例集合，展示各个模块的功能。

## 目录结构

```
cookbook/
├── README.md                          # 本文件
├── 01_models_basic_inference.py      # 基础模型推理
├── 02_models_masked_inference.py     # 掩盖注意力head的推理
├── 03_rendering_basic.py             # 文本渲染为图像（基础）
├── 04_rendering_with_evidence.py     # 文本渲染为图像（带evidence高亮）
├── 05_retrieval_attention.py         # 基于注意力的检索
├── 06_retrieval_qwen_embedding.py    # Qwen embedding检索
├── 07_analysis_heatmap.py            # 热力图生成
├── 08_analysis_full_pipeline.py      # 完整分析流程
├── 09_end_to_end_rag.py              # 端到端RAG示例
├── data/                             # 示例数据
│   ├── sample_text.txt
│   ├── sample_evidence.json
│   └── sample_image.png
└── output/                           # 输出目录
```

## 快速开始

### 1. 基础推理 - 模型初始化和使用
```bash
python 01_models_basic_inference.py
```
展示如何初始化Qwen3-VL模型并进行基础推理。

### 2. 掩盖注意力head的推理
```bash
python 02_models_masked_inference.py
```
展示如何使用QwenEngineMasked来掩盖特定的注意力头。

### 3. 文本渲染为图像
```bash
python 03_rendering_basic.py
python 04_rendering_with_evidence.py
```
展示如何将文本渲染为图像，以及如何高亮显示evidence。

### 4. 检索功能
```bash
python 05_retrieval_attention.py
python 06_retrieval_qwen_embedding.py
```
展示不同的检索方法：基于注意力的检索和基于embedding的检索。

### 5. 分析功能
```bash
python 07_analysis_heatmap.py
python 08_analysis_full_pipeline.py
```
展示如何生成热力图和进行完整的分析流程。

### 6. 端到端RAG示例
```bash
python 09_end_to_end_rag.py
```
完整的RAG流程：渲染 → 推理 → 捕获注意力 → 提取evidence → 重新推理。

## 模块功能概览

### vera.models - 模型推理模块

**功能**:
- `models.initialize()` - 初始化模型引擎
- `engine.run()` - 运行推理
- 支持标准版本和带mask的版本

**使用场景**:
- 单轮/多轮图像问答
- 注意力分析
- 注意力头掩码实验

### vera.rendering - 文本渲染模块

**功能**:
- `rendering.text_to_image()` - 将文本渲染为图像
- 支持evidence高亮显示

**使用场景**:
- 文档视觉化
- 证据高亮显示
- 为视觉模型准备输入

### vera.retrieval - 检索模块

**功能**:
- `retrieval.extract_evidence_from_patches()` - 从patch提取证据
- `retrieval.qwen_embedding()` - Qwen embedding检索
- `retrieval.colpali()` - ColPali检索
- `retrieval.retrieve_by_attention()` - 完整的注意力检索流程

**使用场景**:
- 从文档中检索相关证据
- 多种检索方法对比
- RAG系统的检索部分

### vera.analysis - 分析模块

**功能**:
- `analysis.create_heatmap()` - 创建注意力热力图
- `analysis.get_top_k_patches()` - 获取Top-K patches
- `analysis.run_full_analysis()` - 完整的三阶段分析

**使用场景**:
- 可视化注意力分布
- 批量分析多个样本
- 生成全局统计信息

## 依赖说明

确保已安装以下依赖：

```bash
pip install torch transformers
pip install opencv-python numpy pillow tqdm
pip install matplotlib seaborn
```

可选依赖（用于ColPali检索）:
```bash
pip install colpali-engine
```

## 注意事项

1. **模型路径**: 在运行示例前，请确保在代码中设置了正确的模型路径
2. **CUDA内存**: 某些示例需要GPU，确保有足够的显存
3. **配置文件**: 渲染模块需要config文件，默认使用`config/config_en.json`
4. **输出目录**: 所有输出都会保存在`cookbook/output/`目录下

## 进阶使用

查看各个示例文件中的详细注释，了解每个函数的参数和用法。

## 问题反馈

如有问题，请查看VERA文档或提交issue。
