# VERA Cookbook - 完整总结

## 📚 Cookbook内容概览

VERA Cookbook是一个完整的示例集合，展示了VERA框架的所有功能模块。

## 📂 目录结构

```
cookbook/
├── README.md                      # 总览文档（从这开始）
├── QUICKSTART.md                  # 快速开始指南
├── requirements.txt               # Python依赖
├── run_all_examples.py            # 批量运行脚本
│
├── data/                          # 示例数据
│   ├── sample_text.txt            # 示例文档文本
│   └── sample_evidence.json       # 示例evidence
│
├── output/                        # 输出目录（运行后生成）
│   ├── 01_models_basic_inference/
│   ├── 02_models_masked_inference/
│   ├── 03_rendering_basic/
│   ├── 04_rendering_with_evidence/
│   ├── 05_retrieval_attention/
│   ├── 06_retrieval_qwen_embedding/
│   ├── 07_analysis_heatmap/
│   ├── 08_analysis_full_pipeline/
│   └── 09_end_to_end_rag/
│
└── 示例脚本（01-09）
    ├── 01_models_basic_inference.py
    ├── 02_models_masked_inference.py
    ├── 03_rendering_basic.py
    ├── 04_rendering_with_evidence.py
    ├── 05_retrieval_attention.py
    ├── 06_retrieval_qwen_embedding.py
    ├── 07_analysis_heatmap.py
    ├── 08_analysis_full_pipeline.py
    └── 09_end_to_end_rag.py
```

## 🎯 各示例功能说明

### 📦 模块模块 (vera.models)

#### 01 - 基础模型推理
- **文件**: `01_models_basic_inference.py`
- **功能**:
  - 初始化Qwen3-VL模型
  - 加载图像并运行推理
  - 获取答案和注意力数据
- **需要模型**: ✅
- **输出**: 答案文本、input tokens

#### 02 - 掩盖注意力头的推理
- **文件**: `02_models_masked_inference.py`
- **功能**:
  - 使用QwenEngineMasked掩盖特定heads
  - 对比有mask和无mask的推理结果
- **需要模型**: ✅
- **输出**: 对比结果文件

### 🎨 渲染模块 (vera.rendering)

#### 03 - 文本渲染为图像（基础版）
- **文件**: `03_rendering_basic.py`
- **功能**:
  - 将文本渲染为文档图像
  - 使用配置文件控制字体和布局
- **需要模型**: ❌
- **输出**: 渲染的PNG图像

#### 04 - 文本渲染为图像（带evidence高亮）
- **文件**: `04_rendering_with_evidence.py`
- **功能**:
  - 渲染文本并高亮显示evidence
  - 对比有/无高亮的效果
- **需要模型**: ❌
- **输出**: 两组渲染图像（带/不带高亮）

### 🔍 检索模块 (vera.retrieval)

#### 05 - 基于注意力的检索
- **文件**: `05_retrieval_attention.py`
- **功能**:
  - 从注意力数据提取Top-K patches
  - 从patches提取evidence文本
  - 展示完整的检索流程
- **需要模型**: ❌（使用模拟数据）
- **输出**: Patch坐标、提取的文本

#### 06 - Qwen Embedding检索
- **文件**: `06_retrieval_qwen_embedding.py`
- **功能**:
  - 使用Qwen模型计算embedding
  - 基于相似度检索相关句子
  - 返回Top-K相关内容
- **需要模型**: ✅
- **输出**: 提取的相关句子

### 📊 分析模块 (vera.analysis)

#### 07 - 热力图生成
- **文件**: `07_analysis_heatmap.py`
- **功能**:
  - 创建注意力热力图叠加
  - 创建Top-K patches高亮图
  - 生成可视化结果
- **需要模型**: ❌（使用模拟数据）
- **输出**: 热力图PNG、patch坐标JSON

#### 08 - 完整分析流程
- **文件**: `08_analysis_full_pipeline.py`
- **功能**:
  - 运行三阶段分析（scan + viz + global）
  - 确定全局Top heads
  - 生成全局统计和热力图
- **需要模型**: ❌（使用模拟数据）
- **输出**: 多个可视化文件和统计数据

### 🚀 端到端应用

#### 09 - 端到端RAG示例
- **文件**: `09_end_to_end_rag.py`
- **功能**:
  - 完整的RAG流程演示
  - 渲染 → 推理 → 检索 → 再推理
  - 对比RAG前后的效果
- **需要模型**: ✅
- **输出**: 多个中间结果和对比文件

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r cookbook/requirements.txt
```

### 2. 运行简单示例（不需要模型）
```bash
# 文本渲染
python cookbook/03_rendering_basic.py

# 热力图生成
python cookbook/07_analysis_heatmap.py
```

### 3. 运行完整示例（需要模型）
```bash
# 确保修改了MODEL_PATH
python cookbook/01_models_basic_inference.py

# 端到端RAG
python cookbook/09_end_to_end_rag.py
```

### 4. 批量运行
```bash
# 查看帮助
python cookbook/run_all_examples.py --help

# 运行所有示例（不需要模型的）
python cookbook/run_all_examples.py --example all --skip-model

# 运行单个示例
python cookbook/run_all_examples.py --example 03
```

## 📖 学习路径

### 初学者路径（不需要模型）
```
03渲染基础 → 04渲染+高亮 → 05注意力检索 → 07热力图 → 08完整分析
```

### 进阶路径（需要模型）
```
01基础推理 → 02Mask推理 → 06Embedding检索 → 09端到端RAG
```

## 📊 VERA模块使用对照表

| Cookbook示例 | 使用的VERA模块 | 关键API |
|--------------|---------------|---------|
| 01 | `vera.models` | `models.initialize()` |
| 02 | `vera.models` | `models.initialize("qwen-img-masked")` |
| 03 | `vera.rendering` | `rendering.text_to_image()` |
| 04 | `vera.rendering` | `rendering.text_to_image(evidence_text=...)` |
| 05 | `vera.retrieval`, `vera.analysis` | `retrieval.extract_evidence_from_patches()` |
| 06 | `vera.retrieval` | `retrieval.qwen_embedding()` |
| 07 | `vera.analysis` | `analysis.create_heatmap()`, `analysis.get_top_k_patches()` |
| 08 | `vera.analysis` | `analysis.run_full_analysis()` |
| 09 | `vera.models`, `vera.rendering`, `vera.retrieval` | 完整RAG流程 |

## 💡 关键特性演示

### 1. 模型初始化和推理
```python
from vera import models

# 标准版本
engine = models.initialize(model_path, model_type="qwen-img")
result = engine.run(prompt_context, question, image_paths)

# 带mask的版本
engine = models.initialize(model_path, model_type="qwen-img-masked")
result = engine.run(prompt_context, question, image_paths,
                   is_mask_heads=True, heads_positions={(24, 29)})
```

### 2. 文本渲染
```python
from vera import rendering

# 基础渲染
image_paths = rendering.text_to_image(text, output_dir, config)

# 带evidence高亮
image_paths = rendering.text_to_image(text, output_dir, config,
                                       evidence_text=["重要文本"])
```

### 3. 检索
```python
from vera import retrieval

# 注意力检索
text = retrieval.extract_evidence_from_patches(patch_bounds, word_mapping_path, output_path)

# Embedding检索
stats = retrieval.qwen_embedding(model_path, data_path, save_dir, top_k=10)
```

### 4. 分析
```python
from vera import analysis

# 热力图
analysis.create_heatmap(image_path, attention_data, output_path, mode="overlay")

# 完整分析
stats = analysis.run_full_analysis(root_dir, output_folder_name, top_k=10)
```

## 🔧 自定义和扩展

### 修改模型路径
在每个示例文件中找到并修改：
```python
MODEL_PATH = "/your/path/to/Qwen3-VL-8B-Instruct"
```

### 修改配置文件
文本渲染使用`config/config_en.json`，可以修改：
- 字体路径
- 字体大小
- 图像尺寸
- 颜色配置

### 调整Top-K数量
大多数示例都有`TOP_K`或`top_k`参数：
```python
TOP_K = 20  # 提取Top-20 patches
```

## 📝 注意事项

1. **模型路径**: 所有需要模型的示例都要求修改`MODEL_PATH`
2. **CUDA内存**: 确保有足够的GPU内存
3. **配置文件**: 确保`config/config_en.json`存在
4. **输出目录**: 所有输出保存在`cookbook/output/`下

## 🎓 推荐阅读顺序

1. **README.md** (本文件) - 了解整体结构
2. **QUICKSTART.md** - 快速开始指南
3. **03_rendering_basic.py** - 最简单的示例
4. **09_end_to_end_rag.py** - 完整应用示例
5. **其他示例** - 按需学习

## 🤝 贡献

如果你想添加新的示例：
1. 遵循命名规范：`XX_description.py`
2. 添加详细注释
3. 更新本README

## 📧 获取帮助

- 查看VERA文档: `vera/USAGE.md`
- 查看实验代码: `experiments/`目录
- 提交issue到项目仓库

---

**版本**: 1.0.0
**最后更新**: 2025-01-XX
