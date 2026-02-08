# VERA Cookbook - 快速开始指南

本指南帮助你快速运行VERA cookbook中的示例。

## 前置要求

### 1. 安装依赖

```bash
# 基础依赖
pip install -r cookbook/requirements.txt

# 可选：ColPali检索
pip install colpali-engine
```

### 2. 准备模型

某些示例需要Qwen3-VL模型。请修改示例文件中的`MODEL_PATH`变量指向你的模型路径：

```python
MODEL_PATH = "/path/to/your/Qwen3-VL-8B-Instruct"
```

### 3. 准备配置文件

确保`config/config_en.json`存在，用于文本渲染配置。

## 快速测试

### 方式1：使用运行脚本

```bash
# 查看帮助
python cookbook/run_all_examples.py --help

# 只运行文本渲染示例（不需要模型）
python cookbook/run_all_examples.py --example 03

# 运行所有示例
python cookbook/run_all_examples.py --example all

# 跳过需要模型的示例
python cookbook/run_all_examples.py --example all --skip-model
```

### 方式2：直接运行单个示例

```bash
# 文本渲染（推荐从这里开始）
python cookbook/03_rendering_basic.py

# 带evidence高亮的渲染
python cookbook/04_rendering_with_evidence.py

# 注意力检索
python cookbook/05_retrieval_attention.py

# 热力图生成
python cookbook/07_analysis_heatmap.py
```

## 示例说明

### 不需要模型的示例（推荐先运行这些）

| 示例 | 文件名 | 说明 |
|------|--------|------|
| 03 | `03_rendering_basic.py` | 文本渲染为图像 |
| 04 | `04_rendering_with_evidence.py` | 渲染并高亮evidence |
| 05 | `05_retrieval_attention.py` | 基于注意力的检索 |
| 07 | `07_analysis_heatmap.py` | 热力图生成 |
| 08 | `08_analysis_full_pipeline.py` | 完整分析流程 |

### 需要模型的示例

| 示例 | 文件名 | 说明 |
|------|--------|------|
| 01 | `01_models_basic_inference.py` | 基础模型推理 |
| 02 | `02_models_masked_inference.py` | 掩盖注意力头的推理 |
| 06 | `06_retrieval_qwen_embedding.py` | Qwen Embedding检索 |
| 09 | `09_end_to_end_rag.py` | 端到端RAG示例 |

## 输出文件

所有示例的输出都保存在 `cookbook/output/` 目录下：

```
cookbook/output/
├── 01_*/                          # 示例01的输出
│   ├── basic_inference_answer.txt
│   └── basic_inference_tokens.txt
├── 03_*/                          # 示例03的输出
│   └── *.png                      # 渲染的图像
├── 04_*/                          # 示例04的输出
│   ├── without_evidence/          # 不带高亮的图像
│   └── with_evidence/             # 带高亮的图像
└── ...
```

## 推荐的学习路径

### 初学者路径

1. **文本渲染** → 了解如何将文档转换为图像
   ```bash
   python cookbook/03_rendering_basic.py
   ```

2. **evidence高亮** → 了解如何标注重要信息
   ```bash
   python cookbook/04_rendering_with_evidence.py
   ```

3. **注意力检索** → 了解如何从注意力中提取信息
   ```bash
   python cookbook/05_retrieval_attention.py
   ```

4. **热力图可视化** → 了解注意力可视化
   ```bash
   python cookbook/07_analysis_heatmap.py
   ```

### 进阶路径

5. **完整分析流程** → 了解批量分析
   ```bash
   python cookbook/08_analysis_full_pipeline.py
   ```

6. **端到端RAG** → 了解完整的应用流程
   ```bash
   python cookbook/09_end_to_end_rag.py
   ```

## 常见问题

### Q1: 模型加载失败

**错误**: `FileNotFoundError` 或 `OSError`

**解决**: 修改示例文件中的`MODEL_PATH`变量，指向正确的模型目录。

### Q2: CUDA内存不足

**错误**: `CUDA out of memory`

**解决**:
- 使用更小的模型
- 或者只在CPU上运行（会非常慢）
- 或者减少`max_new_tokens`参数

### Q3: 配置文件找不到

**错误**: `FileNotFoundError: config/config_en.json`

**解决**:
- 确保在项目根目录下运行
- 或者修改示例中的`CONFIG_PATH`变量

### Q4: 渲染字体错误

**错误**: 字体相关的错误

**解决**:
- 检查`config/config_en.json`中的`font-path`是否正确
- 确保字体文件存在且可读

## 下一步

完成示例学习后，你可以：

1. 查看实际的实验代码：`experiments/`目录
2. 阅读分析脚本：`anylasis/`目录
3. 集成VERA到你的项目中

## 获取帮助

- 查看完整文档：`vera/USAGE.md`
- 查看示例代码：`cookbook/`目录
- 查看实验代码：`experiments/qasper_qwen_RAG_VER_vera.py`
