# 基于注意力的检索方法

VERA 的 `retrieval` 模块现在包含基于注意力的文本检索功能。

## 概述

基于注意力的检索通过分析模型的 attention weights 来定位文档中的相关区域，然后提取对应的文本内容。这种方法特别适合：

- 📄 视觉文档理解（VDU）
- 🔍 证据检索和提取
- 🎯 关键信息定位
- 📊 Attention 分析和可视化

## 核心 API

### 1. `extract_evidence_from_patches()`

根据 attention patch 的像素位置提取证据文本。

```python
from vera import retrieval
import numpy as np

# 假设已有 attention 数据和 top-k patches
patch_bounds = [(x1, y1, x2, y2), ...]  # Top-K patch 边界

# 提取证据文本
evidence_text = retrieval.extract_evidence_from_patches(
    patch_bounds=patch_bounds,
    word_mapping_path="word_mapping.json",
    output_path="extracted_evidence.txt"
)

print(f"提取的文本:\n{evidence_text}")
```

**参数:**
- `patch_bounds` (List[Tuple[int, int, int, int]]): Patch 边界列表
- `word_mapping_path` (str): word_mapping.json 文件路径
- `output_path` (str): 输出文本文件路径

**返回:**
- `str`: 提取的文本内容

### 2. `find_word_mapping_path()`

查找 word_mapping.json 文件的路径。

```python
word_mapping_path = retrieval.find_word_mapping_path(
    folder_path="path/to/question/folder",
    root_dir="tem/qasper_qwen_img"
)

if word_mapping_path:
    print(f"找到 word_mapping.json: {word_mapping_path}")
else:
    print("未找到 word_mapping.json")
```

**参数:**
- `folder_path` (str): 文件夹路径
- `root_dir` (str): 根目录

**返回:**
- `Optional[str]`: word_mapping.json 的完整路径，找不到则返回 None

### 3. `retrieve_by_attention()`

**高层 API** - 完整的注意力检索流程。

```python
from vera import retrieval
import numpy as np

# 假设已有 attention 数据
attention_data = np.array([...])  # 从模型获取的 attention weights

# 一键检索
text, patches = retrieval.retrieve_by_attention(
    attention_data=attention_data,
    image_height=1000,
    image_width=800,
    word_mapping_path="word_mapping.json",
    top_k=10,
    output_path="retrieved_evidence.txt"
)

print(f"检索到的文本:\n{text}")
print(f"Top-K patches: {patches}")
```

**参数:**
- `attention_data` (np.array): Attention 数据
- `image_height` (int): 图像高度
- `image_width` (int): 图像宽度
- `word_mapping_path` (str): word_mapping.json 文件路径
- `top_k` (int): 提取 Top K 个 patches（默认 10）
- `output_path` (Optional[str]): 输出文件路径

**返回:**
- `Tuple[str, List[Tuple[int, int, int, int]]]`: (提取的文本, patch 边界列表)

## 完整示例

### 示例 1: 从模型推理结果中检索

```python
from vera import models, retrieval, analysis

# 1. 运行模型推理
engine = models.initialize(
    model_path="/path/to/Qwen3-VL-8B-Instruct",
    model_type="qwen-img"
)

result = engine.run(
    prompt_context="Answer based on the document",
    question_text="What is the main contribution?",
    image_paths=["document.png"],
    is_mask_heads=False,
    heads_positions=None
)

# 2. 获取 attention 数据
attn_data = result['attn_data']  # 多层 attention

# 3. 聚合 attention (如果需要)
# 这里使用 analysis 模块提供的工具
from vera.analysis.full_analysis import aggregate_attention_data
import numpy as np

# 假设已有 visual token 信息
visual_start = 100  # 从 input_tokens 中获取
visual_end = 600
visual_token_count = 500

aggregated_attn = aggregate_attention_data(
    attn_data, visual_start, visual_end, visual_token_count
)

# 4. 获取 Top-K patches
img_width, img_height = 800, 1000
patches = analysis.get_top_k_patches(
    attention_data=aggregated_attn,
    image_height=img_height,
    image_width=img_width,
    k=10
)

# 5. 提取证据文本
word_mapping_path = retrieval.find_word_mapping_path(
    folder_path="path/to/question/folder",
    root_dir="tem/qasper_qwen_img"
)

if word_mapping_path:
    evidence = retrieval.extract_evidence_from_patches(
        patch_bounds=patches,
        word_mapping_path=word_mapping_path,
        output_path="evidence.txt"
    )
    print(f"检索到的证据:\n{evidence}")
```

### 示例 2: 使用高层 API 简化流程

```python
from vera import retrieval
import numpy as np

# 假设已有 attention 数据和图像信息
attention_data = np.array([...])
image_height, image_width = 1000, 800

# 一键完成检索
text, patches = retrieval.retrieve_by_attention(
    attention_data=attention_data,
    image_height=image_height,
    image_width=image_width,
    word_mapping_path="word_mapping.json",
    top_k=10
)

print(f"文本:\n{text}")
print(f"Patches: {len(patches)} 个")
```

### 示例 3: 在完整分析中使用

在 `data_anylasis_vera_api.py` 调用的 `run_full_analysis()` 中，retrieval 功能被自动调用：

```python
from vera import analysis

# 运行完整分析（包括文本检索）
stats = analysis.run_full_analysis(
    root_dir="tem/qasper_qwen_img",
    output_folder_name="result",
    top_k_patches=10
)

# 提取的证据文本会保存到:
# - {root_dir}/{output_folder_name}/extracted_evidence.txt
```

## word_mapping.json 格式

`word_mapping.json` 是渲染文本时生成的映射文件，格式如下：

```json
{
  "words": [
    {
      "word": "This",
      "bbox": [x1, y1, x2, y2],
      "line": 0
    },
    {
      "word": "is",
      "bbox": [x1, y1, x2, y2],
      "line": 0
    },
    ...
  ]
}
```

- `word`: 单词文本
- `bbox`: 边界框 [x_min, y_min, x_max, y_max]
- `line`: 行号

## 工作原理

```
Attention Weights
       ↓
Top-K Patches (像素坐标)
       ↓
与 word_mapping.json 中的 bbox 进行重叠检测
       ↓
提取重叠行的文本
       ↓
返回检索到的证据文本
```

## 与其他检索方法的对比

| 方法 | 优势 | 适用场景 |
|------|------|---------|
| **Qwen Embedding** | 语义相似度检索 | 基于语义的相关性 |
| **ColPali** | 多模态检索 | 图像-文本匹配 |
| **Attention-based** | 模型感知的位置检索 | 证据定位、可解释性 |

## 注意事项

1. **word_mapping.json 必须存在**: 此文件由文本渲染模块生成，包含单词位置信息

2. **Attention 数据格式**: 应该是视觉 token 的 attention weights，而非全部 token

3. **坐标系一致性**: 确保 attention patch 的像素坐标与 word_mapping.json 中的 bbox 坐标系一致

4. **性能考虑**: 对于大量 patches，文本提取可能需要一些时间

## API 快速参考

```python
# 方法 1: 分步骤
patches = analysis.get_top_k_patches(...)  # 获取 patches
evidence = retrieval.extract_evidence_from_patches(...)  # 提取文本

# 方法 2: 高层 API
text, patches = retrieval.retrieve_by_attention(...)  # 一步完成

# 方法 3: 完整分析
stats = analysis.run_full_analysis(...)  # 包含检索
```

## 迁移指南

如果你之前在 `anylasis` 文件夹中使用这些功能：

**旧代码:**
```python
# 在 analysis 脚本中直接调用
from vera.analysis.full_analysis import extract_evidence_from_patches
```

**新代码:**
```python
# 从 retrieval 模块导入
from vera import retrieval

evidence = retrieval.extract_evidence_from_patches(...)
```

## 更多信息

- 完整分析流程: `vera/analysis/full_analysis.py`
- 热力图生成: `vera/analysis/heatmap.py`
- 使用指南: `vera/USAGE.md`
