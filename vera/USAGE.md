# VERA 包使用指南

VERA (Visual Evidence Retrieval and Analysis) 是一个用于视觉文档理解和分析的 Python 包。

## 安装

```bash
cd /path/to/VERA
pip install -e .
```

## 快速开始

### 1. 模型推理

```python
from vera import models

# 初始化模型
engine = models.initialize(
    model_path="/path/to/Qwen3-VL-8B-Instruct",
    model_type="qwen-img"  # 或 "qwen-img-masked", "glm-img"
)

# 运行推理
result = engine.run(
    prompt_context="Please answer based on the document images.",
    question_text="What is the main contribution?",
    image_paths=["/path/to/image1.png", "/path/to/image2.png"],
    is_mask_heads=False,
    heads_positions=None
)

# 结果
print(f"Answer: {result['answer']}")
print(f"Input length: {result['input_len']}")
print(f"Attention data shape: {len(result['attn_data'])} layers")
```

### 2. 文本渲染

```python
from vera import rendering

# 渲染文本为图像
image_paths = rendering.text_to_image(
    text="This is the document content to render...",
    output_dir="output/images",
    config_path="config/config_en.json",
    evidence_text=["important evidence 1", "evidence 2"]  # 可选，用于高亮
)

print(f"Generated {len(image_paths)} images")
```

### 3. 检索

#### Qwen Embedding 检索

```python
from vera import retrieval

# 使用 Qwen embedding 进行检索
stats = retrieval.qwen_embedding(
    model_path="/path/to/Qwen3-VL-8B-Instruct",
    data_path="data/qasper/qasper-test-sample.json",
    save_dir="tem/qasper_qwen_img",
    top_k=10
)

print(f"检索完成: {stats['total']}")
```

#### ColPali 检索

```python
# 使用 ColPali 进行检索
stats = retrieval.colpali(
    model_name="vidore/colpali-v1.2",
    data_path="data/qasper/qasper-test-sample.json",
    save_dir="tem/qasper_qwen_img",
    top_k=20
)

print(f"检索完成: {stats['total']}")
```

### 4. 分析与可视化

#### 生成热力图

```python
from vera import analysis
import numpy as np

# 生成热力图
analysis.create_heatmap(
    image_path="/path/to/image.png",
    attention_data=np.array([...]),  # attention weights
    output_path="/path/to/heatmap.png",
    mode="overlay",  # 或 "top_k"
    alpha=0.5
)
```

#### 获取 Top-K Patches

```python
# 获取注意力最高的 patches
patches = analysis.get_top_k_patches(
    attention_data=np.array([...]),
    image_height=1000,
    image_width=800,
    k=10
)

# patches 格式: [(x_min, y_min, x_max, y_max), ...]
```

#### 完整的三阶段分析

```python
from vera import analysis

# 运行完整分析（Phase 1: 扫描, Phase 2: 可视化, Phase 3: 全局热力图）
stats = analysis.run_full_analysis(
    root_dir="tem/qasper_qwen_img",
    output_folder_name="result",
    top_k_patches=10,
    num_workers=100
)

print(f"分析完成: {stats['num_folders']} 文件夹")
```

## 命令行使用

### 使用 vera API 进行完整分析

```bash
# 基本用法
python anylasis/data_anylasis_vera_api.py \
    --root_dir tem/qasper_qwen_img \
    --output_folder result \
    --top_k 10

# 仅运行扫描阶段
python anylasis/data_anylasis_vera_api.py \
    --root_dir tem/qasper_qwen_img \
    --mode scan

# 仅运行可视化阶段
python anylasis/data_anylasis_vera_api.py \
    --root_dir tem/qasper_qwen_img \
    --mode viz
```

## API 参考

### models 模块

#### `models.initialize(model_path, model_type)`

初始化模型引擎。

**参数:**
- `model_path` (str): 模型路径
- `model_type` (str): 模型类型
  - `"qwen-img"`: Qwen 图像模型
  - `"qwen-img-masked"`: Qwen masked 版本
  - `"glm-img"`: GLM 图像模型

**返回:** Engine 对象

#### `engine.run(prompt_context, question_text, image_paths, is_mask_heads, heads_positions)`

运行推理。

**参数:**
- `prompt_context` (str): 提示上下文
- `question_text` (str): 问题文本
- `image_paths` (List[str]): 图像路径列表
- `is_mask_heads` (bool): 是否使用 attention mask
- `heads_positions` (Set[Tuple[int, int]]): 要 mask 的头位置 {(layer, head), ...}

**返回:** Dict
```python
{
    "answer": str,           # 模型答案
    "input_tokens": List[str],  # 输入 tokens
    "attn_data": List,       # attention 数据
    "attn_error": str,       # attention 错误信息
    "input_len": int         # 输入长度
}
```

### rendering 模块

#### `rendering.text_to_image(text, output_dir, config, evidence_text)`

将文本渲染为图像。

**参数:**
- `text` (str): 要渲染的文本
- `output_dir` (str): 输出目录
- `config` (str | Dict): 配置文件路径或配置字典
- `evidence_text` (List[str] | None): 证据文本列表，用于高亮

**返回:** List[str] - 生成的图像路径列表

### retrieval 模块

#### `retrieval.qwen_embedding(model_path, data_path, save_dir, top_k)`

使用 Qwen embedding 进行检索。

**参数:**
- `model_path` (str): Qwen 模型路径
- `data_path` (str): 数据集路径
- `save_dir` (str): 保存目录
- `top_k` (int): 返回 Top K 个句子

**返回:** Dict - 统计信息

#### `retrieval.colpali(model_name, data_path, save_dir, top_k, overlap, qwen_model_path)`

使用 ColPali 进行检索。

**参数:**
- `model_name` (str): HuggingFace 模型 ID
- `data_path` (str): 数据集路径
- `save_dir` (str): 保存目录
- `top_k` (int): 返回 Top K 个 patches
- `overlap` (int): patches 重叠数量
- `qwen_model_path` (str | None): Qwen 模型路径（用于验证）

**返回:** Dict - 统计信息

### analysis 模块

#### `analysis.create_heatmap(image_path, attention_data, output_path, mode, alpha, top_k)`

生成 attention 热力图。

**参数:**
- `image_path` (str): 基础图像路径
- `attention_data` (np.array): attention 数据
- `output_path` (str): 输出路径
- `mode` (str): "overlay" 或 "top_k"
- `alpha` (float): 叠加透明度 (0-1)
- `top_k` (int): Top K patches 数量（mode="top_k" 时）

**返回:** None

#### `analysis.get_top_k_patches(attention_data, image_height, image_width, k)`

获取 Top-K attention patches 的像素边界。

**参数:**
- `attention_data` (np.array): attention 数据
- `image_height` (int): 图像高度
- `image_width` (int): 图像宽度
- `k` (int): 返回 Top K 个

**返回:** List[Tuple[int, int, int, int]] - [(x_min, y_min, x_max, y_max), ...]

#### `analysis.run_full_analysis(root_dir, output_folder_name, top_k_patches, num_workers, mode, ...)`

运行完整的三阶段分析。

**参数:**
- `root_dir` (str): 根目录
- `output_folder_name` (str): 输出文件夹名称
- `top_k_patches` (int): Top K patches 数量
- `num_workers` (int): 并行工作进程数
- `mode` (str): "scan", "viz", 或 "all"
- `heatmap_color` (str): 热力图颜色
- `top_k_color` (str): Top K patches 颜色
- `kernel_size` (Tuple[int, int]): 高斯核大小
- `heatmap_alpha` (float): 热力图透明度
- `target_size` (int): 目标图像大小
- `dpi` (int): 图像 DPI

**返回:** Dict - 统计信息

## 示例

### 完整的实验流程

```python
from vera import models, rendering, retrieval, analysis

# 1. 初始化模型
engine = models.initialize(
    model_path="/path/to/Qwen3-VL-8B-Instruct",
    model_type="qwen-img"
)

# 2. 渲染文档
images = rendering.text_to_image(
    text=document_text,
    output_dir="output/images",
    config_path="config/config_en.json"
)

# 3. 运行推理
result = engine.run(
    prompt_context="Answer based on the images",
    question_text="What is the main contribution?",
    image_paths=images,
    is_mask_heads=False,
    heads_positions=None
)

# 4. 生成热力图
if result['attn_data']:
    import numpy as np
    attn_array = np.array(result['attn_data'][0][0][0])  # 第一层、第一头
    analysis.create_heatmap(
        image_path=images[0],
        attention_data=attn_array,
        output_path="output/heatmap.png",
        mode="overlay"
    )

print(f"Answer: {result['answer']}")
```

### 使用 Masked Heads

```python
from vera import models

# 初始化 masked 版本
engine = models.initialize(
    model_path="/path/to/Qwen3-VL-8B-Instruct",
    model_type="qwen-img-masked"
)

# 定义要 mask 的 heads
MASK_CONFIG = {
    (29, 24), (11, 21), (8, 24), (26, 26), (13, 24)
}

# 运行推理
result = engine.run(
    prompt_context="...",
    question_text="...",
    image_paths=["..."],
    is_mask_heads=True,
    heads_positions=MASK_CONFIG
)
```

## 迁移指南

### 从旧代码迁移

**旧代码:**
```python
from models.wrapper import QwenEngine_img, InferenceConfig

config = InferenceConfig(model_path, dataset_path, save_base_dir)
engine = QwenEngine_img(config)
result = engine.run(prompt_context, question_text, image_paths)
```

**新代码:**
```python
from vera import models

engine = models.initialize(
    model_path=model_path,
    model_type="qwen-img"
)
result = engine.run(
    prompt_context=prompt_context,
    question_text=question_text,
    image_paths=image_paths,
    is_mask_heads=False,
    heads_positions=None
)
```

## 配置文件

### 模型配置

`config/model_config.json`:
```json
{
    "qwen_model_path": "/path/to/Qwen3-VL-8B-Instruct",
    "glm_model_path": "/path/to/GLM-4V-9B"
}
```

### 渲染配置

`config/config_en.json`:
```json
{
    "font_path": "fonts/Arial.ttf",
    "font_size": 16,
    "line_spacing": 1.5,
    "margin": 50,
    "image_width": 800
}
```

## 常见问题

### Q: 如何选择模型类型?

A:
- 使用 `"qwen-img"` 进行标准推理
- 使用 `"qwen-img-masked"` 进行 attention head masking 实验
- 使用 `"glm-img"` 使用 GLM 模型

### Q: Attention 数据如何理解?

A: `attn_data` 是一个嵌套列表:
```python
attn_data = [
    # Layer 0
    [
        # Head 0
        [attention_weights, ...],
        # Head 1
        [attention_weights, ...],
        ...
    ],
    # Layer 1
    [...],
    ...
]
```

### Q: 如何处理多层 attention?

A: 使用 `aggregate_attention_data()` 函数（在 `vera/analysis/full_analysis.py` 中）来平均所有层的 attention。

## 更多信息

- 查看示例: `experiments/` 目录
- 查看分析脚本: `anylasis/` 目录
- API 文档: `vera/` 目录
