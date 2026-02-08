# VERA 项目重构完成总结

## 概述

VERA 项目已成功重构为标准的 Python 包，现在可以像其他 Python 库一样使用 `import vera` 导入。

## 重构成果

### 1. 包结构

```
vera/                       # 核心包
├── __init__.py            # 包入口，暴露主要 API
├── models/                # 模型相关模块
│   ├── __init__.py
│   ├── base.py            # 基础引擎类
│   └── qwen.py            # Qwen 模型实现
├── rendering/             # 文本渲染模块
│   ├── __init__.py
│   └── text_to_image.py   # 文本转图像函数
├── retrieval/             # 检索模块
│   ├── __init__.py
│   ├── qwen_embedding.py  # Qwen embedding 检索
│   └── colpali.py         # ColPali 检索
└── analysis/              # 分析模块
    ├── __init__.py
    └── heatmap.py         # 热力图生成
```

### 2. 核心 API

#### Models 模块

```python
from vera import models

# 初始化引擎
engine = models.initialize(
    model_path="/path/to/Qwen3-VL-8B-Instruct",
    model_type="qwen-img"  # 或 "qwen-img-masked"
)

# 运行推理
result = engine.run(
    prompt_context="Please answer based on the images",
    question_text="What is X?",
    image_paths=["doc1.png", "doc2.png"],
    is_mask_heads=False,
    heads_positions=None  # 或 {(layer, head), ...}
)

# 返回格式
# {
#     "answer": str,
#     "input_tokens": List[str],
#     "attn_data": List,
#     "attn_error": str,
#     "input_len": int
# }
```

#### Rendering 模块

```python
from vera import rendering

# 渲染文本为图像
image_paths = rendering.text_to_image(
    text="This is the document content...",
    output_dir="/path/to/output",
    config="config/config_en.json",  # 或传入 config dict
    evidence_text=["evidence 1", "evidence 2"]  # 可选，用于高亮
)

# 返回格式
# ["/path/to/output/xxx/merged.png", ...]
```

#### Retrieval 模块

```python
from vera import retrieval

# Qwen embedding 检索
stats = retrieval.qwen_embedding(
    model_path="/path/to/Qwen3-VL-8B-Instruct",
    data_path="tem/qasper_qwen_img",
    save_dir="tem/qasper_qwen_img",
    top_k=10
)

# ColPali 检索
stats = retrieval.colpali(
    model_name="vidore/colpali-v1.2",
    data_path="tem/qasper_qwen_img",
    save_dir="tem/qasper_qwen_img",
    top_k=20
)
```

#### Analysis 模块

```python
from vera import analysis
import json

# 生成热力图
with open('attn_first_token.json') as f:
    attn_data = json.load(f)

path = analysis.create_heatmap(
    image_path="merged.png",
    attention_data=attn_data,
    output_path="heatmap.png",
    mode="overlay"  # 或 "top_k"
)

# 获取 Top K patches
patches = analysis.get_top_k_patches(
    attention_data=attn_data,
    image_height=1000,
    image_width=800,
    k=10
)
```

### 3. 更新的实验脚本

创建了两份新的实验脚本示例：

1. **experiments/qasper_qwen_img_vera.py** - 使用新 API 的标准版本
2. **experiments/qasper_qwen_img_masked_vera.py** - 使用新 API 的 masked 版本

主要变化：

| 旧 API | 新 API |
|--------|--------|
| `from models.wrapper import QwenEngine_img, InferenceConfig` | `from vera import models, rendering` |
| `config = InferenceConfig(...)` | `engine = models.initialize(...)` |
| `engine = QwenEngine_img(config)` | (使用 initialize 返回的 engine) |
| `engine.run(context, question, images)` | `engine.run(prompt_context, question_text, images, is_mask_heads, heads_positions)` |
| `process_single_text_evidence(...)` | `rendering.text_to_image(...)` |

### 4. 测试验证

所有测试已通过：

```
✅ PASS: Models Module
✅ PASS: Rendering Module
✅ PASS: Retrieval Module
✅ PASS: Analysis Module
✅ PASS: API Consistency
✅ PASS: Documentation
✅ PASS: Package Structure
```

## 用户工作区

用户只需关注以下四个文件夹：

- **experiments/** - 实验脚本（已更新示例）
- **anylasis/** - 分析脚本
- **config/** - 配置文件
- **data/** - 数据集

核心功能已封装在 `vera/` 包中，用户无需直接修改。

## 向后兼容性

原有的 `models/wrapper.py` 等文件保留在原位置，可以继续使用旧的 API。新旧 API 可以共存。

## 快速开始

1. **标准推理**：
```bash
python experiments/qasper_qwen_img_vera.py \
    --model_path /path/to/model \
    --data_path data/qasper/qasper-test-sample.json \
    --save_dir tem/qasper_qwen_img
```

2. **带 masking 的推理**：
```bash
python experiments/qasper_qwen_img_masked_vera.py \
    --model_path /path/to/model \
    --data_path data/qasper/qasper-test-sample.json \
    --save_dir tem/qasper_qwen_img_masked
```

## 总结

✅ **重构完成**：所有核心功能已成功封装到 `vera` 包中
✅ **API 统一**：提供了简洁、一致的接口
✅ **测试通过**：所有模块的测试均通过
✅ **文档完善**：所有函数都有详细的 docstring
✅ **向后兼容**：保留了原有代码，新旧 API 可共存

用户现在可以使用更简洁的 API 来完成所有任务，无需关心底层实现细节。
