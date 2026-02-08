# VERA 工具函数提取总结

## 概述

将 `experiments/qasper_qwen_RAG_VER_vera.py` 中的自定义函数提取到 vera 包中，使其可以被其他脚本复用。

## 提取的函数

### 1. Token 工具函数 (`vera/utils/`)

#### `get_visual_token_indices()`
- **功能**: 从 input_tokens.json 获取视觉 token 的索引范围
- **位置**: `vera/utils/token_utils.py`
- **用法**:
  ```python
  from vera.utils import get_visual_token_indices

  indices = get_visual_token_indices("input_tokens.json")
  if indices:
      start, end = indices
      print(f"Visual tokens: {start} to {end}")
  ```

#### `extract_question_from_tokens()`
- **功能**: 从 input_tokens.json 提取问题文本
- **位置**: `vera/utils/token_utils.py`
- **用法**:
  ```python
  from vera.utils import extract_question_from_tokens
  from transformers import AutoProcessor

  processor = AutoProcessor.from_pretrained("/path/to/model")
  question = extract_question_from_tokens(
      "input_tokens.json",
      processor.tokenizer
  )
  ```

### 2. Attention 分析函数 (`vera/analysis/`)

#### `aggregate_attention_with_target_heads()`
- **功能**: 使用预定义的 target heads 聚合 attention 数据
- **位置**: `vera/analysis/attention_analysis.py`
- **用法**:
  ```python
  from vera.analysis import aggregate_attention_with_target_heads

  target_heads = [(24, 29), (21, 11), (24, 8)]
  avg_attn = aggregate_attention_with_target_heads(
      attn_data=attn_data,
      visual_start=0,
      visual_end=100,
      visual_token_count=100,
      target_heads=target_heads
  )
  ```

#### `calculate_patch_distribution()`
- **功能**: 根据图片比例和 token 数量计算最合适的 patch 网格分布
- **位置**: `vera/analysis/attention_analysis.py`
- **用法**:
  ```python
  from vera.analysis import calculate_patch_distribution

  grid_w, grid_h = calculate_patch_distribution(
      img_width=1920,
      img_height=1080,
      total_visual_tokens=256
  )
  ```

#### `get_top_patches_from_attn()`
- **功能**: 从聚合的 attention 向量获取 top-k patches 的像素坐标
- **位置**: `vera/analysis/attention_analysis.py`
- **用法**:
  ```python
  from vera.analysis import get_top_patches_from_attn

  patches = get_top_patches_from_attn(
      attn_vector=avg_attn,
      grid_w=16,
      grid_h=16,
      img_width=1920,
      img_height=1080,
      top_k=10
  )
  ```

### 3. 高级检索函数 (`vera/retrieval/`)

#### `extract_top_patches_with_attention_retrieve()`
- **功能**: 完整的 top-k patches 文本提取流程
- **位置**: `vera/retrieval/attention.py`
- **用法**:
  ```python
  from vera.retrieval import extract_top_patches_with_attention_retrieve

  text, patches = extract_top_patches_with_attention_retrieve(
      attn_data=attn_data,
      visual_indices=(0, 256),
      word_mapping_path="word_mapping.json",
      grid_w=16,
      grid_h=16,
      img_width=1920,
      img_height=1080,
      top_k=10
  )
  ```

## 代码对比

### 旧代码（自定义函数）

```python
# experiments/qasper_qwen_RAG_VER_vera.py

def get_visual_token_indices(input_tokens_path):
    """自定义函数"""
    # ... 30 行代码
    pass

def aggregate_attention_with_target_heads(...):
    """自定义函数"""
    # ... 20 行代码
    pass

def calculate_patch_distribution(...):
    """自定义函数"""
    # ... 30 行代码
    pass

# 使用自定义函数
visual_indices = get_visual_token_indices(input_tokens_path)
avg_attn = aggregate_attention_with_target_heads(...)
grid_w, grid_h = calculate_patch_distribution(...)
```

### 新代码（使用 VERA API）

```python
# experiments/qasper_qwen_RAG_VER_vera_refactored.py

from vera import utils, analysis

# 直接使用 VERA API
visual_indices = utils.get_visual_token_indices(input_tokens_path)
avg_attn = analysis.aggregate_attention_with_target_heads(...)
grid_w, grid_h = analysis.calculate_patch_distribution(...)
```

## 优势

### 1. 代码复用
- 一次实现，多处使用
- 减少代码重复

### 2. 更清晰的职责划分
- **实验脚本**: 负责业务逻辑
- **VERA 库**: 提供工具函数

### 3. 更易于维护
- 修复 bug 只需要在一个地方
- 改进功能自动惠及所有使用者

### 4. 更易于测试
- 可以单独测试每个工具函数
- 提高代码质量

## 迁移指南

### 步骤 1: 导入新模块

```python
# 旧代码
# 无需导入（函数在本地）

# 新代码
from vera import utils, analysis, retrieval
```

### 步骤 2: 替换函数调用

| 旧函数 | 新函数 |
|--------|--------|
| `get_visual_token_indices(...)` | `utils.get_visual_token_indices(...)` |
| `aggregate_attention_with_target_heads(...)` | `analysis.aggregate_attention_with_target_heads(...)` |
| `calculate_patch_distribution(...)` | `analysis.calculate_patch_distribution(...)` |
| `get_top_patches_from_attn(...)` | `analysis.get_top_patches_from_attn(...)` |
| `extract_top_patches_with_retrieval_module(...)` | `retrieval.extract_top_patches_with_attention_retrieve(...)` |

### 步骤 3: 删除本地函数定义

删除实验脚本中的所有自定义函数定义，因为它们现在在 vera 包中。

## 完整示例

### 旧代码

```python
import os
import json
import cv2
import numpy as np

# 定义 5 个自定义函数（约 150 行代码）
def get_visual_token_indices(input_tokens_path):
    # ... 30 行代码
    pass

def aggregate_attention_with_target_heads(...):
    # ... 20 行代码
    pass

def calculate_patch_distribution(img_path, total_visual_tokens):
    # ... 30 行代码
    pass

def extract_top_patches_with_retrieval_module(...):
    # ... 60 行代码
    pass

def is_question_processed(...):
    # ... 10 行代码
    pass

# 使用自定义函数
visual_indices = get_visual_token_indices(input_tokens_path)
# ... 更多代码
```

### 新代码

```python
from vera import utils, analysis, retrieval

# 直接使用 VERA API，无需定义函数
visual_indices = utils.get_visual_token_indices(input_tokens_path)
grid_w, grid_h = analysis.calculate_patch_distribution(
    img_width=img_width,
    img_height=img_height,
    total_visual_tokens=visual_token_count
)

text, patches = retrieval.extract_top_patches_with_attention_retrieve(
    attn_data=attn_data,
    visual_indices=visual_indices,
    word_mapping_path=word_mapping_path,
    grid_w=grid_w,
    grid_h=grid_h,
    img_width=img_width,
    img_height=img_height,
    top_k=10
)
```

## 新增的文件

1. **`vera/utils/__init__.py`** - utils 模块初始化
2. **`vera/utils/token_utils.py`** - token 工具函数
3. **`vera/analysis/attention_analysis.py`** - attention 分析函数
4. **`experiments/qasper_qwen_RAG_VER_vera_refactored.py`** - 使用新 API 的实验脚本

## 更新的文件

1. **`vera/__init__.py`** - 导出 utils 模块
2. **`vera/analysis/__init__.py`** - 导出新的 attention 分析函数
3. **`vera/retrieval/attention.py`** - 添加高级检索函数
4. **`vera/retrieval/__init__.py`** - 导出新的高级检索函数

## 测试

运行测试脚本验证新 API:

```bash
python test_utils_api.py
```

预期输出:
```
✓ All API tests passed!
```

## 向后兼容性

- ✅ 原实验脚本仍然可用
- ✅ 新旧 API 可以共存
- ✅ 可以逐步迁移到新 API

## 总结

这次重构将约 150 行的自定义代码提取到 vera 包中，使其成为可复用的工具函数。新代码更简洁、更易维护、更易于测试。

**关键改进**:
- 减少代码重复
- 提高代码质量
- 改善可维护性
- 增强可测试性
