# VERA 检索 API 迁移指南

## 概述

VERA retrieval 模块已经重构，从"批量处理工具"转变为"检索库"。新的 API 设计更加清晰、灵活，将文件遍历逻辑移到用户代码中。

## 主要变化

### 旧 API (仍在工作，但已废弃)

```python
from vera import retrieval

# 旧 API - vera 内部遍历文件系统
stats = retrieval.colpali(
    model_name="vidore/colpali-v1.2",
    data_path="tem/qasper_qwen_img",  # 要求特定的文件夹结构
    save_dir="tem/qasper_qwen_img",
    top_k=20
)
```

**问题**：
- 要求特定的文件夹结构
- 文件遍历逻辑在库代码中
- 不够灵活

### 新 API (推荐)

```python
from vera import retrieval

# 新 API - 用户直接传入数据
context = retrieval.colpali_retrieve(
    model_name="vidore/colpali-v1.2",
    image_path="/path/to/image.png",
    word_mapping_path="/path/to/word_mapping.json",
    query="What is the main contribution?",
    top_k=20
)

# 返回：提取的文本上下文
print(context)
```

**优势**：
- 清晰的输入输出
- 不要求特定的文件夹结构
- 用户完全控制文件组织
- 易于测试和调试

## API 映射

### ColPali 检索

| 旧 API | 新 API |
|--------|--------|
| `retrieval.colpali(data_path, save_dir, ...)` | `retrieval.colpali_retrieve(image_path, word_mapping_path, query, ...)` |
| 返回 `dict` (统计信息) | 返回 `str` (提取的文本) |

#### 迁移示例

```python
# 旧代码
stats = retrieval.colpali(
    model_name="vidore/colpali-v1.2",
    data_path="tem/qasper_qwen_img",
    save_dir="tem/qasper_qwen_img",
    top_k=20
)

# 新代码
import os

# 用户负责遍历文件夹
for folder in find_folders("tem/qasper_qwen_img"):
    image_path = find_image(folder)
    word_mapping_path = find_word_mapping(folder)
    query = extract_question(f"{folder}/input_tokens.json")

    # 调用 vera API
    context = retrieval.colpali_retrieve(
        model_name="vidore/colpali-v1.2",
        image_path=image_path,
        word_mapping_path=word_mapping_path,
        query=query,
        top_k=20
    )

    # 用户负责保存结果
    with open(f"{folder}/result.txt", "w") as f:
        f.write(context)
```

### Qwen Embedding 检索

| 旧 API | 新 API |
|--------|--------|
| `retrieval.qwen_embedding(data_path, save_dir, ...)` | `retrieval.qwen_embedding_retrieve(context_text, query, ...)` |
| 返回 `dict` (统计信息) | 返回 `str` (提取的文本) |

#### 迁移示例

```python
# 旧代码
stats = retrieval.qwen_embedding(
    model_path="/path/to/Qwen3-VL-8B-Instruct",
    data_path="tem/qasper_qwen_img",
    save_dir="tem/qasper_qwen_img",
    top_k=10
)

# 新代码
import os

# 用户负责遍历文件夹
for folder in find_folders("tem/qasper_qwen_img"):
    context_path = f"{folder}/context.txt"
    query = extract_question(f"{folder}/input_tokens.json")

    # 读取上下文
    with open(context_path, "r") as f:
        context_text = f.read()

    # 调用 vera API
    result = retrieval.qwen_embedding_retrieve(
        model_path="/path/to/Qwen3-VL-8B-Instruct",
        context_text=context_text,
        query=query,
        top_k=10
    )

    # 用户负责保存结果
    with open(f"{folder}/result.txt", "w") as f:
        f.write(result)
```

### Attention 检索

| 旧 API | 新 API |
|--------|--------|
| `retrieval.retrieve_by_attention(image_height, image_width, ...)` | `retrieval.attention_retrieve(image_width, image_height, ...)` |
| 参数顺序：(height, width) | 参数顺序：(width, height) |

**注意**：新 API 的参数顺序改为 `(width, height)` 以符合常见惯例。

#### 迁移示例

```python
# 旧代码
text, patches = retrieval.retrieve_by_attention(
    attention_data=attn_data,
    image_height=1080,
    image_width=1920,
    word_mapping_path="word_mapping.json",
    top_k=10
)

# 新代码 (注意参数顺序变化)
text, patches = retrieval.attention_retrieve(
    attention_data=attn_data,
    image_width=1920,   # width 在前
    image_height=1080,  # height 在后
    word_mapping_path="word_mapping.json",
    top_k=10
)
```

### 辅助函数

以下函数**仍然可用**，无需迁移：

- `retrieval.extract_evidence_from_patches()` - 从 patch 提取文本
- `retrieval.find_word_mapping_path()` - 查找 word_mapping.json (已不推荐使用)

## 实用工具函数

### 文件查找辅助函数

如果你需要遍历文件夹，可以使用以下辅助函数：

```python
import os
from pathlib import Path

def find_image_in_folder(folder_path: str) -> str | None:
    """在文件夹中查找图像文件"""
    # Try merged.png
    merged_path = os.path.join(folder_path, "merged.png")
    if os.path.exists(merged_path):
        return merged_path

    # Try to find in subdirectories
    for root, dirs, files in os.walk(folder_path):
        if "merged_evidence.png" in files:
            return os.path.join(root, "merged_evidence.png")
        elif "merged.png" in files:
            return os.path.join(root, "merged.png")
        elif files:
            png_files = [f for f in files if f.endswith('.png')]
            if png_files:
                return os.path.join(root, png_files[0])

    return None


def find_word_mapping_in_folder(folder_path: str) -> str | None:
    """在文件夹中查找 word_mapping.json"""
    for root, dirs, files in os.walk(folder_path):
        if "word_mapping.json" in files:
            return os.path.join(root, "word_mapping.json")
    return None


def find_all_question_folders(data_path: str) -> list[str]:
    """查找所有包含问题的文件夹"""
    folders = []
    for root, dirs, files in os.walk(data_path):
        if "result" in root:
            continue
        if "input_tokens.json" in files:
            folders.append(root)
    return folders
```

### 问题提取辅助函数

```python
import json
from typing import Optional
from transformers import AutoProcessor

def extract_question_from_tokens(
    input_tokens_path: str,
    tokenizer,
    clean: bool = True
) -> Optional[str]:
    """从 input_tokens.json 提取问题"""
    if not os.path.exists(input_tokens_path) or tokenizer is None:
        return None

    try:
        with open(input_tokens_path, 'r', encoding='utf-8') as f:
            tokens = json.load(f)

        vision_end_idx = None
        for i, token in enumerate(tokens):
            if token == '<|vision_end|>':
                vision_end_idx = i
                break

        if vision_end_idx is None:
            return None

        question_tokens = []
        for i in range(vision_end_idx + 1, len(tokens)):
            token = tokens[i]
            if token == '<|im_end|>':
                break
            question_tokens.append(token)

        token_ids = tokenizer.convert_tokens_to_ids(question_tokens)
        question_text = tokenizer.decode(token_ids, skip_special_tokens=True)

        if clean:
            # 清理问题文本
            prefix_marker = "Please answer the question based on the document images provided."
            if question_text.startswith(prefix_marker):
                question_text = question_text[len(prefix_marker):].strip()
            cutoff_marker = "Please output your answer **directly**"
            if cutoff_marker in question_text:
                question_text = question_text.split(cutoff_marker)[0].strip()

        return question_text.strip()

    except Exception:
        return None
```

## 完整示例

### 使用新 API 的完整脚本

```python
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor
from vera import retrieval

# 配置
DATA_PATH = "tem/qasper_qwen_img"
MODEL_NAME = "vidore/colpali-v1.2"
QWEN_MODEL_PATH = "/path/to/Qwen3-VL-8B-Instruct"
TOP_K = 20

# 加载 Qwen tokenizer
qwen_processor = AutoProcessor.from_pretrained(
    QWEN_MODEL_PATH,
    trust_remote_code=True
)

# 查找所有问题文件夹
folders = find_all_question_folders(DATA_PATH)

# 处理每个文件夹
stats = {"total": 0, "success": 0, "failed": 0}

for folder_path in tqdm(folders):
    try:
        # 查找文件
        image_path = find_image_in_folder(folder_path)
        word_mapping_path = find_word_mapping_in_folder(folder_path)
        input_tokens_path = os.path.join(folder_path, "input_tokens.json")

        if not all([image_path, word_mapping_path]):
            stats["failed"] += 1
            continue

        # 提取问题
        query = extract_question_from_tokens(
            input_tokens_path,
            qwen_processor.tokenizer
        )

        if not query:
            stats["failed"] += 1
            continue

        # 调用新的检索 API
        context = retrieval.colpali_retrieve(
            model_name=MODEL_NAME,
            image_path=image_path,
            word_mapping_path=word_mapping_path,
            query=query,
            top_k=TOP_K
        )

        # 保存结果
        output_path = os.path.join(folder_path, "extracted_evidence.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(context)

        stats["success"] += 1

    except Exception as e:
        print(f"Error processing {folder_path}: {e}")
        stats["failed"] += 1

    stats["total"] += 1

# 打印统计
print(f"Total: {stats['total']}")
print(f"Success: {stats['success']}")
print(f"Failed: {stats['failed']}")
```

## 向后兼容性

旧的 API 函数仍然可用，但会显示 `DeprecationWarning`：

```python
import warnings

# 这会显示弃用警告
stats = retrieval.colpali(
    model_name="vidore/colpali-v1.2",
    data_path="tem/qasper_qwen_img",
    save_dir="tem/qasper_qwen_img",
    top_k=20
)
# DeprecationWarning: retrieval.colpali() is deprecated.
# Use retrieval.colpali_retrieve() instead.
```

## 迁移检查清单

- [ ] 将 `retrieval.colpali()` 替换为 `retrieval.colpali_retrieve()`
  - [ ] 在用户代码中添加文件遍历逻辑
  - [ ] 在用户代码中添加问题提取逻辑
  - [ ] 在用户代码中添加结果保存逻辑

- [ ] 将 `retrieval.qwen_embedding()` 替换为 `retrieval.qwen_embedding_retrieve()`
  - [ ] 在用户代码中添加文件遍历逻辑
  - [ ] 在用户代码中添加上下文读取逻辑
  - [ ] 在用户代码中添加结果保存逻辑

- [ ] 将 `retrieval.retrieve_by_attention()` 替换为 `retrieval.attention_retrieve()`
  - [ ] 注意参数顺序变化：(height, width) → (width, height)

- [ ] 测试迁移后的代码
- [ ] 更新文档和注释

## 常见问题

### Q: 为什么要这样重构？

A: 新的设计遵循"单一职责原则"：
- **库代码**：负责核心检索逻辑
- **用户代码**：负责文件处理、数据准备、结果保存

这样设计更清晰、更灵活、更易于测试。

### Q: 我必须立即迁移吗？

A: 不必须。旧 API 仍然可用，但建议尽快迁移到新 API。旧 API 会在未来版本中移除。

### Q: 新 API 性能如何？

A: 新 API 性能与旧 API 相同，因为底层使用相同的检索逻辑。新 API 还支持模型缓存，多次调用时性能更好。

### Q: 如何处理大批量数据？

A: 新 API 更适合大批量处理，因为：
1. 你可以并行处理多个文件
2. 你可以更好地控制错误处理
3. 你可以添加进度条和日志

```python
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_single_folder(folder_path):
    # 处理单个文件夹
    context = retrieval.colpali_retrieve(...)
    return context

# 并行处理
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(tqdm(
        executor.map(process_single_folder, folders),
        total=len(folders)
    ))
```

## 获取帮助

如果你在迁移过程中遇到问题：

1. 查看 `test_new_retrieval_api.py` 了解新 API 的使用方法
2. 查看 `experiments/` 中的示例脚本
3. 提交 issue 到 VERA 仓库

## 总结

这次重构将 VERA 从一个"批量处理工具"转变为一个"检索库"，符合用户的期望：

- **库负责**：核心检索逻辑（embedding 计算、相似度匹配、文本提取）
- **用户负责**：文件遍历、数据准备、结果保存

这样设计更清晰、更灵活、更易于使用。
