# VERA 检索 API 重构总结

## 完成时间

2025-02-08

## 目标

将 `vera/retrieval/` 模块从"批量处理工具"重构为"检索库"，使 API 更加清晰、灵活、易于使用。

## 核心原则

**职责分离**：
- **库代码**：负责核心检索逻辑
- **用户代码**：负责文件遍历、数据准备、结果保存

## 主要变化

### 1. 新增函数

#### ColPali 检索
```python
# vera/retrieval/colpali.py
def colpali_retrieve(
    model_name: str,
    image_path: str,
    word_mapping_path: str,
    query: str,
    top_k: int = 20,
    overlap: int = 0
) -> str:
    """使用 ColPali 进行单样本检索"""
```

**特点**：
- 接受直接的参数（image_path, word_mapping_path, query）
- 返回提取的文本（str）
- 支持模型缓存，提高性能

#### Qwen Embedding 检索
```python
# vera/retrieval/qwen_embedding.py
def qwen_embedding_retrieve(
    model_path: str,
    context_text: str,
    query: str,
    top_k: int = 10
) -> str:
    """使用 Qwen embedding 进行单样本检索"""
```

**特点**：
- 接受直接的参数（context_text, query）
- 返回提取的文本（str）
- 支持模型缓存，提高性能

#### Attention 检索
```python
# vera/retrieval/attention.py
def attention_retrieve(
    attention_data,
    image_width: int,
    image_height: int,
    word_mapping_path: str,
    top_k: int = 10,
    output_path: Optional[str] = None
) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    """使用注意力数据进行检索"""
```

**特点**：
- 参数顺序改为 (width, height)，符合常见惯例
- 返回 (文本, patch坐标)
- 与 `extract_evidence_from_patches` 配合使用

### 2. 废弃函数

以下函数标记为 `deprecated`，但仍然可用以保持向后兼容：

```python
# 旧 API - 仍然可用，但会显示 DeprecationWarning
retrieval.colpali(data_path, save_dir, ...)          # 使用 colpali_retrieve() 代替
retrieval.qwen_embedding(data_path, save_dir, ...)   # 使用 qwen_embedding_retrieve() 代替
retrieval.retrieve_by_attention(...)                 # 使用 attention_retrieve() 代替
```

### 3. 保留的辅助函数

以下函数仍然可用，无需迁移：

```python
retrieval.extract_evidence_from_patches(...)  # 从 patch 提取文本
```

### 4. 更新的导出

```python
# vera/retrieval/__init__.py
__all__ = [
    # 新 API (推荐)
    "colpali_retrieve",
    "qwen_embedding_retrieve",
    "attention_retrieve",

    # 旧 API (deprecated, 保持向后兼容)
    "colpali",
    "qwen_embedding",
    "retrieve_by_attention",

    # 工具函数
    "find_word_mapping_path",
    "extract_evidence_from_patches",
]
```

## 更新的文件

### 核心库文件

1. **`vera/retrieval/colpali.py`**
   - 新增 `colpali_retrieve()` 函数
   - 标记 `colpali()` 为 deprecated
   - 添加模型缓存机制

2. **`vera/retrieval/qwen_embedding.py`**
   - 新增 `qwen_embedding_retrieve()` 函数
   - 标记 `qwen_embedding()` 为 deprecated
   - 添加模型缓存机制

3. **`vera/retrieval/attention.py`**
   - 新增 `attention_retrieve()` 函数
   - 标记 `retrieve_by_attention()` 为 deprecated
   - 改进参数顺序 (width, height)

4. **`vera/retrieval/__init__.py`**
   - 导出新 API 函数
   - 更新 `__all__` 列表

### 实验脚本

5. **`experiments/calculte_colpali_embedding_retrieval.py`**
   - 使用新的 `colpali_retrieve()` API
   - 文件遍历逻辑移到用户代码中
   - 添加辅助函数（find_image_in_folder, find_word_mapping_in_folder）

6. **`experiments/calculate_qwen_embedding_retrieval.py`**
   - 使用新的 `qwen_embedding_retrieve()` API
   - 文件遍历逻辑移到用户代码中

7. **`experiments/qasper_qwen_RAG_VER_vera.py`**
   - 无需修改（已经直接使用 `extract_evidence_from_patches`）

### 新增文件

8. **`test_new_retrieval_api.py`**
   - 测试新 API 的可用性
   - 提供使用示例

9. **`vera/retrieval/MIGRATION_GUIDE.md`**
   - 详细的迁移指南
   - API 映射表
   - 完整示例

10. **`vera/retrieval/REFACTORING_SUMMARY.md`**
    - 本文件
    - 重构总结

## API 对比

### ColPali 检索

| 方面 | 旧 API | 新 API |
|------|--------|--------|
| 函数名 | `colpali()` | `colpali_retrieve()` |
| 输入 | `data_path`, `save_dir` | `image_path`, `word_mapping_path`, `query` |
| 输出 | `dict` (统计信息) | `str` (提取的文本) |
| 文件遍历 | 库内部 | 用户代码 |
| 批量处理 | 支持 | 用户自行循环 |
| 模型缓存 | 否 | 是 |

### Qwen Embedding 检索

| 方面 | 旧 API | 新 API |
|------|--------|--------|
| 函数名 | `qwen_embedding()` | `qwen_embedding_retrieve()` |
| 输入 | `data_path`, `save_dir` | `context_text`, `query` |
| 输出 | `dict` (统计信息) | `str` (提取的文本) |
| 文件遍历 | 库内部 | 用户代码 |
| 批量处理 | 支持 | 用户自行循环 |
| 模型缓存 | 否 | 是 |

### Attention 检索

| 方面 | 旧 API | 新 API |
|------|--------|--------|
| 函数名 | `retrieve_by_attention()` | `attention_retrieve()` |
| 参数顺序 | (height, width) | (width, height) |
| 输出 | (文本, patch坐标) | (文本, patch坐标) |
| 向后兼容 | - | 旧函数调用新函数 |

## 优势

### 对用户的好处

1. **清晰的输入输出**：用户知道需要提供什么，得到什么
2. **灵活的文件组织**：不要求特定的文件夹结构
3. **更好的控制**：用户可以在自己的代码中处理特殊情况
4. **易于测试**：可以轻松测试单个检索调用
5. **并行处理**：可以更容易地实现并行处理

### 对开发者的好处

1. **职责分离**：库代码负责检索逻辑，用户代码负责文件处理
2. **更简单的 API**：函数参数更少，逻辑更清晰
3. **易于维护**：不需要维护复杂的文件遍历逻辑
4. **向后兼容**：保留旧函数，标记为 deprecated

## 性能优化

新 API 引入了模型缓存机制：

```python
# 第一次调用：加载模型
context1 = retrieval.colpali_retrieve(
    model_name="vidore/colpali-v1.2",
    ...
)

# 后续调用：复用已加载的模型
context2 = retrieval.colpali_retrieve(
    model_name="vidore/colpali-v1.2",  # 相同的模型
    ...
)
```

这显著提高了批量处理的性能。

## 测试

运行测试脚本验证新 API：

```bash
python test_new_retrieval_api.py
```

预期输出：
```
============================================================
VERA Retrieval API Test Suite
Testing new API design
============================================================

✓ colpali_retrieve() is available
✓ qwen_embedding_retrieve() is available
✓ attention_retrieve() is available
✓ extract_evidence_from_patches() is still available
✓ Old API functions are still available for backward compatibility

============================================================
All API tests passed!
============================================================
```

## 向后兼容性

- ✅ 旧 API 函数仍然可用
- ✅ 旧 API 函数会显示 `DeprecationWarning`
- ✅ 现有代码可以继续工作
- ⚠️ 建议尽快迁移到新 API
- 📅 旧 API 会在未来版本中移除

## 迁移建议

1. **立即行动**：
   - 阅读迁移指南：`vera/retrieval/MIGRATION_GUIDE.md`
   - 查看示例脚本：`experiments/calculte_colpali_embedding_retrieval.py`
   - 运行测试：`python test_new_retrieval_api.py`

2. **逐步迁移**：
   - 先在新项目中使用新 API
   - 在现有项目中逐步替换
   - 充分测试后再完全切换

3. **获取帮助**：
   - 查看迁移指南中的常见问题
   - 参考完整的示例代码
   - 提交 issue 寻求帮助

## 未来计划

1. 在下个版本中：
   - 添加更多文档和示例
   - 改进错误处理
   - 添加更多的单元测试

2. 在未来版本中：
   - 完全移除旧的 API 函数
   - 添加更多检索方法
   - 优化性能

## 总结

这次重构成功地将 VERA 从一个"批量处理工具"转变为一个"检索库"，符合用户的期望：

- **库负责**：核心检索逻辑（embedding 计算、相似度匹配、文本提取）
- **用户负责**：文件遍历、数据准备、结果保存

新设计更加清晰、灵活、易于使用，同时保持了向后兼容性。

## 相关文档

- [迁移指南](MIGRATION_GUIDE.md) - 详细的迁移指南和示例
- [测试脚本](../../test_new_retrieval_api.py) - 验证新 API 的测试脚本
- [示例脚本](../../experiments/calculte_colpali_embedding_retrieval.py) - 使用新 API 的完整示例
