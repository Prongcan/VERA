# VERA 检索模块重构总结

## 重构时间
2025-02-08

## 重构目标

将基于注意力的文本检索方法从 `analysis` 模块剥离到 `retrieval` 模块，实现更清晰的模块职责分离。

## 架构改进

### 重构前

```
vera/analysis/
└── full_analysis.py
    ├── find_word_mapping_path()        # ❌ 文本检索相关
    ├── extract_evidence_from_patches() # ❌ 文本检索相关
    └── run_full_analysis()             # ✓ 调用检索功能
```

**问题:**
- 文本检索逻辑混在分析模块中
- 职责不清晰
- 难以单独使用检索功能

### 重构后

```
vera/retrieval/
├── __init__.py                           # ✅ 导出所有检索方法
├── qwen_embedding.py                     # ✓ Qwen embedding 检索
├── colpali.py                            # ✓ ColPali 检索
├── attention.py                          # ✅ 新增：基于注意力的检索
│   ├── find_word_mapping_path()          # ✅ 查找 word_mapping.json
│   ├── extract_evidence_from_patches()   # ✅ 提取证据文本
│   └── retrieve_by_attention()           # ✅ 高层 API
└── ATTENTION_RETRIEVAL.md               # ✅ 使用文档

vera/analysis/
└── full_analysis.py
    └── run_full_analysis()               # ✓ 从 retrieval 导入并调用
```

**优势:**
- ✅ 模块职责清晰：retrieval 专注于检索，analysis 专注于分析
- ✅ 代码复用：检索功能可独立使用
- ✅ 易于维护：相关功能集中在一个模块
- ✅ 扩展性强：添加新的检索方法更容易

## 新增的 API

### 1. `retrieval.find_word_mapping_path()`

查找 word_mapping.json 文件路径。

```python
from vera import retrieval

path = retrieval.find_word_mapping_path(
    folder_path="path/to/folder",
    root_dir="tem/qasper_qwen_img"
)
```

### 2. `retrieval.extract_evidence_from_patches()`

根据 patch 边界提取证据文本。

```python
from vera import retrieval

evidence = retrieval.extract_evidence_from_patches(
    patch_bounds=[(x1, y1, x2, y2), ...],
    word_mapping_path="word_mapping.json",
    output_path="evidence.txt"
)
```

### 3. `retrieval.retrieve_by_attention()`

**新增的高层 API** - 完整的注意力检索流程。

```python
from vera import retrieval

text, patches = retrieval.retrieve_by_attention(
    attention_data=np.array([...]),
    image_height=1000,
    image_width=800,
    word_mapping_path="word_mapping.json",
    top_k=10,
    output_path="retrieved_evidence.txt"
)
```

## 文件变更

### 新增文件

1. **`vera/retrieval/attention.py`** (新文件)
   - 基于注意力的检索方法实现
   - 217 行代码
   - 包含 3 个主要函数

2. **`vera/retrieval/ATTENTION_RETRIEVAL.md`** (新文件)
   - 使用指南和示例
   - API 文档
   - 迁移指南

### 修改文件

1. **`vera/retrieval/__init__.py`**
   - 添加了新的导出函数
   - 现在 `__all__` 包含 5 个函数（原来 2 个）

   ```python
   __all__ = [
       "qwen_embedding",
       "colpali",
       "find_word_mapping_path",          # 新增
       "extract_evidence_from_patches",   # 新增
       "retrieve_by_attention"            # 新增
   ]
   ```

2. **`vera/analysis/full_analysis.py`**
   - 添加了从 `vera.retrieval` 导入这两个函数
   - 删除了原来的函数定义（约 50 行代码）

   ```python
   # 新增导入
   from vera.retrieval import find_word_mapping_path, extract_evidence_from_patches
   ```

### 删除文件

1. **`anylasis/generate_heatmaps_vera.py`**
   - 已删除
   - 功能被 `data_anylasis_vera_api.py` 完全覆盖
   - 存在已知缺陷（缺少 attention 聚合）

## 代码统计

| 指标 | 数值 |
|------|------|
| 新增文件 | 2 |
| 修改文件 | 2 |
| 删除文件 | 1 |
| 新增代码行数 | ~350 |
| 删除代码行数 | ~70 |
| 新增 API 函数 | 3 |

## 测试验证

所有测试通过：

```bash
$ python test_vera_api.py
🎉 All tests passed!
✅ PASS: Imports
✅ PASS: API Signature
✅ PASS: Engine Classes
✅ PASS: All Modules API
✅ PASS: run_full_analysis
```

新函数导入测试：

```bash
$ python -c "from vera import retrieval"
✓ retrieval.extract_evidence_from_patches: True
✓ retrieval.find_word_mapping_path: True
✓ retrieval.retrieve_by_attention: True
```

## 使用示例对比

### 旧方式（直接从 analysis 导入）

```python
# ❌ 旧方式 - 职责不清晰
from vera.analysis.full_analysis import extract_evidence_from_patches

evidence = extract_evidence_from_patches(...)
```

### 新方式（从 retrieval 导入）

```python
# ✅ 新方式 - 模块职责清晰
from vera import retrieval

# 方式 1: 分步骤
patches = analysis.get_top_k_patches(...)
evidence = retrieval.extract_evidence_from_patches(...)

# 方式 2: 使用高层 API
text, patches = retrieval.retrieve_by_attention(...)
```

## 模块职责划分

### retrieval 模块职责

专注于**检索**相关功能：
- ✅ Qwen embedding 检索
- ✅ ColPali 检索
- ✅ **基于注意力的检索**（新增）

### analysis 模块职责

专注于**分析**和**可视化**功能：
- ✅ 热力图生成
- ✅ Top-K patches 计算
- ✅ 完整的三阶段分析流程
- ✅ Attention 聚合
- ✅ 统计和评估

## 向后兼容性

✅ **完全向后兼容**

- `full_analysis.py` 内部自动从 `retrieval` 模块导入函数
- 外部使用 `run_full_analysis()` 的代码无需修改
- 原有的分析脚本继续正常工作

## 未来扩展

现在可以方便地添加新的检索方法：

```python
# vera/retrieval/dense.py
def dense_retrieval(...):
    """密集检索方法"""
    pass

# vera/retrieval/sparse.py
def sparse_retrieval(...):
    """稀疏检索方法"""
    pass

# vera/retrieval/__init__.py
from vera.retrieval.dense import dense_retrieval
from vera.retrieval.sparse import sparse_retrieval

__all__ = [
    "qwen_embedding",
    "colpali",
    "dense_retrieval",      # 新增
    "sparse_retrieval",     # 新增
    ...
]
```

## 总结

✅ **成功完成检索模块重构**

- ✅ 模块职责更清晰
- ✅ 代码组织更合理
- ✅ API 更简洁易用
- ✅ 向后兼容
- ✅ 所有测试通过
- ✅ 文档完善

现在 VERA 的检索模块包含三种检索方法：
1. **Qwen Embedding** - 语义相似度
2. **ColPali** - 多模态检索
3. **Attention-based** - 模型感知的检索 ✨ 新增
